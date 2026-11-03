"""PV design + hourly simulation API.  Run: uvicorn pv_api:app --reload
Deps: pip install fastapi uvicorn pvlib pandas
"""
from __future__ import annotations

import calendar
from functools import lru_cache
from math import ceil, pi, sin, sqrt
from pathlib import Path
from typing import Literal, Optional

import pvlib
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, model_validator

app = FastAPI(title="PV Simulation API")

TARIFFS_AS_OF = "unset - update before relying on results"
DU_RATES = {
    "meralco": {"retail": 14.50, "blended_export": 6.00},
    "veco": {"retail": 12.80, "blended_export": 5.50},
    "dlpc": {"retail": 11.50, "blended_export": 5.20},
    "generic_coop": {"retail": 13.00, "blended_export": 5.80},
}
DAYS = [calendar.monthrange(2023, m)[1] for m in range(1, 13)]
MONTHS = [calendar.month_abbr[m] for m in range(1, 13)]


def _norm(w):
    t = sum(w)
    return [x / t for x in w]


PROFILES = {  # share of daily kWh by hour (0-23)
    "aircon_day": _norm([2, 2, 2, 2, 2, 2, 3, 4, 5, 6, 7, 8, 8, 8, 8, 8, 7, 6, 6, 6, 5, 4, 3, 2]),
    "evening_peak": _norm([3, 3, 3, 3, 3, 3, 4, 5, 4, 4, 4, 4, 5, 5, 5, 5, 5, 6, 8, 9, 9, 8, 6, 4]),
    "flat": _norm([1] * 24),
}
SUN = range(7, 18)  # hours counted as "daytime"


class Appliance(BaseModel):
    name: str = Field("", max_length=60)
    watts: float = Field(gt=0, le=20000, description="running watts per unit")
    qty: int = Field(1, ge=1, le=50)
    hours: float = Field(gt=0, le=24, description="hours per day")
    start: int = Field(0, ge=0, le=23, description="hour of day it starts (wraps past midnight)")
    surge: float = Field(1, ge=1, le=10, description="start-up current multiple")


def appliance_profile(apps, daily):
    """Hourly shape from an appliance list. Unlisted load is a flat baseline so the total matches the bill."""
    e, pk, extra = [0.0] * 24, [0.0] * 24, 0.0
    for a in apps:
        kw = a.watts * a.qty / 1000
        extra = max(extra, a.watts * (a.surge - 1) / 1000)  # biggest single-unit start surge
        for h in range(ceil(a.hours)):
            i = (a.start + h) % 24
            e[i] += kw * min(1.0, a.hours - h)
            pk[i] += kw
    tot = sum(e)
    base = max(0.0, daily - tot)
    prof = _norm([x + base / 24 for x in e])
    return prof, max(x + base / 24 for x in pk), extra, tot, base


class SimRequest(BaseModel):
    topology: Literal["grid-tied", "hybrid-no-export", "hybrid-export", "off-grid"] = "hybrid-no-export"
    du_provider: Literal["meralco", "veco", "dlpc", "generic_coop"] = "meralco"
    load_profile: Literal["aircon_day", "evening_peak", "flat"] = "aircon_day"
    appliances: list[Appliance] = Field(default_factory=list, max_length=30, description="optional; overrides load_profile")
    monthly_bill: Optional[float] = Field(None, gt=0)
    monthly_kwh: Optional[float] = Field(None, gt=0)
    lat: float = Field(14.5995, ge=-90, le=90)
    lon: float = Field(120.9842, ge=-180, le=180)
    tilt: float = Field(15, ge=0, le=90)
    azimuth: float = Field(180, ge=0, le=360)
    target_offset: float = Field(0.8, gt=0, le=1.2)
    system_kw: Optional[float] = Field(None, gt=0)
    inverter_kw: Optional[float] = Field(None, gt=0, le=200, description="override the auto-sized inverter")
    panel_watts: int = Field(450, ge=100, le=800)
    battery_kwh: Optional[float] = Field(None, ge=0, description="blank = auto-size")
    autonomy_days: float = Field(1, ge=0.5, le=5)
    battery_roundtrip_eff: float = Field(0.90, gt=0, le=1)
    battery_cutoff_pct: float = Field(20, ge=0, lt=80, description="discharge stops at this % state of charge")
    retail_rate: Optional[float] = Field(None, gt=0)
    export_rate: Optional[float] = Field(None, ge=0)
    cost_per_kw_php: float = Field(45_000, gt=0)
    battery_cost_per_kwh_php: float = Field(30_000, ge=0)
    degradation_pct: float = Field(0.5, ge=0, le=5)
    tariff_escalation_pct: float = Field(3.0, ge=0, le=15)
    inverter_replacement_pct: float = Field(8.0, ge=0, le=50)
    years: int = Field(25, ge=1, le=40)

    @model_validator(mode="after")
    def need_consumption(self):
        if self.monthly_bill is None and self.monthly_kwh is None:
            raise ValueError("Provide monthly_kwh or monthly_bill")
        return self


@lru_cache(maxsize=64)
def _pvgis_hourly(lat, lon, tilt, azimuth) -> tuple:
    """Typical year: 8760 values of kWh per kWp per hour (local solar time), 14% losses."""
    import pandas as pd
    res = pvlib.iotools.get_pvgis_hourly(
        lat, lon, start=2015, end=2020, raddatabase="PVGIS-ERA5", pvcalculation=True,
        peakpower=1, loss=14, surface_tilt=tilt, surface_azimuth=azimuth, timeout=30)
    p = res[0]["P"] / 1000
    idx = p.index.tz_localize(None) + pd.Timedelta(hours=round(lon / 15))
    g = p.groupby([idx.month, idx.day, idx.hour]).mean()
    lv = g.index.get_level_values
    g = g[~((lv(0) == 2) & (lv(1) == 29))]
    if len(g) != 8760:
        raise ValueError("unexpected PVGIS length")
    return tuple(float(x) for x in g.tolist())


def hourly_yield(req):
    key = (round(req.lat, 2), round(req.lon, 2), round(req.tilt), round(req.azimuth))
    try:
        return list(_pvgis_hourly(*key)), "PVGIS-ERA5 hourly (2015-2020 typical year)"
    except Exception:
        shape = [max(0.0, sin(pi * (h - 6) / 12)) for h in range(24)]
        k = 4.5 * 0.86 / sum(shape)
        return [s * k for _ in range(365) for s in shape], "FALLBACK synthetic 4.5 PSH day (PVGIS unavailable)"


def dispatch(pv, kw, monthly_kwh, prof, bat_kwh, rt, usable, power_kw, grid, cut=0.0):
    e, cap = sqrt(rt), bat_kwh * usable
    soc = cap * 0.5
    daily = monthly_kwh * 12 / 365
    keys = ("gen", "direct", "discharge", "export", "curtail", "grid_import", "unserved")
    mon = [dict.fromkeys(keys, 0.0) for _ in range(12)]
    day = [{k: [0.0] * 24 for k in ("pv", "load", "soc", "grid")} for _ in range(12)]
    i = 0
    for m, days in enumerate(DAYS):
        for _ in range(days):
            for h in range(24):
                g, l = pv[i] * kw, daily * prof[h]
                i += 1
                r = mon[m]
                r["gen"] += g
                d = min(g, l)
                g, l = g - d, l - d
                r["direct"] += d
                dd, dis = day[m], 0.0
                dd["pv"][h] += (g + d)
                dd["load"][h] += (l + d)
                if cap > 0:
                    if g > 0:
                        c = min(g, power_kw, (cap - soc) / e)
                        soc += c * e
                        g -= c
                    if l > 0:
                        dis = min(l, power_kw, soc * e)
                        soc -= dis / e
                        l -= dis
                r["discharge"] += dis
                r["export"] += g if grid == "export" else 0.0
                r["curtail"] += g if grid != "export" else 0.0
                if grid == "offgrid":
                    r["unserved"] += l
                else:
                    r["grid_import"] += l
                    dd["grid"][h] += l
                dd["soc"][h] += (cut + soc / cap * (100 - cut)) if cap else 0.0
        for k in ("pv", "load", "soc", "grid"):
            day[m][k] = [round(v / days, 3) for v in day[m][k]]
    return mon, day


def cash_flow(y1, pv_cost, total, req):
    deg, esc, cum, pay, out = req.degradation_pct / 100, req.tariff_escalation_pct / 100, -total, None, []
    for y in range(1, req.years + 1):
        gross = y1 * (1 - deg) ** (y - 1) * (1 + esc) ** (y - 1)
        capex = pv_cost * req.inverter_replacement_pct / 100 if y == 12 else 0.0
        net, prev = gross - capex, cum
        cum += net
        if pay is None and cum >= 0:
            pay = (y - 1) + (-prev / net if net > 0 else 1)
        out.append({"year": y, "savings_php": round(gross), "replacement_php": round(capex), "cumulative_php": round(cum)})
    return out, pay


@app.post("/api/simulate")
def simulate(req: SimRequest):
    rates = DU_RATES[req.du_provider]
    retail = req.retail_rate or rates["retail"]
    exp_rate = req.export_rate if req.export_rate is not None else rates["blended_export"]
    usable = 1 - req.battery_cutoff_pct / 100
    grid = {"grid-tied": "export", "hybrid-export": "export", "hybrid-no-export": "zero", "off-grid": "offgrid"}[req.topology]
    notes = [f"Tariffs as of: {TARIFFS_AS_OF}"]

    monthly_kwh = req.monthly_kwh
    if monthly_kwh is None:
        monthly_kwh = req.monthly_bill / retail
        notes.append("kWh estimated as bill / retail rate; prefer monthly_kwh")
    annual = monthly_kwh * 12
    daily = annual / 365
    peak_run = surge_extra = None
    if req.appliances:
        prof, peak_run, surge_extra, app_kwh, base = appliance_profile(req.appliances, daily)
        notes.append(f"Appliance list covers {app_kwh:.1f} kWh/day ({app_kwh / daily * 100:.0f}% of your {daily:.1f} kWh/day); the rest is a flat baseline of {base:.1f} kWh/day")
        if app_kwh > daily * 1.05:
            notes.append("Appliance list exceeds your monthly kWh: hours or watts are probably overestimated. Energy was scaled down to match your bill, but the inverter is sized from the listed loads")
    else:
        prof = PROFILES[req.load_profile]
    day_frac = sum(prof[h] for h in SUN)

    pv, src = hourly_yield(req)
    spec_yield = sum(pv)

    # battery
    has_bat = req.topology != "grid-tied"
    if not has_bat:
        bat = 0.0
    elif req.battery_kwh is not None:
        bat = req.battery_kwh
    elif req.topology == "off-grid":
        bat = daily * req.autonomy_days / usable
    else:
        bat = daily * (1 - day_frac) / (usable * req.battery_roundtrip_eff)
    bat = round(bat * 2) / 2 if bat else 0.0

    # PV size (whole panels)
    if req.system_kw is None:
        basis = annual * (1.0 if bat > 0 else day_frac)
        raw = basis * req.target_offset / spec_yield
    else:
        raw = req.system_kw
    panels = max(1, ceil(raw * 1000 / req.panel_watts))
    kw = panels * req.panel_watts / 1000

    # inverter: surge-aware (1.5x peak hour) but never below array/1.25
    if peak_run is not None:
        inv_kw = max(1, ceil(max((peak_run + surge_extra) * 1.1, kw / 1.25)))
        notes.append(f"Inverter sized for peak running load {peak_run:.1f} kW + largest start surge {surge_extra:.1f} kW (+10% margin)")
    else:
        inv_kw = max(1, ceil(max(daily * max(prof) * 1.5, kw / 1.25)))
    if req.inverter_kw:
        inv_kw = round(req.inverter_kw, 2)
        notes.append(f"Inverter size set by method/override: {inv_kw} kW")

    mon, day = dispatch(pv, kw, monthly_kwh, prof, bat, req.battery_roundtrip_eff,
                        usable, inv_kw, grid, req.battery_cutoff_pct)
    rows = []
    for m, r in enumerate(mon):
        sav = (r["direct"] + r["discharge"]) * retail + r["export"] * exp_rate
        rows.append({"month": MONTHS[m], **{k: round(v, 1) for k, v in r.items()}, "savings_php": round(sav, 2)})
    y1 = sum(r["savings_php"] for r in rows)
    gen = sum(r["gen"] for r in rows)
    onsite = sum(r["direct"] + r["discharge"] for r in rows)

    pv_cost = kw * req.cost_per_kw_php
    total = pv_cost + bat * req.battery_cost_per_kwh_php
    flows, pay = cash_flow(y1, pv_cost, total, req)
    if pay is None:
        notes.append(f"No payback within {req.years} years under these assumptions")
    if bat:
        notes.append(f"Battery discharge stops at {req.battery_cutoff_pct:g}% state of charge ({100 - req.battery_cutoff_pct:g}% of nominal kWh is usable)")
        notes.append("Battery replacement is not modeled")
    if req.topology == "off-grid":
        notes.append("Off-grid: savings = avoided grid energy; unserved load needs a generator or a larger system")
    notes.append(f"Load profile: {'custom (appliance list)' if req.appliances else req.load_profile}, {round(day_frac * 100)}% of use in daylight hours; flat across months")

    return {
        "summary": {
            "topology": req.topology, "system_kw": round(kw, 2), "panel_count": panels, "panel_watts": req.panel_watts,
            "battery_kwh": bat, "battery_cutoff_pct": req.battery_cutoff_pct, "peak_running_kw": round(peak_run, 2) if peak_run is not None else None, "inverter_kw": inv_kw, "specific_yield_kwh_per_kwp": round(spec_yield),
            "annual_generation_kwh": round(gen), "solar_coverage_pct": round(100 * onsite / annual, 1),
            "self_consumption_pct": round(100 * onsite / gen, 1) if gen else 0,
            "unserved_kwh": round(sum(r["unserved"] for r in rows)),
            "year1_savings_php": round(y1), "pv_cost_php": round(pv_cost),
            "battery_cost_php": round(bat * req.battery_cost_per_kwh_php), "total_cost_php": round(total),
            "payback_years": round(pay, 1) if pay is not None else None,
            "net_gain_over_horizon_php": flows[-1]["cumulative_php"],
        },
        "monthly": rows, "typical_day": day, "cash_flow": flows,
        "assumptions": {"monthly_kwh": round(monthly_kwh, 1), "retail_rate": retail, "export_rate": exp_rate,
                        "yield_source": src, "notes": notes},
    }


_static = Path(__file__).parent / "static"
if _static.is_dir():  # keep LAST so it doesn't shadow /api
    app.mount("/", StaticFiles(directory=_static, html=True), name="static")
