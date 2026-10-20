"""Deterministic risk engine. All weights/tables are assumptions to tune (see literature: FMEA/RPN, NREL degradation ~0.5%/yr)."""
from datetime import date
CW = {"clean": 0, "dust": .4, "bird_droppings": .5, "shading": .5, "snow": .3, "physical_damage": .9, "electrical_damage": 1.0}
SEV = {"low": .33, "medium": .66, "high": 1.0}
SOILING = {"dust", "bird_droppings", "shading", "snow"}
LOSS = {True: {"low": 2, "medium": 8, "high": 20}, False: {"low": 5, "medium": 15, "high": 35}}  # % output loss
W = {True: dict(severity=.30, overdue=.25, impact=.20, recurrence=.15, age=.10),
     False: dict(severity=.40, age=.20, impact=.20, recurrence=.15, overdue=.05)}
PSH, LIFE, INTERVAL, REF = 5.0, 25, 60, 20.0  # sun hours/day, design life (yrs), cleaning interval (days), ref loss kWh/day
LEVELS = ["low", "medium", "high", "critical"]

def level(s): return LEVELS[0 if s < 25 else 1 if s < 50 else 2 if s < 75 else 3]

def risk_score(asset, obs, n_similar=0, today=None):
    today = today or date.today()
    cls, sev = obs["issue_class"], obs["severity"]
    if cls == "clean":
        return dict(score=0, level="low", completeness=1.0, missing=[], factor_contributions={}, energy_loss_kwh_day=0.0, safety_override=False)
    soil = cls in SOILING
    cap, ins, lc = asset.get("capacity_kw"), asset.get("install_date"), asset.get("last_cleaned_date")
    loss = cap * PSH * LOSS[soil][sev] / 100 if cap else None
    f = {"severity": CW[cls] * SEV[sev],
         "age": min((today - ins).days / 365.25 / LIFE, 1) if ins else None,
         "overdue": min((today - lc).days / INTERVAL / 2, 1) if lc else None,
         "recurrence": min(n_similar / 3, 1),
         "impact": min(loss / REF, 1) if loss is not None else None}
    w = W[soil]
    av = {k: v for k, v in f.items() if v is not None}
    ws = sum(w[k] for k in av)  # renormalise over available factors
    s = sum(w[k] * v for k, v in av.items()) / ws * 100
    lv, ov = level(s), cls == "electrical_damage"
    if ov and LEVELS.index(lv) < 2: lv = "high"  # safety override
    return dict(score=round(s), level=lv, completeness=round(ws / sum(w.values()), 2),
                missing=[k for k, v in f.items() if v is None],
                factor_contributions={k: round(w[k] * v / ws * 100, 1) for k, v in av.items()},
                energy_loss_kwh_day=round(loss, 1) if loss is not None else None, safety_override=ov)
