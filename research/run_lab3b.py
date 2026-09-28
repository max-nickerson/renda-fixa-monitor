"""Lab round 4b — follow-ups on round-4 findings:
  1. 'Hold everything but avoid' — universe excluding issuers whose stock fell ≥15% in 4 weeks or that were
     downgraded in the last 180 days (the two signals with strong, robust post-event drift).
  2. P4 + only the robust filters (stock + rating).
  3. Regulator filter placebo: shift regulator events in time by random offsets. If P4 + regulator filter still
     'works' with shifted news, the gain is a sector tilt, not information.
"""
import sys as _sys
from pathlib import Path as _P
_sys.path.insert(0, str(_P(__file__).resolve().parent.parent))  # run from anywhere

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from rfmonitor.config import DATA_DIR
from rfmonitor.ml import features_ext as fx, lab, lab_models as lm
from rfmonitor.ml.lab import Rule

warnings.filterwarnings("ignore")
OUT = Path(__file__).parent / "out"
t0 = time.time()
log = lambda *a: print(f"[{time.time() - t0:5.0f}s]", *a, flush=True)

g = lm.prepare(pd.read_pickle(DATA_DIR / "history" / "lab_weekly.pkl"), "ret_hedged")
pw = pd.read_pickle(DATA_DIR / "history" / "press_weekly.pkl")
g = g.merge(pw, on=["cnpj8", "week"], how="left")
for c in ["press_neg_7d", "press_neg_30d", "press_neg_90d", "press_spike"]:
    g[c] = g[c].fillna(0.0)
g["cdi_pct"] = lm.weekly_pct(g, "cdi_bps")
keys = g[["cnpj8", "week"]].copy()
g = pd.concat([g, fx.all_blocks(keys)], axis=1)
log("features ready")

nz = lambda x, d=0.0: d if x is None or x != x else x
rich = lambda r: nz(r["resid_z"]) <= -1.5
p4_in = lambda r: nz(r["cdi_pct"], 1) <= 0.3 and not rich(r) and r["press_neg_30d"] < 1
eq_drop = lambda r: nz(r["eq_ret_4w"]) <= -0.15
downgraded = lambda r: nz(r["rat_days_since_down"], 9999) <= 180
avoid = lambda r: eq_drop(r) or downgraded(r)
REGIME = (g.groupby("week")["mkt_mom_21"].first() > 0).astype(float)

RULES = [
    ("U0 Universo", Rule("U0", lambda r: True, lambda r, st: False), None),
    ("U7 Universo evitando queda da ação ≥15% ou rebaixamento ≤180d (entrada)",
     Rule("U7", lambda r: not avoid(r), lambda r, st: False), None),
    ("U8 Universo: não entra E sai com queda da ação/rebaixamento",
     Rule("U8", lambda r: not avoid(r), lambda r, st: avoid(r), 13), None),
    ("U9 U8 + regime momentum", Rule("U9", lambda r: not avoid(r), lambda r, st: avoid(r), 13), REGIME),
    ("P4 (atual)", Rule("P4", p4_in, lambda r, st: rich(r)), REGIME),
    ("P5 P4 + filtros robustos (ação + rating)", Rule("P5", lambda r: p4_in(r) and not avoid(r), lambda r, st: rich(r)), REGIME),
    ("P6 P5 + saída ação/rating", Rule("P6", lambda r: p4_in(r) and not avoid(r), lambda r, st: rich(r) or avoid(r), 13), REGIME),
]
res, curves, bench = {}, {}, {}
for label, rule, ov in RULES:
    row = {}
    for lag, cost in ((1, 25), (2, 25), (1, 50)):
        r = lab.run_rule(g, rule, lag=lag, cost_bps=cost, overlay=ov)
        if label.startswith("U0"):
            bench[(lag, cost)] = r["weekly"]
        st = lab.stats(r["weekly"], bench[(lag, cost)])
        row[f"lag{lag}_{cost}"] = {**st, "avg_bonds": r["avg_n"]}
        if (lag, cost) == (1, 25):
            curves[label] = r["weekly"]
            for sn, (a, b) in {"2022–23": ("2022", "2023"), "2024": ("2024", "2024"), "2025–26": ("2025", "2026")}.items():
                w = r["weekly"][a:b]
                row[f"sub_{sn}"] = (w - bench[(1, 25)].reindex(w.index).fillna(0)).mean() * 5200
    res[label] = row
    a = row["lag1_25"]
    log(f"{label:<70} vsU0 {a['vs_bench_ann_%']:6.2f}% | lag2 {row['lag2_25']['vs_bench_ann_%']:6.2f}% | 50bps "
        f"{row['lag1_50']['vs_bench_ann_%']:6.2f}% | SR {a['sharpe']:5.2f} DD {a['max_dd_%']:6.2f}% n {a['avg_bonds']:4.0f}")

# paired t vs P4 for P5/P6 and vs U0 for U7–U9
def paired(a, b):
    d = (curves[a] - curves[b].reindex(curves[a].index).fillna(0)).dropna()
    return {"diff_ann_%": d.mean() * 5200, "t": d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))}
pt = {k: paired(k, "P4 (atual)") for k in curves if k.startswith(("P5", "P6"))}
pt.update({k: paired(k, "U0 Universo") for k in curves if k.startswith(("U7", "U8", "U9"))})
log("paired: " + json.dumps({k: {kk: round(vv, 2) for kk, vv in v.items()} for k, v in pt.items()}, ensure_ascii=False))

# regulator placebo: shift events by random ±(13..52) weeks, recompute features, rerun P4 + regulator filter
reg_rule = Rule("P4reg", lambda r: p4_in(r) and not (nz(r["reg_neg_30d"]) >= 1), lambda r, st: rich(r))
real = lab.stats(lab.run_rule(g, reg_rule, overlay=REGIME)["weekly"], bench[(1, 25)])["vs_bench_ann_%"]
ev = pd.read_pickle(DATA_DIR / "history" / "regulator_events.pkl")
rng = np.random.default_rng(11)
shifted = []
orig = fx._load
for k in range(8):
    off = int(rng.choice([-1, 1]) * rng.integers(13, 53))
    ev_s = ev.assign(date=pd.to_datetime(ev["date"]) + pd.Timedelta(weeks=off))
    fx._load = lambda name, ev_s=ev_s: ev_s if name == "regulator_events.pkl" else orig(name)
    gp = g.copy()
    gp[fx.REG_FEATURES] = fx.regulator_features(keys).to_numpy()
    shifted.append(lab.stats(lab.run_rule(gp, reg_rule, overlay=REGIME)["weekly"], bench[(1, 25)])["vs_bench_ann_%"])
    log(f"reg placebo shift {off:+d}w → vsU0 {shifted[-1]:.2f}%")
fx._load = orig
plac = {"real": real, "shifted": shifted, "pctile": float((np.array(shifted) < real).mean() * 100)}
log(f"regulator filter real {real:.2f}% vs shifted mean {np.mean(shifted):.2f}% (beats {plac['pctile']:.0f}%)")

json.dump({"strategies": res, "paired": pt, "regulator_placebo": plac}, open(OUT / "lab3b_results.json", "w",
          encoding="utf-8"), indent=1, default=float, ensure_ascii=False)
pd.DataFrame(curves).to_pickle(DATA_DIR / "history" / "lab3b_curves.pkl")
log("done")
