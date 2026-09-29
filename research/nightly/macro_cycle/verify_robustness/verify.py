"""Adversarial robustness check of macro_cycle's best variant ov_mom63_inout (pre-2026 only, holdout untouched
except re-reading their reported number)."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[4]; sys.path.insert(0, str(ROOT))
from research.nightly import harness as H
from research.nightly.macro_cycle import run as R
OUT = Path(__file__).resolve().parent
F = pd.read_pickle(ROOT / "data/history/nightly/macro_cycle/macro_daily_used.pkl")
b25 = H.baseline("P4Q"); b50 = H.baseline("P4Q", cost_bps=50)
bd = b25["daily"]
res = {}
def ov_from(x, thr=0.0, weekly=True):
    e = (x > thr).astype(float).where(x.notna())
    return R.weekly_hold(e) if weekly else e
def st(r, bench=bd):
    s = H.stats(r["daily"], bench=bench)
    return {k: s[k] for k in ["ann_excess_%", "vol_%", "max_dd_%", "diff_ann_%", "diff_t_nw", "diff_h1_%", "diff_h2_%"]}
# 1) reproduction
ov63 = ov_from(F["IDADI_x63"])
rep = R.apply_overlay(b25, ov63)
res["repro_ov_mom63_inout_25"] = st(rep)
res["repro_50bps"] = st(R.apply_overlay(b50, ov63, cost_bps=50), b50["daily"])
res["repro_fund100"] = st(R.apply_overlay(b25, ov63, fund_bps=100))
# 2) window/threshold/frequency perturbations (IDA-DI excess momentum computed from x1)
rx = F["IDADI_rx1"]
grid = {}
for w in [21, 42, 63, 84, 126]:
    x = rx.rolling(w).sum()
    for thr in [-0.001, 0.0, 0.001]:
        for wk in [True, False]:
            k = f"w{w}_thr{thr}_{'wk' if wk else 'daily'}"
            grid[k] = st(R.apply_overlay(b25, ov_from(x, thr, wk)))["diff_ann_%"]
res["perturb_grid_diff_vs_p4q"] = grid
for lag in [1, 5, 10]:
    res[f"extra_lag{lag}"] = st(R.apply_overlay(b25, ov63, extra_lag=lag))
# 3) excluding Americanas crisis months (Jan-Mar 2023) and all 2023
o = ov63.reindex(ov63.index.union(H.days())).ffill().reindex(H.days())
diff_m = H.monthly(rep["daily"][rep["daily"].index < H.HOLDOUT]) - H.monthly(bd.reindex(rep["daily"].index).fillna(0)[lambda s: s.index < H.HOLDOUT])
diff_m = diff_m.dropna()
res["monthly_diff_top5_%"] = (diff_m.sort_values(ascending=False).head(5) * 100).round(3).rename(lambda d: str(d.date())).to_dict()
res["monthly_diff_bottom5_%"] = (diff_m.sort_values().head(5) * 100).round(3).rename(lambda d: str(d.date())).to_dict()
def ann_t(d):
    return {"ann_%": round(d.mean() * 1200, 3), "t_nw": round(H.nw_t(d, 6), 2), "n": int(len(d))}
res["diff_all"] = ann_t(diff_m)
res["diff_ex_JanMar2023"] = ann_t(diff_m[~((diff_m.index >= "2023-01-01") & (diff_m.index < "2023-04-01"))])
res["diff_ex_2023"] = ann_t(diff_m[diff_m.index.year != 2023])
res["diff_ex_top5_months"] = ann_t(diff_m.drop(diff_m.sort_values(ascending=False).head(5).index))
# 4) placebo: circular shift of mom63 exposure path (they did not run a placebo for mom63)
rng = np.random.default_rng(1)
win = o[(o.index >= "2021-06-01") & (o.index < H.HOLDOUT)].dropna()
bb = bd[bd.index < H.HOLDOUT]
def fast(net):
    d = net.reindex(bb.index) - bb
    return float(d.groupby([bb.index.year, bb.index.month]).sum().mean() * 1200)
act = fast(rep["daily"]); vals = []
for _ in range(200):
    sh = pd.Series(np.roll(win.to_numpy(), rng.integers(60, len(win) - 60)), index=win.index)
    vals.append(fast(R.apply_overlay(b25, sh)["daily"]))
vals = np.array(vals)
res["placebo_circshift"] = {"actual": round(act, 3), "mean": round(vals.mean(), 3), "p95": round(float(np.quantile(vals, .95)), 3),
                            "p": round(float((vals >= act).mean()), 3)}
# 5) switch count & avg exposure
oo = o[(o.index >= "2022-01-01") & (o.index < H.HOLDOUT)]
res["avg_expo_2022_25"] = round(float(oo.mean()), 3); res["switches_2022_25"] = int((oo.diff().abs() > 0).sum())
# 6) exposure-matched: static book at avg exposure
a = float(oo.mean())
res["static_matched_diff"] = st(R.apply_overlay(b25, pd.Series(a, index=H.days())))["diff_ann_%"]
# 7) multiple-testing: rank of mom63 among grid
g = pd.Series(grid)
res["grid_summary"] = {"n": len(g), "mean": round(g.mean(), 3), "median": round(g.median(), 3), "min": round(g.min(), 3),
                       "max": round(g.max(), 3), "frac_positive": round(float((g > 0).mean()), 3)}
(OUT / "results.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
print(json.dumps(res, indent=1, default=str))
