"""Adversarial verification of marks_liquidity P4Q_gapfilter (leakage lens). Read-only on their files."""
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from research.nightly import harness as H
from research.nightly.marks_liquidity import signals as SG
OUT = Path(__file__).resolve().parent
T0 = time.time()
def log(*a): print(f"[{time.time()-T0:6.1f}s]", *a, flush=True)

PM = SG.signals(H.load_panel("M"))
log("panel", PM.shape, PM["day"].max())
assert PM["day"].max() < pd.Timestamp("2026-01-01")
U = PM[PM["univ"]]
b = H.baseline("P4Q")
b50 = H.baseline("P4Q", cost_bps=50)

def p4like(extra=None):
    def f(x):
        c = x["cdi_bps"]; pct = c.rank(pct=True, ascending=False)
        m = (pct <= 0.3) & ~(x["resid_z"].fillna(0) <= -1.5) & (x["press_neg_30d"].fillna(0) < 1) & ~x["worstQ"] & c.notna()
        if extra is not None: m = m & extra(x)
        return m.to_numpy()
    return f

# how many P4Q names does the filter drop?
p4q = U[U["p4q"]]
drop = p4q["kf_gap_z"] < -2
log("P4Q rows", len(p4q), "dropped by gap filter", int(drop.sum()), "share", round(drop.mean(), 4),
    "kf_gap_z coverage", round(p4q["kf_gap_z"].notna().mean(), 3))
# fresh print on decision day? (gap is only non-trivial when the last print is at/near the decision day)
if "fresh" in p4q.columns:
    log("dropped rows fresh share", round(p4q.loc[drop, "fresh"].mean(), 3), "all", round(p4q["fresh"].mean(), 3))

res = {}
V = {"P4Q_rebuild": p4like(),
     "gap_z<-2 (claimed)": p4like(lambda x: ~(x["kf_gap_z"] < -2)),
     "gap_z<-1.5": p4like(lambda x: ~(x["kf_gap_z"] < -1.5)),
     "gap_z<-2.5": p4like(lambda x: ~(x["kf_gap_z"] < -2.5)),
     "gap_z<-3": p4like(lambda x: ~(x["kf_gap_z"] < -3)),
     "gap_z>+2 (mirror: skip cheap)": p4like(lambda x: ~(x["kf_gap_z"] > 2)),
     }
# lagged marks: use kf_gap_z from the previous grid day's marks (tests same-day-close dependence)
M = pd.read_pickle(ROOT / "data/history/nightly/marks_liquidity/marks.pkl")[["codigo", "day", "kf_s", "kf_sd"]]
M["codigo"] = M["codigo"].astype(str)
M = M.sort_values(["codigo", "day"])
M["kf_s_l1"] = M.groupby("codigo")["kf_s"].shift(1); M["kf_sd_l1"] = M.groupby("codigo")["kf_sd"].shift(1)
PM = PM.merge(M[["codigo", "day", "kf_s_l1", "kf_sd_l1"]], on=["codigo", "day"], how="left")
PM["gapz_l1"] = (PM["cdi_bps"] - PM["kf_s_l1"]) / np.sqrt(PM["kf_sd_l1"] ** 2 + 1)
V["gap_z_prior (fair from t-1)<-2"] = p4like(lambda x: ~(x["gapz_l1"] < -2))
for nm, f in V.items():
    res[nm] = H.backtest(f, panel=PM, name=nm)
    s = H.stats(res[nm]["daily"], bench=b["daily"])
    log(nm, {k: round(s[k], 3) for k in ("ann_excess_%", "diff_ann_%", "diff_t_nw", "diff_h1_%", "diff_h2_%")})
tab = H.compare({**res, "P4Q": b}, bench="P4Q")
print(tab.to_string())
r50 = H.backtest(V["gap_z<-2 (claimed)"], panel=PM, cost_bps=50)
s50 = H.stats(r50["daily"], bench=b50["daily"])
log("50bps diff", round(s50["diff_ann_%"], 3), round(s50["diff_t_nw"], 2))

# random-drop placebo: drop the same number of P4Q names per date at random
rng = np.random.default_rng(0)
nd = PM[PM["univ"] & PM["p4q"]].assign(d=lambda z: z["kf_gap_z"] < -2).groupby("day")["d"].sum()
pl = []
for i in range(30):
    rows = []
    for d, z in PM[PM["univ"] & PM["p4q"]].groupby("day"):
        k = int(nd.get(d, 0)); keep = z["codigo"].to_numpy()
        if k: keep = np.setdiff1d(keep, rng.choice(keep, k, replace=False))
        rows.append(pd.DataFrame({"day": d, "codigo": keep, "select": True}))
    r = H.backtest(pd.concat(rows), panel=PM)
    pl.append(H.stats(r["daily"], bench=b["daily"])["diff_ann_%"])
pl = np.array(pl)
claimed = H.stats(res["gap_z<-2 (claimed)"]["daily"], bench=b["daily"])["diff_ann_%"]
log("random-drop placebo diff mean", pl.mean().round(3), "sd", pl.std().round(3), "p95", np.quantile(pl, .95).round(3),
    "frac >= claimed", (pl >= claimed).mean())
out = {"variants": {k: {kk: (None if not np.isfinite(v) else float(v)) for kk, v in H.stats(r["daily"], bench=b["daily"]).items()
                        if isinstance(v, (float, int, np.floating))} for k, r in res.items()},
       "diff_50bps": float(s50["diff_ann_%"]), "random_drop_placebo": pl.tolist(), "claimed_rerun": float(claimed),
       "n_dropped_rows": int(drop.sum()), "share_dropped": float(drop.mean())}
json.dump(out, open(OUT / "verify_results.json", "w"), indent=1, default=str)
log("done")
