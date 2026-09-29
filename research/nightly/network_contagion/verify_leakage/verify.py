"""Adversarial leakage check of network_contagion's best filter (pre-2026 only)."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))
import run as R, features as FT
from research.nightly import harness as H

PM = R.load("M")
V = R.variants()
F = FT.build()
out = {}
# 1) reproduce
runs = {"best_repro": H.backtest(V["avoid_exposure_stress_q5"], panel=PM, name="best")}
# 2) lagged features: use features from the previous W/F date (>= 1 week stale)
fd = np.sort(F["day"].unique())
prev = {d: fd[i - 1] for i, d in enumerate(fd) if i > 0}
Fl = F[["day", "codigo", "i_exp_r4"]].copy()
nxt = {v: k for k, v in prev.items()}
Fl["day"] = Fl["day"].map(nxt)
Fl = Fl.dropna(subset=["day"]).rename(columns={"i_exp_r4": "lag_exp_r4"})
PM2 = PM.merge(Fl, on=["day", "codigo"], how="left")
PM2["s_lag"] = -PM2["lag_exp_r4"]
runs["lag1period"] = H.backtest(lambda x: (x["p4q"] & ~R.q_by_day(x, "s_lag", 0.8)).to_numpy(), panel=PM2, name="lag")
# 3) threshold sensitivity
for q in (0.7, 0.9):
    runs[f"q{q}"] = H.backtest(lambda x, q=q: (x["p4q"] & ~R.q_by_day(x, "s_exp_r4", q)).to_numpy(), panel=PM, name=f"q{q}")
# 4) sign flip (drop the least stressed quintile) as a placebo
runs["flip"] = H.backtest(lambda x: (x["p4q"] & ~R.q_by_day(x.assign(o=x["i_exp_r4"]), "o", 0.8)).to_numpy(), panel=PM, name="flip")
# 5) within-date shuffle of the feature among covered names (10 draws)
rng = np.random.default_rng(7)
sh = []
for k in range(10):
    P3 = PM.copy()
    P3["s_sh"] = P3.groupby("day")["s_exp_r4"].transform(lambda s: pd.Series(rng.permutation(s.to_numpy()), index=s.index))
    r = H.backtest(lambda x: (x["p4q"] & ~R.q_by_day(x, "s_sh", 0.8)).to_numpy(), panel=P3, name="sh")
    sh.append(H.stats(r["daily"], bench=H.baseline("P4Q")["daily"])["diff_ann_%"])
tab = H.compare(runs, bench="P4Q")
print(tab.round(3).to_string())
out["table"] = tab.round(4).to_dict(orient="index")
out["shuffle_diff_vs_p4q"] = {"mean": float(np.mean(sh)), "p95": float(np.percentile(sh, 95)), "draws": [float(s) for s in sh]}
# feature coverage within p4q
X = PM[PM["univ"] & PM["p4q"] & (PM["day"] >= H.START)]
out["cov_in_p4q"] = float(X["s_exp_r4"].notna().mean())
out["distinct_values_per_date_median"] = float(X.groupby("day")["s_exp_r4"].nunique().median())
print(out["shuffle_diff_vs_p4q"], out["cov_in_p4q"], out["distinct_values_per_date_median"])
(HERE / "verify_results.json").write_text(json.dumps(R.jf(out), indent=1))
