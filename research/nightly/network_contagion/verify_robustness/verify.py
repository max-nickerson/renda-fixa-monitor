"""Adversarial robustness check of network_contagion best variant (avoid_exposure_stress_q5) vs P4+Q. Pre-2026 only."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))
import run as R
from research.nightly import harness as H

PM = R.load("M")
V = R.variants()
q = R.q_by_day
out = {}
def st(r, b):
    s = H.stats(r["daily"], bench=b)
    return {k: s[k] for k in ("ann_excess_%", "diff_ann_%", "diff_t_nw", "diff_p", "diff_h1_%", "diff_h2_%")}

b25 = H.baseline("P4Q")["daily"]
best = H.backtest(V["avoid_exposure_stress_q5"], panel=PM, name="best")
out["reproduce"] = st(best, b25)
print("reproduce", out["reproduce"], flush=True)

# perturbations
pert = {}
for qq in (0.7, 0.75, 0.85, 0.9):
    f = (lambda qq: lambda x: (x["p4q"] & ~q(x, "s_exp_r4", qq)).to_numpy())(qq)
    pert[f"exp_r4_q{qq}"] = st(H.backtest(f, panel=PM), b25)
pert["exp_any_q0.8"] = st(H.backtest(lambda x: (x["p4q"] & ~q(x, "i_exp_any", 0.8)).to_numpy(), panel=PM), b25)
pert["exp_ret_q0.8"] = st(H.backtest(lambda x: (x["p4q"] & ~q(x, "i_exp_ret", 0.8)).to_numpy(), panel=PM), b25)
# keep NaN-feature names? (baseline keeps them). Variant that also drops uncovered:
pert["exp_r4_q0.8_weekly"] = None
PW = R.load("W")
bW = H.baseline("P4Q", freq="W")["daily"]
pert["exp_r4_q0.8_weekly"] = st(H.backtest(V["avoid_exposure_stress_q5"], panel=PW, freq="W"), bW)
out["perturbations"] = pert
print(pd.DataFrame(pert).T.round(3).to_string(), flush=True)

# costs
b50 = H.baseline("P4Q", cost_bps=50)["daily"]
out["cost50"] = st(H.backtest(V["avoid_exposure_stress_q5"], panel=PM, cost_bps=50), b50)

# excluding 2023 (monthly diff excluding calendar 2023)
m = H.monthly(best["daily"]); mb = H.monthly(b25)
d = (m - mb).dropna(); d = d[d.index < "2026-01-01"]
d_ex = d[(d.index.year != 2023)]
out["excl_2023"] = {"diff_ann_%": d_ex.mean() * 1200, "t_nw": H.nw_t(d_ex, 6), "n": len(d_ex)}
out["full_monthly_diff"] = {"diff_ann_%": d.mean() * 1200, "t_nw": H.nw_t(d, 6), "n": len(d)}
yr = d.groupby(d.index.year).mean() * 1200
out["diff_by_year"] = yr.to_dict()
print("excl2023", out["excl_2023"], "by year", yr.round(3).to_dict(), flush=True)

# top-5 contributing issuers: issuers of removed names with worst forward 126d relative to p4q mean
X = PM[PM["univ"] & (PM["day"] >= H.START)]
rows = []
for dday, x in X.groupby("day"):
    keep = np.asarray(V["avoid_exposure_stress_q5"](x))
    rem = x[x["p4q"].to_numpy() & ~keep]
    base_mean = x.loc[x["p4q"], "fwd_126"].mean()
    for _, r_ in rem.iterrows():
        if pd.notna(r_["fwd_126"]):
            rows.append((r_["cnpj8"], r_["fwd_126"] - base_mean))
C = pd.DataFrame(rows, columns=["cnpj8", "rel"]).groupby("cnpj8")["rel"].sum().sort_values()
top5 = list(C.index[:5])
out["top5_helpful_removed_issuers_relsum"] = C.iloc[:5].round(4).to_dict()
ex = lambda f: (lambda x: (np.asarray(f(x)) & ~x["cnpj8"].isin(top5).to_numpy()))
bq = H.backtest(ex(lambda x: x["p4q"].to_numpy()), panel=PM, name="p4q_ex5")
bv = H.backtest(ex(V["avoid_exposure_stress_q5"]), panel=PM, name="best_ex5")
out["excl_top5_issuers"] = st(bv, bq["daily"])
print("ex top5", out["excl_top5_issuers"], flush=True)

# placebo: 60 random drops
pl = R.drop_placebo(PM, V["avoid_exposure_stress_q5"], n=60, seed=7)
act = H.stats(best["daily"])["ann_excess_%"]
out["placebo60"] = {"actual": act, "mean": pl.mean(), "p95": np.percentile(pl, 95), "share_ge": float((pl >= act).mean())}
print("placebo", out["placebo60"], flush=True)

# liquidity: names removed vs kept trades_30d
if "trades_30d" in X.columns:
    lk = []
    for dday, x in X.groupby("day"):
        keep = np.asarray(V["avoid_exposure_stress_q5"](x)); p = x["p4q"].to_numpy()
        lk.append((x.loc[p & ~keep, "trades_30d"].median(), x.loc[p & keep, "trades_30d"].median()))
    lk = np.array(lk, float)
    out["liq_median_trades30d_removed_vs_kept"] = [np.nanmean(lk[:, 0]), np.nanmean(lk[:, 1])]

# sector momentum FM reproduction
PW["own_shock"] = PW["i_any"].fillna(False).astype(float)
out["fm_sec_r13_fwd63"] = R.fama_macbeth(PW, "s_sec_r13", "fwd_63")
out["fm_sec_r13_fwd63_p4q"] = R.fama_macbeth(PW, "s_sec_r13", "fwd_63", sub="p4q")
print(out["fm_sec_r13_fwd63"], flush=True)
(HERE / "verify_results.json").write_text(json.dumps(R.jf(out), indent=1, default=str))
print("done")
