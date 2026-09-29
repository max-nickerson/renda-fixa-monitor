"""Adversarial ROBUSTNESS verification of issuer_curve_rv (pre-2026 only; no holdout touched)."""
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from research.nightly import harness as H
from research.nightly.issuer_curve_rv import signals as S

OUT = Path(__file__).resolve().parent
T0 = time.time()
R = {}
def log(*a): print(f"[{time.time()-T0:6.0f}s]", *a, flush=True)

PM = S.attach(H.load_panel("M"))
nz = lambda s, v: s.fillna(v)

def tilt(lam, col="e_iss", noise=None):
    def f(x):
        e = x[col] if noise is None else noise.reindex(x.index)
        sc = x["cdi_bps"] + lam * e.fillna(0).clip(-300, 300)
        thr = sc.quantile(0.7)
        return ((sc >= thr) & ~(nz(x["resid_z"], 0) <= -1.5) & (nz(x["press_neg_30d"], 0) < 1) & ~x["worstQ"]).to_numpy()
    return f

def mdiff(a, b):
    return (H.monthly(a) - H.monthly(b.reindex(a.index).fillna(0))).dropna()

def summ(d, lag=6):
    d = d[d.index < H.HOLDOUT]
    return {"ann_%": round(d.mean()*1200, 4), "t": round(H.nw_t(d, lag), 2), "n": int(len(d))}

def run_pair(panel, lam, **kw):
    r = H.backtest(tilt(lam), panel=panel, **kw)
    b = H.backtest(lambda x: x["p4q"].to_numpy(), panel=panel, **kw)
    t0 = H.backtest(tilt(0.0), panel=panel, **kw)
    return r, b, t0

# 1) reproduce + lambda grid (vs P4Q and vs lambda-0 control which uses same quantile rule)
lam_res = {}
BASE = {}
for lam in (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0):
    r = H.backtest(tilt(lam), panel=PM)
    BASE[lam] = r
b = H.backtest(lambda x: x["p4q"].to_numpy(), panel=PM)
for lam, r in BASE.items():
    d = mdiff(r["daily"], b["daily"]); d0 = mdiff(r["daily"], BASE[0.0]["daily"])
    lam_res[str(lam)] = {"vs_P4Q": summ(d), "vs_lam0": summ(d0),
                         "h1": round(d[d.index < H.SPLIT].mean()*1200, 3), "h2": round(d[(d.index >= H.SPLIT)&(d.index<H.HOLDOUT)].mean()*1200, 3),
                         "ex2023": summ(d[d.index.year != 2023])}
R["lambda_grid_25bps"] = lam_res
log("lambda", json.dumps(lam_res))

# 2) 50 bps
b50 = H.backtest(lambda x: x["p4q"].to_numpy(), panel=PM, cost_bps=50)
r50 = H.backtest(tilt(0.5), panel=PM, cost_bps=50)
R["cost50"] = summ(mdiff(r50["daily"], b50["daily"]))
# 3) overlap: how different are the books?
h1, h2 = BASE[0.5]["holdings"], b["holdings"]
g1 = h1.groupby("day")["codigo"].apply(set); g2 = h2.groupby("day")["codigo"].apply(set)
ov = [(len(g1[d] - g2.get(d, set())), len(g1[d])) for d in g1.index]
R["names_diff_per_date"] = {"mean_new_names": float(np.mean([a for a, _ in ov])), "mean_n": float(np.mean([n for _, n in ov]))}
log("overlap", R["names_diff_per_date"], R["cost50"])

# 4) top-5 issuer contributors to the diff: cohort attribution with fwd_126, then drop from universe and rerun
X = PM[PM["univ"] & PM["fwd_126"].notna()][["day", "codigo", "cnpj8", "fwd_126"]]
w1 = h1.groupby(["day", "codigo"])["weight"].sum().rename("w1")
w2 = h2.groupby(["day", "codigo"])["weight"].sum().rename("w2")
A = X.set_index(["day", "codigo"]).join(w1, how="left").join(w2, how="left").fillna({"w1": 0, "w2": 0})
A["c"] = (A["w1"] - A["w2"]) * A["fwd_126"]
contrib = A.groupby("cnpj8")["c"].sum().sort_values(ascending=False)
top5 = list(contrib.index[:5])
R["top5_contrib_issuers_share"] = round(float(contrib.iloc[:5].sum() / contrib.sum()), 3) if contrib.sum() != 0 else None
R["contrib_total"] = round(float(contrib.sum()), 4)
PX = PM.copy(); PX.loc[PX["cnpj8"].astype(str).isin([str(c) for c in top5]), "univ"] = False
rx = H.backtest(tilt(0.5), panel=PX); bx = H.backtest(lambda x: x["p4q"].to_numpy(), panel=PX)
R["ex_top5_issuers"] = summ(mdiff(rx["daily"], bx["daily"]))
log("ex top5", R["ex_top5_issuers"], R["top5_contrib_issuers_share"])

# 5) placebo: permute e_iss within date among defined rows (20 draws) -> null of diff vs P4Q
rng = np.random.default_rng(11)
null = []
for i in range(20):
    e = PM["e_iss"].copy()
    m = e.notna()
    e[m] = PM[m].groupby("day")["e_iss"].transform(lambda s: rng.permutation(s.to_numpy()))
    rp = H.backtest(tilt(0.5, noise=e), panel=PM)
    null.append(summ(mdiff(rp["daily"], b["daily"]))["ann_%"])
act = lam_res["0.5"]["vs_P4Q"]["ann_%"]
R["placebo_perm_eiss"] = {"actual": act, "null_mean": round(float(np.mean(null)), 4), "null_p95": round(float(np.percentile(null, 95)), 4),
                          "frac_null_ge_actual": round(float(np.mean(np.array(null) >= act)), 3)}
log("placebo", R["placebo_perm_eiss"])

# 6) pair trades (vectorized): cheap-minus-rich resid_bps, fwd_126; robustness
P = PM[PM["univ"] & PM["dok_126"] & PM["fwd_126"].notna() & PM["resid_bps"].notna()].copy()
P = P[P.groupby(["day", "cnpj8"])["codigo"].transform("count") >= 2]
hi = P.loc[P.groupby(["day", "cnpj8"])["resid_bps"].idxmax()].set_index(["day", "cnpj8"])
lo = P.loc[P.groupby(["day", "cnpj8"])["resid_bps"].idxmin()].set_index(["day", "cnpj8"])
pt = pd.DataFrame({"dret": hi["fwd_126"] - lo["fwd_126"], "dcarry": hi["cdi_bps"] - lo["cdi_bps"],
                   "gap": hi["resid_bps"] - lo["resid_bps"]}).reset_index()
def pts(q):
    s = q.groupby("day")["dret"].mean()
    s = s[s.index < H.HOLDOUT]
    return {"ann_%": round(float(s.mean()*200), 3), "t": round(H.nw_t(s, 6), 2), "n_pairs": int(len(q))}
pr = {"all": pts(pt), "ex2023": pts(pt[pt["day"].dt.year != 2023])}
ic_ = pt.groupby("cnpj8")["dret"].sum().sort_values(ascending=False)
pr["ex_top5_issuers"] = pts(pt[~pt["cnpj8"].isin(ic_.index[:5])])
pr["ex_top10_issuers"] = pts(pt[~pt["cnpj8"].isin(ic_.index[:10])])
pr["top5_share_of_sum"] = round(float(ic_.iloc[:5].sum() / ic_.sum()), 3)
pr["winsor_1pct"] = pts(pt.assign(dret=pt["dret"].clip(pt["dret"].quantile(.01), pt["dret"].quantile(.99))))
pr["median_pair_ann_%"] = round(float(pt["dret"].median()*200), 3)
# carry-neutral: residual of dret on dcarry (carry predicted by 126d accrual = dcarry/1e4*0.5)
pr["beyond_carry_ann_%"] = pts(pt.assign(dret=pt["dret"] - pt["dcarry"]/1e4*0.5))
pr["small_carry_gap_lt25bps"] = pts(pt[pt["dcarry"].abs() < 25])
# net of switching cost: one round trip per 126d switch
for c in (25, 50):
    pr[f"net_{c}bps_ann_%"] = round(pr["all"]["ann_%"] - c/100*2, 3)
R["pair_trades_robust"] = pr
log("pairs", json.dumps(pr))

# 7) multiple testing: effective tried variants
R["variants_note"] = ("README counts 12 book variants; ICs across 7 residuals x 2 horizons x 3 scopes, 3 sorts x 2 horizons x 4 "
                      "pair types and 2 basis trades were also tried (~80 tests). Best of 12 picked ex post by vs_bench.")
json.dump(R, open(OUT / "verify_results.json", "w"), indent=1, default=str)
log("done")
