"""Robustness verification of marks_liquidity P4Q_gapfilter vs P4+Q. Read-only on their files."""
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
assert PM["day"].max() < pd.Timestamp("2026-01-01")
R = {}

def p4like(extra=None, excl=None):
    def f(x):
        c = x["cdi_bps"]; pct = c.rank(pct=True, ascending=False)
        m = (pct <= 0.3) & ~(x["resid_z"].fillna(0) <= -1.5) & (x["press_neg_30d"].fillna(0) < 1) & ~x["worstQ"] & c.notna()
        if extra is not None: m = m & extra(x)
        if excl is not None: m = m & ~x["cnpj8"].astype(str).isin(excl)
        return m.to_numpy()
    return f
gap = lambda x: ~(x["kf_gap_z"] < -2)

def pair(kw_a, kw_b, label, **bt):
    ra = H.backtest(p4like(**kw_a), panel=PM, **bt); rb = H.backtest(p4like(**kw_b), panel=PM, **bt)
    s = H.stats(ra["daily"], bench=rb["daily"])
    out = {k: float(s[k]) for k in ("ann_excess_%", "diff_ann_%", "diff_t_nw", "diff_p", "diff_h1_%", "diff_h2_%")}
    d = (ra["daily"] - rb["daily"]); d = d[d.index < "2026-01-01"]
    m = H.monthly(d) if hasattr(H, "monthly") else d.resample("M").sum()
    ex23 = m[(m.index.year != 2023)]
    out["diff_ex2023_ann_%"] = float(ex23.mean() * 12 * 100) if abs(ex23.mean()) < 1 else float(ex23.mean() * 12)
    out["diff_ex2023_t"] = float(H.nw_t(ex23, 6))
    out["by_year_%"] = {str(y): float(v.sum() * 100) for y, v in m.groupby(m.index.year)}
    R[label] = out; log(label, json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in out.items()}))
    return ra, rb

ra, rb = pair({"extra": gap}, {}, "base_25bps")
pair({"extra": gap}, {}, "cost_50bps", cost_bps=50)
pair({"extra": gap}, {}, "rec40", scenario="rec40")
pair({"extra": gap}, {}, "hold63", hold=63)
pair({"extra": gap}, {}, "overlay_ida", overlay="ida")

# issuer contributions: bonds dropped by the filter, their fwd_126 vs P4Q mean on that date
U = PM[PM["univ"] & PM["p4q"] & (PM["day"] >= H.START)].copy()
U["rel"] = U["fwd_126"].fillna(0) - U.groupby("day")["fwd_126"].transform(lambda s: s.fillna(0).mean())
dr = U[U["kf_gap_z"] < -2]
R["n_dropped_rows"] = int(len(dr)); R["share_dropped"] = float(len(dr) / len(U)); R["n_dropped_issuers"] = int(dr["cnpj8"].nunique())
contrib = (-dr.groupby(dr["cnpj8"].astype(str))["rel"].sum()).sort_values(ascending=False)
R["top_issuer_contrib_(sum -rel fwd126)"] = contrib.head(10).round(4).to_dict()
R["contrib_share_top5"] = float(contrib.head(5).sum() / contrib.clip(lower=0).sum()) if contrib.clip(lower=0).sum() > 0 else None
log("dropped", R["n_dropped_rows"], R["share_dropped"], R["n_dropped_issuers"], contrib.head(8).round(3).to_dict())
top5 = list(contrib.head(5).index)
pair({"extra": gap, "excl": top5}, {"excl": top5}, "ex_top5_issuers_both_books")
top10 = list(contrib.head(10).index)
pair({"extra": gap, "excl": top10}, {"excl": top10}, "ex_top10_issuers_both_books")

# threshold neighbourhood (to test knife-edge)
for th in (-1.5, -2.5, -3.0):
    pair({"extra": (lambda t: (lambda x: ~(x["kf_gap_z"] < t)))(th)}, {}, f"thresh_{th}")

# random-drop placebo: same number of P4Q names dropped per date
nd = U.assign(d=U["kf_gap_z"] < -2).groupby("day")["d"].sum()
rng = np.random.default_rng(7); pl = []
Pp = PM[PM["univ"] & PM["p4q"]]
for i in range(20):
    rows = []
    for d, z in Pp.groupby("day"):
        k = int(nd.get(d, 0)); keep = z["codigo"].to_numpy()
        if k: keep = np.setdiff1d(keep, rng.choice(keep, k, replace=False))
        rows.append(pd.DataFrame({"day": d, "codigo": keep, "select": True}))
    r = H.backtest(pd.concat(rows), panel=PM)
    pl.append(H.stats(r["daily"], bench=H.baseline("P4Q")["daily"])["diff_ann_%"])
pl = np.array(pl)
R["random_drop_placebo"] = {"mean": float(pl.mean()), "sd": float(pl.std()), "p95": float(np.quantile(pl, .95)),
                            "frac_ge_actual": float((pl >= R["base_25bps"]["diff_ann_%"]).mean())}
log("placebo", R["random_drop_placebo"])
json.dump(R, open(OUT / "verify_results.json", "w"), indent=1, default=str)
log("done")
