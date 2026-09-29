"""Adversarial robustness check of alt_signals' P4Q_ex_supply_or_nc_expo (pre-2026 only; holdout untouched)."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from research.nightly import harness as H
from research.nightly.alt_signals import signals as S

OUT = Path(__file__).resolve().parent
Q = pd.read_pickle(S.CACHE / "panel_alt_M.pkl")
o = S.load_offers()
res = {}

def diff_stats(r, b=None, drop_years=(), drop_months=None):
    b = b if b is not None else H.baseline("P4Q")
    d = (H.monthly(r["daily"]) - H.monthly(b["daily"]))
    d = d[(d.index < "2026-01-01") & (d.index >= "2022-01-01")]
    if drop_years:
        d = d[~d.index.year.isin(drop_years)]
    return {"diff_ann_%": round(float(d.mean()) * 1200, 3), "t_nw": round(H.nw_t(d, 6), 2), "n_m": int(len(d))}

def mk(bad_fn, base="p4q"):
    def f(x):
        return x[base].to_numpy() & ~bad_fn(x)
    return f

sup = lambda x, c="sup_iss_90d": x[c].fillna(0).to_numpy() > 0
ncb = lambda x, thr=-1.0: x["nc_expo"].to_numpy() < thr
best = mk(lambda x: sup(x) | ncb(x))

rb = H.backtest(best, panel=Q); b = H.baseline("P4Q")
res["reproduce"] = {**diff_stats(rb), "exCDI": H.stats(rb["daily"])["ann_excess_%"], "n_avg": rb["n_avg"],
                    "stats_vs_p4q": {k: H.stats(rb["daily"], bench=b["daily"])[k] for k in ("diff_ann_%", "diff_t_nw", "diff_h1_%", "diff_h2_%")}}
print("reproduce", res["reproduce"], flush=True)

# --- monthly diff series, excl 2023, excl 2022, by year
res["excl_2023"] = diff_stats(rb, drop_years=(2023,))
res["excl_2022"] = diff_stats(rb, drop_years=(2022,))
d = (H.monthly(rb["daily"]) - H.monthly(b["daily"])); d = d[(d.index >= "2022-01-01") & (d.index < "2026-01-01")]
res["by_year"] = {int(y): round(float(g.mean()) * 1200, 3) for y, g in d.groupby(d.index.year)}
res["top_months"] = {str(k.date()): round(v * 100, 3) for k, v in d.sort_values(ascending=False).head(6).items()}
res["share_top3_months_of_total"] = round(float(d.sort_values(ascending=False).head(3).sum() / d.sum()), 3)
res["excl_top3_months"] = {"diff_ann_%": round(float(d.sort_values().iloc[:-3].mean()) * 1200, 3),
                           "t_nw": round(H.nw_t(d.sort_values().iloc[:-3].sort_index(), 6), 2)}
print("years", res["by_year"], res["excl_2023"], res["top_months"], flush=True)

# --- costs
b50 = H.baseline("P4Q", cost_bps=50)
res["cost50"] = diff_stats(H.backtest(best, panel=Q, cost_bps=50), b50)
res["cost100"] = diff_stats(H.backtest(best, panel=Q, cost_bps=100), H.baseline("P4Q", cost_bps=100))
print("costs", res["cost50"], res["cost100"], flush=True)

# --- issuer contribution: cohort fwd_126 of excluded names relative to P4Q mean, summed by issuer
U = Q[Q["univ"] & Q["p4q"] & (Q["day"] >= H.START) & Q["dok_126"]].copy()
U["fwd"] = U["fwd_126"].fillna(0)
U["xs"] = U["fwd"] - U.groupby("day")["fwd"].transform("mean")
U["bad"] = (U["sup_iss_90d"].fillna(0) > 0) | (U["nc_expo"] < -1)
ndate = U.groupby("day")["codigo"].transform("size")
U["contrib"] = -U["xs"] / ndate  # removing a bad name with negative xs helps
cb = U[U["bad"]].groupby("cnpj8")["contrib"].sum().sort_values(ascending=False)
names = pd.read_csv(S.ROOT / "research" / "data" / "issuer_sectors.csv", dtype={"cnpj8": str}).set_index("cnpj8")
res["contrib_total"] = round(float(cb.sum()), 4)
res["top_contributors"] = [{"cnpj8": c, "name": str(names["issuer_name"].get(c, "?"))[:35],
                            "sector": str(names["sector"].get(c, "?")), "contrib": round(float(v), 4),
                            "share": round(float(v / cb.sum()), 3)} for c, v in cb.head(10).items()]
res["share_top5_contrib"] = round(float(cb.head(5).sum() / cb.sum()), 3)
print("contrib", res["share_top5_contrib"], res["top_contributors"][:6], flush=True)
top5 = set(cb.index[:5]); top10 = set(cb.index[:10])
# (a) top-5 contributors never flagged; (b) drop top-5 contributor issuers from BOTH books (universe)
for k, s in (("top5", top5), ("top10", top10)):
    r = H.backtest(mk(lambda x, s=s: (sup(x) | ncb(x)) & ~x["cnpj8"].isin(s).to_numpy()), panel=Q)
    res[f"unflag_{k}_contrib"] = diff_stats(r)
    Qd = Q.copy(); Qd["p4q2"] = Qd["p4q"] & ~Qd["cnpj8"].isin(s)
    r1 = H.backtest(mk(lambda x: sup(x) | ncb(x), base="p4q2"), panel=Qd)
    r0 = H.backtest("p4q2", panel=Qd)
    res[f"drop_{k}_both_books"] = diff_stats(r1, r0)
    print(k, res[f"unflag_{k}_contrib"], res[f"drop_{k}_both_books"], flush=True)

# --- new-bond mechanism: is the flagged bond itself the new issue?
U["age_d"] = U["age"]
res["age_col_sample"] = U["age"].describe().round(2).to_dict()
(OUT / "verify1.json").write_text(json.dumps(res, indent=1, default=str))
