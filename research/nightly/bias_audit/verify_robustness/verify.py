"""Adversarial robustness verification of bias_audit's honest P4+Q (pre-2026 only; no holdout touched)."""
import json, time
import numpy as np, pandas as pd
from research.nightly import harness as H
from research.nightly.bias_audit import run as RN, build as B, engine as E

T0 = time.time()
OUT = {}
def log(*a): print(f"[{time.time()-T0:6.0f}s]", *a, flush=True)

P = RN.P
Pv = P.copy()
Pv["wq_lag90"] = RN.worstq_variant(P, lag_days=90)
Pv["p4q_lag90"] = Pv["p4"] & ~Pv["wq_lag90"]
Pv["carry30"] = Pv["univ"] & (Pv["cdi_pct"].fillna(1) <= 0.3)
T = lambda f: E.targets(lambda x: x[f].to_numpy(), Pv)
TU, TP4, TQ, TQ0, TC = T("univ"), T("p4"), T("p4q_lag90"), T("p4q"), T("carry30")

ev, stopped = RN.distress_lists()
Rd, _ = RN.R_recovery(RN.R0, stopped.loc[stopped["dist_issuer"], "last_ratio"])
Rcum = B.R_hol(Rd)
liq = RN.bond_liquidity()
COST = RN.cost_vectors(liq, None, RN.range_table(liq))
EX = RN.EX
kw = dict(R=Rcum, pen_buy=RN.pen(EX["SB"], 1), pen_sell=RN.pen(EX["SB"], -1), exit_fresh=True, cost_b=COST["liq_inst"])

base = {"U": E.tranche_book(TU), "P4Q": E.tranche_book(TQ0)}
hon = {"U": E.tranche_book(TU, **kw), "P4": E.tranche_book(TP4, **kw), "P4Q": E.tranche_book(TQ, **kw)}
def d(a, b):
    s = H.stats(a["daily"] if isinstance(a, dict) else a, bench=b["daily"] if isinstance(b, dict) else b)
    return {k: s[k] for k in ["ann_excess_%", "diff_ann_%", "diff_t_nw", "diff_h1_%", "diff_h2_%"]}
OUT["reproduce"] = {"honest_P4Q_vs_U": d(hon["P4Q"], hon["U"]), "honest_P4Q_vs_base_P4Q": d(hon["P4Q"], base["P4Q"]),
                    "honest_P4_vs_U": d(hon["P4"], hon["U"]), "honest_P4Q_vs_P4": d(hon["P4Q"], hon["P4"]),
                    "base_P4Q_vs_U": d(base["P4Q"], base["U"])}
log("reproduce", OUT["reproduce"])

def exyear(a, b, years=(2023,)):
    ma, mb = H.monthly(a["daily"]), H.monthly(b["daily"])
    x = (ma - mb).dropna(); x = x[(x.index < H.HOLDOUT) & (~x.index.year.isin(years))]
    return {"diff_ann_%": x.mean() * 1200, "t_nw6": H.nw_t(x, 6), "n_months": len(x)}
def by_year(a, b):
    x = (H.monthly(a["daily"]) - H.monthly(b["daily"])).dropna(); x = x[x.index < H.HOLDOUT]
    return (x.groupby(x.index.year).mean() * 1200).round(3).to_dict()
OUT["ex2023"] = {"honest_P4Q_vs_U": exyear(hon["P4Q"], hon["U"]), "honest_P4Q_vs_P4": exyear(hon["P4Q"], hon["P4"]),
                 "honest_P4_vs_U": exyear(hon["P4"], hon["U"])}
OUT["by_year"] = {"honest_P4Q_vs_U": by_year(hon["P4Q"], hon["U"]), "honest_P4Q_vs_P4": by_year(hon["P4Q"], hon["P4"]),
                  "honest_P4Q_vs_base_P4Q": by_year(hon["P4Q"], base["P4Q"])}
log("ex2023", OUT["ex2023"], OUT["by_year"])

# cost perturbations on the honest stack
for lab, ck in {"flat50": dict(cost_b=None, cost_bps=50), "bucket_x1.5": dict(cost_b=COST["liq_inst"] * 1.5),
                "bucket_x0.5": dict(cost_b=COST["liq_inst"] * 0.5)}.items():
    k2 = {**kw, **ck}
    u, q = E.tranche_book(TU, **k2), E.tranche_book(TQ, **k2)
    OUT["cost_" + lab] = {"P4Q_vs_U": d(q, u), "U_level": H.stats(u["daily"])["ann_excess_%"]}
    log(lab, OUT["cost_" + lab])

# U without trading costs (passive-index-like benchmark) vs honest P4Q
u0 = E.tranche_book(TU, **{**kw, "cost_b": None, "cost_bps": 0})
OUT["honest_P4Q_vs_costless_U"] = d(hon["P4Q"], u0)
log("vs costless U", OUT["honest_P4Q_vs_costless_U"])

# carry-only placebo (top-30% carry, no rich/press/Q filter) under the honest stack
c = E.tranche_book(TC, **kw)
OUT["carry_only_honest"] = {"vs_U": d(c, hon["U"]), "honest_P4Q_vs_carry": d(hon["P4Q"], c)}
log("carry", OUT["carry_only_honest"])

# top-5 issuer contributors (P&L of honest P4Q tranches) excluded
iss = H._issuer_of_b()
rows = hon["P4Q"]["rows"]; rows = rows[rows["e"] > rows["k"]]
contrib = {}
for p, b, w, k, e in rows.itertuples(index=False):
    r = Rcum[int(k):int(e), int(b)]
    contrib[iss[int(b)]] = contrib.get(iss[int(b)], 0.0) + w * (np.prod(1 + r) - 1)
cs = pd.Series(contrib).sort_values(ascending=False)
top5 = list(cs.index[:5])
OUT["top5_issuers"] = {str(k): float(v) for k, v in cs.head(5).items()}
def drop(Tg, bad):
    out = {}
    for p, w in Tg.items():
        if not len(w): out[p] = w; continue
        keep = w[~np.isin(iss[w.index.to_numpy()], bad)]
        out[p] = keep * (w.sum() / keep.sum()) if len(keep) and keep.sum() > 0 else keep
    return out
q5 = E.tranche_book(drop(TQ, top5), **kw)
u5 = E.tranche_book(drop(TU, top5), **kw)
OUT["ex_top5_P4Q_only"] = d(q5, hon["U"])
OUT["ex_top5_both"] = d(q5, u5)
log("top5", OUT["top5_issuers"], OUT["ex_top5_P4Q_only"], OUT["ex_top5_both"])

# capacity proxy: only the 3 most liquid lifetime-ADV quintiles tradable, for both books
liqq = liq["q"].reindex(RN.CODES).fillna(1).to_numpy()
ill = np.nonzero(liqq <= 2)[0]
def dropb(Tg):
    out = {}
    for p, w in Tg.items():
        if not len(w): out[p] = w; continue
        keep = w[~np.isin(w.index.to_numpy(), ill)]
        out[p] = keep * (w.sum() / keep.sum()) if len(keep) and keep.sum() > 0 else keep
    return out
qL, uL = E.tranche_book(dropb(TQ), **kw), E.tranche_book(dropb(TU), **kw)
OUT["liquid_q3to5_only"] = d(qL, uL)
log("liquid", OUT["liquid_q3to5_only"])

json.dump(OUT, open("research/nightly/bias_audit/verify_robustness/results.json", "w"), indent=1, default=float)
log("done")
