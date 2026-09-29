"""Reproduce the bias_audit honest vs v4 numbers and test the look-ahead pieces of the cost model.
Writes only to research/nightly/bias_audit/verify_leakage/."""
import json, sys, glob
import numpy as np, pandas as pd
sys.argv = ["x"]
from research.nightly import harness as H
from research.nightly.bias_audit import build as B, engine as E
from research.nightly.bias_audit import run as RN

P, R0, EX, lm, CODES = RN.P, RN.R0, RN.EX, RN.lm, RN.CODES
out = {}
ev, stopped = RN.distress_lists()
Rd, nd = RN.R_recovery(R0, stopped.loc[stopped["dist_issuer"], "last_ratio"])
Rcum = B.R_hol(Rd)
liq = RN.bond_liquidity()
COST = RN.cost_vectors(liq, None, RN.range_table(liq))
Pv = P.copy()
Pv["wq_lag90"] = RN.worstq_variant(P, lag_days=90)
Pv["p4q_lag90"] = Pv["p4"] & ~Pv["wq_lag90"]
SIG = {"U": lambda x: x["univ"].to_numpy(), "P4": lambda x: x["p4"].to_numpy(), "P4Q": lambda x: x["p4q"].to_numpy(),
       "P4Q_lag90": lambda x: x["p4q_lag90"].to_numpy()}
T = {k: E.targets(v, Pv) for k, v in SIG.items()}
pen = RN.pen
base = {n: E.tranche_book(T[n]) for n in ["U", "P4", "P4Q"]}
# harness bit-for-bit check
hb = {n: H.baseline(n) for n in ["U", "P4Q"]} if hasattr(H, "baseline") else {}
for n, v in hb.items():
    try:
        d = v["daily"] if isinstance(v, dict) else v
        out[f"maxabs_engine_vs_harness_{n}"] = float((base[n]["daily"] - d.reindex(base[n]["daily"].index)).abs().max())
    except Exception as e:
        out[f"harness_cmp_{n}"] = str(e)[:200]

def honest_kw(cost_b):
    return dict(R=Rcum, pen_buy=pen(EX["SB"], 1), pen_sell=pen(EX["SB"], -1), exit_fresh=True, cost_b=cost_b)

def book(kw, qname="P4Q_lag90"):
    return {"U": E.tranche_book(T["U"], **kw), "P4": E.tranche_book(T["P4"], **kw), "P4Q": E.tranche_book(T[qname], **kw)}

def summ(r, tag):
    s = {n: H.stats(r[n]["daily"])["ann_excess_%"] for n in r}
    d1 = H.stats(r["P4Q"]["daily"], bench=r["U"]["daily"])
    d0 = H.stats(r["P4Q"]["daily"], bench=base["P4Q"]["daily"])
    s.update({"P4Q-U": d1["diff_ann_%"], "t(P4Q-U)": d1["diff_t_nw"], "P4Q-U_h1": d1["diff_h1_%"], "P4Q-U_h2": d1["diff_h2_%"],
              "vs_v4_P4Q": d0["diff_ann_%"], "t_vs_v4_P4Q": d0["diff_t_nw"],
              "P4Q_cost_%": r["P4Q"]["cost_ann_%"], "U_cost_%": r["U"]["cost_ann_%"],
              "P4Q_turn": r["P4Q"]["turnover_ann"], "U_turn": r["U"]["turnover_ann"]})
    out[tag] = {k: round(float(v), 3) for k, v in s.items()}
    print(tag, out[tag], flush=True)

summ(base, "v4_base")
summ(book(honest_kw(COST["liq_inst"])), "honest_repro")

# --- leakage test 1: liquidity quintile from lifetime ADV (look-ahead). Replace by (a) flat per group, (b) ADV measured
# only over each bond's first 126 business days on the grid (known early), (c) ADV using trades strictly before 2022-01-01
# where available else first-126d.
ml = {"DI": {1: 65, 2: 60, 3: 55, 4: 50, 5: 45}, "IPCA_PRE": {1: 169, 2: 161, 3: 152, 4: 144, 5: 136}}
L = liq.reindex(CODES)
grp = L["grp"].fillna("IPCA_PRE").to_numpy()
flat = np.array([55.0 if g == "DI" else 152.0 for g in grp])
summ(book(honest_kw(flat)), "honest_flat_group_cost")
fs = sorted(glob.glob(str(H.HIST / "snd_trades_20*.csv.gz")))
s = pd.concat([pd.read_csv(f, parse_dates=["date"], usecols=["date", "codigo", "qty", "pu_avg"]) for f in fs], ignore_index=True)
s = s[s["date"] < H.HOLDOUT]
s["vol"] = s["qty"] * s["pu_avg"]
first = lm.groupby("codigo")["day"].min()
s["first"] = s["codigo"].map(first)
e = s[(s["first"].notna()) & (s["date"] >= s["first"]) & (s["date"] < s["first"] + pd.tseries.offsets.BDay(126))]
adv = e.groupby("codigo")["vol"].sum() / 126
q = np.ceil(adv.rank(pct=True) * 5).clip(1, 5).reindex(CODES).fillna(1).astype(int).to_numpy()
early = np.array([ml[g][k] for g, k in zip(grp, q)], dtype=float)
summ(book(honest_kw(early)), "honest_cost_early_adv_quintile")
# cross-check which quintile P4Q/U hold on average (lifetime vs early)
ql = L["q"].fillna(1).astype(int).to_numpy()
for n in ["U", "P4Q_lag90"]:
    ws = pd.concat([w for w in T[n].values() if len(w)])
    b = ws.index.to_numpy()
    out[f"avg_cost_bps_{n}"] = {"lifetime": float(np.average(COST["liq_inst"][b], weights=ws.values)),
                                "early": float(np.average(early[b], weights=ws.values)),
                                "flat": float(np.average(flat[b], weights=ws.values)),
                                "share_IPCA_PRE": float(np.average(grp[b] != "DI", weights=ws.values))}
print(out["avg_cost_bps_U"], out["avg_cost_bps_P4Q_lag90"])
# --- leakage test 2: distress recovery uses post-event silence (future info); drop it
summ(book(dict(R=B.R_hol(R0), pen_buy=pen(EX["SB"], 1), pen_sell=pen(EX["SB"], -1), exit_fresh=True, cost_b=COST["liq_inst"])),
     "honest_without_distress_recovery")
# --- leakage test 3: exit_fresh sells at the next real trade (future timing knowledge is fine for execution) - drop it
summ(book(dict(R=Rcum, pen_buy=pen(EX["SB"], 1), pen_sell=pen(EX["SB"], -1), cost_b=COST["liq_inst"])), "honest_without_exit_fresh")
json.dump(out, open("research/nightly/bias_audit/verify_leakage/verify_results.json", "w"), indent=1)
