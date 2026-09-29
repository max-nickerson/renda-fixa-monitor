"""Part 3: does lgb_peer:tilt add beyond simple (pre-specified, 3) non-ML within-peer tilts? + issuer attribution."""
import json
import numpy as np, pandas as pd
from pathlib import Path
from research.nightly import harness as H
from research.nightly.ml_ranking import run as RUN

OUT = Path("research/nightly/ml_ranking/verify_leakage")
P = pd.read_pickle("data/history/nightly/ml_ranking/features_M.pkl")
pr = pd.read_pickle("data/history/nightly/ml_ranking/pred_lgb_peer.pkl")["pred"]
b = H.baseline("P4Q")
best_sig = RUN.overlay_signals("lgb_peer", pr, P)["tilt"]
best = H.backtest(best_sig)
U0 = P[P["univ"] & P["dpos"].isin(pr["dpos"].unique())].copy()
U0["peer_carry"] = U0.groupby(["dpos", "peer"])["cdi_bps"].rank(pct=True)
U0["combo"] = U0.groupby("dpos")["cdi_bps"].rank(pct=True) + U0.groupby("dpos")["resid_z"].rank(pct=True)
ctrls = {"peer_carry": "peer_carry", "cdi_x_dur": "cdi_x_dur", "carry+resid": "combo"}
res = {"best_vs_p4q": H.stats(best["daily"], bench=b["daily"])}
C = {}
for k, col in ctrls.items():
    d = U0[["day", "dpos", "codigo", "cnpj8"]].assign(pred=U0[col].fillna(U0[col].median()).to_numpy())
    s = RUN.overlay_signals(k, d, P)["tilt"]
    C[k] = H.backtest(s)
    st = H.stats(C[k]["daily"], bench=b["daily"])
    st2 = H.stats(best["daily"], bench=C[k]["daily"])
    res[k] = {"ctrl_vs_p4q": {x: st[x] for x in ("diff_ann_%", "diff_t_nw", "diff_h1_%", "diff_h2_%")},
              "best_vs_ctrl": {x: st2[x] for x in ("diff_ann_%", "diff_t_nw", "diff_h1_%", "diff_h2_%")}}
    print(k, res[k], flush=True)
# ---- issuer attribution of the tilt-vs-P4Q gap (label-based approx: (w_tilt - w_p4q) x fwd_126 per decision)
def wts(r):
    h = r["holdings"].copy()
    h["w"] = h["weight"] / h.groupby("day")["weight"].transform("sum")
    return h[["day", "codigo", "w"]]
hb, hq = wts(best), wts(b)
m = hb.merge(hq, on=["day", "codigo"], how="outer", suffixes=("_t", "_q")).fillna({"w_t": 0, "w_q": 0})
m = m.merge(P[["day", "codigo", "cnpj8", "fwd_126"]], on=["day", "codigo"], how="left")
m["contrib"] = (m["w_t"] - m["w_q"]) * m["fwd_126"].fillna(0)
m["yr"] = m["day"].dt.year
ci = m.groupby("cnpj8")["contrib"].sum() / m["day"].nunique() * 2 * 100  # approx %/yr (126d label, 2 per yr)
res["approx_total_%yr"] = float(ci.sum())
res["top_issuers_%yr"] = ci.sort_values(ascending=False).head(8).round(4).to_dict()
res["bottom_issuers_%yr"] = ci.sort_values().head(5).round(4).to_dict()
print("attr", res["approx_total_%yr"], res["top_issuers_%yr"], flush=True)
# ---- leave-top-k-issuers-out: drop top 1/3 contributing issuers from BOTH books
for k in (1, 3):
    drop = set(ci.sort_values(ascending=False).index[:k])
    keep = ~P["cnpj8"].isin(drop)
    sT = best_sig.merge(P[["day", "codigo", "cnpj8"]], on=["day", "codigo"])
    sT = sT[~sT["cnpj8"].isin(drop)][["day", "codigo", "weight"]]
    sQ = P[P["univ"] & keep][["day", "codigo"]].assign(select=P.loc[P["univ"] & keep, "p4q"].to_numpy())
    rt, rq = H.backtest(sT), H.backtest(sQ)
    st = H.stats(rt["daily"], bench=rq["daily"])
    res[f"drop_top{k}_issuers"] = {x: st[x] for x in ("diff_ann_%", "diff_t_nw", "diff_h1_%", "diff_h2_%")}
    print(k, res[f"drop_top{k}_issuers"], flush=True)
# ---- monthly diff concentration
dm = H.monthly(best["daily"]) - H.monthly(b["daily"])
res["monthly_diff_top5_share"] = float(dm.sort_values(ascending=False).head(5).sum() / dm.sum())
res["monthly_diff_ann_excl_top3"] = float(dm.sort_values(ascending=False).iloc[3:].mean() * 12 * 100)
json.dump(res, open(OUT / "v3_results.json", "w"), indent=1, default=float)
print(json.dumps({k: v for k, v in res.items() if k.startswith("monthly")}))
