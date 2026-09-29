"""Diagnostics: where does the mean-variance (empirical cov) edge come from? Pre-2026 only."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

from research.nightly import harness as H
from research.nightly.portfolio_construction import signals as SG, build_weights as BW
from research.nightly.portfolio_construction.data import build as build_liq

OUTD = Path(__file__).resolve().parent


def main():
    P = H.load_panel("M")
    S = SG.cached("M")
    L = build_liq()[["codigo", "day", "vol91_brl", "adv_brl"]]
    X = P[P["univ"] & (P["day"] >= H.START)].merge(S, on=["codigo", "day"], how="left").merge(L, on=["codigo", "day"], how="left")
    X["is_di"] = (X["kind"] == "DI_SPREAD").astype(float)
    X["is_ipca"] = (X["kind"] == "IPCA").astype(float)
    X["cov_"] = X["covered"].astype(float)
    X["lst"] = X["listed"].astype(float)
    X["inc"] = X["incent"].astype(float)
    X["lv91"] = np.log10(X["vol91_brl"].fillna(1).clip(lower=1))
    out = {}
    # 1) ICs (universe and within P4+Q), fwd_126, and rec40
    for f in ("carry_over_var", "emp_vol", "vol_hat", "dts", "cdi_bps"):
        Xi = X.copy()
        Xi["neg"] = -Xi[f] if f in ("emp_vol", "vol_hat", "dts") else Xi[f]
        a = H.ic(Xi, "neg", "fwd_126")
        q = H.ic(Xi[Xi["p4q"]].assign(univ=True), "neg", "fwd_126", min_n=20)
        out[f"IC_{f}"] = {"sign": "-" if f in ("emp_vol", "vol_hat", "dts") else "+", "univ": round(a["mean"], 3),
                          "t_univ": round(a["t_nw"], 2), "p4q": round(q["mean"], 3), "t_p4q": round(q["t_nw"], 2)}
    # 2) double sort inside P4+Q: spread terciles x emp-vol terciles, mean fwd_126 (annualised %) per cell
    Q = X[X["p4q"] & X["dok_126"] & X["fwd_126"].notna() & X["emp_vol"].notna()].copy()
    Q["s3"] = Q.groupby("day")["cdi_bps"].transform(lambda v: pd.qcut(v.rank(method="first"), 3, labels=False))
    Q["v3"] = Q.groupby(["day", "s3"])["emp_vol"].transform(lambda v: pd.qcut(v.rank(method="first"), 3, labels=False))
    cell = Q.groupby(["day", "s3", "v3"])["fwd_126"].mean().unstack("v3")
    lowmhigh = (cell[0] - cell[2]).groupby(level="s3")
    out["double_sort_lowvol_minus_highvol_fwd126_ann%"] = {
        f"spread_T{int(k) + 1}": {"mean": round(v.mean() * 200, 2), "t_nw": round(H.nw_t(v.groupby(level=0).mean(), 6), 2)}
        for k, v in lowmhigh}
    Q["fr40"] = Q["fwd_126_rec40"]
    c40 = Q.groupby(["day", "s3", "v3"])["fr40"].mean().unstack("v3")
    out["double_sort_rec40_lowmhigh_ann%"] = {f"spread_T{int(k) + 1}": round(v.mean() * 200, 2)
                                              for k, v in (c40[0] - c40[2]).groupby(level="s3")}
    out["share_p4q_with_emp_vol"] = round(float(X.loc[X["p4q"], "emp_vol"].notna().mean()), 3)
    # 3) who does MV-emp buy vs EW?  holdings characteristics (weighted)
    W, M = BW.build(tag="pre")
    ch = {}
    for v in ("EW_cap10", "mv_emp_g20", "mv_emp_g20_fund500m", "carry_over_var_tilt", "top50_comp"):
        h = W[W["variant"] == v].merge(X, on=["day", "codigo"], how="left")
        wt = h["weight"] / h.groupby("day")["weight"].transform("sum")
        f = lambda c: float((wt * h[c].astype(float)).groupby(h["day"]).sum().mean())
        ex = []
        for d, g in h.groupby("dpos"):
            ep = H.exec_pos(int(d))
            ex.append(float((g["weight"] * (ep[g["b"].to_numpy()] >= 0)).sum() / g["weight"].sum()))
        ch[v] = {"DI_SPREAD_share": f("is_di"), "IPCA_share": f("is_ipca"), "emp_vol": f("emp_vol"),
                 "trades_30d": f("trades_30d"), "log10_vol91": f("lv91"), "covered": f("cov_"), "listed": f("lst"),
                 "incent": f("inc"), "cdi_bps": f("cdi_bps"), "dur": f("dur"), "age_days": f("age"),
                 "executed_weight_share": float(np.mean(ex))} if len(h) else {}
    out["holdings_characteristics"] = {k: {a: round(b, 3) for a, b in v.items()} for k, v in ch.items()}
    json.dump(out, open(OUTD / "diag.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
