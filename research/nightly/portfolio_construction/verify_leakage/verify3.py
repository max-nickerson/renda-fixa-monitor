"""Issuer concentration of the top25 edge (ex-post robustness, contract-corrected R): drop the k issuers that
contribute most (by weight x fwd_126) from BOTH books and re-run."""
import json
import numpy as np, pandas as pd
from pathlib import Path
from research.nightly import harness as H
from research.nightly.portfolio_construction.verify_leakage import verify as V
OUTD = Path(__file__).resolve().parent

def main():
    P = H.load_panel("M")
    P["gap_pit"] = V.pit_gap_panel(P)
    P["cdi_corr"] = P["cdi_bps"] - 100 * P["gap_pit"]
    W1 = V.topk(P, "cdi_corr")
    h = W1.merge(P[["day", "codigo", "cnpj8", "fwd_126"]], on=["day", "codigo"])
    h["c"] = h["weight"] * h["fwd_126"].fillna(0)
    Q = P[P["univ"] & P["p4q"] & (P["day"] >= H.START)].copy()
    Q["w"] = 1 / Q.groupby("day")["codigo"].transform("count")
    Q["c"] = Q["w"] * Q["fwd_126"].fillna(0)
    ci = h.groupby(h["cnpj8"].astype(str))["c"].sum().sub(Q.groupby(Q["cnpj8"].astype(str))["c"].sum(), fill_value=0)
    ci = ci.sort_values(ascending=False)
    out = {"top_issuer_contrib_share": {k: round(float(v / ci.sum()), 3) for k, v in ci.head(8).items()}}
    Rc, _ = V.corrected_R()
    H._MEM[("R64", "R")] = Rc
    for k in (0, 3, 5, 10):
        drop = set(ci.index[:k])
        P2 = P.copy(); P2["p4q"] = P2["p4q"] & ~P2["cnpj8"].astype(str).isin(drop)
        for kk in [kk for kk in list(H._MEM) if isinstance(kk, tuple) and kk[0] in ("LC", "baseline")]:
            del H._MEM[kk]
        bq = H.backtest("p4q", freq="M", hold=126, panel=P2)
        r = H.backtest(V.topk(P2, "cdi_corr"), freq="M", hold=126, issuer_cap=1.0)
        out[f"drop_top{k}"] = V.st(r, bq)
    print(json.dumps(out, indent=1))
    (OUTD / "verify3_results.json").write_text(json.dumps(out, indent=1))

if __name__ == "__main__":
    main()
