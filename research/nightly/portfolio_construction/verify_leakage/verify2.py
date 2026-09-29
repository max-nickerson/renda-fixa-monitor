"""Stress the top25-carry edge for execution-contingent selection and survivorship (on contract-corrected R)."""
import json
import numpy as np, pandas as pd
from pathlib import Path
from research.nightly import harness as H
from research.nightly.portfolio_construction.verify_leakage import verify as V

OUTD = Path(__file__).resolve().parent
_orig_exec = H.exec_pos

def clear():
    for k in [k for k in list(H._MEM) if isinstance(k, tuple) and k[0] in ("LC", "baseline", "EX", "SP")]:
        del H._MEM[k]

def forced_exec(p, max_wait=H.ENTRY_MAX):
    ep = _orig_exec(p, max_wait).copy()
    ND = len(H.days())
    ep[ep < 0] = min(p + max_wait, ND - 1)
    return ep

def harsh_R(R, thr=0.98, rec=0.40):
    C = H._core()
    last, dd, codes = C["last"], C["days"], C["codes"]
    st = last[(last["last_day"] < dd[-1] - pd.Timedelta(days=30))
              & (last["maturity"].isna() | (last["maturity"] > last["last_day"] + pd.Timedelta(days=60)))
              & (last["last_ratio"] < thr)]
    R = R.copy()
    TD = C["TD"]
    n = 0
    for c, r in st.iterrows():
        if c not in codes: continue
        b = codes.get_loc(c)
        rows = np.nonzero(TD[:, b] >= 0)[0]
        if not len(rows): continue
        p = int(rows.max())
        R[p, b] = (1 + R[p, b]) * (rec / r["last_ratio"]) - 1
        n += 1
    return R, n

def run(W, tag, out):
    bq = H.baseline("P4Q")
    r = H.backtest(W, freq="M", hold=126, issuer_cap=1.0)
    out[tag] = V.st(r, bq)
    out[tag]["P4Q_exCDI"] = round(float(H.stats(bq["daily"])["ann_excess_%"]), 3)

def main():
    P = H.load_panel("M")
    P["gap_pit"] = V.pit_gap_panel(P)
    P["cdi_corr"] = P["cdi_bps"] - 100 * P["gap_pit"]
    W0 = V.topk(P)
    W1 = V.topk(P, "cdi_corr")
    Rc, _ = V.corrected_R()
    out = {}
    # execution share of top25 vs P4Q
    ex = []
    for d, g in W1.merge(P[["day", "codigo", "b", "dpos", "p4q"]], on=["day", "codigo"]).groupby("dpos"):
        ep = _orig_exec(int(d))
        q = P[(P["dpos"] == d) & P["univ"] & P["p4q"]]
        ex.append((float((g["weight"] * (ep[g["b"].to_numpy()] >= 0)).sum() / g["weight"].sum()),
                   float((ep[q["b"].to_numpy()] >= 0).mean())))
    ex = np.array(ex)
    out["executed_share_top25_vs_p4q"] = [round(float(ex[:, 0].mean()), 3), round(float(ex[:, 1].mean()), 3)]
    H._MEM[("R64", "R")] = Rc; clear()
    run(W1, "corrR", out)
    H.exec_pos = forced_exec; clear()
    run(W1, "corrR_forced_exec", out)
    H.exec_pos = _orig_exec
    Rh, n = harsh_R(Rc); out["n_harsh_bonds"] = n
    H._MEM[("R64", "R")] = Rh; clear()
    run(W1, "corrR_harsh_surv", out)
    H.exec_pos = forced_exec; clear()
    run(W1, "corrR_harsh_surv_forced_exec", out)
    run(W0, "corrR_harsh_surv_forced_exec_origsel", out)
    Rh2, n2 = harsh_R(Rc, thr=1.01, rec=0.40)   # extreme: every early stop below par gets 40% (upper bound)
    out["n_extreme_bonds"] = n2
    H._MEM[("R64", "R")] = Rh2; clear()
    run(W1, "EXTREME_all_stopped_below_101_forced_exec", out)
    H.exec_pos = _orig_exec
    print(json.dumps(out, indent=1))
    (OUTD / "verify2_results.json").write_text(json.dumps(out, indent=1))

if __name__ == "__main__":
    main()
