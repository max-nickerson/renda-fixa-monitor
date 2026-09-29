"""Re-run top25_carry_cap5 vs P4+Q with (a) returns corrected for the static-contract look-ahead and
(b) PIT-corrected carry for the ranking. Pre-2026 only (holdout untouched)."""
import json, sys
import numpy as np, pandas as pd
from pathlib import Path
from research.nightly import harness as H
from research.nightly.portfolio_construction import run as RUN, construct as CO

OUTD = Path(__file__).resolve().parent
CACHE = Path(__file__).resolve().parents[4] / "data/history/nightly/portfolio_construction_verify_leakage"
OFF = 0.06     # systematic offset of implied vs contract (CDI accrual convention), common to all bonds
THR = 0.15     # pp: ignore smaller gaps (noise)

def truth_gap():
    """per (codigo, date): centred 90D rolling median of step-implied spread -> gap = contract - s - OFF."""
    S = pd.read_pickle(CACHE / "implied_steps.pkl")
    out = []
    for c, s in S.groupby("codigo"):
        s = s.set_index("date").sort_index()
        m = s["s_imp"].rolling("90D", min_periods=1, center=True).median()
        gp = s["contract"] - m - OFF
        out.append(pd.DataFrame({"codigo": c, "date": s.index, "gap": gp.to_numpy()}))
    return pd.concat(out, ignore_index=True)

def corrected_R():
    C = H._core()
    R = H._Rmat("base").copy()
    dd, codes = C["days"], C["codes"]
    G = truth_gap()
    G = G[G["codigo"].isin(codes)]
    B = np.zeros_like(R)
    for c, g in G.groupby("codigo"):
        b = codes.get_loc(c)
        s = g.set_index("date")["gap"].sort_index()
        s = s[~s.index.duplicated()]
        v = s.reindex(dd, method="nearest", tolerance=pd.Timedelta(days=200)).to_numpy()
        v = np.where(np.abs(v) > THR, v, 0.0)
        v = np.nan_to_num(v)
        alive = (C["TD"][:, b] >= 0)
        B[:, b] = np.where(alive, v / 100 / 252, 0.0)
    return R - B, B

def pit_gap_panel(P):
    O = pd.read_pickle(CACHE / "implied_contract_pit.pkl").dropna().sort_values("date")
    O["date"] = O["date"].astype("datetime64[ns]"); O["codigo"] = O["codigo"].astype(str)
    X = P[["codigo", "day"]].copy().reset_index()
    X["day"] = X["day"].astype("datetime64[ns]"); X["codigo"] = X["codigo"].astype(str); X = X.sort_values("day")
    m = pd.merge_asof(X, O.rename(columns={"date": "d"}), left_on="day", right_on="d", by="codigo",
                      direction="backward", tolerance=pd.Timedelta(days=180))
    m["gap"] = m["contract"] - m["s_pit"] - OFF
    m["gap"] = np.where(m["gap"].abs() > THR, m["gap"], 0.0)
    return m.set_index("index")["gap"].reindex(P.index).fillna(0.0)

def topk(P, col="cdi_bps", k=25):
    X = P[P["univ"] & (P["day"] >= H.START)].copy()
    X["cnpj8"] = X["cnpj8"].astype(str)
    rows = []
    for d, x in X.groupby("day", sort=True):
        s = x[x["p4q"]].sort_values(col, ascending=False).head(k).reset_index(drop=True)
        if len(s) < 5: continue
        w = CO.project(s, np.ones(len(s)), {"issuer_cap": 0.05})
        rows.append(pd.DataFrame({"day": d, "codigo": s["codigo"], "weight": w}))
    return pd.concat(rows)

def st(r, b):
    s = H.stats(r["daily"], bench=b["daily"])
    return {k: round(float(s[k]), 3) for k in ("ann_excess_%", "diff_ann_%", "diff_t_nw", "diff_h1_%", "diff_h2_%")}

def main():
    P = H.load_panel("M")
    P["gap_pit"] = pit_gap_panel(P)
    P["cdi_corr"] = P["cdi_bps"] - 100 * P["gap_pit"]
    out = {}
    W0 = topk(P)
    # holdings exposure to the contract look-ahead
    Q = P[P["univ"] & P["p4q"] & (P["day"] >= H.START)]
    h = W0.merge(P[["day", "codigo", "gap_pit", "kind"]], on=["day", "codigo"])
    out["top25_weight_share_gap_pos"] = round(float((h["weight"] * (h["gap_pit"] > 0)).sum() / h["weight"].sum()), 3)
    out["top25_wavg_gap_pp"] = round(float((h["weight"] * h["gap_pit"]).sum() / h["weight"].sum()), 3)
    out["p4q_share_gap_pos"] = round(float((Q["gap_pit"] > 0).mean()), 3)
    out["p4q_avg_gap_pp"] = round(float(Q["gap_pit"].mean()), 3)
    # A: original
    bq = H.baseline("P4Q")
    out["A_original"] = st(H.backtest(W0, freq="M", hold=126, issuer_cap=1.0), bq)
    # swap in corrected returns
    Rc, B = corrected_R()
    out["bias_matrix_sum_abs"] = float(np.abs(B).sum())
    H._MEM[("R64", "R")] = Rc
    for k in [k for k in list(H._MEM) if isinstance(k, tuple) and k[0] in ("LC", "baseline")]:
        del H._MEM[k]
    bqc = H.baseline("P4Q")
    bu = H.baseline("U")
    out["P4Q_exCDI_orig_vs_corr"] = [round(H.stats(bq["daily"])["ann_excess_%"], 3), round(H.stats(bqc["daily"])["ann_excess_%"], 3)]
    out["B_corrR_orig_selection"] = st(H.backtest(W0, freq="M", hold=126, issuer_cap=1.0), bqc)
    W1 = topk(P, "cdi_corr")
    r1 = H.backtest(W1, freq="M", hold=126, issuer_cap=1.0)
    out["C_corrR_pit_carry"] = st(r1, bqc)
    out["C_vs_U"] = round(float(H.stats(r1["daily"], bench=bu["daily"])["diff_ann_%"]), 3)
    out["C_50bps"] = st(H.backtest(W1, freq="M", hold=126, issuer_cap=1.0, cost_bps=50), H.baseline("P4Q", cost_bps=50))
    # D: also drop names whose corrected carry falls out of the top 30% of the universe
    P["pct_corr"] = P[P["univ"]].groupby("day")["cdi_corr"].rank(pct=True, ascending=False)
    P2 = P.copy(); P2["p4q"] = P2["p4q"] & (P2["pct_corr"] <= 0.30)
    out["D_corrR_pit_carry_p4q_recut"] = st(H.backtest(topk(P2, "cdi_corr"), freq="M", hold=126, issuer_cap=1.0), bqc)
    # which names drive it
    h2 = W0.merge(P[["day", "codigo", "gap_pit"]], on=["day", "codigo"])
    top = h2[h2["gap_pit"] > 0].groupby("codigo")["weight"].sum().sort_values(ascending=False).head(15)
    out["top_gap_names_weightsum"] = {k: round(float(v), 3) for k, v in top.items()}
    print(json.dumps(out, indent=1))
    (OUTD / "verify_results.json").write_text(json.dumps(out, indent=1))

if __name__ == "__main__":
    main()
