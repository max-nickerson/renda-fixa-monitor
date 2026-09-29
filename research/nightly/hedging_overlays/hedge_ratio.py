"""Follow-up (pre-2026 only; the sealed holdout was already spent by run.py on its pre-registered best variant):
the harness's IPCA+ rate hedge (hedge ratio 1 x model duration vs NTN-B at the bond's duration) is measured to
OVER-hedge: the hedged IPCA sub-book has a +1.2 %/pp beta to the 5y real yield, the raw one -2.4 %/pp.
Hypothesis: IPCA+ debenture yields move less than 1:1 with NTN-B (spreads compress when real yields rise).

1) Point-in-time 'yield beta': pooled OLS of d(debenture yield) on d(NTN-B at its duration) over consecutive
   fresh trades (<= 10 bdays apart) of the same IPCA+ bond, trailing 365 days, re-estimated at each month end and
   applied from the next grid day.
2) Books: IPCA+ sub-book of P4+Q and full P4+Q with hedge ratio h in {1 (harness), PIT beta, 0.5 fixed} and
   with / without the carry swap. Rate beta, vol, maxDD, crisis window. pre-2026 only.
Signal exported: data/history/nightly/hedging_overlays/ipca_yield_beta.pkl  (date, value) - the PIT hedge ratio.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from research.nightly import harness as H
from research.nightly.hedging_overlays import data as D
from research.nightly.hedging_overlays import rates as RT
from research.nightly.hedging_overlays.run_helpers import crisis, rate_betas

OUT = D.CACHE


def yield_beta() -> pd.Series:
    path = OUT / "ipca_yield_beta.pkl"
    if path.exists():
        return pd.read_pickle(path)
    g = pd.read_pickle(H.HIST / "lab_daily.pkl")
    g = g[(g["kind"] == "IPCA") & g["fresh"]][["codigo", "date", "ratio", "dur", "contract", "bench_rate"]]
    g = g.drop_duplicates(["codigo", "date"]).sort_values(["codigo", "date"])
    g["y"] = ((1 + g["contract"] / 100) * g["ratio"] ** (-1 / g["dur"]) - 1) * 100
    grp = g.groupby("codigo")
    g["dy"] = grp["y"].diff()
    g["db"] = grp["bench_rate"].diff()
    g["gap"] = grp["date"].diff().dt.days
    x = g[(g["gap"] <= 14) & g["dy"].abs().lt(1.5) & g["db"].notna()]
    out = {}
    for t in pd.date_range("2021-06-30", "2026-09-30", freq="ME"):
        w = x[(x["date"] > t - pd.Timedelta(days=365)) & (x["date"] <= t)]
        if len(w) < 500:
            continue
        X = np.c_[np.ones(len(w)), w["db"]]
        out[t] = float(np.linalg.lstsq(X, w["dy"], rcond=None)[0][1])
    s = pd.Series(out)
    s.to_pickle(path)
    return s


if __name__ == "__main__":
    yb = yield_beta()
    print("PIT IPCA yield beta (month ends):", yb.round(2).to_dict())
    dd = H.days()
    h_grid = yb.copy()
    h_grid.index = h_grid.index + pd.Timedelta(days=1)       # known after month end, used from the next day
    h_grid = h_grid.reindex(h_grid.index.union(dd)).ffill().reindex(dd).fillna(1.0).clip(0, 1.2).to_numpy()
    RA = RT.build()
    raw = RA["mats"]["raw"].astype(np.float64)          # = -(dur * d bench) on IPCA/Pre rows
    swap = RA["mats"]["swap"].astype(np.float64)
    P = H.load_panel("W", holdout=True)
    kind_b = P.drop_duplicates("b", keep="last").set_index("b")["kind"]
    ipca_cols = np.zeros(raw.shape[1], bool)
    ipca_cols[kind_b[kind_b == "IPCA"].index.to_numpy()] = True
    variants = {
        "h=1 (harness)": None,
        "h=PIT beta": (1 - h_grid)[:, None] * raw * ipca_cols,
        "h=0.5 fixed (hindsight)": 0.5 * raw * ipca_cols,
        "h=1 + swap": swap,
        "h=PIT beta + swap": (1 - h_grid)[:, None] * raw * ipca_cols + swap,
    }
    ipca_sig = lambda x: (x["p4q"] & x["kind"].eq("IPCA")).to_numpy()
    b0 = H.baseline("P4Q")["daily"]
    res = {}
    for nm, adj in variants.items():
        with RT.use_R(None if adj is None else adj.astype(np.float32)):
            rb = {c: H.backtest("p4q", cost_bps=c) for c in (25, 50)}
            ri = H.backtest(ipca_sig)
        row = {}
        for lab, s in (("book", rb[25]["daily"]), ("ipca_subbook", ri["daily"])):
            st = H.stats(s, bench=b0)
            row[lab] = {"exCDI_%": st["ann_excess_%"], "vol_%": st["vol_%"], "maxDD_%": st["max_dd_%"],
                        "worst_month_%": st["worst_month_%"], "sharpe": st["sharpe"], "crisis_%": crisis(s),
                        **rate_betas(s), "h1": st["h1_2022_23_%"], "h2": st["h2_2024_25_%"]}
            if lab == "book":
                row[lab].update({"vs_P4Q_%": st["diff_ann_%"], "t_vs_P4Q": st["diff_t_nw"], "p_vs_P4Q": st["diff_p"],
                                 "exCDI_50bps_%": H.stats(rb[50]["daily"])["ann_excess_%"]})
        res[nm] = row
        print(nm, json.dumps(row, default=float))
        if nm == "h=PIT beta + swap":
            pd.to_pickle(rb[25]["daily"], OUT / "book_pitbeta_swap_daily.pkl")
    out = {"yield_beta_month_end": {str(k.date()): v for k, v in yb.items()},
           "yield_beta_mean_pre2026": float(yb[yb.index < H.HOLDOUT].mean()), "variants": res,
           "note": "pre-2026 only; the holdout was spent in run.py on E_gate_ibov and is not re-used here"}
    (D.ROOT / "research" / "nightly" / "hedging_overlays" / "hedge_ratio_results.json").write_text(
        json.dumps(out, indent=1, default=float))
