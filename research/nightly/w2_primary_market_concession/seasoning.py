"""Stage 5: harness-native (secondary, market-marked, next-trade entry) test of the post-first-print drift found in
the event study: new issues bought at/after their first SND print outperform their peer cohort over 126 bdays.

new_d = calendar days since the bond's first grid row (first SND print), known at the decision close.
Variants (monthly decisions, 126-bday tranches, 25 bps, harness v4):
  C1  universe & new_d <= 31                      (all recently-printed new issues)
  C2  C1 & P4 (top-30% carry, not rich, no neg press)
  C3  P4+Q  union  (new_d <= 31 & p4f-like: top-50% carry, not rich, press ok, not worstQ)
  C4  P4+Q minus new issues (new_d <= 92)         (does P4+Q already own the effect?)
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from research.nightly import harness as H

RES = H._ROOT / "research" / "nightly" / "w2_primary_market_concession"


def first_print() -> pd.Series:
    g = pd.read_pickle(H.HIST / "lab_daily.pkl")[["codigo", "day"]]
    return g.groupby("codigo")["day"].min()


def add_new(P: pd.DataFrame, fp: pd.Series) -> pd.DataFrame:
    P = P.copy()
    f = fp.reindex(P["codigo"]).to_numpy()
    P["new_d"] = (P["day"].to_numpy() - f).astype("timedelta64[D]").astype(float)
    # bonds that entered the grid in its first weeks (2021-01) are not new issues
    P.loc[pd.Series(f).lt(pd.Timestamp("2021-03-01")).to_numpy(), "new_d"] = np.nan
    return P


def rules():
    new = lambda x, d=31: (x["new_d"] <= d).fillna(False).to_numpy()
    p4f_like = lambda x: ((x["cdi_pct"] <= 0.5).to_numpy() & (x["resid_z"].fillna(0) > -1.5).to_numpy()
                          & (x["press_neg_30d"].fillna(0) == 0).to_numpy() & ~x["worstQ"].fillna(False).to_numpy())
    return {
        "C1_new31_all": lambda x: new(x),
        "C2_new31_p4": lambda x: new(x) & x["p4"].to_numpy(),
        "C3_p4q_plus_new31": lambda x: x["p4q"].to_numpy() | (new(x) & p4f_like(x)),
        "C4_p4q_ex_new92": lambda x: x["p4q"].to_numpy() & ~new(x, 92),
    }


def main(holdout: bool = False):
    fp = first_print()
    P = add_new(H.load_panel("M", holdout=holdout), fp)
    U = P[P["univ"]]
    share = U.groupby("day").apply(lambda x: (x["new_d"] <= 31).mean()).mean()
    print("avg share of universe that is new (<=31d):", round(float(share), 3))
    out = {}
    res = {k: H.backtest(f, panel=P, name=k, holdout=holdout) for k, f in rules().items()}
    if holdout:
        bh = H.baseline("P4Q", holdout=True)["daily"]
        for k, v in res.items():
            s = H.stats(v["daily"], bench=bh, holdout="only")
            out[k] = {"exCDI_%": s["ann_excess_%"], "vsP4Q_%": s["diff_ann_%"], "t": s["diff_t_nw"]}
        return out, res
    T = H.compare(res, bench="P4Q")
    T50 = H.compare({k: H.backtest(f, panel=P, cost_bps=50) for k, f in rules().items()}, bench="P4Q", cost_bps=50)
    T["vs_bench_50bps"] = T50["vs_bench_%"]
    Trec = H.compare({k: H.backtest(f, panel=P, scenario="rec40") for k, f in rules().items()}, bench="P4Q",
                     scenario="rec40")
    T["vs_bench_rec40"] = Trec["vs_bench_%"]
    pd.set_option("display.width", 250)
    print(T.round(3).to_string())
    # IC-style cohort check: C1 names vs universe per decision (fwd_126 cohort)
    coh = H.cohort_excess(rules()["C1_new31_all"], H=126, panel=P)
    out["C1_cohort126_ann_%"] = float(coh.mean() / 0.5 * 100)
    out["C1_cohort126_t"] = float(H.nw_t(coh, 6))
    pl = H.placebo(rules()["C1_new31_all"], n=20, panel=P)
    out["C1_placebo"] = {"mean": pl["mean"], "p95": pl["p95"], "actual": pl["actual"]}
    out["table"] = T.round(4).reset_index().rename(columns={"index": "variant"}).to_dict("records")
    out["avg_new_share_universe"] = float(share)
    print({k: v for k, v in out.items() if k != "table"})
    R = json.loads((RES / "results.json").read_text(encoding="utf-8"))
    R["seasoning_harness"] = out
    (RES / "results.json").write_text(json.dumps(R, indent=1, default=str), encoding="utf-8")
    return out, res


if __name__ == "__main__":
    main()
