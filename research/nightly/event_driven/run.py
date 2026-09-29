"""Event-driven overlays / sleeves on P4+Q (harness v4). Pre-2026 only unless --holdout.

  PYTHONPATH=. PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
      .venv/Scripts/python.exe research/nightly/event_driven/run.py [--holdout]

Reusable PIT signal: signals(panel) -> DataFrame [codigo, day, cnpj8, ev_*]  (days since the issuer's last event of
each type, strictly before the decision day; months since the bond's distribution start; new-issue flags).
Cached for the harness M and W panels at data/history/nightly/event_driven/event_signals_{M,W}.pkl.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from research.nightly import harness as H
from research.nightly.event_driven.events import build

OUTD = Path(__file__).resolve().parent
CACHE = H.HIST / "nightly" / "event_driven"
ETYPES = ["agd", "agd_waiver", "resgate", "deb_buyback", "equity_raise", "ma", "rj", "rating_up", "rating_down",
          "ipe_new_deb", "supply_new_issue"]
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)


def signals(panel: pd.DataFrame) -> pd.DataFrame:
    """Point-in-time event features for every (codigo, day) row of a harness panel."""
    ref, E = build()
    X = panel[["codigo", "day", "cnpj8"]].copy()
    X["day"] = X["day"].astype("datetime64[ns]"); X["cnpj8"] = X["cnpj8"].astype(str)
    X["_i"] = np.arange(len(X))
    X = X.sort_values("day")
    for et in ETYPES:
        e = E[E["etype"] == et][["cnpj8", "date"]].sort_values("date").rename(columns={"date": "_ed"})
        e["_ed"] = e["_ed"].astype("datetime64[ns]"); e["cnpj8"] = e["cnpj8"].astype(str)
        m = pd.merge_asof(X, e, left_on="day", right_on="_ed", by="cnpj8", allow_exact_matches=False,
                          direction="backward")          # last event dated STRICTLY before the decision day
        X[f"ev_{et}_days"] = (m["day"] - m["_ed"]).dt.days.to_numpy()
    X = X.sort_values("_i").drop(columns="_i")
    ds = X["codigo"].map(ref.set_index("codigo")["dist_start"])
    X["months_since_issue"] = ((X["day"] - ds).dt.days / 30.44).to_numpy()
    X["new_issue_3m"] = (X["months_since_issue"] >= 0) & (X["months_since_issue"] <= 3)
    X["callable"] = X["codigo"].map(ref.set_index("codigo")["callable"]).fillna(False).astype(bool)
    return X.reset_index(drop=True)


def load(freq: str, holdout: bool = False) -> pd.DataFrame:
    P = H.load_panel(freq, holdout=holdout)
    p = CACHE / f"event_signals_{freq}{'_ho' if holdout else ''}.pkl"
    if p.exists():
        S = pd.read_pickle(p)
    else:
        S = signals(P)
        S.to_pickle(p)
    S = S.drop(columns=["cnpj8"])
    return P.merge(S, on=["codigo", "day"], how="left")


def within(x, et, d):
    v = x[f"ev_{et}_days"]
    return (v <= d).fillna(False).to_numpy()


def notrich_clean(x):
    return ((x["resid_z"] > -1.5) & (x["press_neg_30d"].fillna(0) == 0)).to_numpy()


def neg(x):
    return within(x, "ma", 180) | within(x, "rating_down", 180) | within(x, "agd", 180) | \
        within(x, "resgate", 126) | within(x, "rj", 365)


RULES = {
    "P4Q_exMA180": lambda x: x["p4q"].to_numpy() & ~within(x, "ma", 180),
    "P4Q_exRatingDown180": lambda x: x["p4q"].to_numpy() & ~within(x, "rating_down", 180),
    "P4Q_exAGD180": lambda x: x["p4q"].to_numpy() & ~within(x, "agd", 180),
    "P4Q_exResgate126": lambda x: x["p4q"].to_numpy() & ~within(x, "resgate", 126),
    "P4Q_exNegEvents": lambda x: x["p4q"].to_numpy() & ~neg(x),
    "P4Q_plusNewIssue": lambda x: x["p4q"].to_numpy() | (x["new_issue_3m"].to_numpy() & notrich_clean(x)),
    "P4Q_plusNewIncent": lambda x: x["p4q"].to_numpy() | (x["new_issue_3m"].to_numpy() & (x["incent"] == 1).to_numpy()
                                                          & notrich_clean(x)),
    "P4Q_plusRatingUp": lambda x: x["p4q"].to_numpy() | (within(x, "rating_up", 180) & notrich_clean(x)),
    "P4Q_combo": lambda x: (x["p4q"].to_numpy() | (x["new_issue_3m"].to_numpy() & notrich_clean(x))
                            | (within(x, "rating_up", 180) & notrich_clean(x))) & ~neg(x),
    "Sleeve_NewIssue": lambda x: x["new_issue_3m"].to_numpy() & notrich_clean(x),
    "Sleeve_NewIncent": lambda x: x["new_issue_3m"].to_numpy() & (x["incent"] == 1).to_numpy() & notrich_clean(x),
}
# weekly, short-hold new-issue sleeve (the premium decays within ~3 months)
WEEKLY = {"Sleeve_NewIssue_W63": ("Sleeve_NewIssue", 63)}


def main(holdout: bool = False):
    PM = load("M")
    PW = load("W")
    log("panels", PM.shape, PW.shape)
    res = {}
    for k, f in RULES.items():
        res[k] = H.backtest(f, panel=PM, name=k)
        log(k, round(res[k]["n_avg"], 1), H.stats(res[k]["daily"])["ann_excess_%"])
    for k, (base, hold) in WEEKLY.items():
        res[k] = H.backtest(RULES[base], freq="W", panel=PW, hold=hold, name=k)
        log(k, round(res[k]["n_avg"], 1), H.stats(res[k]["daily"])["ann_excess_%"])
    tried = dict(res)
    tab = H.compare({**tried, "P4": H.baseline("P4"), "U": H.baseline("U")}, bench="P4Q")
    pd.set_option("display.width", 250, "display.max_columns", 40)
    print(tab)
    return PM, PW, res, tab


if __name__ == "__main__":
    main()
