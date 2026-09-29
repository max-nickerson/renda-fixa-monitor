"""Reusable point-in-time structural-credit signals.

    from research.nightly.structural_credit.signals import signals, load_daily
    P = H.load_panel("M");  P = signals(P)          # adds sc_* columns

Columns added (all known at the close of the panel day; ticker = the panel's PIT `eq_ticker`, own or listed parent):
    sc_dd          Merton/KMV distance to default (1y, default point = ST debt + 0.5 LT debt, risk-neutral drift)
    sc_dd_naive    Bharath-Shumway naive DD
    sc_pd          N(-sc_dd)
    sc_cg          CreditGrades 5y equity-implied spread, bps (L=0.5, lambda=0.3, R=0.5, D = gross debt)
    sc_lcg         log(10 + sc_cg)
    sc_mlev        market leverage gross debt / (gross debt + market cap)
    sc_dd_chg21/63 change in DD over 21/63 trading days
    sc_lcg_chg63   change in log CG spread over 63 trading days
    sc_div         per-date rank(cdi_bps) - rank(sc_cg) among universe rows with sc_cg  (>0: bond wide vs equity)
    sc_div_res     per-date OLS residual of log(cdi_bps) on log(10+cg) and duration (>0: bond wide vs equity)
Rows without a listed ticker / debt / market cap are NaN.
Underlying daily table: data/history/nightly/structural_credit/structural_daily.pkl (build_data.py).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DAILY = ROOT / "data" / "history" / "nightly" / "structural_credit" / "structural_daily.pkl"


def load_daily() -> pd.DataFrame:
    D = pd.read_pickle(DAILY).sort_values(["ticker", "date"]).reset_index(drop=True)
    g = D.groupby("ticker")
    D["lcg"] = np.log(10 + D["cg_spread_bps"])
    for n in (21, 63):
        D[f"dd_chg{n}"] = D["dd"] - g["dd"].shift(n)
    D["lcg_chg63"] = D["lcg"] - g["lcg"].shift(63)
    return D


def signals(panel: pd.DataFrame, daily: pd.DataFrame | None = None) -> pd.DataFrame:
    D = load_daily() if daily is None else daily
    cols = {"dd": "sc_dd", "dd_naive": "sc_dd_naive", "pd1y": "sc_pd", "cg_spread_bps": "sc_cg", "lcg": "sc_lcg",
            "mkt_lev": "sc_mlev", "dd_chg21": "sc_dd_chg21", "dd_chg63": "sc_dd_chg63", "lcg_chg63": "sc_lcg_chg63",
            "mcap": "sc_mcap", "sigE": "sc_sigE"}
    R = D[["ticker", "date"] + list(cols)].rename(columns=cols).rename(columns={"ticker": "eq_ticker"})
    R["date"] = R["date"].astype("datetime64[ns]")
    R = R.sort_values("date")
    X = panel.drop(columns=[c for c in panel.columns if c.startswith("sc_")]).copy()
    X["_i"] = np.arange(len(X))
    L = X[["_i", "day", "eq_ticker"]].copy()
    L["day"] = L["day"].astype("datetime64[ns]")
    L = L[L["eq_ticker"].notna()].sort_values("day")
    M = pd.merge_asof(L, R, left_on="day", right_on="date", by="eq_ticker", direction="backward",
                      tolerance=pd.Timedelta(days=7))
    M = M.set_index("_i").reindex(X["_i"])
    for c in cols.values():
        X[c] = M[c].to_numpy()
    # divergence vs the bond's own spread (same-day cross-section of universe rows with an equity-implied spread)
    X["sc_div"] = np.nan
    X["sc_div_res"] = np.nan
    ok = X["univ"] & X["sc_cg"].notna() & X["cdi_bps"].notna()
    for d, x in X[ok].groupby("day"):
        if len(x) < 20:
            continue
        X.loc[x.index, "sc_div"] = x["cdi_bps"].rank(pct=True) - x["sc_cg"].rank(pct=True)
        y = np.log(x["cdi_bps"].clip(lower=20))
        A = np.c_[np.ones(len(x)), x["sc_lcg"].to_numpy(), x["dur"].fillna(x["dur"].median()).to_numpy()]
        beta, *_ = np.linalg.lstsq(A, y.to_numpy(), rcond=None)
        X.loc[x.index, "sc_div_res"] = y.to_numpy() - A @ beta
    return X.drop(columns=["_i"])
