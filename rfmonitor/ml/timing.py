"""Study A — credit market timing: when to be long credit vs sitting in CDI.

Series (ANBIMA IDA indices + BCB CDI, daily since 2009):
  DI credit    : IDA-DI − CDI                                    (floating-rate debentures, pure credit)
  IPCA credit  : IDA-IPCA − CDI − β·(IMA-B − CDI), β = D_IDA/D_IMAB  (duration-hedged IPCA credit)

Target: next-H-day cumulative excess return > 0. Features are strictly trailing. Walk-forward: expanding
window, retrain every RETRAIN days, PURGE-day gap between train and test so labels never overlap the
test period. Positions decided at close t, applied from t+1; switching costs charged per change.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .. import history as h

H = 21            # forecast horizon (business days)
PURGE = H
RETRAIN = 63      # retrain quarterly
TEST_START = pd.Timestamp("2013-01-01")


def excess_series() -> dict[str, pd.DataFrame]:
    cdi = h.bcb_series(12, date(2008, 12, 1)) / 100            # % per day -> decimal per day
    idadi, idaipca, imab = h.ida("IDADI"), h.ida("IDAIPCA"), h.ida("IMAB")
    out = {}
    df = pd.DataFrame({"ida": idadi["index"].pct_change(), "cdi": cdi}).dropna()
    df["ex"] = df["ida"] - df["cdi"]
    df["dur"] = idadi["duration_du"].reindex(df.index) / 252
    out["DI credit (IDA-DI vs CDI)"] = df
    j = pd.DataFrame({"ida": idaipca["index"].pct_change(), "imab": imab["index"].pct_change(), "cdi": cdi,
                      "d_ida": idaipca["duration_du"], "d_imab": imab["duration_du"]}).dropna()
    beta = (j["d_ida"] / j["d_imab"]).shift(1)
    j["ex"] = j["ida"] - j["cdi"] - beta * (j["imab"] - j["cdi"])
    j["dur"] = j["d_ida"] / 252
    out["IPCA credit (IDA-IPCA, hedged w/ IMA-B)"] = j
    return out


def features(df: pd.DataFrame) -> pd.DataFrame:
    ex, cdi = df["ex"], df["cdi"]
    cum = ex.cumsum()
    f = pd.DataFrame(index=df.index)
    for n in (5, 21, 63, 126):
        f[f"mom_{n}"] = ex.rolling(n).sum()
    for n in (21, 63):
        f[f"vol_{n}"] = ex.rolling(n).std() * np.sqrt(252)
    f["dd_252"] = cum - cum.rolling(252).max()
    f["mom_21_z"] = f["mom_21"] / (f["vol_63"] / np.sqrt(252 / 21))
    f["cdi_ann"] = (1 + cdi) ** 252 - 1
    f["cdi_chg_63"] = f["cdi_ann"] - f["cdi_ann"].shift(63)
    f["dur"] = df["dur"]
    f["dur_chg_63"] = df["dur"] - df["dur"].shift(63)
    return f


@dataclass
class Result:
    name: str
    daily: pd.DataFrame      # per-day excess returns by strategy
    positions: pd.DataFrame
    stats: pd.DataFrame
    latest: dict


def _stats(r: pd.Series, pos: pd.Series | None = None) -> dict:
    eq = (1 + r).cumprod()
    yrs = len(r) / 252
    vol = r.std() * np.sqrt(252)
    ann = eq.iloc[-1] ** (1 / yrs) - 1 if yrs > 0 else np.nan
    out = {"excess_ann_%": ann * 100, "vol_%": vol * 100, "sharpe": ann / vol if vol else np.nan,
           "max_dd_%": ((eq / eq.cummax()) - 1).min() * 100, "hit_%": (r > 0).mean() * 100}
    if pos is not None:
        out["time_in_%"] = pos.mean() * 100
        out["switches_per_yr"] = pos.diff().abs().sum() / yrs
    return out


def run_one(name: str, df: pd.DataFrame, cost_bps: float = 15.0, only: tuple[str, ...] | None = None) -> Result:
    f = features(df)
    y_fwd = df["ex"].rolling(H).sum().shift(-H)  # forward H-day excess (label)
    data = f.join(y_fwd.rename("y")).dropna(subset=list(f.columns))
    test_idx = data.index[data.index >= TEST_START]
    all_models = {
        "Logistic": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=500)),
        "Gradient boosting": lambda: HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200,
                                                                    min_samples_leaf=100, l2_regularization=1.0),
    }
    models = {k: v for k, v in all_models.items() if only is None or k in only}
    prob = {k: pd.Series(np.nan, index=test_idx) for k in models}
    for start in range(0, len(test_idx), RETRAIN):
        block = test_idx[start:start + RETRAIN]
        train = data[data.index < block[0]].iloc[:-PURGE].dropna(subset=["y"])
        X, yy = train[f.columns], (train["y"] > 0).astype(int)
        for k, mk in models.items():
            if yy.nunique() < 2:  # e.g. early DI-credit history: every window was positive
                prob[k].loc[block] = float(yy.mean())
                continue
            m = mk().fit(X, yy)
            prob[k].loc[block] = m.predict_proba(data.loc[block, f.columns])[:, 1]

    ex = df["ex"].reindex(test_idx)
    pos = pd.DataFrame(index=test_idx)
    pos["Buy & hold crédito"] = 1.0
    pos["Momentum 21d > 0"] = (data.loc[test_idx, "mom_21"] > 0).astype(float)
    for k in models:
        # Hysteresis: enter above 0.55, exit below 0.45 — cuts churn from noise around 50%.
        p, s, st = prob[k], [], 1.0
        for v in p:
            st = 1.0 if v >= 0.55 else 0.0 if v <= 0.45 else st
            s.append(st)
        pos[k] = s
    held = pos.shift(1).fillna(1.0)  # decide at t, earn from t+1
    daily = held.mul(ex, axis=0) - held.diff().abs().fillna(0) * cost_bps / 1e4
    stats = pd.DataFrame({c: _stats(daily[c], held[c]) for c in daily.columns}).T
    last = data.index[-1]
    latest = {"date": str(last.date()), "prob": {k: float(prob[k].dropna().iloc[-1]) for k in models},
              "position": {c: float(pos[c].iloc[-1]) for c in pos.columns},
              "features": {k: round(float(v), 5) for k, v in data.loc[last, f.columns].items()}}
    return Result(name, daily, pos, stats, latest)


def run_all() -> list[Result]:
    return [run_one(n, df) for n, df in excess_series().items()]
