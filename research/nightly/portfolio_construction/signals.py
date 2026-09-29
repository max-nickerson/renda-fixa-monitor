"""Reusable PIT signals from this study (for a combiner).

signals(panel) -> DataFrame [codigo, day, emp_vol, dts, vol_hat, carry_over_var, n_obs4w]
  emp_vol         de-smoothed annualised vol of the bond's excess return: overlapping 4-week sums of weekly returns
                  over the trailing 104 weeks realised by the decision close (NaN if < 26 obs)
  dts             duration x CDI+ spread (decimal)
  vol_hat         0.5 emp_vol + 0.5 kappa*DTS (kappa = cross-sectional median emp_vol/DTS of the date) or kappa*DTS
  carry_over_var  0.5 * spread / vol_hat^2 (the mean-variance tilt that drove the 'mv_emp' books)
PIT: uses only R rows realised by the decision close (see risk.py). Cached for the harness monthly/weekly panels
in data/history/nightly/portfolio_construction/signals_{M,W}.pkl (all dates, holdout included, as data only).
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from research.nightly import harness as H
from research.nightly.portfolio_construction import risk as RK
from research.nightly.portfolio_construction.data import OUT


def signals(panel: pd.DataFrame) -> pd.DataFrame:
    X = panel[panel["univ"]] if "univ" in panel else panel
    rows = []
    for p, x in X.groupby("dpos", sort=True):
        b = x["b"].to_numpy()
        ev, S4 = RK.emp_vol(int(p), b)
        n4 = np.isfinite(S4).sum(axis=0)
        dts = x["dur"].clip(0.25, 15).to_numpy() * np.clip(x["cdi_bps"].to_numpy(), 30, 3000) / 1e4
        ok = np.isfinite(ev)
        kappa = float(np.nanmedian(ev[ok] / dts[ok])) if ok.sum() >= 10 else 0.35
        vh = np.where(ok, 0.5 * ev + 0.5 * kappa * dts, kappa * dts)
        rows.append(pd.DataFrame({"codigo": x["codigo"].to_numpy(), "day": x["day"].to_numpy(), "emp_vol": ev,
                                  "dts": dts, "vol_hat": vh, "n_obs4w": n4,
                                  "carry_over_var": 0.5 * x["cdi_bps"].to_numpy() / 1e4 / vh ** 2}))
    return pd.concat(rows, ignore_index=True)


def cached(freq: str = "M") -> pd.DataFrame:
    path = OUT / f"signals_{freq}.pkl"
    if path.exists():
        return pd.read_pickle(path)
    S = signals(H.load_panel(freq, holdout=True))
    S.to_pickle(path)
    return S


if __name__ == "__main__":
    S = cached("M")
    print(S.describe())
