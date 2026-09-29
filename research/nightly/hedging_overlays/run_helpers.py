"""Small helpers shared by hedge_ratio.py (kept separate so importing does not execute run.py)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from research.nightly import harness as H
from research.nightly.hedging_overlays import data as D


def crisis(s: pd.Series) -> float:
    m = H.monthly(s[(s.index >= "2022-12-01") & (s.index < "2023-07-01")])
    return float(((1 + m).prod() - 1) * 100)


def _mlvl():
    pre, real = D.pre_curve(), D.real_curve()
    lvl = pd.DataFrame({"real5y": real["DIC_1260"], "pre3y": pre["PRE_756"]})
    return lvl.resample("MS").last().diff()


def rate_betas(s: pd.Series) -> dict:
    """Monthly excess (% per +1pp) regressed on the month's change in the 5y real / 3y nominal yield, pre-2026."""
    m = H.monthly(s[s.index < H.HOLDOUT])
    j = pd.concat([m.rename("y"), _mlvl()], axis=1, join="inner").dropna()
    out = {}
    for c in ("real5y", "pre3y"):
        X = np.c_[np.ones(len(j)), j[c]]
        out[f"beta_%_per_pp_{c}"] = float(np.linalg.lstsq(X, j["y"], rcond=None)[0][1] * 100)
    out["corr_real5y"] = float(j["y"].corr(j["real5y"]))
    return out
