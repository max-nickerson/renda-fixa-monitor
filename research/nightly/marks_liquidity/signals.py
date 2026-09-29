"""Reusable point-in-time signals from marks_liquidity (load onto a harness panel).

    from research.nightly.marks_liquidity.signals import signals
    P = signals(H.load_panel("M"))      # adds the columns below, all known at the close of each decision day

Columns added (source: data/history/nightly/marks_liquidity/marks.pkl, built by build_marks.py):
  kf_s        Kalman fair CDI+ spread (bps): issuer common + bond random walks around the peer factor, fed by
              every print <= day with print-size-dependent noise and robust gating.
  kf_sd       posterior sd of kf_s (bps). Large = stale / noisy.
  kf_gap      cdi_bps (last print) - kf_s. > 0: last print cheaper (higher spread) than fair.
  kf_gap_z    kf_gap / sqrt(kf_sd^2 + 1).
  kf_resid_bps / kf_resid_z   kf_s vs the peer curve fitted on kf_s (same buckets / robust z as lab_daily).
  vw_s        volume-weighted spread of the last <= 5 clean prints (no odd lots, no outliers) within 10 bdays.
  adv63       average R$ traded per bday over the past 63 bdays (SND), tdays63 = share of bdays with a print,
  ticket63    average ticket (R$ per trade), roll_hs = bond Roll half-spread (price bps, 252 bdays, NaN if < 15).
  liq_q       ADV quintile within that date's universe (1 = least liquid, 5 = most liquid).
  cost_rt_bps estimated round-trip cost (price bps) = 2 x half-spread of the bond's ADV quintile x indexer group
              (Roll estimator on prints dated < 2024, tails included; table in results.json).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
CACHE = ROOT / "data" / "history" / "nightly" / "marks_liquidity"
HERE = Path(__file__).resolve().parent
BUCKETS = [0, 1, 2, 3, 4, 5, 7, 10, 40]   # duration buckets, as rfmonitor.ml.selection.BUCKETS (lab_daily resid)


def _peer_resid(df: pd.DataFrame, col: str, out: str) -> pd.DataFrame:
    """Per (day, peer) median-by-duration-bucket curve on `col` over universe rows (>= 8 names); robust z by day."""
    df[out + "_bps"] = np.nan
    X = df[df["univ"] & df[col].notna()]
    for (d, pe), grp in X.groupby(["day", "peer"]):
        if len(grp) < 8:
            continue
        b = pd.cut(grp["dur"], BUCKETS)
        med = grp.groupby(b, observed=True).agg(x=("dur", "median"), y=(col, "median")).dropna()
        fair = np.interp(grp["dur"], med["x"], med["y"]) if len(med) > 1 else med["y"].iloc[0]
        df.loc[grp.index, out + "_bps"] = grp[col] - fair
    dd = df.groupby("day")[out + "_bps"]
    mad = dd.transform(lambda s: (s - s.median()).abs().median() * 1.4826)
    df[out + "_z"] = ((df[out + "_bps"] - dd.transform("median")) / mad.replace(0, np.nan)).clip(-5, 5)
    return df


def cost_table() -> dict:
    p = HERE / "cost_table.json"
    return json.load(open(p)) if p.exists() else {}


def signals(panel: pd.DataFrame) -> pd.DataFrame:
    M = pd.read_pickle(CACHE / "marks.pkl")
    P = panel.copy()
    P["codigo"] = P["codigo"].astype(str)
    keys = P[["codigo", "day"]].drop_duplicates()
    M = keys.merge(M, on=["codigo", "day"], how="left")
    cols = ["kf_s", "kf_sd", "kf_prior", "kf_z", "cl_s", "vw_s", "adv63", "ntr63", "tdays63", "ticket63", "roll_hs",
            "roll_n"]
    P = P.drop(columns=[c for c in cols if c in P.columns]).merge(M[["codigo", "day"] + cols], on=["codigo", "day"],
                                                                   how="left")
    for c in cols:
        P[c] = P[c].astype(float)
    P["kf_gap"] = P["cdi_bps"] - P["kf_s"]
    P["kf_gap_z"] = P["kf_gap"] / np.sqrt(P["kf_sd"] ** 2 + 1)
    P = _peer_resid(P, "kf_s", "kf_resid")
    P["vw_s2"] = P["vw_s"].fillna(P["cdi_bps"])
    P = _peer_resid(P, "vw_s2", "vw_resid")
    U = P["univ"]
    P["liq_q"] = np.nan
    P.loc[U, "liq_q"] = np.ceil(P[U].groupby("day")["adv63"].rank(pct=True) * 5).clip(1, 5)
    ct = cost_table()
    if ct:
        grp = np.where(P["kind"].astype(str) == "DI_SPREAD", "DI", "IPCA_PRE")
        q = P["liq_q"].fillna(1).astype(int).astype(str)
        P["cost_rt_bps"] = [2 * ct[g][k] for g, k in zip(grp, q)]
    return P
