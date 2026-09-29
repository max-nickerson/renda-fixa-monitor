"""Issuer-curve & cross-indexer relative-value fair values (point-in-time, same-day cross-sections).

Every quantity for decision day d uses only the FRESH marks (<= 7 days old) of day d, exactly like the existing
`resid_bps` / `resid_z` (peer curve = kind x incent duration-bucket medians of the same day). Nothing is pooled over
time, so the signal is known at the close of d and is executed by the harness at the next trade.

Fair-value models (all residuals in CDI+-equivalent bps, positive = CHEAP):
  r        = resid_bps (existing peer curve: kind x Lei-12.431 flag, duration buckets)          -> 'resid_bps'
  e_iss    = r - u_(-j)           u_(-j) = leave-one-out mean of the OTHER fresh bonds' r of the same issuer (cnpj8),
                                  shrunk toward 0 with k=1 pseudo-bond. Only defined when the issuer has >=2 fresh bonds.
  e_grp    = same, issuer key = economic group (parent ticker root from research/data/equity_map.csv, static map)
  e_isec   = r - u'              u' = issuer LOO mean shrunk toward the sector LOO mean (sector mean excl. the issuer);
                                  defined for every bond (single-bond issuers -> r - sector mean)            ('issuer+sector')
  r_sec    = r - sector LOO mean (sector-aware peer curve, no issuer information)
  e_slope  = e_iss minus a same-day pooled within-issuer duration-slope correction
             (r_j - rbar_i) = g1 (dur_j - dbar_i) + g2 (dur_j - dbar_i) * rbar_i       (riskier issuers steeper)
  *_z      = (x - median_d) / (1.4826 MAD_d) over the fresh rows of the day, clipped to +-5.
Pair-type flags per bond: whether its issuer has other fresh bonds of the same kind/incent ('curve'), another indexer
('xidx') or another tax status ('xinc').

`signals(panel)` returns one row per (day, codigo) with these columns; `load_cached()` returns the cached weekly+monthly
signal table (day, codigo, cnpj8, value columns) written by run.py.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
CACHE = ROOT / "data" / "history" / "nightly" / "issuer_curve_rv"
WIN = 500.0     # winsorise residuals used to estimate issuer / sector effects (distressed prints)
K_SHRINK = 1.0  # pseudo-bonds of shrinkage toward the prior


def _group_map() -> pd.Series:
    mp = pd.read_csv(ROOT / "research" / "data" / "equity_map.csv", dtype=str)
    mp = mp[mp["ticker"].notna() & mp["mapping_type"].isin(["direct", "parent"])].copy()
    mp["cnpj8"] = mp["cnpj8"].str.replace(r"\D", "", regex=True).str.zfill(8)
    mp["grp"] = "G_" + mp["ticker"].str[:4]
    return mp.drop_duplicates("cnpj8").set_index("cnpj8")["grp"]


def _robust_z(x: pd.Series, by: pd.Series) -> pd.Series:
    med = x.groupby(by).transform("median")
    mad = (x - med).abs().groupby(by).transform("median") * 1.4826
    return ((x - med) / mad.replace(0, np.nan)).clip(-5, 5)


def _loo(df: pd.DataFrame, key: str, val: str) -> tuple[pd.Series, pd.Series]:
    """leave-one-out sum and count of `val` within (day, key)."""
    g = df.groupby(["day", key])[val]
    s = g.transform("sum") - df[val]
    n = g.transform("count") - 1
    return s, n


def signals(panel: pd.DataFrame) -> pd.DataFrame:
    """Fair values / RV residuals for every fresh row of `panel` (any harness panel: weekly or monthly, with or
    without holdout rows). Returns DataFrame [day, codigo, cnpj8, ...signal columns]."""
    F = panel.loc[panel["fresh"] & panel["cdi_bps"].notna() & panel["resid_bps"].notna(),
                  ["day", "codigo", "cnpj8", "kind", "incent", "dur", "cdi_bps", "resid_bps", "sector"]].copy()
    F["cnpj8"] = F["cnpj8"].astype(str)
    F["sector"] = F["sector"].astype(str)
    gm = _group_map()
    F["grp"] = F["cnpj8"].map(gm).fillna("I_" + F["cnpj8"])
    F["rw"] = F["resid_bps"].clip(-WIN, WIN)
    r = F["resid_bps"]

    # issuer leave-one-out, shrunk to 0 (peer curve)
    s, n = _loo(F, "cnpj8", "rw")
    F["n_iss_other"] = n
    u = s / (n + K_SHRINK)
    F["e_iss"] = (r - u).where(n >= 1)
    # economic group
    sg, ng = _loo(F, "grp", "rw")
    F["n_grp_other"] = ng
    F["e_grp"] = (r - sg / (ng + K_SHRINK)).where(ng >= 1)
    # sector LOO mean excluding the whole issuer
    ss = F.groupby(["day", "sector"])["rw"].transform("sum")
    ns = F.groupby(["day", "sector"])["rw"].transform("count")
    si = F.groupby(["day", "cnpj8"])["rw"].transform("sum")
    ni = F.groupby(["day", "cnpj8"])["rw"].transform("count")
    sec_mean = ((ss - si) / (ns - ni)).where((ns - ni) >= 3).fillna(0.0)
    sec_mean = sec_mean.where(F["sector"] != "unknown", 0.0)
    F["r_sec"] = r - sec_mean
    F["e_isec"] = r - (s + K_SHRINK * sec_mean) / (n + K_SHRINK)

    # within-issuer duration slope correction (pooled, same day)
    F["dbar"] = F.groupby(["day", "cnpj8"])["dur"].transform("mean")
    F["rbar"] = F.groupby(["day", "cnpj8"])["rw"].transform("mean")
    F["e_slope"] = np.nan
    m = ni >= 2
    for d, x in F[m].groupby("day"):
        dd = (x["dur"] - x["dbar"]).to_numpy()
        X = np.c_[dd, dd * x["rbar"].to_numpy() / 100]
        y = (x["rw"] - x["rbar"]).to_numpy()
        if len(x) < 20:
            continue
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        # LOO issuer level + slope-implied position of bond j on its issuer curve
        idx = x.index
        dj = (x["dur"] - (x["dbar"] * ni[idx] - x["dur"]) / (ni[idx] - 1)).to_numpy()
        uj = (s[idx] / (n[idx] + K_SHRINK)).to_numpy()
        adj = beta[0] * dj + beta[1] * dj * uj / 100
        F.loc[idx, "e_slope"] = F.loc[idx, "e_iss"].to_numpy() - adj

    # pair-type flags
    ki = F.groupby(["day", "cnpj8", "kind", "incent"])["rw"].transform("count")
    kk = F.groupby(["day", "cnpj8", "kind"])["rw"].transform("count")
    F["has_curve_pair"] = ki >= 2
    F["has_xidx_pair"] = F.groupby(["day", "cnpj8"])["kind"].transform("nunique") >= 2
    F["has_xinc_pair"] = (kk > ki)

    for c in ("resid_bps", "e_iss", "e_grp", "e_isec", "r_sec", "e_slope"):
        F[c + "_z"] = _robust_z(F[c], F["day"])
    keep = ["day", "codigo", "cnpj8", "grp", "n_iss_other", "n_grp_other", "e_iss", "e_grp", "e_isec", "r_sec",
            "e_slope", "resid_bps_z", "e_iss_z", "e_grp_z", "e_isec_z", "r_sec_z", "e_slope_z",
            "has_curve_pair", "has_xidx_pair", "has_xinc_pair"]
    return F[keep].reset_index(drop=True)


def attach(panel: pd.DataFrame) -> pd.DataFrame:
    """panel + signal columns (left join on day, codigo)."""
    S = signals(panel)
    S = S.drop(columns=["cnpj8"])
    out = panel.merge(S, on=["day", "codigo"], how="left")
    return out


def load_cached(freq: str = "W") -> pd.DataFrame:
    """Cached signal table (day, codigo, cnpj8, value columns) for the weekly ('W') or monthly ('M') panel dates,
    all dates incl. 2026 (the signals are same-day cross-sections, so holdout rows carry no future info)."""
    return pd.read_pickle(CACHE / f"signals_{freq}.pkl")
