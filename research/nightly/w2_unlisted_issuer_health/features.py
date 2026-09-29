"""Extended issuer-health features for the monthly harness panel (point-in-time).

attach(P) adds, for every panel row:
  xq_dd252, xq_vol63, xq_r126   equity-health inputs: harness own/parent equity when present (tier 'own'),
                                else the PIT FRE parent/controller's equity (tier 'fre'), else the static hand-verified
                                name-token parent (tier 'name').  Same formulas and ticker-eligibility rule as the
                                harness (price at d and >= 150 prices in the prior 252 grid days).
  eq_tier                       own / fre / name / none
  px_sector                     listed sector-peer composite: mean pct-rank (within the day's universe) of the
                                sector medians of eq_dd252, -eq_vol63, eq_r126 over the universe's harness-listed rows
                                (>= 3 distinct listed issuers in the sector, else NaN)
  px_fund                       fundamentals health = harness f_quality (CVM available_date_strict; NaN if uncovered)
  px_bond                       bond-mark health: mean pct-rank of issuer means of -ds_63, rmom_126, ratio_from_hi_252
                                (factor_zoo features_v4.pkl, known at the decision close)
Nothing here uses information after the row's decision day (FRE edges by DT_RECEB <= day).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from research.nightly import harness as H

CACHE = H.HIST / "nightly" / "unlisted_issuer_health"
EQ = ("eq_dd252", "eq_vol63", "eq_r126")
XQ = ("xq_dd252", "xq_vol63", "xq_r126")


def _eq_feats(tickers: list[str], p: int, tick, TI, SPX, lr):
    for t in tickers:
        if t not in TI:
            continue
        j = TI[t]
        if np.isnan(SPX[p, j]) or np.isfinite(SPX[max(0, p - 252):p, j]).sum() < 150:
            continue
        px = SPX[:, j]
        r126 = px[p] / px[p - 126] - 1 if p - 126 >= 0 and px[p - 126] > 0 else np.nan
        win = lr[max(1, p - 62):p + 1, j]
        vol = np.nanstd(win) * np.sqrt(252) if np.isfinite(win).sum() >= 40 else np.nan
        hi = np.nanmax(px[max(0, p - 251):p + 1])
        return t, px[p] / hi - 1, vol, r126
    return None


def attach(P: pd.DataFrame) -> pd.DataFrame:
    P = P.copy()
    for a, b in zip(EQ, XQ):
        P[b] = P[a]
    P["eq_tier"] = np.where(P[list(EQ)].notna().all(axis=1), "own", "none")
    P["x_ticker"] = P["eq_ticker"]
    tick, TI, SPX, ADTV, cands = H._equity_grid()
    lr = np.diff(np.log(SPX), axis=0, prepend=np.nan)
    F = pd.read_pickle(CACHE / "parent_map_fre_pit.pkl")
    fmap = {(c, d): t.split("|") for c, d, t in zip(F["cnpj8"], F["day"], F["parent_tickers"])}
    N = pd.read_pickle(CACHE / "name_map_static.pkl")
    N = N[N["accepted"]]
    nmap = {c: t.split("|") for c, t in zip(N["cnpj8"], N["parent_tickers"])}
    need = P["eq_tier"].eq("none")
    cache = {}
    vals = np.full((len(P), 3), np.nan)
    tier = P["eq_tier"].to_numpy().astype(object)
    xt = P["x_ticker"].to_numpy().astype(object)
    for i in np.flatnonzero(need.to_numpy()):
        c, d, p = P["cnpj8"].iat[i], P["day"].iat[i], int(P["dpos"].iat[i])
        for route, tl in (("fre", fmap.get((c, d))), ("name", nmap.get(c))):
            if not tl:
                continue
            key = (tuple(tl), p)
            if key not in cache:
                cache[key] = _eq_feats(tl, p, tick, TI, SPX, lr)
            r = cache[key]
            if r is not None and np.isfinite(r[1:]).all():
                vals[i] = r[1:]
                tier[i] = route
                xt[i] = r[0]
                break
    m = ~np.isnan(vals[:, 0])
    for k, b in enumerate(XQ):
        P.loc[m, b] = vals[m, k]
    P["eq_tier"] = tier
    P["x_ticker"] = xt

    # --- sector-peer composite (from harness-listed universe rows of the same day)
    U = P[P["univ"] & P["eq_tier"].eq("own")]
    g = U.groupby(["day", "sector"])
    sec = g[list(EQ)].median()
    sec["n_iss"] = g["cnpj8"].nunique()
    sec = sec[(sec["n_iss"] >= 3)]
    sec = sec.drop(columns="n_iss").reset_index()
    sec = sec[sec["sector"] != "unknown"]
    P = P.merge(sec.rename(columns={c: "sec_" + c for c in EQ}), on=["day", "sector"], how="left")
    P["px_sector"] = _comp(P, ["sec_eq_dd252", "sec_eq_vol63", "sec_eq_r126"], [1, -1, 1])

    # --- fundamentals proxy
    P["px_fund"] = P["f_quality"]

    # --- bond-mark health (issuer means of factor_zoo features at the decision close)
    Z = pd.read_pickle(H.HIST / "nightly" / "factor_zoo" / "features_v4.pkl")[["day", "codigo", "ds_63", "rmom_126", "ratio_from_hi_252"]]
    P = P.merge(Z, on=["day", "codigo"], how="left")
    gi = P.groupby(["day", "cnpj8"])
    for c in ("ds_63", "rmom_126", "ratio_from_hi_252"):
        P["iss_" + c] = gi[c].transform("mean")
    P["px_bond"] = _comp(P, ["iss_ds_63", "iss_rmom_126", "iss_ratio_from_hi_252"], [-1, 1, 1])
    return P


def _comp(P: pd.DataFrame, cols, signs) -> pd.Series:
    """Mean pct-rank within each day's universe rows (NaN where any input is missing or outside the universe)."""
    ok = P[cols].notna().all(axis=1) & P["univ"]
    out = pd.Series(np.nan, index=P.index)
    acc = pd.Series(0.0, index=P.index[ok])
    for c, s in zip(cols, signs):
        acc += (s * P.loc[ok, c]).groupby(P.loc[ok, "day"]).rank(pct=True)
    out[ok] = acc / len(cols)
    return out
