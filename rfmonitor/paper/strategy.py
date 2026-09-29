"""Live versions of the research rules (research/nightly/combined/p7.py, research/nightly/harness.py), computed only
from data in paper.db and data/history/fundamentals_pit.pkl as of the decision day.

P4+Q = top 30% CDI+ carry, not rich vs peers (resid_z > -1.5), no negative press in 30 days, minus the worst
       quintile of balance-sheet quality (issuers without financials are kept). Equal weight, issuer cap 10%.
P7   = P4+Q minus the bottom 20% equity-health issuers (listed issuer or parent; unlisted kept), weights
       proportional to carry, issuer cap 5%, each bond <= 25% of its 91-day SND traded volume / AUM.
Live universe = bonds ANBIMA prices that day (the screener universe) with a known DI+/IPCA+/Pré contract.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import history as h
from . import store
from .collect import equity_map

QSIGN = {"f_lev": -1, "f_cov": 1, "f_cash_st": 1, "f_eq_ratio": 1, "f_d_lev": -1}
FUND_STALE_DAYS = 460
KINDS = ("DI_SPREAD", "IPCA", "PRE")
MIN_NAMES = 5


def universe(day: str) -> pd.DataFrame:
    u = store.df("SELECT * FROM snap_bonds WHERE date = ?", (day,))
    return u[u["kind"].isin(KINDS) & u["contract"].notna() & u["ratio"].notna() & u["spread_bps"].notna()].copy()


def quality(u: pd.DataFrame, day: str) -> pd.Series:
    """Composite balance-sheet quality rank (higher = better) from the latest financials public by `day`."""
    p = h.HIST / "fundamentals_pit.pkl"
    if not p.exists():
        return pd.Series(np.nan, index=u.index)
    f = pd.read_pickle(p)
    f = f[pd.to_datetime(f["available_date_strict"]) <= pd.Timestamp(day)]
    f = f.sort_values(["cnpj8", "period_end"]).drop_duplicates("cnpj8", keep="last").set_index("cnpj8")
    f = f[(pd.Timestamp(day) - pd.to_datetime(f["period_end"])).dt.days <= FUND_STALE_DAYS]
    e, nd, fe = f["ebitda_ltm"], f["net_debt"], f["fin_exp_ltm"]
    x = pd.DataFrame(index=f.index)
    x["f_lev"] = np.clip(np.where(e > 0, nd / e, 15.0), -3, 15)
    x.loc[nd.isna() | e.isna(), "f_lev"] = np.nan
    x["f_cov"] = np.clip(np.where(fe > 0, e / fe, 30.0), -5, 30)
    x.loc[fe.isna() | e.isna(), "f_cov"] = np.nan
    x["f_cash_st"] = np.log1p(f["cash_to_st_debt"].clip(0, 50))
    x["f_eq_ratio"] = f["equity_ratio"].clip(-1, 1)
    x["f_d_lev"] = f["d_net_debt_ebitda_4q"].clip(-10, 10)
    m = x.reindex(u["cnpj8"].to_numpy())
    m.index = u.index
    covered = m["f_lev"].notna() & m["f_cov"].notna()
    rk = pd.DataFrame({c: m[c].where(covered).rank(pct=True) * s for c, s in QSIGN.items()})
    return rk.mean(axis=1, skipna=True).where(covered)


def equity_health(u: pd.DataFrame, day: str) -> pd.DataFrame:
    """Per bond: stock drawdown vs 252d high, 63d vol, 126d return of the issuer/parent ticker, and the composite."""
    e = store.df("SELECT date, ticker, close FROM eq_close WHERE date <= ? AND date >= date(?, '-400 day')", (day, day))
    out = pd.DataFrame(index=u.index, columns=["ticker", "dd252", "vol63", "r126", "eqh"], dtype=object)
    if e.empty:
        return out
    px = e.pivot(index="date", columns="ticker", values="close").sort_index().ffill(limit=5)
    feats = {}
    for t in px.columns:
        s = px[t].dropna()
        if len(s) < 150 or s.index[-1] < (pd.Timestamp(day) - pd.Timedelta(days=10)).strftime("%Y-%m-%d"):
            continue
        lr = np.log(s).diff()
        feats[t] = (s.iloc[-1] / s.iloc[-252:].max() - 1, lr.iloc[-63:].std() * np.sqrt(252),
                    s.iloc[-1] / s.iloc[-127] - 1 if len(s) > 127 else np.nan)
    cands = equity_map().groupby("cnpj8")["ticker"].apply(list).to_dict()
    for i, c8 in zip(u.index, u["cnpj8"]):
        t = next((t for t in cands.get(c8, []) if t in feats), None)
        if t:
            out.loc[i, ["ticker", "dd252", "vol63", "r126"]] = [t, *feats[t]]
    ok = out[["dd252", "vol63", "r126"]].notna().all(axis=1)
    if ok.sum() >= 10:
        r = (out.loc[ok, "dd252"].astype(float).rank(pct=True) - out.loc[ok, "vol63"].astype(float).rank(pct=True)
             + 1 + out.loc[ok, "r126"].astype(float).rank(pct=True))
        out.loc[ok, "eqh"] = r / 3
    return out


def liquidity(u: pd.DataFrame, day: str) -> pd.Series:
    """R$ traded per bond in the SND over the 91 days up to `day`."""
    t = store.df("SELECT codigo, sum(qty * pu_avg) v FROM snd_trades WHERE date <= ? AND date > date(?, '-91 day') "
                 "GROUP BY codigo", (day, day))
    return u["codigo"].map(t.set_index("codigo")["v"]).fillna(0.0)


def waterfill(w0: np.ndarray, cap_row: np.ndarray, issuer: np.ndarray, icap: float) -> np.ndarray:
    """Weights proportional to w0 summing to 1, with per-bond caps and an issuer cap (same as research p7.py)."""
    w, fixed, iss = np.zeros(len(w0)), np.zeros(len(w0), bool), pd.Series(issuer)
    for _ in range(200):
        free = ~fixed
        rem = 1.0 - w[fixed].sum()
        if not free.any() or w0[free].sum() <= 0 or rem <= 1e-12:
            break
        w[free] = rem * w0[free] / w0[free].sum()
        over_r = free & (w > cap_row + 1e-12)
        over_i = free & (pd.Series(w).groupby(iss).transform("sum").to_numpy() > icap + 1e-12)
        if not over_r.any() and not over_i.any():
            break
        if over_r.any():
            w[over_r] = cap_row[over_r]
            fixed |= over_r
            continue
        for g in np.unique(issuer[over_i]):
            gi = issuer == g
            fr = gi & ~fixed
            s = w[fr].sum()
            if s > 0:
                w[fr] *= max(icap - w[gi & fixed].sum(), 0) / s
            fixed |= fr
    return w / w.sum() if w.sum() > 0 else w


def targets(book: str, day: str, aum: float) -> pd.DataFrame:
    """Target weights for a new tranche decided at the close of `day`."""
    u = universe(day)
    if u.empty:
        return u
    u["cdi_pct"] = u["spread_bps"].rank(pct=True, ascending=False)
    u["quality"] = quality(u, day)
    thr = u["quality"].quantile(0.2)
    u["worstQ"] = u["quality"].notna() & (u["quality"] <= thr)
    p4 = (u["cdi_pct"] <= 0.3) & ~(u["resid_z"].fillna(0) <= -1.5) & (u["press_neg_30d"].fillna(0) < 1)
    sel = p4 & ~u["worstQ"]
    u["issuer"] = u["cnpj8"].fillna(u["codigo"].str[:4])
    E = equity_health(u, day)
    u = u.join(E)
    note = np.where(u["worstQ"], "qualidade fraca", "")
    if book == "P7":
        eqh = pd.to_numeric(u["eqh"], errors="coerce")
        bad = eqh.notna() & (eqh <= eqh.quantile(0.2))
        note = np.where(sel & bad, "ação em estresse", note)
        sel = sel & ~bad
        idx = np.flatnonzero(sel.to_numpy())
        w = np.zeros(len(u))
        if len(idx) >= MIN_NAMES:
            xs = u.iloc[idx]
            cap = 0.25 * liquidity(xs, day).to_numpy() / aum
            issuer = xs["issuer"].to_numpy()
            k = 1.0
            for _ in range(12):  # relax liquidity caps x1.5 until feasible (as in research)
                if pd.Series(np.minimum(cap * k, 1)).groupby(issuer).sum().clip(upper=0.05).sum() >= 1 - 1e-9:
                    break
                k *= 1.5
            w[idx] = waterfill(np.clip(xs["spread_bps"].to_numpy(float), 1, None), np.minimum(cap * k, 1), issuer, 0.05)
    else:  # P4Q: equal weight, issuer cap 10%
        idx = np.flatnonzero(sel.to_numpy())
        w = np.zeros(len(u))
        if len(idx) >= MIN_NAMES:
            xs = u.iloc[idx]
            w[idx] = waterfill(np.ones(len(idx)), np.ones(len(idx)), xs["issuer"].to_numpy(), 0.10)
    u["weight"] = w
    u["note"] = note
    return u
