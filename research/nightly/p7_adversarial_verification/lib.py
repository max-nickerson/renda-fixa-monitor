"""Building blocks for the adversarial verification of P7 (research/nightly/combined).

PRE-DECLARED before any test in this folder was run (2026-09-29):
  * Honest alpha = P7 minus the MEAN of the matched placebo that is closest to P7's mechanics
    (issuer-level random drop of the same number of LISTED P4+Q issuers, stratified by issuer carry quintile x sector,
    identical carry/liquidity construction). The bond-level stratified and the plain listed-random placebos are reported
    alongside.
  * New variant (only one family, pre-declared): FLAG-VOTE screen. Flags = {EQH bottom 20% (universe-listed),
    factor_zoo composite bottom 20% within P4+Q, event_driven NegEvents, alt_signals supply (new offer 90d)}.
    Drop P4+Q names with >= 2 flags (primary: FV2 with the P7 carry/liquidity construction) and >= 3 flags (FV3).
    EW (issuer cap 10%) versions are secondary.  Nothing else is tuned.
  * Verdict rule: 'robust' if the honest alpha is > 0 with NW t > 2, survives the multiplicity threshold, and is not
    carried by <= 3 issuers or <= 3 months; 'fragile' if it is positive but fails any of those; 'refuted' if the honest
    alpha is <= 0 or the placebo p-value > 0.10.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from research.nightly import harness as H
from research.nightly.combined import p7

KW = dict(as_weights=True, issuer_cap=1.0)
SIGNS = {"eq_dd252": 1.0, "eq_vol63": -1.0, "eq_r126": 1.0}


def bt(fn, P, **kw):
    return H.backtest(fn, panel=P, **KW, **kw)


def listed(x) -> np.ndarray:
    return x[list(p7.EQ_COLS)].notna().all(axis=1).to_numpy()


def mser(a: pd.Series, b: pd.Series | None = None, holdout=False) -> pd.Series:
    """Monthly excess (or paired monthly difference a-b), pre-2026 unless holdout='only'/True."""
    a = a.dropna()
    if holdout is False:
        a = a[a.index < H.HOLDOUT]
    elif holdout == "only":
        a = a[a.index >= H.HOLDOUT]
    ma = H.monthly(a)
    if b is None:
        return ma
    mb = H.monthly(b.reindex(a.index).fillna(0))
    return (ma - mb).dropna()


def ann(m: pd.Series) -> float:
    return float(m.mean() * 1200)


# ------------------------------------------------------------------------------------------------ screens
def eqh_g(x, cols=p7.EQ_COLS):
    """EQH composite with a configurable component set; the eligible 'listed' set is always rows with all three
    equity columns (so single-component variants screen the same population)."""
    ok = listed(x)
    out = np.full(len(x), np.nan)
    if ok.sum() < 10:
        return out
    r = np.zeros(ok.sum())
    for c in cols:
        r += (SIGNS[c] * x.loc[ok, c]).rank(pct=True).to_numpy()
    out[ok] = r / len(cols)
    return out


def flag_eqh_g(x, q=p7.Q_SCREEN, cols=p7.EQ_COLS):
    s = eqh_g(x, cols)
    ok = np.isfinite(s)
    if ok.sum() < 10:
        return np.zeros(len(x), bool)
    return ok & (s <= np.quantile(s[ok], q))


def votes(x) -> np.ndarray:
    return (p7.flag_eqh(x).astype(int) + p7.flag_zoo(x).astype(int) + p7.flag_events(x).astype(int)
            + p7.flag_supply(x).astype(int))


def mk(drop=None, construct="carry", base="p4q", aum=p7.AUM, icap=0.10):
    """Generic P7-family signal. drop(x, sel)->bool mask of names to remove; construct None = EW issuer-cap 10%
    (identical to the P4+Q baseline construction), else p7.carry_weights mode."""
    def fn(x):
        sel = x[base].to_numpy().astype(bool)
        if drop is not None:
            sel = sel & ~np.asarray(drop(x, sel), bool)
        if construct is None:
            w = np.zeros(len(x))
            if sel.sum() >= H.MIN_NAMES:
                w[sel] = H.cap_weights(x.loc[sel, "cnpj8"].to_numpy(), icap)
            return w
        return p7.carry_weights(x, sel, aum=aum, mode=construct)
    return fn


def d_eqh(q=p7.Q_SCREEN, cols=p7.EQ_COLS):
    return lambda x, sel: flag_eqh_g(x, q, cols)


# ------------------------------------------------------------------------------------------------ placebos
STATS = {"bond_lvl1": 0, "bond_lvl2": 0, "bond_lvl3": 0, "iss_lvl1": 0, "iss_lvl2": 0, "iss_lvl3": 0}


def _rng(seed, d, salt):
    return np.random.default_rng([int(seed), int(d.strftime("%Y%m")), int(salt)])


def _draw(rng, pool, k):
    m = min(k, len(pool))
    return rng.choice(pool, m, replace=False) if m else np.array([], int), k - m


def drop_strat_bond(seed, q=p7.Q_SCREEN, cols=p7.EQ_COLS, pool_listed=True, target=None, exclude=False, ndec=10,
                    sector=True):
    """Bond-level placebo: drop as many P4+Q bonds as the real screen, drawn at random from (listed) P4+Q bonds in the
    same (carry decile within P4+Q, sector) cell; shortfall -> same carry decile -> anywhere in the pool.
    target(x, sel) gives the real drop mask (default: EQH)."""
    target = target or d_eqh(q, cols)

    def f(x, sel):
        d = x["day"].iloc[0]
        rng = _rng(seed, d, 11)
        D = sel & np.asarray(target(x, sel), bool)
        out = np.zeros(len(x), bool)
        if D.sum() == 0:
            return out
        avail = sel & (listed(x) if pool_listed else np.ones(len(x), bool))
        if exclude:          # matched CONTROLS: never draw a name the real screen flags
            avail &= ~D
        idx = np.flatnonzero(sel)
        dec = np.full(len(x), -1)
        dec[idx] = np.clip(np.ceil(pd.Series(x["cdi_bps"].to_numpy(float)[idx]).rank(pct=True).to_numpy() * ndec), 1, ndec)
        sec = x["sector"].astype(str).to_numpy() if sector else np.full(len(x), "all")
        need = pd.Series(1, index=pd.MultiIndex.from_arrays([dec[D], sec[D]])).groupby(level=[0, 1]).sum()
        short = {}
        for (dv, sv), k in need.items():
            ch, s = _draw(rng, np.flatnonzero(avail & (dec == dv) & (sec == sv)), int(k))
            out[ch] = True
            avail[ch] = False
            STATS["bond_lvl1"] += len(ch)
            if s:
                short[dv] = short.get(dv, 0) + s
        rest = 0
        for dv, k in short.items():
            ch, s = _draw(rng, np.flatnonzero(avail & (dec == dv)), k)
            out[ch] = True
            avail[ch] = False
            STATS["bond_lvl2"] += len(ch)
            rest += s
        if rest:
            ch, _ = _draw(rng, np.flatnonzero(avail), rest)
            out[ch] = True
            STATS["bond_lvl3"] += len(ch)
        return out
    return f


def drop_strat_issuer(seed, q=p7.Q_SCREEN, cols=p7.EQ_COLS, exclude=False, ndec=5, sector=True):
    """Issuer-level placebo (EQH is an issuer-level screen): drop all P4+Q bonds of as many LISTED P4+Q issuers as the
    screen drops, drawn within (issuer carry quintile, sector) cells; shortfall -> same quintile -> any listed issuer."""
    def f(x, sel):
        d = x["day"].iloc[0]
        rng = _rng(seed, d, 23)
        D = sel & flag_eqh_g(x, q, cols)
        out = np.zeros(len(x), bool)
        if D.sum() == 0:
            return out
        iss = x["cnpj8"].astype(str).to_numpy()
        L = sel & listed(x)
        it = pd.DataFrame({"iss": iss[L], "carry": x["cdi_bps"].to_numpy(float)[L],
                           "sec": x["sector"].astype(str).to_numpy()[L]}).groupby("iss").agg(
            carry=("carry", "mean"), sec=("sec", "first"))
        it["dec"] = np.clip(np.ceil(it["carry"].rank(pct=True) * ndec), 1, ndec)
        if not sector:
            it["sec"] = "all"
        dropped = pd.Index(sorted(set(iss[D])))
        need = it.loc[dropped].groupby(["dec", "sec"]).size()
        names = it.index.to_numpy()
        avail = ~it.index.isin(dropped) if exclude else np.ones(len(it), bool)
        dec, sec = it["dec"].to_numpy(), it["sec"].to_numpy()
        chosen, short = [], {}
        for (dv, sv), k in need.items():
            ch, s = _draw(rng, np.flatnonzero(avail & (dec == dv) & (sec == sv)), int(k))
            avail[ch] = False
            chosen += list(ch)
            STATS["iss_lvl1"] += len(ch)
            if s:
                short[dv] = short.get(dv, 0) + s
        rest = 0
        for dv, k in short.items():
            ch, s = _draw(rng, np.flatnonzero(avail & (dec == dv)), k)
            avail[ch] = False
            chosen += list(ch)
            STATS["iss_lvl2"] += len(ch)
            rest += s
        if rest:
            ch, _ = _draw(rng, np.flatnonzero(avail), rest)
            chosen += list(ch)
            STATS["iss_lvl3"] += len(ch)
        out = sel & np.isin(iss, names[np.array(chosen, int)])
        return out
    return f


# ------------------------------------------------------------------------------------------------ fundamentals lag
def worstq_lag(panel, lag_days=0, q=0.2):
    """Copy of bias_audit.run.worstq_variant (strict dates): worst-quintile quality flag with fundamentals lagged."""
    fu = H._fundamentals(strict=True).copy()
    fu["avail"] = pd.to_datetime(fu["avail"]) + pd.Timedelta(days=lag_days)
    fu = fu.sort_values("avail")
    g = panel[["day", "cnpj8", "univ"]].copy()
    g["cnpj8"] = g["cnpj8"].astype(str)
    fu["cnpj8"] = fu["cnpj8"].astype(str)
    m = pd.merge_asof(g.reset_index().sort_values("day"), fu, left_on="day", right_on="avail", by="cnpj8",
                      direction="backward").set_index("index").reindex(g.index)
    stale = (m["day"] - m["period_end"]).dt.days > H.FUND_STALE_DAYS
    cov = m["f_lev"].notna() & m["f_cov"].notna() & ~stale
    U = g["univ"] & cov
    rk = pd.DataFrame({c: m.loc[U, c].groupby(g.loc[U, "day"]).rank(pct=True) * s for c, s in H.QSIGN.items()})
    fq = rk.mean(axis=1, skipna=True).reindex(g.index)
    thr = fq.groupby(g["day"]).quantile(q)
    return (cov & g["univ"] & (fq <= g["day"].map(thr))).fillna(False).to_numpy()


# ------------------------------------------------------------------------------------------------ cohort attribution
def cohort_matrix(fnA, fnB, P):
    """Per decision date (complete 126d labels only) and issuer: sum over bonds of (wA - wB) x fwd_126 (NaN = cash)."""
    U = P[P["univ"] & (P["day"] >= H.START)]
    rows = []
    for d, x in U.groupby("day"):
        if not x["dok_126"].all():
            continue
        wa, wb = fnA(x), fnB(x)
        f = np.nan_to_num(x["fwd_126"].to_numpy(float))
        c = pd.Series((wa - wb) * f).groupby(x["cnpj8"].astype(str).to_numpy()).sum()
        c = c[(c != 0)]
        rows.append(pd.DataFrame({"day": d, "iss": c.index, "c": c.to_numpy()}))
    M = pd.concat(rows).pivot_table(index="day", columns="iss", values="c", aggfunc="sum").fillna(0.0)
    return M


def p4q_ew(x):
    s = x["p4q"].to_numpy().astype(bool)
    w = np.zeros(len(x))
    if s.sum() >= H.MIN_NAMES:
        w[s] = H.cap_weights(x.loc[s, "cnpj8"].to_numpy(), 0.10)
    return w
