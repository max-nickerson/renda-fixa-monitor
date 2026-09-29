"""P7: the combined strategy built from the wave-1 nightly research (harness v4).

PRE-REGISTERED SPEC (written before any P7 backtest was run; parameters are the wave-1 defaults, no tuning here)
--------------------------------------------------------------------------------------------------------------
1. Alpha / selection   P4+Q candidates (harness column `p4q`).
2. Avoid screen (EQH)  equity-health composite = mean percentile rank, within the universe's listed rows of that
                       decision date, of  eq_dd252 (+, drawdown vs 252d high), -eq_vol63, eq_r126 (+).
                       P4+Q names whose composite is in the universe-listed bottom 20% are dropped; unlisted names
                       are kept.  Source: the common robust core of structural_credit (C2 eq_dd252 ~ Merton DD),
                       factor_zoo (the equity block was its only robust part) and ml_ranking SHAP.
3. Construction        weights proportional to carry (cdi_bps) inside the screened set (portfolio_construction:
                       the carry slope inside P4+Q, IC +0.33, is what construction can harvest), issuer cap 5%,
                       liquidity cap w_i <= 25% x 91-day SND R$ volume / AUM with AUM = R$500m (capacity-feasible;
                       if infeasible the liquidity caps are relaxed x1.5 until feasible, as in portfolio_construction).
4. Timing / hedging    none.  Every wave-1 timing overlay and hedge was <= 0 vs P4+Q net of costs (macro_cycle,
                       hedging_overlays); they are tested here only as ablation variants.
5. Engine              harness monthly decisions, 126-bday overlapping tranches, next-trade execution, 25 bps.

Everything is point-in-time: every input is a harness panel column known at the decision close, or the
portfolio_construction liquidity table (SND trades dated <= decision day).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from research.nightly import harness as H

LIQ_PATH = H.HIST / "nightly" / "portfolio_construction" / "liquidity.pkl"
ALT_HOLD = H.HIST / "nightly" / "alt_signals" / "panel_alt_M_hold.pkl"
EVT_HOLD = H.HIST / "nightly" / "event_driven" / "event_signals_M_ho.pkl"
ZOO = H.HIST / "nightly" / "factor_zoo" / "zoo_composite.pkl"

AUM = 500e6
PART = 0.25
ICAP = 0.05
Q_SCREEN = 0.20
EQ_COLS = ("eq_dd252", "eq_vol63", "eq_r126")
EQ_SIGN = (1.0, -1.0, 1.0)


# ------------------------------------------------------------------------------------------------ data
def load(holdout: bool = False) -> pd.DataFrame:
    """Monthly harness panel + liquidity + (for extension variants) supply/event/zoo columns."""
    P = H.load_panel("M", holdout=holdout)
    L = pd.read_pickle(LIQ_PATH)[["codigo", "day", "vol91_brl", "adv_brl"]]
    P = P.merge(L, on=["codigo", "day"], how="left")
    try:
        A = pd.read_pickle(ALT_HOLD)[["codigo", "day", "sup_iss_90d"]]
        P = P.merge(A, on=["codigo", "day"], how="left")
    except Exception:
        P["sup_iss_90d"] = np.nan
    E = pd.read_pickle(EVT_HOLD)
    ecols = ["ev_ma_days", "ev_rating_down_days", "ev_agd_days", "ev_resgate_days", "ev_rj_days"]
    P = P.merge(E[["codigo", "day"] + ecols], on=["codigo", "day"], how="left")
    Z = pd.read_pickle(ZOO)
    Z = Z[Z["freq"] == "M"][["codigo", "day", "value"]].rename(columns={"value": "zoo"})
    P = P.merge(Z, on=["codigo", "day"], how="left")
    if not holdout:
        P = P[P["day"] < H.HOLDOUT]
    return P.reset_index(drop=True)


# ------------------------------------------------------------------------------------------------ screens
def eqh_score(x: pd.DataFrame) -> np.ndarray:
    """Equity-health composite on one decision cross-section (NaN for unlisted / incomplete rows)."""
    ok = x[list(EQ_COLS)].notna().all(axis=1).to_numpy()
    out = np.full(len(x), np.nan)
    if ok.sum() < 10:
        return out
    r = np.zeros(ok.sum())
    for c, s in zip(EQ_COLS, EQ_SIGN):
        r += (s * x.loc[ok, c]).rank(pct=True).to_numpy()
    out[ok] = r / len(EQ_COLS)
    return out


def flag_eqh(x: pd.DataFrame, q: float = Q_SCREEN) -> np.ndarray:
    s = eqh_score(x)
    ok = np.isfinite(s)
    if ok.sum() < 10:
        return np.zeros(len(x), bool)
    thr = np.quantile(s[ok], q)
    return ok & (s <= thr)


def flag_zoo(x: pd.DataFrame, q: float = Q_SCREEN) -> np.ndarray:
    """factor_zoo spec: worst 20% of the zoo composite WITHIN P4+Q."""
    v = x["zoo"].to_numpy(float)
    m = x["p4q"].to_numpy() & np.isfinite(v)
    out = np.zeros(len(x), bool)
    if m.sum() < 10:
        return out
    thr = np.quantile(v[m], q)
    out[m] = v[m] <= thr
    return out


def _within(x, col, d):
    return (x[col] <= d).fillna(False).to_numpy()


def flag_events(x: pd.DataFrame) -> np.ndarray:
    """event_driven 'NegEvents': M&A 180d, downgrade 180d, AGD 180d, redemption 126d, RJ 365d."""
    return (_within(x, "ev_ma_days", 180) | _within(x, "ev_rating_down_days", 180) | _within(x, "ev_agd_days", 180)
            | _within(x, "ev_resgate_days", 126) | _within(x, "ev_rj_days", 365))


def flag_supply(x: pd.DataFrame) -> np.ndarray:
    """alt_signals: issuer started a new debenture offer in the last 90 days."""
    return (x["sup_iss_90d"].fillna(0) > 0).to_numpy()


# ------------------------------------------------------------------------------------------------ construction
def waterfill(w0: np.ndarray, cap_row: np.ndarray, issuer: np.ndarray, icap: float) -> np.ndarray:
    """Weights proportional to w0, sum 1, with per-row caps and an issuer cap (iterative clamp + redistribute).
    Assumes feasibility was checked by the caller."""
    n = len(w0)
    w = np.zeros(n)
    fixed = np.zeros(n, bool)
    iss = pd.Series(issuer)
    for _ in range(200):
        free = ~fixed
        rem = 1.0 - w[fixed].sum()
        if not free.any() or w0[free].sum() <= 0 or rem <= 1e-12:
            break
        w[free] = rem * w0[free] / w0[free].sum()
        over_r = free & (w > cap_row + 1e-12)
        tot = pd.Series(w).groupby(iss).transform("sum").to_numpy()
        over_i = free & (tot > icap + 1e-12)
        if not over_r.any() and not over_i.any():
            break
        if over_r.any():
            w[over_r] = cap_row[over_r]
            fixed |= over_r
            continue
        # issuer cap: scale that issuer's free rows so the issuer total equals icap, then fix them
        for g in np.unique(issuer[over_i]):
            gi = issuer == g
            fx = w[gi & fixed].sum()
            fr = gi & ~fixed
            s = w[fr].sum()
            if s > 0:
                w[fr] *= max(icap - fx, 0) / s
            fixed |= fr
    return w / w.sum() if w.sum() > 0 else w


def carry_weights(x: pd.DataFrame, sel: np.ndarray, aum: float | None = AUM, part: float = PART,
                  icap: float = ICAP, mode: str = "carry") -> np.ndarray:
    """Weights on the rows of x (0 outside sel). mode 'carry' = proportional to cdi_bps; 'ew' = equal;
    'top25' = equal weight on the 25 highest-carry names (extended down the ranking if liquidity binds)."""
    out = np.zeros(len(x))
    idx = np.flatnonzero(sel)
    if len(idx) < H.MIN_NAMES:
        return out
    xs = x.iloc[idx]
    carry = xs["cdi_bps"].to_numpy(float)
    if mode == "carry":
        w0 = np.clip(carry, 1.0, None)
    elif mode == "ew":
        w0 = np.ones(len(idx))
    elif mode == "top25":
        w0 = np.zeros(len(idx))
        order = np.argsort(-carry)
        w0[order[:25]] = 1.0
    else:
        raise ValueError(mode)
    issuer = xs["cnpj8"].astype(str).to_numpy()
    if aum is None:
        cap = np.ones(len(idx))
    else:
        cap = part * np.nan_to_num(xs["vol91_brl"].to_numpy(float), nan=0.0) / aum
    k = 1.0
    for _ in range(12):
        c = np.minimum(cap * k, 1.0)
        cand = w0 > 0
        if mode == "top25" and aum is not None:
            # extend down the carry ranking until the capped capacity of the chosen names reaches 1
            order = np.argsort(-carry)
            w0 = np.zeros(len(idx))
            capsum = 0.0
            per_iss = {}
            for j in order:
                w0[j] = 1.0
                g = issuer[j]
                add = min(c[j], ICAP - per_iss.get(g, 0.0)) if icap < 1 else c[j]
                per_iss[g] = per_iss.get(g, 0.0) + max(add, 0)
                capsum += max(add, 0)
                if w0.sum() >= 25 and capsum >= 1.0:
                    break
            cand = w0 > 0
        feas = pd.Series(c[cand]).groupby(issuer[cand]).sum().clip(upper=icap).sum()
        if feas >= 1.0 - 1e-9:
            break
        k *= 1.5
    w = waterfill(np.where(w0 > 0, w0, 0.0), np.minimum(cap * k, 1.0), issuer, icap)
    out[idx] = w
    return out


# ------------------------------------------------------------------------------------------------ strategy
def make_signal(screen=("eqh",), construct: str | None = "carry", aum: float | None = AUM, base: str = "p4q",
                gate=None, icap: float = ICAP, q: float = Q_SCREEN):
    """Returns a callable for H.backtest(as_weights=True, issuer_cap=1.0).
    screen: tuple of 'eqh' / 'zoo' / 'events' / 'supply' / 'rand:<seed>' (random placebo screen);
    construct: None (capped EW, harness style with icap) / 'carry' / 'ew' / 'top25';
    gate: None or callable(day)->bool (True = the decision's new tranche stays in cash)."""
    def fn(x: pd.DataFrame):
        d = x["day"].iloc[0]
        if gate is not None and gate(d):
            return np.zeros(len(x))
        sel = x[base].to_numpy().astype(bool)
        drop = np.zeros(len(x), bool)
        for s in screen:
            if s == "eqh":
                drop |= flag_eqh(x, q)
            elif s == "zoo":
                drop |= flag_zoo(x, q)
            elif s == "events":
                drop |= flag_events(x)
            elif s == "supply":
                drop |= flag_supply(x)
            elif s.startswith("rand"):
                # placebo: drop as many LISTED base names as eqh would, chosen at random
                seed = int(s.split(":")[1])
                n_drop = int((sel & flag_eqh(x, q)).sum())
                lst = np.flatnonzero(sel & x[list(EQ_COLS)].notna().all(axis=1).to_numpy())
                rng = np.random.default_rng(seed * 100003 + int(d.strftime("%Y%m")))
                if n_drop and len(lst):
                    drop[rng.choice(lst, size=min(n_drop, len(lst)), replace=False)] = True
        sel = sel & ~drop
        if construct is None:
            w = np.zeros(len(x))
            if sel.sum() >= H.MIN_NAMES:
                w[sel] = H.cap_weights(x.loc[sel, "cnpj8"].to_numpy(), icap)
            return w
        return carry_weights(x, sel, aum=aum, icap=icap, mode=construct)
    return fn


def weights_frame(fn, P: pd.DataFrame, start=H.START) -> pd.DataFrame:
    """Materialise a signal as a (day, codigo, weight) frame (for export / diagnostics)."""
    U = P[P["univ"] & (P["day"] >= pd.Timestamp(start))]
    rows = []
    for d, x in U.groupby("day"):
        w = fn(x)
        m = w > 0
        rows.append(pd.DataFrame({"day": d, "codigo": x["codigo"].to_numpy()[m], "weight": w[m]}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["day", "codigo", "weight"])


def signals(panel: pd.DataFrame | None = None, holdout: bool = False) -> pd.DataFrame:
    """Reusable export: P7 target weights per monthly decision (day, codigo, weight), known at the decision close."""
    P = load(holdout) if panel is None else panel
    return weights_frame(make_signal(), P)
