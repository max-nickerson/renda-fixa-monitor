"""Relative-value credit strategy for fixed income: signals + walk-forward backtest.

Signal components (all computed with trailing windows only — no look-ahead):
  value      z-score of the spread vs its own history (wide = cheap = +)
  momentum   recent spread trend (still widening = −), avoids catching falling knives
  equity     equity-credit divergence: spread change vs what the issuer's stock move implies
             (spread wider than the stock justifies = cheap = +)
  carry      breakeven: spread per unit of duration vs the spread's own volatility
Risk filters (block BUY): recent material fact / negative news, leverage deterioration, stock crash.

When a bond has no spread (e.g. distressed paper with no ANBIMA rate) the price is used instead.
Output is a research signal, not investment advice.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from . import db

WINDOW = 120
MIN_OBS = 40
ENTRY, EXIT = 0.75, -0.25
WEIGHTS = {"value": 0.35, "momentum": 0.25, "equity": 0.25, "carry": 0.15}


@dataclass
class Signal:
    label: str
    score: float | None
    components: dict = field(default_factory=dict)
    blocked_by: list[str] = field(default_factory=list)
    basis: str = "spread"
    asof: str | None = None
    obs: int = 0

    def as_dict(self) -> dict:
        return asdict(self)


def _z(s: pd.Series, w: int = WINDOW) -> pd.Series:
    mu = s.rolling(w, min_periods=MIN_OBS).mean()
    sd = s.rolling(w, min_periods=MIN_OBS).std()
    return (s - mu) / sd.replace(0, np.nan)


def market_frame(isin: str) -> pd.DataFrame:
    """Market series only (fundamental snapshots excluded so they don't create phantom dates)."""
    df = db.series(isin)
    if df.empty:
        return df
    return df[[c for c in df.columns if not c.startswith("f_")]].dropna(how="all")


def event_blocks(isin: str, index: pd.DatetimeIndex, days: int = 10) -> pd.Series:
    """True on dates within `days` after a material fact or negative-keyword news (point-in-time)."""
    blocked = pd.Series(False, index=index)
    for e in db.events(isin, limit=5000):
        if e["severity"] != "high" or e["kind"] not in ("material_fact", "news"):
            continue
        try:
            t = pd.Timestamp(e["ts"][:10])
        except ValueError:
            continue
        blocked |= (index >= t) & (index <= t + pd.Timedelta(days=days))
    return blocked


def components(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Per-date signal components from a wide series frame (see collect.py metrics)."""
    out = pd.DataFrame(index=df.index)
    if "spread_bps" in df and df["spread_bps"].notna().sum() >= MIN_OBS:
        base, basis = df["spread_bps"].ffill(), "spread"
    elif "price" in df and df["price"].notna().sum() >= MIN_OBS:
        base, basis = -df["price"].ffill(), "price"  # lower price ~ wider spread
    else:
        return out, "insufficient"

    out["value"] = _z(base).clip(-3, 3)
    d = base.diff()
    vol20 = d.rolling(WINDOW, min_periods=MIN_OBS).std() * np.sqrt(20)
    out["momentum"] = (-(base.diff(20)) / vol20.replace(0, np.nan)).clip(-3, 3)

    if "stock_close" in df and df["stock_close"].notna().sum() >= MIN_OBS:
        px = df["stock_close"].ffill()
        rs = np.log(px).diff()
        beta = d.rolling(250, min_periods=60).cov(rs) / rs.rolling(250, min_periods=60).var()
        implied = beta * np.log(px).diff(20)
        resid = base.diff(20) - implied
        out["equity"] = (resid / resid.rolling(WINDOW, min_periods=MIN_OBS).std()).clip(-3, 3)

    if basis == "spread" and "duration" in df:
        dur = df["duration"].ffill().clip(lower=0.25)
        vol_ann = d.rolling(WINDOW, min_periods=MIN_OBS).std() * np.sqrt(252)
        out["carry"] = ((base / dur) / vol_ann.replace(0, np.nan) - 1).clip(-2, 2)
    return out, basis


def composite(comp: pd.DataFrame) -> pd.Series:
    if comp.empty:
        return pd.Series(dtype=float)
    w = pd.Series({k: v for k, v in WEIGHTS.items() if k in comp.columns})
    num = comp[w.index].mul(w, axis=1).sum(axis=1, min_count=1)
    den = comp[w.index].notna().mul(w, axis=1).sum(axis=1).replace(0, np.nan)
    return num / den * 1.5  # rescale so a ~2σ value move alone crosses the entry threshold


def _risk_blocks(isin: str, df: pd.DataFrame, asof: pd.Timestamp) -> list[str]:
    blocks = []
    recent = [e for e in db.events(isin, limit=200)
              if e["severity"] == "high" and e["ts"][:10] >= (asof - pd.Timedelta(days=10)).strftime("%Y-%m-%d")]
    if any(e["kind"] == "material_fact" for e in recent):
        blocks.append("material fact in last 10 days")
    if any(e["kind"] == "news" for e in recent):
        blocks.append("negative-keyword news in last 10 days")
    lev = df.get("f_net_debt_ebitda")
    if lev is not None and lev.notna().sum() >= 1:
        last = lev.dropna().iloc[-1]
        old = lev.dropna()[lev.dropna().index <= asof - pd.Timedelta(days=90)]
        if last > 4.5:
            blocks.append(f"net debt/EBITDA {last:.1f}x > 4.5x")
        elif not old.empty and last - old.iloc[-1] > 0.5:
            blocks.append(f"leverage up {last - old.iloc[-1]:.1f}x in 90d")
    if "stock_close" in df and df["stock_close"].notna().sum() > 20:
        px = df["stock_close"].dropna()
        dd = px.iloc[-1] / px.tail(20).max() - 1
        if dd < -0.25:
            blocks.append(f"issuer stock {dd:.0%} from 20d high")
    return blocks


def label_for(score: float | None) -> str:
    if score is None or score != score:
        return "N/A"
    return "BUY" if score >= ENTRY else "SELL" if score <= -ENTRY else "HOLD"


def current(isin: str) -> Signal:
    df = market_frame(isin)
    if df.empty:
        return Signal("N/A", None, basis="insufficient")
    comp, basis = components(df)
    if basis == "insufficient" or comp.dropna(how="all").empty:
        return Signal("N/A", None, basis="insufficient", obs=len(df))
    score_s = composite(comp).dropna()
    if score_s.empty:
        return Signal("N/A", None, basis=basis, obs=len(df))
    asof = score_s.index[-1]
    score = float(score_s.iloc[-1])
    lab = label_for(score)
    blocks = _risk_blocks(isin, db.series(isin), asof)
    if lab == "BUY" and blocks:
        lab = "HOLD*"
    row = comp.loc[asof]
    return Signal(lab, round(score, 2), {k: round(float(v), 2) for k, v in row.items() if v == v},
                  blocks, basis, asof.strftime("%Y-%m-%d"), int(len(df)))


# ---------------------------------------------------------------- backtest
def backtest(isin: str, cost_bps: float = 30.0, entry: float = ENTRY, exit_: float = EXIT,
             use_event_filter: bool = True) -> dict:
    """Long-only: enter when score >= entry, exit when score <= exit. Trades at next day's level.

    Daily bond return ≈ carry − duration × Δyield (or price return when no yield). Flat = earns CDI
    (BRL assets) or 0 (USD). Costs charged on every position change. With `use_event_filter`, no new
    entries within 10 days of a material fact / negative news (as the live signal does). Leverage
    filters are not applied historically (only today's fundamentals snapshot is known).
    """
    info = db.get_asset(isin) or {}
    df = market_frame(isin)
    if df.empty:
        return {"error": "no data"}
    comp, basis = components(df)
    score = composite(comp)
    if score.dropna().empty:
        return {"error": f"insufficient history ({len(df)} obs, need ~{MIN_OBS + 20})"}

    dt = df.index.to_series().diff().dt.days.fillna(1) / 365.0
    macro = db.series("_MACRO", ["cdi"])
    cdi = (macro["cdi"].reindex(df.index, method="ffill") if not macro.empty else pd.Series(0.0, index=df.index))
    cash = (cdi.fillna(0) / 100 * dt) if info.get("currency", "BRL") == "BRL" else pd.Series(0.0, index=df.index)

    if "yield" in df and df["yield"].notna().sum() > MIN_OBS and "duration" in df:
        y = df["yield"].ffill()
        dur = df["duration"].ffill().fillna(0)
        carry_y = y + (cdi.fillna(0) if info.get("index") in ("DI_SPREAD",) else 0)
        if info.get("index") == "DI_PCT":
            carry_y = y / 100 * cdi.fillna(0)
        bond_ret = carry_y.shift(1) / 100 * dt - dur.shift(1) * y.diff() / 100
    else:
        bond_ret = df["price"].ffill().pct_change()
    bond_ret = bond_ret.fillna(0)

    blocked = event_blocks(isin, df.index).shift(1, fill_value=False) if use_event_filter \
        else pd.Series(False, index=df.index)
    pos, state = [], 0
    for s, blk in zip(score.shift(1).reindex(df.index), blocked):  # decide on yesterday's info, trade today
        if s == s:
            if state == 0 and s >= entry and not blk:
                state = 1
            elif state == 1 and s <= exit_:
                state = 0
        pos.append(state)
    pos = pd.Series(pos, index=df.index)
    trades = pos.diff().abs().fillna(pos.iloc[0])
    strat = pos * bond_ret + (1 - pos) * cash - trades * cost_bps / 10000

    def stats(r: pd.Series) -> dict:
        eq = (1 + r).cumprod()
        yrs = max((r.index[-1] - r.index[0]).days / 365.25, 1 / 365)
        vol = r.std() * np.sqrt(252)
        ex = (r - cash).mean() * 252
        return {"total_return_pct": round((eq.iloc[-1] - 1) * 100, 2),
                "cagr_pct": round((eq.iloc[-1] ** (1 / yrs) - 1) * 100, 2),
                "vol_pct": round(vol * 100, 2),
                "sharpe_vs_cash": round(ex / vol, 2) if vol > 0 else None,
                "max_drawdown_pct": round(((eq / eq.cummax()) - 1).min() * 100, 2)}

    start = score.first_valid_index()
    rng = slice(start, None)
    entries = pos[rng].diff().fillna(0) == 1
    trade_rets = []
    cur = None
    for d, p in pos[rng].items():
        if p == 1 and cur is None:
            cur = []
        if cur is not None:
            cur.append(strat[d])
            if p == 0:
                trade_rets.append(np.prod([1 + x for x in cur]) - 1)
                cur = None
    return {
        "isin": isin, "basis": basis, "from": str(start.date()), "to": str(df.index[-1].date()),
        "days": int(len(df[rng])), "trades": int(entries.sum()),
        "time_in_market_pct": round(pos[rng].mean() * 100, 1),
        "hit_rate_pct": round(np.mean([t > 0 for t in trade_rets]) * 100, 1) if trade_rets else None,
        "strategy": stats(strat[rng]), "buy_and_hold": stats(bond_ret[rng]), "cash": stats(cash[rng]),
        "params": {"entry": entry, "exit": exit_, "cost_bps": cost_bps, "window": WINDOW, "weights": WEIGHTS,
                   "event_filter": use_event_filter},
        "equity_curve": {
            "dates": [d.strftime("%Y-%m-%d") for d in df.index[df.index >= start]],
            "strategy": ((1 + strat[rng]).cumprod()).round(5).tolist(),
            "buy_and_hold": ((1 + bond_ret[rng]).cumprod()).round(5).tolist(),
        },
        "caveats": [
            "ANBIMA/indicative prices are not executable; real bid-ask can be much wider on illiquid paper.",
            "Short history → few trades → results are statistically weak; treat as a sanity check.",
            "Coupons/amortisation approximated via carry; defaults and events are not modelled.",
        ],
    }
