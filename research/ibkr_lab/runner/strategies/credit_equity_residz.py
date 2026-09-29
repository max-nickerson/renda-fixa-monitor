"""B3 long/short from the debenture spread-level signal resid_z (control), weekly.

Signal/basket: research/ibkr_lab/credit_equity/paper_check.basket() -- latest resid_z per bond from
data/history/lab_daily.pkl (issuer-level, bond-count-weighted median, mapped to the issuer's most liquid B3 stock),
ADTV >= R$5m, quintiles: LONG the lowest resid_z (spread tight vs peers), SHORT the highest. Equal weight,
USD 125k per side (250k gross), so each name is ~USD 10k (<= 48k cap).
Shorts: kept (the backtest is L/S, 'residz|ls|h1' weekly) but IBKR reported shortable level 2.0 on every short name
(borrow needs a locate, not confirmed) -- the forward shadow book assumes borrow at the backtest's 3%/yr, which the
runner does NOT charge; subtract ~3%/yr x 125k (~USD 70/week) when judging. If borrow is refused in real paper
trading, switch SHORTS=False (long-only Q1, not the tested strategy).
Data freshness: lab_daily.pkl is rebuilt by the repo's nightly pipeline; the signal day is stored in each fill note.
"""
from __future__ import annotations

from datetime import date

from research.ibkr_lab.runner.strategies import _common as C

NAME = "credit_equity_residz"
DESCRIPTION = "B3 L/S quintiles of debenture resid_z (tight-vs-peers long, wide short), 125k/side, weekly"
REBALANCE = "weekly"
EXPECTED = {"ann_return": 8.5, "vol": 9.3, "sharpe": 0.92, "status": "control",
            "note": "credit_equity residz|ls|h1 2022-26: 8.5%/yr, 9.3% vol, SR 0.92 (NW t 2.0, perm p 0.015) with 10bp "
                    "costs + 3% borrow; 2026 YTD SR 0.47. Mostly equity low-vol/distress (partial t -2.07), 5 names "
                    "drive it, nothing survives Holm. Runner charges no borrow: deduct ~0.3%/yr of NAV."}
SHORTS = True


def targets(ib, asof: date):
    from research.ibkr_lab.credit_equity import paper_check as P
    from research.ibkr_lab.setup import lib
    t, _ = P.basket()
    if not SHORTS:
        t = t[t.side == "LONG"]
    cs = lib.b3_stocks(ib, list(t.ticker))
    per_side = C.CAPITAL / 2
    out = []
    for side, g in t.groupby("side"):
        g = g[g.ticker.isin(cs)]
        for _, r in g.iterrows():
            out.append({"contract": cs[r.ticker], "key": f"B3:{r.ticker}",
                        "target_notional_usd": min(per_side / len(g), C.POS_CAP), "side": side,
                        "note": f"residz={r.residz:+.2f} signal_day={str(r.signal_day)[:10]}"})
    return out
