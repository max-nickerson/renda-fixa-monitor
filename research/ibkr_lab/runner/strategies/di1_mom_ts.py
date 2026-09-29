"""DI1 time-series momentum (MOM_TS) with the look-ahead FIX, monthly, on B3 DI1 futures.

Signal: research/ibkr_lab/di_curve/backtest.py sig_mom(bret.shift(1), ...) -- the backtest's bucket excess returns are
indexed at t for the move t->t+1, so the original sig_mom used one day of the future; shifting by one day makes the
signal use only returns realised by t (verifier's correction: Sharpe 0.45 -> 0.435, OOS 2020-26 0.30 -> 0.28).
Per bucket (1y/2y/3y/5y = 252/504/756/1260 du): s = mean(sign of 3/6/12m excess return) in {-1,-1/3,1/3,1};
held contract = the F/J/N/V DI1 whose du is closest to the bucket (du >= 0.6 x bucket), as in bucket_books().
Sizing by DV01: backtest weight = s / 4 / modified duration (fraction of capital in PU notional), i.e. each bucket
carries capital x |s|/4 x 1bp of DV01. PU notional per bucket = 250k x |w|; the whole book is scaled down by one
factor so the largest leg is <= 48k (keeps DV01 proportions; the 1y leg binds). Contracts = round(notional / PU).
s > 0 = receive the rate (long PU) = SELL DI1 on IBKR (price is the rate) -> side SHORT.
Valuation (unit_value hook): one DI1 contract bought (= long rate) is worth -PU(rate, du) x CDI accrual since
2026-09-29, in BRL. With the CDI accrual the P&L of a held contract is exactly the B3 daily settlement
(PU_t x (1+CDI) - PU_t+1), i.e. carry vs CDI is captured and pull-to-par is not a fake P&L.
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from research.ibkr_lab.runner.strategies import _common as C

NAME = "di1_mom_ts"
DESCRIPTION = "DI1 MOM_TS (3/6/12m sign ensemble per 1/2/3/5y bucket), look-ahead fixed, DV01-sized, <=48k per leg"
REBALANCE = "monthly"
EXPECTED = {"ann_return": 0.5, "vol": 1.1, "sharpe": 0.435, "status": "candidate",
            "note": "di_curve MOM_TS after look-ahead fix (bret.shift(1)): SR 0.435 full 2013-26, OOS 2020-26 0.28, "
                    "0.36 with 2-day lag; placebo p<0.01; deflated-Sharpe prob 0.07. Raw book 0.73%/yr on 1.63% vol "
                    "per unit capital; the 48k-per-leg cap scales it by ~0.7 -> ~0.5%/yr, ~1.1% vol on 250k."}
BUCKETS = [252, 504, 756, 1260]
REF_DATE = pd.Timestamp("2026-09-29")


def _maturity(local_symbol: str) -> date:
    from research.ibkr_lab.di_curve import curve
    m = "FGHJKMNQUVXZ".index(local_symbol[3]) + 1
    return curve.first_bday(2000 + int(local_symbol[4:6]), m)


def _du(asof: date, mat: date) -> int:
    from research.ibkr_lab.data import b3cal
    return int(b3cal.bdays_between(np.datetime64(asof), np.datetime64(mat)))


def _acc(asof: date) -> float:
    """CDI accrual factor from REF_DATE to asof (B3 days), missing recent days filled with the last CDI."""
    from research.ibkr_lab.di_curve import curve
    try:
        s = curve.cdi()
        n = _du(REF_DATE.date(), asof) if asof > REF_DATE.date() else 0
        if n <= 0:
            return 1.0
        k = s[(s.index >= REF_DATE) & (s.index < pd.Timestamp(asof))]
        return float(np.prod(1 + k.values / 100) * (1 + s.iloc[-1] / 100) ** max(0, n - len(k)))
    except Exception:
        return 1.0


def unit_value(contract, px, asof):
    """BRL value of ONE contract held long in IBKR terms (long rate = short PU)."""
    from research.ibkr_lab.di_curve import curve
    du = max(_du(asof, _maturity(contract.localSymbol)), 1)
    return -float(curve.pu(px, du)) * _acc(asof)


def signal() -> dict:
    from research.ibkr_lab.di_curve import backtest as B
    p, cm, dates, reb = B.load()
    sel, bret = B.bucket_books(p, dates, reb)
    reb2 = pd.DatetimeIndex(sorted(set(reb) | {dates[-1]}))
    mom = B.sig_mom(bret.shift(1), reb2).iloc[-1]          # <- the look-ahead fix
    last = p[p.date == dates[-1]]
    out = {}
    for T in BUCKETS:
        g = last[last.du >= 0.6 * T]
        if g.empty or not np.isfinite(mom.get(T, np.nan)):
            continue
        c = g.iloc[(g.du - T).abs().argmin()]
        dur = c.du / 252 / (1 + c.rate / 100)
        out[T] = {"code": c.code, "s": float(mom[T]), "dur": float(dur), "rate": float(c.rate)}
    return {"asof_data": str(dates[-1].date()), "buckets": out}


def targets(ib, asof: date):
    from ib_async import Contract
    sg = signal()
    fx = C.brl_per_usd()
    legs = {}
    for T, b in sg["buckets"].items():
        w = b["s"] / len(BUCKETS) / b["dur"]                 # fraction of capital (PU notional), signed
        legs.setdefault(b["code"], 0.0)
        legs[b["code"]] += w
    if not legs:
        return []
    k = min(1.0, C.POS_CAP / (C.CAPITAL * max(abs(w) for w in legs.values())))
    out = []
    for code, w in legs.items():
        if abs(w) < 1e-9:
            continue
        c = ib.qualifyContracts(Contract(secType="FUT", localSymbol=code, exchange="B3", currency="BRL"))
        c = [x for x in c if x is not None and getattr(x, "conId", 0)]
        if not c:
            continue
        c = c[0]
        pu_usd = abs(unit_value(c, sg["buckets"][next(T for T, b in sg["buckets"].items() if b["code"] == code)]["rate"],
                                asof)) / fx
        qty = int(round(C.CAPITAL * abs(w) * k / pu_usd))
        if qty < 1:
            continue
        out.append({"contract": c, "key": f"DI1:{code}", "target_notional_usd": qty * pu_usd, "qty": qty,
                    "side": "SHORT" if w > 0 else "LONG",
                    "note": f"w={w:+.3f} scale={k:.2f} data={sg['asof_data']}"})
    return out
