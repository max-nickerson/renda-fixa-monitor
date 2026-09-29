"""BRL time-series momentum ensemble (1/3/6/12m) on CME 6L (IBKR symbol BRE), monthly.

Signal: research/ibkr_lab/brl_fx/backtest.py signals()['tsmom_ls_ens']['ens'] (mean of sign of the 21/63/126/252-day
carry-inclusive BRL excess return; values in {-1,-0.5,0,0.5,1}), times the backtest's vol-target leverage
min(10% / EWMA63 vol, 2). Long 6L = long BRL / short USD.
Sizing: desired notional = 250k x ens x lev, then capped at 2 x 48k (= two positions of <= 48k each, the 50k
per-position safety cap), split evenly between the two contracts below. Integer contracts (floor), ~USD 19k each.
Roll rule: on each (monthly) rebalance hold the first two 6L contracts whose last trade date is > asof + 35 calendar
days, so nothing held can expire before the next monthly rebalance (6L serial months stop trading ~2 bd before the
contract month; skipping < 35 d also avoids the delivery week). Keys carry the contract, so a roll = close old key
and open the new one at the touch.
Data refresh: BCB/FRED inputs are re-downloaded (brl_fx/fetch.py) when older than 2 days.
Known forward/backtest gap: the backtest lags the signal 2 days (t-2 data); here the latest PTAX (usually t-1) is used.
"""
from __future__ import annotations

import math
from datetime import date, timedelta

from research.ibkr_lab.runner.strategies import _common as C

NAME = "brl_tsmom"
DESCRIPTION = "BRL TS-momentum ensemble 1/3/6/12m on CME 6L front/next, 10% vol target, capped at ~96k gross"
REBALANCE = "monthly"
EXPECTED = {"ann_return": 4.3, "vol": 8.1, "sharpe": 0.53, "status": "candidate",
            "note": "brl_fx walk-forward OOS 2006-26 net 3bp: SR 0.53 (NW t 2.3, placebo p<0.002); since 2013 only ~0.2; "
                    "@10bp 0.25. Book here is capped at ~96k of 250k, so expect return/vol x ~0.4-0.5 of the "
                    "backtest (same Sharpe): ~1.8%/yr, ~3.5% vol on 250k."}
MAX_TOTAL = 2 * C.POS_CAP


def signal() -> dict:
    from research.ibkr_lab.brl_fx import backtest as B
    from research.ibkr_lab.brl_fx import fetch
    if C.file_age_days(B.RAW / "ptax_sell.csv") > 2:
        try:
            fetch.main(force=True)
        except Exception:
            pass
    df = B.load()
    ens = float(B.signals(df)["tsmom_ls_ens"]["ens"].dropna().iloc[-1])
    vol = float(((df["r"] ** 2).ewm(span=63, min_periods=63).mean() * 252).iloc[-1] ** 0.5)
    lev = min(B.VOL_TGT / vol, 2.0)
    return {"ens": ens, "vol": vol, "lev": lev, "asof_data": str(df.index[-1].date())}


def targets(ib, asof: date):
    from ib_async import Future
    s = signal()
    want = C.CAPITAL * s["ens"] * s["lev"]
    total = min(abs(want), MAX_TOTAL)
    if total < 1:
        return []
    cut = (asof + timedelta(days=35)).strftime("%Y%m%d")
    det = ib.reqContractDetails(Future("BRE", exchange="CME", currency="USD"))
    cands = sorted((d.contract for d in det if d.contract.lastTradeDateOrContractMonth[:8] > cut),
                   key=lambda c: c.lastTradeDateOrContractMonth)[:2]
    side = "LONG" if s["ens"] > 0 else "SHORT"
    out = []
    for c in cands:
        q = C.quote(ib, c)
        px = q["mid"] or q["last"] or q["close"]
        if not px:
            continue
        mult = float(c.multiplier or 100000)
        qty = math.floor(total / len(cands) / (px * mult))
        if qty >= 1:
            out.append({"contract": c, "key": f"6L:{c.localSymbol}", "target_notional_usd": qty * px * mult,
                        "qty": qty, "side": side,
                        "note": f"ens={s['ens']:+.2f} lev={s['lev']:.2f} vol={s['vol']:.3f} data={s['asof_data']}"})
    return out
