"""SPX iron condor, short 16-delta / long 5-delta on both sides, ~35 DTE, defined risk (control), monthly.

Legs are returned as 4 per-leg targets with explicit keys 'SPXW:<localSymbol>' and explicit contract counts (the
runner's per-contract logic handles option legs; no combo contract needed). Short legs = side SHORT.
Contract: SPXW (PM-settled, cash, European) because it lists an expiry near every target DTE; strikes on a 50-pt grid.
Strike choice: IBKR model delta of the (delayed / frozen) quotes: short put ~ -0.16, long put ~ -0.05, short call
~ +0.16, long call ~ +0.05 (fallback when no greeks: 1.0 / 1.65 sigma moves with sigma = VIX x sqrt(T)).
Size: n condors with max loss (widest wing - credit) x 100 x n <= 48k (<= 20% of the 250k capital).
Roll/expiry rule (runner only calls targets() on the monthly rebalance): legs are kept while their expiry is > 7 days
away; otherwise (next month's rebalance, ~2-5 days before expiry) they are closed at the touch and a new ~35 DTE
condor is opened. The backtest held to cash settlement instead, so forward costs are a bit higher (closing spread).
"""
from __future__ import annotations

import math
from datetime import date, datetime, timedelta

from research.ibkr_lab.runner.strategies import _common as C

NAME = "spx_condor"
DESCRIPTION = "SPX (SPXW) iron condor 16/5-delta ~35 DTE, max loss <= 48k, rolled at the monthly rebalance"
REBALANCE = "monthly"
EXPECTED = {"ann_return": 0.6, "vol": 0.8, "sharpe": 0.48, "status": "control",
            "note": "vol_premium model-priced backtest 2006-26: 3.5%/yr on max-loss capital, 4.4% vol, SR 0.48 "
                    "(t 2.2) but 0.22 in 2016-26 OOS; skew -3.8, worst month -9.9%. Here max loss ~<=48k of 250k "
                    "-> ~0.6%/yr, ~0.8% vol on 250k. Skew assumption moves SR a lot: forward fills are the test."}
TARGET_DTE, MIN_DTE_KEEP = 35, 7


def _keep_held(ib, asof):
    from ib_async import Contract
    h = C.held(NAME)
    if not h:
        return None
    out = []
    for key, r in h.items():
        c = Contract(conId=r["conid"])
        ib.qualifyContracts(c)
        exp = datetime.strptime(c.lastTradeDateOrContractMonth[:8], "%Y%m%d").date()
        if exp - asof <= timedelta(days=MIN_DTE_KEEP):
            return None                      # roll the whole condor
        out.append({"contract": c, "key": key, "target_notional_usd": 0.0, "qty": abs(r["qty"]),
                    "side": "LONG" if r["qty"] > 0 else "SHORT", "note": "hold"})
    return out


def targets(ib, asof: date):
    from ib_async import Index, Option
    kept = _keep_held(ib, asof)
    if kept:
        return kept
    spx = ib.qualifyContracts(Index("SPX", "CBOE", "USD"))[0]
    sq = C.quote(ib, spx)
    spot = sq["last"] or sq["close"] or sq["mid"]
    vq = C.quote(ib, ib.qualifyContracts(Index("VIX", "CBOE", "USD"))[0])
    vix = (vq["last"] or vq["close"] or 18.0) / 100
    params = [p for p in ib.reqSecDefOptParams("SPX", "", "IND", spx.conId) if p.tradingClass == "SPXW"]
    p = next((x for x in params if x.exchange == "SMART"), params[0])
    exps = sorted(e for e in p.expirations if (datetime.strptime(e, "%Y%m%d").date() - asof).days >= 21)
    exp = min(exps, key=lambda e: abs((datetime.strptime(e, "%Y%m%d").date() - asof).days - TARGET_DTE))
    T = ((datetime.strptime(exp, "%Y%m%d").date() - asof).days) / 365
    sd = vix * math.sqrt(T)
    strikes = sorted(k for k in p.strikes if k % 50 == 0)
    puts = [k for k in strikes if spot * (1 - 2.6 * sd) <= k <= spot * 0.995]
    calls = [k for k in strikes if spot * 1.005 <= k <= spot * (1 + 2.0 * sd)]
    cs = [Option("SPX", exp, k, "P", "SMART", tradingClass="SPXW", currency="USD", multiplier="100") for k in puts] + \
         [Option("SPX", exp, k, "C", "SMART", tradingClass="SPXW", currency="USD", multiplier="100") for k in calls]
    cs = [c for c in ib.qualifyContracts(*cs) if c is not None and c.conId]
    # one batch of quotes (<= ~60 lines), greeks from IBKR's model
    ib.reqMarketDataType(4)
    tks = [ib.reqMktData(c, "", False, False) for c in cs]
    ib.sleep(6)
    rows = []
    for c, t in zip(cs, tks):
        g = t.modelGreeks
        d = g.delta if g is not None and g.delta == g.delta else None
        ok = lambda v: float(v) if v is not None and v == v and v > 0 else None
        rows.append({"c": c, "K": c.strike, "R": c.right, "delta": d, "bid": ok(t.bid), "ask": ok(t.ask)})
        ib.cancelMktData(c)

    def pick(right, target, z):
        cand = [r for r in rows if r["R"] == right]
        wd = [r for r in cand if r["delta"] is not None]
        if len(wd) >= 4:
            return min(wd, key=lambda r: abs(abs(r["delta"]) - target))
        kk = spot * (1 - z * sd) if right == "P" else spot * (1 + z * sd)
        return min(cand, key=lambda r: abs(r["K"] - kk))

    sp, lp, sc, lc = pick("P", 0.16, 1.0), pick("P", 0.05, 1.65), pick("C", 0.16, 1.0), pick("C", 0.05, 1.65)
    if lp["K"] >= sp["K"] or lc["K"] <= sc["K"]:
        return []
    mid = lambda r: (r["bid"] + r["ask"]) / 2 if r["bid"] and r["ask"] else 0.0
    credit = mid(sp) + mid(sc) - mid(lp) - mid(lc)
    width = max(sp["K"] - lp["K"], lc["K"] - sc["K"])
    max_loss = (width - max(credit, 0)) * 100
    n = max(1, math.floor(C.POS_CAP / max_loss)) if max_loss > 0 else 1
    if n * max_loss > C.POS_CAP * 1.05:
        return []
    note = f"exp={exp} spot={spot:.0f} credit_mid={credit:.2f} width={width:.0f} n={n} maxloss={n*max_loss:.0f}"
    out = []
    for r, side in ((sp, "SHORT"), (lp, "LONG"), (sc, "SHORT"), (lc, "LONG")):
        out.append({"contract": r["c"], "key": f"SPXW:{r['c'].localSymbol}", "qty": n, "side": side,
                    "target_notional_usd": n * (mid(r) or 0) * 100, "note": note + f" delta={r['delta']}"})
    return out
