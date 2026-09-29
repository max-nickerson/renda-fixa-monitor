"""SPX box lending-rate MEASUREMENT (log-only variant): targets() always returns [] and appends one record per day to
data/ibkr_lab/forward_box_rates.jsonl with the implied lending rate of a ~1y SPX box vs the maturity-matched Treasury.

Why log-only and not a held box: a <= 50k SPX box (width 400 -> pays USD 40k) is fine as ONE position, but the runner
books options per leg and every 1y SPX box has at least one deep-ITM leg worth > USD 50k (e.g. the K2 put or K1 call
~ 600-1,300 index points x 100), so the per-position cap (runner + safe.place_paper_order) would clip it; the runner
has no BAG/combo contract support, and legging 4 shadow fills at 4 separate delayed touches would add ~4 half-spreads
(~USD 1-2k on a 40k box) that are not what the study measures (the study's result is the combo mid/touch rate).
The daily log gives exactly the quantity to validate: r_mid and r_touch (lend = BUY box at the combined ask) minus
the Treasury CMT interpolated at the same maturity (study: mid ~ +52-63 bp over UST, touch ~ -40 bp).

Box choice (stable, so the series is comparable day to day): SPX monthly (AM, European) expiry closest to 1 year
(>= 0.4y), strikes K1 = spot rounded down to 100 - 200 and K2 = K1 + 400, re-chosen only when the held choice
(stored in the last log line) is < 0.4y from expiry. REBALANCE is 'daily' so the log is written every run day.
"""
from __future__ import annotations

import json
import math
from datetime import date, datetime, timezone

from research.ibkr_lab import safe
from research.ibkr_lab.runner.strategies import _common as C

NAME = "spx_box_lend"
DESCRIPTION = "SPX ~1y box (width 400 = USD 40k) implied lending rate vs UST, logged daily (no positions held)"
REBALANCE = "daily"
EXPECTED = {"ann_return": 0.0, "vol": 0.0, "sharpe": 0.0, "status": "measurement",
            "note": "box_spreads (verified): box mid 52-63 bp over UST and 52-57 bp over term SOFR at 0.5-2y; "
                    "lending at the touch (<=50k) ~40 bp BELOW UST (median). Forward check: median(r_mid - r_tsy) "
                    "and median(r_touch - r_tsy) over >= 20 days vs those numbers. NAV stays flat by design."}
LOG = safe.DATA / "forward_box_rates.jsonl"
WIDTH = 400


def _last_choice():
    try:
        return json.loads(LOG.read_text(encoding="utf-8").splitlines()[-1]).get("box")
    except Exception:
        return None


def targets(ib, asof: date):
    from ib_async import Index, Option
    from research.ibkr_lab.box_spreads.analyze import FEE, fred, tsy_cc
    from research.ibkr_lab.box_spreads.collector import _read_batch
    spx = ib.qualifyContracts(Index("SPX", "CBOE", "USD"))[0]
    box = _last_choice()
    if not box or (datetime.strptime(box[0], "%Y%m%d").date() - asof).days < 0.4 * 365:
        sq = C.quote(ib, spx)
        spot = sq["last"] or sq["close"] or sq["mid"]
        p = [x for x in ib.reqSecDefOptParams("SPX", "", "IND", spx.conId) if x.tradingClass == "SPX"]
        p = next((x for x in p if x.exchange == "SMART"), p[0])
        exps = [e for e in p.expirations if (datetime.strptime(e, "%Y%m%d").date() - asof).days >= 0.4 * 365]
        exp = min(exps, key=lambda e: abs((datetime.strptime(e, "%Y%m%d").date() - asof).days - 365))
        k1 = math.floor(spot / 100) * 100 - 200
        box = [exp, k1, k1 + WIDTH]
    exp, k1, k2 = box
    legs = [Option("SPX", exp, k, r, "SMART", tradingClass="SPX", currency="USD", multiplier="100")
            for k, r in ((k1, "C"), (k2, "C"), (k1, "P"), (k2, "P"))]
    legs = ib.qualifyContracts(*legs)
    rec = {"date": asof.isoformat(), "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "box": box}
    if any(c is None or not c.conId for c in legs):
        rec["error"] = "legs not qualified"
    else:
        q = {(r["strike"], r["right"]): r for r in _read_batch(ib, legs, wait=6.0)}
        c1, c2, p1, p2 = q[(k1, "C")], q[(k2, "C")], q[(k1, "P")], q[(k2, "P")]
        rec["legs"] = {f"{int(k)}{r}": [v["bid"], v["ask"], v["mdType"], v["fresh"]] for (k, r), v in q.items()}
        if all(v["bid"] and v["ask"] for v in q.values()):
            ask = c1["ask"] - c2["bid"] - p1["bid"] + p2["ask"]
            bid = c1["bid"] - c2["ask"] - p1["ask"] + p2["bid"]
            mid = (ask + bid) / 2
            T = (datetime.strptime(exp, "%Y%m%d").date() - asof).days / 365
            fee = 4 * FEE["SPX"]
            r_touch = math.log(WIDTH * 100 / (ask * 100 + fee)) / T
            r_mid = math.log(WIDTH * 100 / (mid * 100 + fee)) / T
            try:
                r_t = tsy_cc(fred().ffill().iloc[-1], T)
            except Exception:
                r_t = None
            rec.update({"T": round(T, 4), "bid": round(bid, 2), "ask": round(ask, 2), "mid": round(mid, 3),
                        "notional_usd": WIDTH * 100, "r_mid": round(r_mid, 5), "r_touch": round(r_touch, 5),
                        "r_tsy": None if r_t is None else round(r_t, 5),
                        "mid_minus_tsy_bp": None if r_t is None else round((r_mid - r_t) * 1e4, 1),
                        "touch_minus_tsy_bp": None if r_t is None else round((r_touch - r_t) * 1e4, 1),
                        "all_fresh": all(v["fresh"] for v in q.values())})
        else:
            rec["error"] = "missing bid/ask on a leg (market closed or no delayed data)"
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, default=str) + "\n")
    return []
