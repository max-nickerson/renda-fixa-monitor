"""Safety layer for everything in research/ibkr_lab: PAPER ONLY.

- connect_paper(): connects only to the paper ports (4002 Gateway, 7497 TWS) and refuses unless EVERY managed
  account is a paper account (IBKR paper account ids start with 'DU'). Live port 4001/7496 is never tried.
- place_paper_order(): the only function allowed to send an order. Re-checks the account on every call, caps
  size, and records everything in data/ibkr_lab/orders.jsonl. If the Gateway is in Read-Only API mode the order
  is rejected by IBKR; callers must then fall back to shadow_fill() (simulated fill at live bid/ask).
- Each agent must use its own client id (IBKR_LAB_CLIENT_ID) to avoid collisions; historical-data pacing is
  shared, so use hist() which caches every request under data/ibkr_lab/cache/.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "ibkr_lab"
CACHE = DATA / "cache"
PAPER_PORTS = (4002, 7497)
MAX_ORDER_USD = float(os.getenv("IBKR_LAB_MAX_ORDER_USD", "50000"))


class NotPaper(RuntimeError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def log(name: str, rec: dict) -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    with open(DATA / f"{name}.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": _now(), **rec}, default=str, ensure_ascii=False) + "\n")


def connect_paper(client_id: int | None = None, timeout: float = 15.0):
    """Paper session only. If the client id is busy (e.g. a stale connection), the next ids are tried."""
    from ib_async import IB
    base = int(client_id or os.getenv("IBKR_LAB_CLIENT_ID", "60"))
    errors = []
    for port in PAPER_PORTS:
        for cid in (base, base + 100, base + 200, base + 300):
            ib = IB()
            ib.RequestTimeout = 30
            try:
                ib.connect("127.0.0.1", port, clientId=cid, timeout=timeout)
            except ConnectionRefusedError as e:
                errors.append(f"{port}: refused")
                break  # nothing listening on this port
            except Exception as e:
                errors.append(f"{port}/{cid}: {e!r}"[:120])
                try:
                    ib.disconnect()
                except Exception:
                    pass
                continue
            accts = ib.managedAccounts()
            if not accts or not all(a.startswith("DU") for a in accts):
                ib.disconnect()
                raise NotPaper(f"refusing: port {port} is not a paper session ({[a[:2] for a in accts]})")
            return ib
    raise ConnectionError(f"no paper Gateway/TWS on {PAPER_PORTS}: {errors}")


def assert_paper(ib) -> None:
    accts = ib.managedAccounts()
    if ib.client.port not in PAPER_PORTS or not accts or not all(a.startswith("DU") for a in accts):
        raise NotPaper("refusing to trade: not a paper session")


def place_paper_order(ib, contract, action: str, qty: float, order_type: str = "LMT", limit: float | None = None,
                      strategy: str = "unknown", est_notional_usd: float | None = None, wait: float = 5.0) -> dict:
    """Send an order to the PAPER account (never live). Returns status dict; rejected orders are logged too."""
    from ib_async import LimitOrder, MarketOrder
    assert_paper(ib)
    if est_notional_usd is not None and est_notional_usd > MAX_ORDER_USD:
        raise ValueError(f"order notional {est_notional_usd:.0f} USD above cap {MAX_ORDER_USD:.0f}")
    order = LimitOrder(action, qty, limit) if order_type == "LMT" else MarketOrder(action, qty)
    order.orderRef = f"lab:{strategy}"[:40]
    try:
        trade = ib.placeOrder(contract, order)
        ib.sleep(wait)
        st = {"status": trade.orderStatus.status, "filled": trade.orderStatus.filled,
              "avg_price": trade.orderStatus.avgFillPrice,
              "log": [f"{e.status}: {e.message}" for e in trade.log][-4:]}
    except Exception as e:
        st = {"status": "error", "error": repr(e)}
    log("orders", {"strategy": strategy, "symbol": getattr(contract, "localSymbol", "") or contract.symbol,
                   "secType": contract.secType, "action": action, "qty": qty, "type": order_type, "limit": limit, **st})
    return st


def shadow_fill(ib, contract, action: str, qty: float, strategy: str, note: str = "") -> dict:
    """Simulated fill at the live touch (buy at ask, sell at bid) when paper orders are not possible."""
    ib.reqMarketDataType(1)
    t = ib.reqMktData(contract, "", False, False)
    ib.sleep(2.5)
    px = t.ask if action == "BUY" else t.bid
    if not px or px != px or px <= 0:
        px = t.last if t.last and t.last == t.last else None
    ib.cancelMktData(contract)
    rec = {"strategy": strategy, "symbol": getattr(contract, "localSymbol", "") or contract.symbol,
           "secType": contract.secType, "action": action, "qty": qty, "price": px, "bid": t.bid, "ask": t.ask,
           "note": note}
    log("shadow_fills", rec)
    return rec


def hist(ib, contract, duration: str = "1 Y", bar: str = "1 day", what: str = "TRADES", rth: bool = True):
    """Cached reqHistoricalData (IBKR pacing: ~60 requests / 10 min shared by everyone)."""
    import pandas as pd
    from ib_async import util
    CACHE.mkdir(parents=True, exist_ok=True)
    key = f"{contract.secType}_{getattr(contract, 'localSymbol', '') or contract.symbol}_{contract.conId}_{duration}_{bar}_{what}_{rth}"
    path = CACHE / (hashlib.md5(key.encode()).hexdigest() + ".pkl")
    if path.exists() and time.time() - path.stat().st_mtime < 6 * 3600:
        return pd.read_pickle(path)
    bars = ib.reqHistoricalData(contract, "", duration, bar, what, rth, 1)
    df = util.df(bars) if bars else pd.DataFrame()
    df.to_pickle(path)
    time.sleep(1.0)  # be polite with pacing
    return df
