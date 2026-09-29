"""Interactive Brokers market data through a locally running IB Gateway / TWS (read-only use: quotes only).

Setup: IB Gateway or TWS logged in → Configure → Settings → API → Settings:
  Enable ActiveX and Socket Clients, Read-Only API (checked), Socket port 4001 (Gateway live) / 7496 (TWS live),
  Trusted IPs 127.0.0.1. Then:  python -m rfmonitor ibkr-test
Data stays on this machine (exchange licences forbid redistribution): never export it to the public site.
"""
from __future__ import annotations

import logging
import os

log = logging.getLogger(__name__)

HOST = os.getenv("IBKR_HOST", "127.0.0.1")
PORTS = [int(p) for p in os.getenv("IBKR_PORT", "4001,7496,4002,7497").split(",")]
CLIENT_ID = int(os.getenv("IBKR_CLIENT_ID", "17"))


def connect(timeout: float = 4.0):
    """First port that answers (Gateway live, TWS live, Gateway paper, TWS paper)."""
    from ib_async import IB
    last = None
    for port in PORTS:
        ib = IB()
        try:
            ib.connect(HOST, port, clientId=CLIENT_ID, timeout=timeout, readonly=True)
            log.info("IBKR connected on %s:%s", HOST, port)
            return ib
        except Exception as e:  # not listening / refused
            last = e
    raise ConnectionError(f"IB Gateway/TWS API not reachable on {HOST}:{PORTS} ({last}). "
                          "Is it logged in with 'Enable ActiveX and Socket Clients' on?")


def b3_stock(symbol: str):
    from ib_async import Stock
    return Stock(symbol, "BOVESPA", "BRL")


def quotes(ib, contracts: list, wait: float = 3.0) -> dict:
    """Snapshot of last/bid/ask/close for qualified contracts: {localSymbol: {...}}."""
    ib.reqMarketDataType(int(os.getenv("IBKR_DATA_TYPE", "1")))  # 1 real-time, 3 delayed
    cs = ib.qualifyContracts(*contracts)
    tickers = [ib.reqMktData(c, "", snapshot=False, regulatorySnapshot=False) for c in cs]
    ib.sleep(wait)
    out = {}
    for t in tickers:
        c = t.contract
        out[c.localSymbol or c.symbol] = {"last": t.last, "bid": t.bid, "ask": t.ask, "close": t.close,
                                          "time": str(t.time) if t.time else None, "exchange": c.exchange,
                                          "currency": c.currency}
        ib.cancelMktData(c)
    return out


def test() -> dict:
    """Connection check + one B3 stock, one B3 DI future search and one eurobond search."""
    ib = connect()
    try:
        res = {"server_version": ib.client.serverVersion(), "accounts": len(ib.managedAccounts())}
        res["PETR4"] = quotes(ib, [b3_stock("PETR4")]).get("PETR4")
        res["DI1_search"] = [f"{d.contract.symbol} {d.contract.secType} {d.contract.exchange} {d.contract.currency}"
                             for d in ib.reqMatchingSymbols("DI1")[:5]]
        res["bond_search_braskem"] = [f"{d.contract.symbol} {d.contract.secType} {d.contract.exchange}"
                                      for d in ib.reqMatchingSymbols("BRASKEM")[:5]]
        return res
    finally:
        ib.disconnect()
