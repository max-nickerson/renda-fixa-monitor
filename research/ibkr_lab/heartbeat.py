"""5-minute heartbeat for the IBKR paper lab: Gateway status, paper account, positions, orders and a watch-list
snapshot, appended to data/ibkr_lab/heartbeat.jsonl (and live quotes to snapshots.jsonl). Runs forever."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from research.ibkr_lab import safe  # noqa: E402

WATCH_B3 = ["PETR4", "VALE3", "ITUB4", "BBDC4", "BPAC11", "B3SA3", "BRKM5", "RENT3", "EQTL3", "SBSP3", "CSAN3", "BOVA11"]
EVERY = 300


def beat():
    from ib_async import Stock, Forex
    rec = {"ok": False}
    try:
        ib = safe.connect_paper(client_id=59, timeout=45)
    except Exception as e:
        rec["error"] = repr(e)[:200]
        safe.log("heartbeat", rec)
        return
    try:
        rec["ok"] = True
        rec["account"] = {v.tag: v.value for v in ib.accountValues()
                          if v.tag in ("NetLiquidation", "TotalCashValue", "UnrealizedPnL", "RealizedPnL",
                                       "GrossPositionValue") and v.currency == "USD"}
        rec["positions"] = [(p.contract.localSymbol or p.contract.symbol, p.position, p.avgCost) for p in ib.positions()]
        rec["open_orders"] = len(ib.openTrades())
        cs = [Stock(s, "B3", "BRL") for s in WATCH_B3] + [Forex("USDBRL")]
        cs = [c for c in ib.qualifyContracts(*cs) if c is not None]
        ib.reqMarketDataType(1)
        ts = [ib.reqMktData(c, "", False, False) for c in cs]
        ib.sleep(3)
        snap = {}
        for c, t in zip(cs, ts):
            snap[c.localSymbol or c.symbol] = {"last": t.last, "bid": t.bid, "ask": t.ask, "close": t.close}
            ib.cancelMktData(c)
        rec["n_quotes"] = sum(1 for v in snap.values() if v["last"] == v["last"] and v["last"])
        safe.log("snapshots", {"quotes": snap})
    except Exception as e:
        rec["error"] = repr(e)[:200]
    finally:
        ib.disconnect()
    safe.log("heartbeat", rec)


if __name__ == "__main__":
    import asyncio
    while True:
        asyncio.set_event_loop(asyncio.new_event_loop())
        beat()
        time.sleep(EVERY)
