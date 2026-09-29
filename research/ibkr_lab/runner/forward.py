"""Forward (live) validation of the IBKR-lab strategies on the PAPER login.

Each strategy is a plugin: research/ibkr_lab/runner/strategies/<name>.py exposing
    NAME, DESCRIPTION, REBALANCE ('daily' | 'weekly' | 'monthly'), EXPECTED (dict: ann_return, vol, sharpe from backtest)
    targets(ib, asof) -> list[dict(contract=<ib contract>, key=str, target_notional_usd=float, side='LONG'|'SHORT')]
The runner (once per market day after the close, plus intraday marks):
  1) calls targets() on rebalance days,
  2) trades the difference: a real paper order through safe.place_paper_order if the paper login accepts orders,
     otherwise safe.shadow_fill (live bid/ask) — every fill is recorded with its quoted spread,
  3) marks all positions with live prices and writes the daily NAV of every strategy,
into data/ibkr_lab/forward.db (SQLite). The /paper page reads it. Never touches the live account (see safe.py).
"""
from __future__ import annotations

import importlib
import json
import logging
import pkgutil
import sqlite3
from contextlib import contextmanager
from datetime import date, datetime, timezone

from research.ibkr_lab import safe

log = logging.getLogger(__name__)
DB = safe.DATA / "forward.db"
CAPITAL_USD = 250_000.0  # per strategy, small on purpose
SCHEMA = """
CREATE TABLE IF NOT EXISTS strategies (name TEXT PRIMARY KEY, description TEXT, rebalance TEXT, expected TEXT,
    started TEXT, capital REAL, active INTEGER DEFAULT 1);
CREATE TABLE IF NOT EXISTS positions (strategy TEXT, key TEXT, conid INTEGER, qty REAL, avg_px REAL, last_px REAL,
    currency TEXT, multiplier REAL DEFAULT 1, PRIMARY KEY (strategy, key));
CREATE TABLE IF NOT EXISTS fills (ts TEXT, strategy TEXT, key TEXT, action TEXT, qty REAL, px REAL, bid REAL, ask REAL,
    mode TEXT, note TEXT);
CREATE TABLE IF NOT EXISTS nav (strategy TEXT, date TEXT, nav REAL, cash REAL, gross REAL, n_pos INTEGER,
    PRIMARY KEY (strategy, date));
CREATE TABLE IF NOT EXISTS runs (ts TEXT, what TEXT, note TEXT);
"""


@contextmanager
def db():
    DB.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB, timeout=30)
    con.row_factory = sqlite3.Row
    try:
        con.executescript(SCHEMA)
        yield con
        con.commit()
    finally:
        con.close()


def plugins() -> list:
    from research.ibkr_lab.runner import strategies as pkg
    mods = []
    for m in pkgutil.iter_modules(pkg.__path__):
        if not m.name.startswith("_"):
            try:
                mods.append(importlib.import_module(f"{pkg.__name__}.{m.name}"))
            except Exception as e:
                log.exception("plugin %s failed to import: %s", m.name, e)
    return mods


def _is_rebalance(freq: str, d: date, last: str | None) -> bool:
    if last is None:
        return True
    ld = date.fromisoformat(last)
    return {"daily": d > ld, "weekly": d.isocalendar()[1] != ld.isocalendar()[1] or d.year != ld.year,
            "monthly": (d.year, d.month) != (ld.year, ld.month)}.get(freq, False)


def _price(ib, contract) -> dict:
    ib.reqMarketDataType(1)
    t = ib.reqMktData(contract, "", False, False)
    ib.sleep(2)
    ib.cancelMktData(contract)
    ok = lambda v: v if v and v == v and v > 0 else None
    mid = (ok(t.bid) + ok(t.ask)) / 2 if ok(t.bid) and ok(t.ask) else None
    return {"bid": ok(t.bid), "ask": ok(t.ask), "last": ok(t.last) or mid or ok(t.close)}


def _orders_accepted(ib) -> bool:
    """Read-Only API → orders rejected; remember the answer for the day (checked by the setup probe log)."""
    try:
        recs = [json.loads(l) for l in open(safe.DATA / "orders.jsonl", encoding="utf-8")][-5:]
        return any(r.get("status") not in ("ValidationError", "error", "Cancelled", "Inactive") for r in recs)
    except Exception:
        return False


def run_day(asof: date | None = None, mark_only: bool = False) -> dict:
    asof = asof or date.today()
    ib = safe.connect_paper(client_id=80)
    out = {}
    try:
        live_orders = _orders_accepted(ib)
        for mod in plugins():
            name = mod.NAME
            with db() as con:
                con.execute("INSERT OR IGNORE INTO strategies VALUES (?,?,?,?,?,?,1)",
                            (name, mod.DESCRIPTION, mod.REBALANCE, json.dumps(getattr(mod, "EXPECTED", {})),
                             asof.isoformat(), CAPITAL_USD))
                last = con.execute("SELECT max(date) d FROM nav WHERE strategy=?", (name,)).fetchone()["d"]
                pos = {r["key"]: dict(r) for r in con.execute("SELECT * FROM positions WHERE strategy=?", (name,))}
                cash_row = con.execute("SELECT cash FROM nav WHERE strategy=? ORDER BY date DESC LIMIT 1", (name,)).fetchone()
            cash = cash_row["cash"] if cash_row else CAPITAL_USD
            note = []
            if not mark_only and _is_rebalance(mod.REBALANCE, asof, last):
                try:
                    tg = mod.targets(ib, asof)
                except Exception as e:
                    log.exception("%s targets failed", name)
                    tg, note = None, [f"targets error: {e!r}"[:150]]
                if tg is not None:
                    want = {t["key"]: t for t in tg}
                    for key in set(want) | set(pos):
                        t = want.get(key)
                        c = t["contract"] if t else None
                        if c is None:  # close a position no longer wanted
                            from ib_async import Contract
                            c = Contract(conId=pos[key]["conid"])
                            ib.qualifyContracts(c)
                        q = _price(ib, c)
                        px = q["last"]
                        if not px:
                            note.append(f"no price {key}")
                            continue
                        mult = float(getattr(c, "multiplier", "") or 1)
                        fx = 1.0 if (c.currency or "USD") == "USD" else _fx_usd(ib, c.currency)
                        tgt_qty = 0.0
                        if t:
                            sign = 1 if t.get("side", "LONG") == "LONG" else -1
                            tgt_qty = sign * round(t["target_notional_usd"] / (px * mult * fx))
                        cur = pos.get(key, {}).get("qty", 0.0)
                        dq = tgt_qty - cur
                        if abs(dq) < 1:
                            continue
                        action = "BUY" if dq > 0 else "SELL"
                        if live_orders:
                            st = safe.place_paper_order(ib, c, action, abs(dq), "LMT",
                                                        q["ask"] if action == "BUY" else q["bid"], strategy=name,
                                                        est_notional_usd=abs(dq) * px * mult * fx)
                            fill_px, mode = (st.get("avg_price") or px), f"paper:{st.get('status')}"
                        else:
                            fill_px = (q["ask"] if action == "BUY" else q["bid"]) or px
                            safe.log("shadow_fills", {"strategy": name, "key": key, "action": action, "qty": abs(dq),
                                                      "price": fill_px, "bid": q["bid"], "ask": q["ask"]})
                            mode = "shadow"
                        cash -= dq * fill_px * mult * fx
                        new_qty = cur + dq
                        with db() as con:
                            con.execute("INSERT INTO fills VALUES (?,?,?,?,?,?,?,?,?,?)",
                                        (datetime.now(timezone.utc).isoformat(timespec="seconds"), name, key, action,
                                         abs(dq), fill_px, q["bid"], q["ask"], mode, ""))
                            if abs(new_qty) < 1e-9:
                                con.execute("DELETE FROM positions WHERE strategy=? AND key=?", (name, key))
                                pos.pop(key, None)
                            else:
                                con.execute("INSERT OR REPLACE INTO positions VALUES (?,?,?,?,?,?,?,?)",
                                            (name, key, c.conId, new_qty, fill_px, fill_px, c.currency, mult))
                                pos[key] = {"key": key, "conid": c.conId, "qty": new_qty, "multiplier": mult, "fx": fx}
            # mark to market
            gross, value = 0.0, 0.0
            with db() as con:
                rows = [dict(r) for r in con.execute("SELECT * FROM positions WHERE strategy=?", (name,))]
            for r in rows:
                from ib_async import Contract
                c = Contract(conId=r["conid"])
                try:
                    ib.qualifyContracts(c)
                    px = _price(ib, c)["last"] or r["last_px"]
                except Exception:
                    px = r["last_px"]
                fx = 1.0 if (r["currency"] or "USD") == "USD" else _fx_usd(ib, r["currency"])
                value += r["qty"] * px * (r["multiplier"] or 1) * fx
                gross += abs(r["qty"] * px * (r["multiplier"] or 1) * fx)
                with db() as con:
                    con.execute("UPDATE positions SET last_px=? WHERE strategy=? AND key=?", (px, name, r["key"]))
            with db() as con:
                con.execute("INSERT OR REPLACE INTO nav VALUES (?,?,?,?,?,?)",
                            (name, asof.isoformat(), cash + value, cash, gross, len(rows)))
            out[name] = {"nav": round(cash + value, 2), "positions": len(rows), "notes": note}
        with db() as con:
            con.execute("INSERT INTO runs VALUES (?,?,?)", (datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                                             "mark" if mark_only else "day", json.dumps(out)[:2000]))
    finally:
        ib.disconnect()
    return out


_FX = {}


def _fx_usd(ib, ccy: str) -> float:
    """USD per 1 unit of ccy (BRL via IBKR if available, else BCB PTAX)."""
    if ccy in _FX:
        return _FX[ccy]
    v = None
    if ccy == "BRL":
        try:
            from rfmonitor.history import bcb_series
            s = bcb_series(1, date(2026, 1, 1))  # BRL per USD
            v = 1.0 / float(s.dropna().iloc[-1])
        except Exception:
            v = None
    _FX[ccy] = v or 1.0
    return _FX[ccy]
