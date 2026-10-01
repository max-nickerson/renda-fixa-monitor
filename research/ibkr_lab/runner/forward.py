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
import math
import pkgutil
import sqlite3
from contextlib import contextmanager
from datetime import date, datetime, timezone

from research.ibkr_lab import safe

log = logging.getLogger(__name__)
DB = safe.DATA / "forward.db"
CAPITAL_USD = 250_000.0  # per strategy, small on purpose
MAX_POS_USD = min(50_000.0, safe.MAX_ORDER_USD)  # per position (valuation notional); targets above are clipped
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
CREATE TABLE IF NOT EXISTS nav_intraday (strategy TEXT, ts TEXT, nav REAL, gross REAL, n_pos INTEGER,
    PRIMARY KEY (strategy, ts));
CREATE TABLE IF NOT EXISTS rebalances (strategy TEXT, date TEXT, PRIMARY KEY (strategy, date));
CREATE TABLE IF NOT EXISTS recon (ts TEXT, conid INTEGER, symbol TEXT, ledger_qty REAL, ibkr_qty REAL);
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


_QCACHE: dict = {}  # conId -> quote, cleared at the start of every run_day (one quote per contract per run)


def _price(ib, contract) -> dict:
    """Quote at the touch. Market-data type 4 = live where subscribed, else 15-min delayed, and the last (frozen)
    values once the market is closed. Type 1 (live only) returned nothing for CME/B3 futures, US options and SPX on
    this paper login. 'last' is the valuation price: MID first (a stale last trade mis-marks options/bonds)."""
    if contract.conId and contract.conId in _QCACHE:
        return _QCACHE[contract.conId]
    ib.reqMarketDataType(4)
    t = ib.reqMktData(contract, "", False, False)
    ib.sleep(3)
    ib.cancelMktData(contract)
    ok = lambda v: v if v and v == v and v > 0 else None
    mid = (ok(t.bid) + ok(t.ask)) / 2 if ok(t.bid) and ok(t.ask) else None
    val = mid or ok(t.last) or ok(t.close)
    if mid and (ok(t.ask) - ok(t.bid)) / mid > 0.015:
        # very wide touch (typically one stale side on a delayed/frozen back month or an illiquid bond): value at the
        # last trade / close if it lies inside the touch, else keep the mid
        inside = [v for v in (ok(t.last), ok(t.close)) if v and ok(t.bid) <= v <= ok(t.ask)]
        val = inside[0] if inside else mid
    q = {"bid": ok(t.bid), "ask": ok(t.ask), "last": val}
    if contract.conId:
        _QCACHE[contract.conId] = q
    return q


def _mult(c) -> float:
    """Value of 1 unit per 1 price point. IBKR bonds: size = face value in USD, price = % of par -> 0.01."""
    if getattr(c, "secType", "") == "BOND":
        return 0.01
    return float(getattr(c, "multiplier", "") or 1)


def _unit(mod, c, px, asof, mult):
    """Optional plugin hook unit_value(contract, px, asof) -> value of ONE unit in the contract currency (multiplier
    included). Used for instruments quoted in something other than price, e.g. DI1 (quoted as a rate)."""
    f = getattr(mod, "unit_value", None)
    if f is None or px is None:
        return px, mult
    return f(c, px, asof), 1.0


def _orders_accepted(ib) -> bool:
    """Read-Only API → orders rejected; remember the answer for the day (checked by the setup probe log)."""
    try:
        recs = [json.loads(l) for l in open(safe.DATA / "orders.jsonl", encoding="utf-8")][-5:]
        return any(r.get("status") not in ("ValidationError", "error", "Cancelled", "Inactive") for r in recs)
    except Exception:
        return False


def run_day(asof: date | None = None, mark_only: bool = False) -> dict:
    asof = asof or date.today()
    _QCACHE.clear()
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
                last = con.execute("SELECT max(date) d FROM rebalances WHERE strategy=?", (name,)).fetchone()["d"]
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
                    with db() as con:
                        con.execute("INSERT OR IGNORE INTO rebalances VALUES (?,?)", (name, asof.isoformat()))
                    want = {t["key"]: t for t in tg}
                    for key in set(want) | set(pos):
                        t = want.get(key)
                        c = t["contract"] if t else None
                        if c is None:  # close a position no longer wanted
                            from ib_async import Contract
                            c = Contract(conId=pos[key]["conid"])
                            try:
                                ib.qualifyContracts(c)
                            except Exception as e:
                                note.append(f"cannot qualify {key}: {e!r}"[:120])
                                continue
                        q = _price(ib, c)
                        px = q["last"]
                        if not px:
                            note.append(f"no price {key}")
                            continue
                        mult = _mult(c)
                        fx = 1.0 if (c.currency or "USD") == "USD" else _fx_usd(ib, c.currency)
                        upx, umult = _unit(mod, c, px, asof, mult)   # valuation price/multiplier (hook-aware)
                        unit_usd = abs(upx * umult * fx)
                        tgt_qty = 0.0
                        if t:
                            sign = 1 if t.get("side", "LONG") == "LONG" else -1
                            # explicit unit count (option legs, DI1, bonds) or notional sizing
                            tgt_qty = sign * (abs(float(t["qty"])) if t.get("qty") is not None
                                              else round(t["target_notional_usd"] / unit_usd))
                            cap = t.get("cap_usd", MAX_POS_USD)  # a plugin may only LOWER the per-position cap
                            if abs(tgt_qty) * unit_usd > min(cap, MAX_POS_USD):
                                tgt_qty = sign * math.floor(min(cap, MAX_POS_USD) / unit_usd)
                                note.append(f"{key} clipped to {abs(tgt_qty):g} (cap)")
                        cur = pos.get(key, {}).get("qty", 0.0)
                        dq = tgt_qty - cur
                        if abs(dq) < 1:
                            continue
                        action = "BUY" if dq > 0 else "SELL"
                        mode = None
                        if live_orders:
                            try:
                                # marketable limit (touch +/- 0.3%), wait up to 45 s, cancel the rest: the ledger only
                                # records what IBKR really filled, so it matches the paper account.
                                lim = (q["ask"] or px) * 1.003 if action == "BUY" else (q["bid"] or px) * 0.997
                                tick = 0.01 if (c.secType == "STK") else None
                                if tick:
                                    lim = round(lim / tick) * tick
                                # futures/options: some are quoted in rate (DI1) or wide, so use a market order on
                                # paper; stocks/bonds use the marketable limit above
                                otype = "LMT" if c.secType in ("STK", "BOND") else "MKT"
                                st = safe.place_paper_order(ib, c, action, abs(dq), otype, lim if otype == "LMT" else None,
                                                            strategy=name,
                                                            est_notional_usd=abs(dq) * unit_usd, wait=3,
                                                            wait_fill=45, cancel_unfilled=True)
                                filled = float(st.get("filled") or 0)
                                if filled < 1:
                                    note.append(f"{key}: paper order not filled ({st.get('status')})"[:120])
                                    continue
                                dq = filled if dq > 0 else -filled
                                fill_px, mode = (st.get("avg_price") or px), f"paper:{st.get('status')}"
                            except ValueError as e:  # safe.py size cap -> never bypass it, shadow instead
                                note.append(f"{key}: {e}"[:120])
                        if mode is None:
                            fill_px = (q["ask"] if action == "BUY" else q["bid"]) or px
                            safe.log("shadow_fills", {"strategy": name, "key": key, "action": action, "qty": abs(dq),
                                                      "price": fill_px, "bid": q["bid"], "ask": q["ask"]})
                            mode = "shadow"
                        fill_u, _ = _unit(mod, c, fill_px, asof, mult)
                        cash -= dq * fill_u * umult * fx
                        new_qty = cur + dq
                        with db() as con:
                            con.execute("INSERT INTO fills VALUES (?,?,?,?,?,?,?,?,?,?)",
                                        (datetime.now(timezone.utc).isoformat(timespec="seconds"), name, key, action,
                                         abs(dq), fill_px, q["bid"], q["ask"], mode,
                                         (t or {}).get("note", "close") if t else "close"))
                            if abs(new_qty) < 1e-9:
                                con.execute("DELETE FROM positions WHERE strategy=? AND key=?", (name, key))
                                pos.pop(key, None)
                            else:
                                # avg_px/last_px are VALUATION prices (after the unit_value hook), multiplier likewise
                                con.execute("INSERT OR REPLACE INTO positions VALUES (?,?,?,?,?,?,?,?)",
                                            (name, key, c.conId, new_qty, fill_u, fill_u, c.currency, umult))
                                pos[key] = {"key": key, "conid": c.conId, "qty": new_qty, "multiplier": umult, "fx": fx}
            # mark to market
            gross, value = 0.0, 0.0
            with db() as con:
                rows = [dict(r) for r in con.execute("SELECT * FROM positions WHERE strategy=?", (name,))]
            _prefetch(ib, [r["conid"] for r in rows])
            for r in rows:
                from ib_async import Contract
                c = Contract(conId=r["conid"])
                try:
                    if r["conid"] not in _QCACHE:
                        ib.qualifyContracts(c)
                    raw = _price(ib, c)["last"]
                    px = _unit(mod, c, raw, asof, r["multiplier"])[0] if raw else r["last_px"]
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
                con.execute("INSERT OR REPLACE INTO nav_intraday VALUES (?,?,?,?,?)",
                            (name, datetime.now(timezone.utc).isoformat(timespec="minutes"), cash + value, gross, len(rows)))
            out[name] = {"nav": round(cash + value, 2), "positions": len(rows), "notes": note}
        try:
            _reconcile(ib)
        except Exception as e:
            log.warning("reconcile failed: %s", e)
        with db() as con:
            con.execute("INSERT INTO runs VALUES (?,?,?)", (datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                                             "mark" if mark_only else "day", json.dumps(out)[:2000]))
    finally:
        ib.disconnect()
    return out


def _prefetch(ib, conids: list) -> None:
    """One batched market-data request for all contracts not yet quoted in this run (instead of 3 s each)."""
    from ib_async import Contract
    todo = [c for c in dict.fromkeys(conids) if c and c not in _QCACHE]
    if not todo:
        return
    cs = [Contract(conId=c) for c in todo]
    try:
        cs = [c for c in ib.qualifyContracts(*cs) if c is not None and c.conId]
    except Exception:
        return
    ib.reqMarketDataType(4)
    ts = [(c, ib.reqMktData(c, "", False, False)) for c in cs]
    ib.sleep(4)
    ok = lambda v: v if v and v == v and v > 0 else None
    for c, t in ts:
        ib.cancelMktData(c)
        mid = (ok(t.bid) + ok(t.ask)) / 2 if ok(t.bid) and ok(t.ask) else None
        val = mid or ok(t.last) or ok(t.close)
        if mid and (ok(t.ask) - ok(t.bid)) / mid > 0.015:
            inside = [v for v in (ok(t.last), ok(t.close)) if v and ok(t.bid) <= v <= ok(t.ask)]
            val = inside[0] if inside else mid
        _QCACHE[c.conId] = {"bid": ok(t.bid), "ask": ok(t.ask), "last": val}


def _reconcile(ib) -> None:
    """Compare the runner's paper fills (sum over strategies per contract) with the paper account's positions."""
    with db() as con:
        paper = {r["conid"]: r["q"] for r in con.execute(
            "SELECT p.conid, sum(p.qty) q FROM positions p JOIN (SELECT DISTINCT strategy, key FROM fills "
            "WHERE mode LIKE 'paper:%') f ON f.strategy=p.strategy AND f.key=p.key GROUP BY p.conid")}
    acct = {p.contract.conId: (p.position, p.contract.localSymbol or p.contract.symbol) for p in ib.positions()}
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    rows = []
    for cid in set(paper) | set(acct):
        lq, (aq, sym) = paper.get(cid, 0.0), acct.get(cid, (0.0, str(cid)))
        if abs((lq or 0) - (aq or 0)) > 1e-6:
            rows.append((ts, cid, sym, lq, aq))
    if rows:
        with db() as con:
            con.executemany("INSERT INTO recon VALUES (?,?,?,?,?)", rows)
        log.warning("paper reconciliation: %d mismatches", len(rows))


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
