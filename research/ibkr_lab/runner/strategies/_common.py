"""Shared helpers for the forward-validation plugins (not a plugin: the leading underscore makes the runner skip it)."""
from __future__ import annotations

import sqlite3
import time
from datetime import date
from pathlib import Path

from research.ibkr_lab import safe

ROOT = Path(__file__).resolve().parents[4]
CAPITAL = 250_000.0     # per strategy (runner.CAPITAL_USD)
POS_CAP = 48_000.0      # per position, a little under the runner/safe.py 50k cap so rounding never trips it


def held(strategy: str) -> dict:
    """{key: row} of what the runner currently holds for a strategy (read-only)."""
    p = safe.DATA / "forward.db"
    if not p.exists():
        return {}
    con = sqlite3.connect(p, timeout=30)
    con.row_factory = sqlite3.Row
    try:
        return {r["key"]: dict(r) for r in con.execute("SELECT * FROM positions WHERE strategy=?", (strategy,))}
    except sqlite3.OperationalError:
        return {}
    finally:
        con.close()


def brl_per_usd() -> float:
    """PTAX (BCB SGS 1) via the repo's cached reader; fallback: IBKR store 6L/DOL."""
    try:
        from rfmonitor.history import bcb_series
        return float(bcb_series(1, date(2026, 1, 1)).dropna().iloc[-1])
    except Exception:
        from research.ibkr_lab.data import load
        u = load.usdbrl()
        for c in ("usdbrl_DOL", "usdbrl_6L"):
            if c in u and u[c].notna().any():
                return float(u[c].dropna().iloc[-1])
    return 5.3


def quote(ib, c, wait: float = 3.0) -> dict:
    """Touch quote, md type 4 (live / delayed / frozen after the close)."""
    ib.reqMarketDataType(4)
    t = ib.reqMktData(c, "", False, False)
    ib.sleep(wait)
    ib.cancelMktData(c)
    ok = lambda v: float(v) if v is not None and v == v and v > 0 else None
    b, a = ok(t.bid), ok(t.ask)
    g = t.modelGreeks
    return {"bid": b, "ask": a, "mid": (b + a) / 2 if b and a else None, "last": ok(t.last), "close": ok(t.close),
            "delta": (g.delta if g is not None and g.delta == g.delta else None), "mdType": t.marketDataType}


def file_age_days(p: Path) -> float:
    return (time.time() - p.stat().st_mtime) / 86400 if p.exists() else 1e9
