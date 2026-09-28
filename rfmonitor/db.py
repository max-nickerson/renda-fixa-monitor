"""SQLite storage: assets, time series observations, events and alerts."""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

import pandas as pd

from .config import DB_PATH, ensure_dirs

SCHEMA = """
CREATE TABLE IF NOT EXISTS assets (
    isin TEXT PRIMARY KEY,
    info TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS observations (
    isin TEXT NOT NULL,
    date TEXT NOT NULL,
    metric TEXT NOT NULL,
    value REAL,
    source TEXT,
    PRIMARY KEY (isin, date, metric)
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    isin TEXT NOT NULL,
    ts TEXT NOT NULL,
    kind TEXT NOT NULL,
    title TEXT NOT NULL,
    url TEXT,
    source TEXT,
    severity TEXT DEFAULT 'info',
    uid TEXT NOT NULL,
    UNIQUE (isin, uid)
);
CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    isin TEXT NOT NULL,
    ts TEXT NOT NULL,
    rule TEXT NOT NULL,
    severity TEXT NOT NULL,
    message TEXT NOT NULL,
    dedup_key TEXT NOT NULL UNIQUE,
    emailed INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS screener (
    date TEXT PRIMARY KEY,
    payload TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_obs_isin_metric ON observations (isin, metric, date);
CREATE INDEX IF NOT EXISTS idx_events_isin ON events (isin, ts);
"""


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextmanager
def connect():
    ensure_dirs()
    con = sqlite3.connect(DB_PATH, timeout=30)
    con.row_factory = sqlite3.Row
    try:
        con.executescript(SCHEMA)
        yield con
        con.commit()
    finally:
        con.close()


def save_asset(info: dict) -> None:
    with connect() as con:
        con.execute(
            "INSERT OR REPLACE INTO assets (isin, info, updated_at) VALUES (?, ?, ?)",
            (info["isin"], json.dumps(info, default=str), now_iso()),
        )


def get_asset(isin: str) -> dict | None:
    with connect() as con:
        row = con.execute("SELECT info FROM assets WHERE isin = ?", (isin,)).fetchone()
    return json.loads(row["info"]) if row else None


def delete_asset(isin: str) -> None:
    with connect() as con:
        for table in ("assets", "observations", "events", "alerts"):
            con.execute(f"DELETE FROM {table} WHERE isin = ?", (isin,))


def put_observations(isin: str, rows: list[tuple[str, str, float | None]], source: str) -> int:
    """rows: (date 'YYYY-MM-DD', metric, value)."""
    clean = [(isin, d, m, float(v), source) for d, m, v in rows if v is not None and v == v]
    if not clean:
        return 0
    with connect() as con:
        con.executemany(
            "INSERT OR REPLACE INTO observations (isin, date, metric, value, source) VALUES (?, ?, ?, ?, ?)",
            clean,
        )
    return len(clean)


def series(isin: str, metrics: list[str] | None = None) -> pd.DataFrame:
    """Wide daily DataFrame indexed by date, one column per metric."""
    q = "SELECT date, metric, value FROM observations WHERE isin = ?"
    params: list = [isin]
    if metrics:
        q += f" AND metric IN ({','.join('?' * len(metrics))})"
        params += metrics
    with connect() as con:
        df = pd.read_sql_query(q, con, params=params)
    if df.empty:
        return pd.DataFrame()
    wide = df.pivot_table(index="date", columns="metric", values="value", aggfunc="last")
    wide.index = pd.to_datetime(wide.index)
    return wide.sort_index()


def put_event(isin: str, ts: str, kind: str, title: str, url: str | None, source: str,
              uid: str, severity: str = "info") -> bool:
    """Insert an event; returns True if it is new."""
    with connect() as con:
        cur = con.execute(
            "INSERT OR IGNORE INTO events (isin, ts, kind, title, url, source, severity, uid)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (isin, ts, kind, title, url, source, severity, uid),
        )
        return cur.rowcount > 0


def events(isin: str | None = None, limit: int = 100, since: str | None = None) -> list[dict]:
    q, params = "SELECT * FROM events WHERE 1=1", []
    if isin:
        q += " AND isin = ?"
        params.append(isin)
    if since:
        q += " AND ts >= ?"
        params.append(since)
    q += " ORDER BY ts DESC LIMIT ?"
    params.append(limit)
    with connect() as con:
        return [dict(r) for r in con.execute(q, params)]


def put_alert(isin: str, rule: str, severity: str, message: str, dedup_key: str) -> bool:
    with connect() as con:
        cur = con.execute(
            "INSERT OR IGNORE INTO alerts (isin, ts, rule, severity, message, dedup_key)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (isin, now_iso(), rule, severity, message, dedup_key),
        )
        return cur.rowcount > 0


def alerts(isin: str | None = None, limit: int = 200, unsent_only: bool = False) -> list[dict]:
    q, params = "SELECT * FROM alerts WHERE 1=1", []
    if isin:
        q += " AND isin = ?"
        params.append(isin)
    if unsent_only:
        q += " AND emailed = 0"
    q += " ORDER BY ts DESC, id DESC LIMIT ?"
    params.append(limit)
    with connect() as con:
        return [dict(r) for r in con.execute(q, params)]


def save_screener(result: dict) -> None:
    with connect() as con:
        con.execute("INSERT OR REPLACE INTO screener (date, payload, created_at) VALUES (?, ?, ?)",
                    (result["date"], json.dumps(result, default=str), now_iso()))


def load_screener() -> dict | None:
    with connect() as con:
        row = con.execute("SELECT payload FROM screener ORDER BY date DESC LIMIT 1").fetchone()
    return json.loads(row["payload"]) if row else None


def mark_emailed(ids: list[int]) -> None:
    if not ids:
        return
    with connect() as con:
        con.execute(f"UPDATE alerts SET emailed = 1 WHERE id IN ({','.join('?' * len(ids))})", ids)
