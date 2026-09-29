"""Paper-trading database (data/paper.db): the daily point-in-time data it collects and the books it runs."""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager

import pandas as pd

from ..config import DATA_DIR

PATH = DATA_DIR / "paper.db"

SCHEMA = """
-- data collection (one row per day; never overwritten after the day closes, so it stays point-in-time)
CREATE TABLE IF NOT EXISTS snap_bonds (
    date TEXT NOT NULL, codigo TEXT NOT NULL, isin TEXT, cnpj8 TEXT, nome TEXT, kind TEXT, contract REAL,
    pu REAL, ratio REAL, taxa REAL, spread_bps REAL, duration REAL, resid_z REAL, cdi_pct REAL,
    press_neg_30d REAL, strategy TEXT, flags TEXT, collected_at TEXT,
    PRIMARY KEY (date, codigo));
CREATE TABLE IF NOT EXISTS snd_trades (
    date TEXT NOT NULL, codigo TEXT NOT NULL, qty REAL, trades REAL, pu_avg REAL, pct_curve REAL,
    PRIMARY KEY (date, codigo));
CREATE TABLE IF NOT EXISTS eq_close (
    date TEXT NOT NULL, ticker TEXT NOT NULL, close REAL, source TEXT, PRIMARY KEY (date, ticker));
CREATE TABLE IF NOT EXISTS rates (date TEXT PRIMARY KEY, cdi REAL, ipca_factor REAL);
CREATE TABLE IF NOT EXISTS collect_log (ts TEXT, what TEXT, n INTEGER, note TEXT);

-- paper books
CREATE TABLE IF NOT EXISTS books (name TEXT PRIMARY KEY, aum REAL, start TEXT, last_day TEXT, cash REAL);
CREATE TABLE IF NOT EXISTS signals (
    book TEXT, date TEXT, codigo TEXT, weight REAL, cdi_bps REAL, eqh REAL, quality REAL, note TEXT,
    PRIMARY KEY (book, date, codigo));
CREATE TABLE IF NOT EXISTS orders (
    id INTEGER PRIMARY KEY AUTOINCREMENT, book TEXT, tranche TEXT, codigo TEXT, side TEXT, value REAL,
    created TEXT, expires TEXT, status TEXT, fill_date TEXT, fill_ratio REAL, mark_ratio REAL,
    slippage_bps REAL, note TEXT);
CREATE TABLE IF NOT EXISTS positions (
    id INTEGER PRIMARY KEY AUTOINCREMENT, book TEXT, tranche TEXT, codigo TEXT, value REAL, ratio REAL,
    entry_date TEXT, exit_due TEXT, status TEXT, exit_date TEXT);
CREATE TABLE IF NOT EXISTS nav (
    book TEXT, date TEXT, nav REAL, cash REAL, invested REAL, n_pos INTEGER, PRIMARY KEY (book, date));
CREATE INDEX IF NOT EXISTS idx_snd_codigo ON snd_trades (codigo, date);
CREATE INDEX IF NOT EXISTS idx_snap_codigo ON snap_bonds (codigo, date);
"""


@contextmanager
def connect():
    PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(PATH, timeout=30)
    con.row_factory = sqlite3.Row
    try:
        con.executescript(SCHEMA)
        yield con
        con.commit()
    finally:
        con.close()


def df(sql: str, params=()) -> pd.DataFrame:
    with connect() as con:
        return pd.read_sql_query(sql, con, params=params)


def rows(sql: str, params=()) -> list[dict]:
    with connect() as con:
        return [dict(r) for r in con.execute(sql, params).fetchall()]


def log(what: str, n: int, note: str = "") -> None:
    with connect() as con:
        con.execute("INSERT INTO collect_log VALUES (datetime('now'), ?, ?, ?)", (what, n, note))


def dumps(x) -> str:
    return json.dumps(x, ensure_ascii=False, default=str)
