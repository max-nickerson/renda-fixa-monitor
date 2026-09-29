"""Paper-trading engine: monthly tranches, fills at the next real SND trade, daily marks at ANBIMA prices.

Mechanics (same as the research backtests, but live and forward only):
  * On the first ANBIMA day of each month the book opens a tranche of AUM/6 with the strategy's target weights.
    Orders fill at the first SND trade dated after the decision day, at that trade's % of the par curve; unfilled
    orders expire after 20 business days (the money stays in cash, earning CDI).
  * A tranche is held 126 business days, then each bond is sold at its next SND trade (or, after 20 days without
    one, at the ANBIMA mark minus 50 bps, flagged as forced).
  * Positions are marked every day on ANBIMA's % PU par; value grows with the contract accrual
    (CDI·(1+spread)^τ, IPCA·(1+c)^τ or (1+c)^τ) × ratio change, so coupons and amortisation don't show as losses.
  * Slippage of every fill is recorded vs ANBIMA's mark of the same day: this is the number the research could not
    measure (how far real trades are from model marks).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from . import store, strategy

log = logging.getLogger(__name__)

BOOKS = {"P7": 100e6, "P4Q": 100e6}
TRANCHES = 6
HOLD_DAYS = 126
EXPIRE_DAYS = 20
FORCED_HAIRCUT_BPS = 50


def _bdays(a: str, b: str) -> int:
    return int(np.busday_count(a, b))


def _add_bdays(d: str, n: int) -> str:
    return str(np.busday_offset(d, n, roll="forward"))


def _ensure_books(start: str) -> None:
    with store.connect() as con:
        for name, aum in BOOKS.items():
            con.execute("INSERT OR IGNORE INTO books VALUES (?,?,?,?,?)", (name, aum, start, None, aum))


class Market:
    """Everything the engine needs for a range of days, loaded once."""

    def __init__(self, start: str, end: str):
        s = store.df("SELECT date, codigo, ratio, kind, contract FROM snap_bonds WHERE date >= date(?, '-10 day') "
                     "AND date <= ?", (start, end))
        self.ratio = s.pivot(index="date", columns="codigo", values="ratio").sort_index()
        self.ref = s.drop_duplicates("codigo", keep="last").set_index("codigo")[["kind", "contract"]]
        t = store.df("SELECT date, codigo, pct_curve FROM snd_trades WHERE date >= ? AND date <= ?", (start, end))
        self.trades = t.pivot_table(index="date", columns="codigo", values="pct_curve", aggfunc="mean").sort_index() / 100
        r = store.df("SELECT * FROM rates ORDER BY date").set_index("date")
        self.C = (1 + r["cdi"]).cumprod()
        self.I = r["ipca_factor"]
        self.days = list(self.ratio.index[(self.ratio.index > start) & (self.ratio.index <= end)])

    def mark(self, code: str, day: str) -> float | None:
        if code in self.ratio.columns and day in self.ratio.index:
            v = self.ratio.at[day, code]
            return None if pd.isna(v) else float(v)
        return None

    def trade(self, code: str, day: str) -> float | None:
        if code in self.trades.columns and day in self.trades.index:
            v = self.trades.at[day, code]
            return None if pd.isna(v) else float(v)
        return None

    def _f(self, s: pd.Series, a: str, b: str) -> float:
        x = s[s.index <= b]
        y = s[s.index <= a]
        return float(x.iloc[-1] / y.iloc[-1]) if len(x) and len(y) else 1.0

    def cdi(self, a: str, b: str) -> float:
        return self._f(self.C, a, b)

    def gpar(self, code: str, a: str, b: str) -> float:
        """Growth of the par value from a to b (contract accrual)."""
        if code not in self.ref.index:
            return self.cdi(a, b)
        kind, c = self.ref.at[code, "kind"], float(self.ref.at[code, "contract"]) / 100
        tau = _bdays(a, b) / 252
        if kind == "DI_SPREAD":
            return self.cdi(a, b) * (1 + c) ** tau
        if kind == "IPCA":
            return self._f(self.I, a, b) * (1 + c) ** tau
        return (1 + c) ** tau


def open_tranche(book: str, day: str) -> int:
    """Create buy orders for a new tranche decided at the close of `day` (idempotent per book and month)."""
    tranche = f"{book}-{day[:7]}"
    if store.rows("SELECT 1 FROM orders WHERE tranche = ? LIMIT 1", (tranche,)):
        return 0
    aum = store.rows("SELECT aum FROM books WHERE name = ?", (book,))[0]["aum"]
    nav = store.rows("SELECT nav FROM nav WHERE book = ? ORDER BY date DESC LIMIT 1", (book,))
    size = (nav[0]["nav"] if nav else aum) / TRANCHES
    t = strategy.targets(book, day, aum)
    if t.empty:
        return 0
    with store.connect() as con:
        for r in t.itertuples():
            if r.weight > 0 or r.note:
                con.execute("INSERT OR REPLACE INTO signals VALUES (?,?,?,?,?,?,?,?)",
                            (book, day, r.codigo, float(r.weight), r.spread_bps,
                             None if pd.isna(r.eqh) else float(r.eqh),
                             None if pd.isna(r.quality) else float(r.quality), r.note))
        buys = t[t["weight"] > 0]
        for r in buys.itertuples():
            con.execute("INSERT INTO orders (book, tranche, codigo, side, value, created, expires, status) "
                        "VALUES (?,?,?,?,?,?,?,?)",
                        (book, tranche, r.codigo, "buy", size * r.weight, day, _add_bdays(day, EXPIRE_DAYS), "open"))
    log.info("paper %s: tranche %s with %d bonds", book, tranche, len(buys))
    return len(buys)


def _new_month(book: str, day: str) -> bool:
    last = store.rows("SELECT max(created) d FROM orders WHERE book = ? AND side = 'buy'", (book,))[0]["d"]
    return last is None or last[:7] < day[:7]


def step(until: str | None = None) -> dict:
    """Advance every book day by day up to the latest collected ANBIMA day (idempotent)."""
    days_all = [r["date"] for r in store.rows("SELECT DISTINCT date FROM snap_bonds ORDER BY date")]
    if not days_all:
        return {"error": "no data collected yet"}
    until = until or days_all[-1]
    _ensure_books(days_all[-1])
    out = {}
    for b in store.rows("SELECT * FROM books"):
        book, last = b["name"], b["last_day"]
        if last is None:  # book starts at the latest collected day: open the first tranche, nothing to mark yet
            open_tranche(book, b["start"])
            with store.connect() as con:
                con.execute("UPDATE books SET last_day = ? WHERE name = ?", (b["start"], book))
                con.execute("INSERT OR REPLACE INTO nav VALUES (?,?,?,?,?,?)", (book, b["start"], b["cash"], b["cash"], 0, 0))
            out[book] = {"started": b["start"]}
            continue
        mkt = Market(last, until)
        cash = b["cash"]
        prev = last
        for day in mkt.days:
            cash *= mkt.cdi(prev, day)
            with store.connect() as con:
                # 1) mark open positions: accrual x ratio change (ratio carried forward when ANBIMA has no price)
                for p in con.execute("SELECT * FROM positions WHERE book = ? AND status IN ('open', 'selling')",
                                     (book,)).fetchall():
                    r1 = mkt.mark(p["codigo"], day) or p["ratio"]
                    val = p["value"] * mkt.gpar(p["codigo"], prev, day) * r1 / p["ratio"]
                    con.execute("UPDATE positions SET value = ?, ratio = ? WHERE id = ?", (val, r1, p["id"]))
                # 2) fills of open orders at the first SND trade after the decision day
                for o in con.execute("SELECT * FROM orders WHERE book = ? AND status = 'open' AND created < ?",
                                     (book, day)).fetchall():
                    px, mk = mkt.trade(o["codigo"], day), mkt.mark(o["codigo"], day)
                    forced = px is None and day >= o["expires"] and o["side"] == "sell"
                    if px is None and not forced:
                        if day >= o["expires"]:
                            con.execute("UPDATE orders SET status = 'expired' WHERE id = ?", (o["id"],))
                        continue
                    if forced:
                        px = (mk or 0) * (1 - FORCED_HAIRCUT_BPS / 1e4) if mk else None
                        if px is None:
                            continue
                    slip = (px / mk - 1) * 1e4 if mk else None
                    if o["side"] == "buy":
                        value = min(o["value"], cash)
                        cash -= value
                        # entered at the trade's ratio; tonight's mark moves it to ANBIMA's (that gap = slippage)
                        val_now = value * (mk / px if mk else 1.0)
                        con.execute("INSERT INTO positions (book, tranche, codigo, value, ratio, entry_date, exit_due, "
                                    "status) VALUES (?,?,?,?,?,?,?, 'open')",
                                    (book, o["tranche"], o["codigo"], val_now, mk or px, day,
                                     _add_bdays(o["created"], HOLD_DAYS)))
                    else:
                        pos = con.execute("SELECT * FROM positions WHERE id = ?", (int(o["note"]),)).fetchone()
                        proceeds = pos["value"] * px / (mk or pos["ratio"])
                        cash += proceeds
                        con.execute("UPDATE positions SET status = 'closed', exit_date = ?, value = ? WHERE id = ?",
                                    (day, proceeds, pos["id"]))
                    con.execute("UPDATE orders SET status = ?, fill_date = ?, fill_ratio = ?, mark_ratio = ?, "
                                "slippage_bps = ? WHERE id = ?",
                                ("forced" if forced else "filled", day, px, mk, slip, o["id"]))
                # 3) positions due to leave: sell order (fills from tomorrow)
                for p in con.execute("SELECT * FROM positions WHERE book = ? AND status = 'open' AND exit_due <= ?",
                                     (book, day)).fetchall():
                    con.execute("UPDATE positions SET status = 'selling' WHERE id = ?", (p["id"],))
                    con.execute("INSERT INTO orders (book, tranche, codigo, side, value, created, expires, status, note) "
                                "VALUES (?,?,?,?,?,?,?,?,?)", (book, p["tranche"], p["codigo"], "sell", p["value"], day,
                                                               _add_bdays(day, EXPIRE_DAYS), "open", str(p["id"])))
                inv = con.execute("SELECT coalesce(sum(value), 0) v, count(*) n FROM positions WHERE book = ? "
                                  "AND status IN ('open', 'selling')", (book,)).fetchone()
                con.execute("INSERT OR REPLACE INTO nav VALUES (?,?,?,?,?,?)",
                            (book, day, cash + inv["v"], cash, inv["v"], inv["n"]))
                con.execute("UPDATE books SET last_day = ?, cash = ? WHERE name = ?", (day, cash, book))
            prev = day
            # 4) monthly tranche, decided at this close
            if _new_month(book, day):
                open_tranche(book, day)
        out[book] = {"to": prev, "days": len(mkt.days)}
    return out


def benchmarks() -> pd.DataFrame:
    """Daily index levels since the first book day: CDI and the equal-weight ANBIMA universe (same accrual maths)."""
    start = store.rows("SELECT min(start) s FROM books")[0]["s"]
    if not start:
        return pd.DataFrame()
    mkt = Market(start, "9999-12-31")
    days = [start] + mkt.days
    cdi, uni = [1.0], [1.0]
    for a, b in zip(days[:-1], days[1:]):
        cdi.append(cdi[-1] * mkt.cdi(a, b))
        r = []
        for c in mkt.ratio.columns:
            r0, r1 = mkt.mark(c, a), mkt.mark(c, b)
            if r0 and r1 and c in mkt.ref.index and mkt.ref.at[c, "kind"] in strategy.KINDS:
                r.append(mkt.gpar(c, a, b) * r1 / r0)
        uni.append(uni[-1] * (float(np.mean(r)) if r else mkt.cdi(a, b)))
    return pd.DataFrame({"CDI": cdi, "Universo": uni}, index=days)


def reset() -> None:
    with store.connect() as con:
        for t in ("books", "signals", "orders", "positions", "nav"):
            con.execute(f"DELETE FROM {t}")


def run_daily(force: bool = False) -> dict:
    from . import collect
    from .. import screener
    try:
        screener.ensure_fresh()  # stores today's ANBIMA snapshot if the app's cycle hasn't yet
    except Exception:
        log.exception("screener refresh failed")
    c = collect.run(force)
    s = step()
    return {"collect": c, "step": s}
