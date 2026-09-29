"""Daily data collection for paper trading. Everything is stored with the date it was known, so the paper
books (and future research) only ever see point-in-time data."""
from __future__ import annotations

import json
import logging
from datetime import date, datetime, timedelta, timezone

import pandas as pd

from .. import db, history as h
from ..config import ROOT
from . import store

log = logging.getLogger(__name__)


def equity_map() -> pd.DataFrame:
    """cnpj8 → candidate tickers (issuer first, then parent), best confidence first."""
    p = ROOT / "research" / "data" / "equity_map.csv"
    if not p.exists():
        return pd.DataFrame(columns=["cnpj8", "ticker", "mapping_type", "rank"])
    m = pd.read_csv(p, dtype=str)
    m = m[m["ticker"].notna() & (m["mapping_type"] != "none")].copy()
    m["cnpj8"] = m["cnpj8"].str.zfill(8)
    m["ticker"] = m["ticker"].str.upper()
    m["rank"] = m["confidence"].map({"high": 0, "med": 1, "low": 2}).fillna(1)
    return m.sort_values(["cnpj8", "rank"])[["cnpj8", "ticker", "mapping_type", "rank"]]


def bonds() -> int:
    """Copy every stored daily screener payload (ANBIMA prices + model fields) that is not in snap_bonds yet."""
    from ..ml.selection import reference
    with db.connect() as con:
        payloads = con.execute("SELECT date, payload FROM screener ORDER BY date").fetchall()
    have = {r["date"] for r in store.rows("SELECT DISTINCT date FROM snap_bonds")}
    todo = [(d, p) for d, p in payloads if d not in have]
    if not todo:
        return 0
    ref = reference().drop_duplicates("codigo").set_index("codigo")
    ref["cnpj8"] = ref["cnpj"].str.replace(r"\D", "", regex=True).str.zfill(14).str[:8]
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    n = 0
    with store.connect() as con:
        for d, payload in todo:
            for r in json.loads(payload).get("rows", []):
                c = r.get("codigo")
                rf = ref.loc[c] if c in ref.index else None
                ratio = r.get("pct_pu_par")
                con.execute(
                    "INSERT OR IGNORE INTO snap_bonds VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (d, c, r.get("isin"), None if rf is None else rf["cnpj8"], r.get("nome"),
                     None if rf is None else rf["kind"], None if rf is None else float(rf["contract"]),
                     r.get("pu"), ratio / 100 if ratio else None, r.get("taxa_indicativa"), r.get("spread_bps"),
                     r.get("duration"), r.get("resid_z"), r.get("cdi_pct"), r.get("press_neg_30d"),
                     r.get("strategy"), store.dumps(r.get("flags") or []), now))
                n += 1
    store.log("bonds", n, f"{len(todo)} day(s)")
    return n


def trades(days: int = 120) -> int:
    """SND secondary trades (average PU and % of the par curve per bond per day)."""
    t = h.snd_trades(date.today() - timedelta(days=days))
    if t.empty:
        return 0
    t = t.dropna(subset=["pct_curve"])
    with store.connect() as con:
        con.executemany("INSERT OR REPLACE INTO snd_trades VALUES (?,?,?,?,?,?)",
                        [(d.strftime("%Y-%m-%d"), c, q, n, pu, pc) for d, c, q, n, pu, pc in
                         t[["date", "codigo", "qty", "trades", "pu_avg", "pct_curve"]].itertuples(index=False)])
    store.log("snd_trades", len(t))
    return len(t)


def equity() -> int:
    """Issuer / parent stock closes: research history once, then brapi daily."""
    tick = sorted(equity_map()["ticker"].unique())
    if not tick:
        return 0
    n = 0
    if not store.rows("SELECT 1 FROM eq_close LIMIT 1"):
        p = h.HIST / "equity_daily.pkl"
        if p.exists():
            e = pd.read_pickle(p)
            e = e[e["ticker"].isin(tick) & (e["date"] >= pd.Timestamp.today() - pd.Timedelta(days=420))]
            with store.connect() as con:
                con.executemany("INSERT OR IGNORE INTO eq_close VALUES (?,?,?,?)",
                                [(d.strftime("%Y-%m-%d"), t, float(c), "research")
                                 for t, d, c in e[["ticker", "date", "close"]].itertuples(index=False)])
            n += len(e)
    from ..sources import brapi
    last = {r["ticker"]: r["d"] for r in store.rows("SELECT ticker, max(date) d FROM eq_close GROUP BY ticker")}
    stale = [t for t in tick if last.get(t, "") < (date.today() - timedelta(days=10)).isoformat()]
    rows = []
    for t in stale[:60]:  # gaps (new tickers, long downtime): 3 months of history, a few per run
        try:
            rows += [(d, t, c, "brapi") for d, c, _ in brapi.history(t, "3mo")]
        except Exception as e:
            log.debug("brapi history %s: %s", t, e)
    try:
        for t, q in brapi.quotes(tick).items():
            ts = q.get("time")
            d = pd.Timestamp(ts).tz_convert("America/Sao_Paulo").date().isoformat() if ts else date.today().isoformat()
            rows.append((d, t, q["price"], "brapi"))
    except Exception as e:
        log.warning("brapi quotes: %s", e)
    with store.connect() as con:
        con.executemany("INSERT OR REPLACE INTO eq_close VALUES (?,?,?,?)", rows)
    n += len(rows)
    store.log("eq_close", n)
    return n


def rates() -> int:
    """CDI daily rate and a daily IPCA accrual factor (monthly IPCA spread over business days)."""
    from ..ml.selection import _accruals
    C, I = _accruals(date(2025, 1, 1))
    cdi = C.pct_change().dropna()
    out = pd.DataFrame({"cdi": cdi, "ipca": I.reindex(cdi.index)}).dropna()
    with store.connect() as con:
        con.executemany("INSERT OR REPLACE INTO rates VALUES (?,?,?)",
                        [(d.strftime("%Y-%m-%d"), float(a), float(b)) for d, a, b in out.itertuples()])
    store.log("rates", len(out))
    return len(out)


def _ran_within(what: str, hours: float) -> bool:
    r = store.rows("SELECT max(ts) t FROM collect_log WHERE what = ?", (what,))[0]["t"]
    return bool(r) and r >= (datetime.now(timezone.utc) - timedelta(hours=hours)).strftime("%Y-%m-%d %H:%M:%S")


def run(force: bool = False) -> dict:
    """Hourly: new ANBIMA days every run; SND trades, stock closes and rates at most every 3 hours."""
    out = {}
    for name, fn, what in (("bonds", bonds, None), ("trades", trades, "snd_trades"), ("equity", equity, "eq_close"),
                           ("rates", rates, "rates")):
        if what and not force and _ran_within(what, 3):
            out[name] = "recent"
            continue
        try:
            out[name] = fn()
        except Exception as e:
            log.exception("paper collect %s failed", name)
            out[name] = f"error: {e}"
    return out


def status() -> list[dict]:
    q = [("Preços ANBIMA + sinais (snap_bonds)", "snap_bonds"), ("Negócios SND (snd_trades)", "snd_trades"),
         ("Ações emissor/controladora (eq_close)", "eq_close"), ("CDI / IPCA (rates)", "rates")]
    out = []
    for label, t in q:
        r = store.rows(f"SELECT count(*) n, count(DISTINCT date) d, min(date) a, max(date) b FROM {t}")[0]
        out.append({"label": label, "table": t, **r})
    return out
