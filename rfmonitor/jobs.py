"""Scheduled work.

- quick cycle (every QUOTES_EVERY_SECONDS, market hours only): live issuer stock quotes + eurobond prices → alerts
- full cycle (every COLLECT_EVERY_MINUTES): ANBIMA/Tesouro prices, spreads, CVM filings, news → alerts → e-mail
"""
from __future__ import annotations

import logging
import threading

from . import alerts, db, isin as isin_mod, screener
from .collect import collect_all, refresh_quotes
from .config import load_watchlist, settings

log = logging.getLogger(__name__)
_lock = threading.Lock()


def _entries() -> dict[str, dict]:
    out = {}
    for e in load_watchlist():
        try:
            out[isin_mod.normalize(e["isin"])] = e
        except ValueError:
            continue
    return out


def run_cycle(backfill_days: int | None = None) -> dict:
    if not _lock.acquire(blocking=False):
        return {"skipped": "a cycle is already running"}
    try:
        results = collect_all(backfill_days)
        entries = _entries()
        fired = []
        for r in results:
            if "error" in r:
                continue
            info = db.get_asset(r["isin"])
            if info:
                fired += alerts.evaluate(info, r["new_events"], entries.get(r["isin"], {}))
        emailed = alerts.send_pending_email()
        log.info("cycle: %d assets, %d alerts, %d e-mailed", len(results), len(fired), emailed)
        screener.ensure_fresh()  # re-rank the universe when ANBIMA publishes a new day
        return {"assets": len(results), "alerts": len(fired), "emailed": emailed,
                "errors": [r for r in results if "error" in r]}
    finally:
        _lock.release()


def run_quick() -> dict:
    if not _lock.acquire(blocking=False):
        return {"skipped": "busy"}
    try:
        updated = refresh_quotes()
        entries = _entries()
        fired = []
        for isin in updated:
            info = db.get_asset(isin)
            if info:
                fired += alerts.evaluate(info, [], entries.get(isin, {}))
        emailed = alerts.send_pending_email() if fired else 0
        if updated:
            log.info("quick: %d updated, %d alerts", len(updated), len(fired))
        return {"updated": updated, "alerts": len(fired), "emailed": emailed}
    finally:
        _lock.release()


def start_scheduler():
    from apscheduler.schedulers.background import BackgroundScheduler

    sched = BackgroundScheduler(daemon=True)
    sched.add_job(run_cycle, "interval", minutes=settings.collect_every_minutes, id="cycle",
                  max_instances=1, coalesce=True)
    sched.add_job(run_quick, "interval", seconds=settings.quotes_every_seconds, id="quick",
                  max_instances=1, coalesce=True)
    sched.start()
    threading.Thread(target=run_cycle, daemon=True).start()  # first run right away
    return sched
