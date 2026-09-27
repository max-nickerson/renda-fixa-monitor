"""One full cycle (collect → alerts → e-mail) and the background scheduler."""
from __future__ import annotations

import logging
import threading

from . import alerts, db, isin as isin_mod
from .collect import collect_all
from .config import load_watchlist, settings

log = logging.getLogger(__name__)
_lock = threading.Lock()


def run_cycle(backfill_days: int | None = None) -> dict:
    if not _lock.acquire(blocking=False):
        return {"skipped": "a cycle is already running"}
    try:
        results = collect_all(backfill_days)
        entries = {isin_mod.normalize(e["isin"]): e for e in load_watchlist()}
        fired = []
        for r in results:
            if "error" in r:
                continue
            info = db.get_asset(r["isin"])
            if info:
                fired += alerts.evaluate(info, r["new_events"], entries.get(r["isin"], {}))
        emailed = alerts.send_pending_email()
        log.info("cycle: %d assets, %d alerts, %d e-mailed", len(results), len(fired), emailed)
        return {"assets": len(results), "alerts": len(fired), "emailed": emailed,
                "errors": [r for r in results if "error" in r]}
    finally:
        _lock.release()


def start_scheduler():
    from apscheduler.schedulers.background import BackgroundScheduler

    sched = BackgroundScheduler(daemon=True)
    sched.add_job(run_cycle, "interval", minutes=settings.collect_every_minutes, id="cycle",
                  max_instances=1, coalesce=True)
    sched.start()
    threading.Thread(target=run_cycle, daemon=True).start()  # first run right away
    return sched
