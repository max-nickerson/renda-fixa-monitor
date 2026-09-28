"""Scheduled work.

- quick cycle (every QUOTES_EVERY_SECONDS, market hours only): live issuer stock quotes + eurobond prices → alerts
- full cycle (every COLLECT_EVERY_MINUTES): ANBIMA/Tesouro prices, spreads, CVM filings, news → alerts → e-mail
"""
from __future__ import annotations

import logging
import threading

from . import alerts, db, isin as isin_mod, screener
from .ml import live
from .collect import collect_all, history_ready, refresh_quotes
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
        live.regime()            # credit-timing model, recomputed once a day
        if settings.enable_ml_selection and history_ready():
            live.selection()     # bond-selection model (SND trades), retrained once a day
        res = screener.ensure_fresh()  # re-rank the universe when ANBIMA publishes a new day
        # Negative-press check (P4) for C3 candidates + holdings, once a day; re-rank to apply it.
        if res and "rows" in res:
            codes = [r["codigo"] for r in res["rows"] if r.get("c3")]
            codes += [(db.get_asset(i) or {}).get("cetip_code", "").strip() for i in _entries()]
            before = len(live.cached_press())
            live.press_now([c for c in codes if c])
            if len(live.cached_press()) != before:
                screener.run()
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


def bootstrap_history() -> None:
    """First boot on a server: download SND trades / B3 curves / IDA (≈30–60 min, throttled) so trade-based
    spread history and the selection model work. The dashboard is usable meanwhile."""
    import runpy
    import sys
    from .config import ROOT
    log.info("bootstrap: downloading research history in the background")
    try:
        sys.argv = ["download_history.py", "2021"]
        runpy.run_path(str(ROOT / "research" / "download_history.py"), run_name="__main__")
    except SystemExit:
        pass
    except Exception:
        log.exception("bootstrap history failed")
        return
    log.info("bootstrap: history ready")
    if settings.enable_ml_selection:
        live.selection(refresh=True)
    run_cycle()


def start_scheduler():
    from apscheduler.schedulers.background import BackgroundScheduler

    sched = BackgroundScheduler(daemon=True)
    sched.add_job(run_cycle, "interval", minutes=settings.collect_every_minutes, id="cycle",
                  max_instances=1, coalesce=True)
    sched.add_job(run_quick, "interval", seconds=settings.quotes_every_seconds, id="quick",
                  max_instances=1, coalesce=True)
    sched.start()
    threading.Thread(target=run_cycle, daemon=True).start()  # first run right away
    if settings.bootstrap_history and not history_ready():
        threading.Thread(target=bootstrap_history, daemon=True).start()
    return sched
