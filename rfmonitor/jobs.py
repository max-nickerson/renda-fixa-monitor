"""Scheduled work.

- quick cycle (every QUOTES_EVERY_SECONDS, market hours only): live issuer stock quotes + eurobond prices → alerts
- full cycle (every COLLECT_EVERY_MINUTES): ANBIMA/Tesouro prices, spreads, CVM filings, news → alerts → e-mail
"""
from __future__ import annotations

import logging
import threading
from datetime import datetime, timedelta

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
            before = (len(live.cached_press()), len(live.cached_risk()))
            live.press_now([c for c in codes if c])
            live.risk_now([c for c in codes if c])  # stock crash / rating downgrade flags (P5)
            if (len(live.cached_press()), len(live.cached_risk())) != before:
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


def _ativos():
    """Live quotes for the researched holdings list, every minute while B3 is open (and once on start)."""
    from .collect import b3_open
    from . import ativos
    if not ativos.FILE.exists() or (not b3_open() and ativos.LIVE.exists()):
        return
    try:
        ativos.refresh_live()
    except Exception:
        log.exception("ativos refresh failed")


def _ibkr_forward(mark_only: bool = False):
    """Forward validation of the IBKR-lab strategies on the PAPER login (research/ibkr_lab/runner)."""
    import asyncio
    import sys
    from .config import ROOT
    sys.path.insert(0, str(ROOT))
    try:
        asyncio.set_event_loop(asyncio.new_event_loop())
        from research.ibkr_lab.runner import forward
        if forward.plugins():
            log.info("ibkr forward: %s", forward.run_day(mark_only=mark_only))
    except Exception:
        log.exception("ibkr forward run failed (is IB Gateway logged into the paper account?)")


def _paper():
    """Paper trading: collect the day's data and advance the books (idempotent; cheap when nothing is new)."""
    try:
        from .paper import engine
        log.info("paper: %s", engine.run_daily())
    except Exception:
        log.exception("paper trading step failed")


def start_scheduler():
    from apscheduler.schedulers.background import BackgroundScheduler

    sched = BackgroundScheduler(daemon=True)
    sched.add_job(run_cycle, "interval", minutes=settings.collect_every_minutes, id="cycle",
                  max_instances=1, coalesce=True)
    sched.add_job(run_quick, "interval", seconds=settings.quotes_every_seconds, id="quick",
                  max_instances=1, coalesce=True)
    sched.add_job(_ativos, "interval", seconds=60, id="ativos", max_instances=1, coalesce=True,
                  next_run_time=datetime.now() + timedelta(seconds=20))
    sched.add_job(_ibkr_forward, "cron", day_of_week="mon-fri", hour=15, minute=40, id="ibkr_forward_day",
                  kwargs={"mark_only": False}, max_instances=1, coalesce=True)
    sched.add_job(_ibkr_forward, "cron", day_of_week="mon-fri", hour="10-16", minute=47, id="ibkr_forward_mark",
                  kwargs={"mark_only": True}, max_instances=1, coalesce=True)
    sched.add_job(_paper, "interval", minutes=60, id="paper", max_instances=1, coalesce=True,
                  next_run_time=datetime.now() + timedelta(minutes=3))
    sched.start()
    threading.Thread(target=run_cycle, daemon=True).start()  # first run right away
    if settings.bootstrap_history and not history_ready():
        threading.Thread(target=bootstrap_history, daemon=True).start()
    return sched
