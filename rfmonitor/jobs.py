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


_IBKR_LOCK = threading.Lock()


def _ibkr_forward(mark_only: bool = False):
    """Forward validation of the IBKR-lab strategies on the PAPER login (research/ibkr_lab/runner).
    One run at a time: a probe that finds the rebalance (or another probe) still running is skipped."""
    if not _IBKR_LOCK.acquire(blocking=not mark_only):
        return
    try:
        _ibkr_forward_run(mark_only)
    finally:
        _IBKR_LOCK.release()


def _ibkr_forward_run(mark_only: bool):
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


def _tv_store_update(retry: bool = False):
    """Append the missing daily bars to the TradingView store (data/tv_lab) from IBKR paper historical data, for the
    series the tv_core2_* plugins read (research/tv_lab/update_ibkr.py). Runs in a subprocess so a stale Gateway
    socket dies with it; the retry only runs when the morning run did not complete (resumes where it stopped)."""
    import os
    import subprocess
    import sys
    from .config import ROOT
    args = [sys.executable, "-m", "research.tv_lab.update_ibkr", "--client-id", "141"]
    if retry:
        args.append("--if-incomplete")
    env = {**os.environ, "PYTHONPATH": str(ROOT), "PYTHONIOENCODING": "utf-8"}
    try:
        r = subprocess.run(args, cwd=str(ROOT), env=env, capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=(18 if retry else 90) * 60,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        log.info("tv store update%s: exit %s %s", " (retry)" if retry else "", r.returncode, r.stdout[-400:])
    except subprocess.TimeoutExpired:
        log.error("tv store update%s timed out (see data/tv_lab/update_ibkr.log)", " (retry)" if retry else "")
    except Exception:
        log.exception("tv store update failed")


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
    # TradingView store top-up from IBKR (tv_core2_* inputs) before the rebalance; retry at 10:15 if incomplete.
    sched.add_job(_tv_store_update, "cron", day_of_week="mon-fri", hour=8, minute=30, id="tv_store_update",
                  max_instances=1, coalesce=True, misfire_grace_time=3600)
    sched.add_job(_tv_store_update, "cron", day_of_week="mon-fri", hour=10, minute=15, id="tv_store_update_retry",
                  kwargs={"retry": True}, max_instances=1, coalesce=True, misfire_grace_time=600)
    # IBKR paper books: rebalance once a day shortly after the B3 open (signals use the previous close), then live
    # probes every 5 minutes while B3/CME/US are open: batched marks, intraday NAV, reconciliation with the account.
    sched.add_job(_ibkr_forward, "cron", day_of_week="mon-fri", hour=10, minute=35, id="ibkr_forward_day",
                  kwargs={"mark_only": False}, max_instances=1, coalesce=True)
    sched.add_job(_ibkr_forward, "cron", day_of_week="mon-fri", hour="10-17", minute="*/5", id="ibkr_forward_mark",
                  kwargs={"mark_only": True}, max_instances=1, coalesce=True, misfire_grace_time=120)
    sched.add_job(_paper, "interval", minutes=60, id="paper", max_instances=1, coalesce=True,
                  next_run_time=datetime.now() + timedelta(minutes=3))
    sched.start()
    threading.Thread(target=run_cycle, daemon=True).start()  # first run right away
    if settings.bootstrap_history and not history_ready():
        threading.Thread(target=bootstrap_history, daemon=True).start()
    return sched
