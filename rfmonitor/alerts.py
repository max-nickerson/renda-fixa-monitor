"""Alert rules evaluated after every collection run, plus e-mail dispatch."""
from __future__ import annotations

import logging
import smtplib
from email.message import EmailMessage
from html import escape

import pandas as pd

from . import db, strategy
from .config import alert_config, settings

log = logging.getLogger(__name__)


def _last_change(s: pd.Series, periods: int = 1) -> tuple[pd.Timestamp, float, float] | None:
    s = s.dropna()
    if len(s) <= periods:
        return None
    return s.index[-1], float(s.iloc[-1]), float(s.iloc[-1] - s.iloc[-1 - periods])


def evaluate(info: dict, new_events: list[dict], entry: dict) -> list[dict]:
    cfg = alert_config(entry)
    isin, name = info["isin"], info.get("name") or info["isin"]
    df = db.series(isin)
    fired: list[dict] = []

    def fire(rule: str, sev: str, msg: str, key_suffix: str) -> None:
        if db.put_alert(isin, rule, sev, f"{name}: {msg}", f"{isin}:{rule}:{key_suffix}"):
            fired.append({"rule": rule, "severity": sev, "message": f"{name}: {msg}"})

    if not df.empty:
        if "price" in df and (c := _last_change(df["price"])):
            d, v, ch = c
            prev = v - ch
            pct = ch / prev * 100 if prev else 0
            if abs(pct) >= cfg["price_move_pct"]:
                fire("price_move", "high" if pct < 0 else "info", f"price {pct:+.2f}% to {v:,.4f}", d.date())
        if "yield" in df and (c := _last_change(df["yield"])):
            d, v, ch = c
            if abs(ch * 100) >= cfg["yield_move_bps"]:
                fire("yield_move", "high" if ch > 0 else "info", f"yield {ch * 100:+.0f} bps to {v:.2f}%", d.date())
        if "spread_bps" in df:
            s = df["spread_bps"]
            if c := _last_change(s):
                d, v, ch = c
                if abs(ch) >= cfg["spread_move_bps"]:
                    fire("spread_move", "high" if ch > 0 else "info", f"spread {ch:+.0f} bps (1d) to {v:.0f} bps", d.date())
            if c := _last_change(s, 5):
                d, v, ch = c
                if abs(ch) >= cfg["spread_move_5d_bps"]:
                    fire("spread_move_5d", "high" if ch > 0 else "info", f"spread {ch:+.0f} bps (5d) to {v:.0f} bps", d.date())
            z = strategy._z(s.dropna()).dropna()
            if not z.empty and abs(z.iloc[-1]) >= cfg["spread_zscore"]:
                fire("spread_zscore", "info", f"spread z-score {z.iloc[-1]:+.1f} (vs {strategy.WINDOW}d)", z.index[-1].date())
        if "stock_close" in df and (c := _last_change(df["stock_close"])):
            d, v, ch = c
            pct = ch / (v - ch) * 100 if v - ch else 0
            if abs(pct) >= cfg["stock_move_pct"]:
                fire("stock_move", "high" if pct < 0 else "info",
                     f"stock {info.get('stock_ticker')} {pct:+.1f}% to R$ {v:.2f}", d.date())
            px = df["stock_close"].dropna()
            if len(px) > 20:
                dd = (px.iloc[-1] / px.tail(20).max() - 1) * 100
                if dd <= -cfg["stock_drawdown_20d_pct"]:
                    fire("stock_drawdown", "high", f"stock {dd:.0f}% below 20d high", px.index[-1].date())

    recent_cutoff = (pd.Timestamp.now() - pd.Timedelta(days=3)).strftime("%Y-%m-%d")
    for ev in new_events:
        if (ev.get("date") or ev.get("ts") or "")[:10] < recent_cutoff:
            continue  # backfilled history: stored for the dashboard, but no alert
        k = ev["kind"]
        if k in ("fundamentals", "material_fact", "filing") and cfg["filings"]:
            label = {"fundamentals": "Fundamentals published", "material_fact": "Material fact",
                     "filing": "CVM filing"}[k]
            fire(k, ev.get("severity", "info"), f"{label}: {ev['title']}", ev["uid"])
        elif k == "news" and cfg["news"]:
            if cfg["news"] == "all" or ev.get("severity") == "high":
                fire("news", ev.get("severity", "info"), f"News: {ev['title']}", ev["uid"])

    if cfg["signal_change"]:
        sig = strategy.current(isin)
        prev = info.get("last_signal")
        if sig.label != "N/A" and prev and prev != sig.label:
            fire("signal_change", "high", f"strategy signal {prev} → {sig.label} (score {sig.score})", sig.asof)
        if sig.label != "N/A":
            info["last_signal"] = sig.label
            db.save_asset(info)
    return fired


def send_pending_email() -> int:
    if not settings.email_enabled:
        return 0
    pending = db.alerts(unsent_only=True, limit=500)
    if not pending:
        return 0
    high = [a for a in pending if a["severity"] == "high"]
    msg = EmailMessage()
    msg["Subject"] = f"[Renda Fixa] {len(pending)} alertas" + (f" ({len(high)} importantes)" if high else "")
    msg["From"] = settings.smtp_user
    msg["To"] = settings.alert_email_to
    rows = "".join(
        f"<tr><td>{escape(a['ts'][:16])}</td><td>{'🔴' if a['severity'] == 'high' else '🔵'}</td>"
        f"<td>{escape(a['rule'])}</td><td>{escape(a['message'])}</td></tr>" for a in pending)
    msg.set_content("\n".join(f"{a['ts'][:16]} [{a['severity']}] {a['message']}" for a in pending))
    msg.add_alternative(
        f"<h3>Alertas de renda fixa</h3><table cellpadding=4 border=1 style='border-collapse:collapse'>"
        f"<tr><th>Quando</th><th></th><th>Regra</th><th>Mensagem</th></tr>{rows}</table>"
        f"<p>Dashboard: http://{settings.host}:{settings.port}/</p>", subtype="html")
    try:
        with smtplib.SMTP(settings.smtp_host, settings.smtp_port, timeout=30) as s:
            s.starttls()
            s.login(settings.smtp_user, settings.smtp_password)
            s.send_message(msg)
    except Exception as e:
        log.error("E-mail failed: %s", e)
        return 0
    db.mark_emailed([a["id"] for a in pending])
    return len(pending)
