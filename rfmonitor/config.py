"""Settings loaded from .env and the watchlist file."""
from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env", encoding="utf-8-sig")

DATA_DIR = Path(os.getenv("RFM_DATA_DIR", ROOT / "data"))
CACHE_DIR = DATA_DIR / "cache"
DB_PATH = DATA_DIR / "monitor.db"
WATCHLIST_PATH = Path(os.getenv("RFM_WATCHLIST", ROOT / "watchlist.yaml"))
WATCHLIST_EXAMPLE = ROOT / "watchlist.example.yaml"

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) renda-fixa-monitor/0.1"

# Alert thresholds; each can be overridden per asset under `alerts:` in watchlist.yaml.
DEFAULT_ALERTS = {
    "price_move_pct": 2.0,        # |daily price change| in %
    "yield_move_bps": 25,         # |daily yield change| in bps
    "spread_move_bps": 25,        # |daily spread change| in bps
    "spread_move_5d_bps": 50,     # |5-day spread change| in bps
    "spread_zscore": 2.0,         # |z-score of spread vs 120d history|
    "stock_move_pct": 5.0,        # |daily issuer stock change| in %
    "stock_drawdown_20d_pct": 15.0,
    "filings": True,              # CVM ITR/DFP (fundamentals) and material facts
    "news": "high",               # "high" = only negative-keyword news; "all"; or false. All news is on the dashboard.
    "signal_change": True,        # strategy signal flips BUY/HOLD/SELL
}

NEGATIVE_KEYWORDS = [
    "recuperação judicial", "recuperacao judicial", "falência", "default", "calote",
    "rebaix", "downgrade", "reestrutura", "vencimento antecipado", "covenant",
    "chapter 11", "waiver", "inadimpl", "fato relevante", "investigação", "multa",
]


def _env(name: str, default: str = "") -> str:
    return (os.getenv(name) or default).strip()


@dataclass
class Settings:
    brapi_token: str = field(default_factory=lambda: _env("BRAPI_TOKEN"))
    anbima_client_id: str = field(default_factory=lambda: _env("ANBIMA_CLIENT_ID"))
    anbima_client_secret: str = field(default_factory=lambda: _env("ANBIMA_CLIENT_SECRET"))
    anbima_env: str = field(default_factory=lambda: _env("ANBIMA_ENV", "production"))
    smtp_host: str = field(default_factory=lambda: _env("SMTP_HOST", "smtp.gmail.com"))
    smtp_port: int = field(default_factory=lambda: int(_env("SMTP_PORT", "587")))
    smtp_user: str = field(default_factory=lambda: _env("SMTP_USER"))
    smtp_password: str = field(default_factory=lambda: _env("SMTP_PASSWORD"))
    alert_email_to: str = field(default_factory=lambda: _env("ALERT_EMAIL_TO"))
    tradingview_mcp: bool = field(default_factory=lambda: _env("TRADINGVIEW_MCP", "false").lower() == "true")
    host: str = field(default_factory=lambda: _env("HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: int(_env("PORT", "8000")))
    collect_every_minutes: int = field(default_factory=lambda: int(_env("COLLECT_EVERY_MINUTES", "15")))
    quotes_every_seconds: int = field(default_factory=lambda: int(_env("QUOTES_EVERY_SECONDS", "60")))
    # Public hosting: viewing is open; changes need this key (empty = everything open, e.g. local use).
    admin_key: str = field(default_factory=lambda: _env("ADMIN_KEY"))
    # Bond-selection ML model needs the SND/B3 history (~1 GB, more RAM). P4, regime and alerts don't.
    enable_ml_selection: bool = field(default_factory=lambda: _env("ENABLE_ML_SELECTION", "true").lower() == "true")
    # First boot on a server: download the research history in the background if it isn't there yet.
    bootstrap_history: bool = field(default_factory=lambda: _env("BOOTSTRAP_HISTORY", "false").lower() == "true")
    backfill_days: int = field(default_factory=lambda: int(_env("BACKFILL_DAYS", "180")))

    @property
    def email_enabled(self) -> bool:
        return bool(self.smtp_user and self.smtp_password and self.alert_email_to)

    @property
    def anbima_api_enabled(self) -> bool:
        return bool(self.anbima_client_id and self.anbima_client_secret)


settings = Settings()


def ensure_dirs() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if not WATCHLIST_PATH.exists() and WATCHLIST_EXAMPLE.exists():
        shutil.copy(WATCHLIST_EXAMPLE, WATCHLIST_PATH)


def load_watchlist() -> list[dict]:
    ensure_dirs()
    if not WATCHLIST_PATH.exists():
        return []
    doc = yaml.safe_load(WATCHLIST_PATH.read_text(encoding="utf-8")) or {}
    return [a for a in (doc.get("assets") or []) if a and a.get("isin")]


def save_watchlist(assets: list[dict]) -> None:
    ensure_dirs()
    WATCHLIST_PATH.write_text(
        yaml.safe_dump({"assets": assets}, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )


def alert_config(entry: dict) -> dict:
    return {**DEFAULT_ALERTS, **(entry.get("alerts") or {})}
