"""Collectors: turn raw source data into daily metrics + events per asset.

Metrics stored per ISIN (observations table):
  price        clean price (PU for BR assets, % of par for eurobonds)
  pct_par      price as % of par (debentures)
  yield        indicative rate / yield-to-maturity (%)
  spread_bps   credit spread vs benchmark (DI, NTN-B, pre curve or UST)
  duration     years
  stock_close  issuer stock close (BRL)
  f_*          fundamentals snapshot (net debt/EBITDA, debt/equity, ...)
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo

import pandas as pd

from . import db, isin as isin_mod
from .bonds import interp, mod_duration, ytm
from .config import load_watchlist, settings
from .resolver import resolve
from .sources import anbima_api, anbima_public, brapi, cvm, fred, news, tesouro_direto, tradingview_mcp

log = logging.getLogger(__name__)
MACRO = "_MACRO"
PUBLIC_DAYS = 12  # ANBIMA keeps only the last ~10 business days of public files


# ---------------------------------------------------------------- market context
class Market:
    """Loads shared daily market files once per collection run."""

    def __init__(self, days: int):
        self.days = days
        self._deb = self._tpf = None
        self._cdi: pd.Series | None = None

    def _combine(self, public_fetch, api_fetch, key: str) -> pd.DataFrame:
        """ANBIMA public files cover ~10 business days; older dates come from the ANBIMA API (if configured)."""
        recent = anbima_public.history(public_fetch, min(self.days, PUBLIC_DAYS))
        frames = [recent]
        if settings.anbima_api_enabled and self.days > PUBLIC_DAYS:
            have = set(recent["date"]) if not recent.empty else set()
            for d in anbima_public.business_days_back(self.days):
                if pd.Timestamp(d) in have:
                    continue
                try:
                    df = api_fetch(d)
                except Exception as e:
                    log.warning("ANBIMA API %s %s: %s", key, d, e)
                    break
                if df is not None and not df.empty:
                    frames.append(df)
        frames = [f for f in frames if f is not None and not f.empty]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    @property
    def deb(self) -> pd.DataFrame:
        if self._deb is None:
            self._deb = self._combine(anbima_public.debentures, anbima_api.debentures, "debentures")
        return self._deb

    @property
    def tpf(self) -> pd.DataFrame:
        """Federal bond curve. Tesouro Direto first (long, consistent history — avoids level jumps
        in spread z-scores), ANBIMA for dates Tesouro Direto has not published."""
        if self._tpf is None:
            start = pd.Timestamp(anbima_public.business_days_back(self.days)[0])
            try:
                td = tesouro_direto.curve_like_tpf(start)
            except Exception as e:
                log.warning("Tesouro Direto history failed: %s", e)
                td = pd.DataFrame()
            anb = self._combine(anbima_public.tpf, anbima_api.tpf, "tpf") if td.empty else \
                anbima_public.history(anbima_public.tpf, min(self.days, PUBLIC_DAYS))
            if not anb.empty and not td.empty:
                anb = anb[~anb["date"].isin(set(td["date"]))]
            frames = [f for f in (anb, td) if not f.empty]
            self._tpf = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        return self._tpf

    @property
    def cdi(self) -> pd.Series:
        """Daily CDI proxy (% a.a.) = SELIC target - 0.10, forward-filled."""
        if self._cdi is None:
            try:
                s = pd.Series(dict(brapi.selic_series(historical=True)), dtype=float)
                s.index = pd.to_datetime(s.index)
                self._cdi = (s.sort_index() - 0.10)
                db.put_observations(MACRO, [(d.strftime("%Y-%m-%d"), "cdi", v) for d, v in self._cdi.tail(400).items()],
                                    "brapi")
            except Exception as e:
                log.warning("SELIC fetch failed: %s", e)
                self._cdi = pd.Series(dtype=float)
        return self._cdi

    def cdi_at(self, d: pd.Timestamp) -> float | None:
        s = self.cdi
        if s.empty:
            return None
        s = s[s.index <= d]
        return float(s.iloc[-1]) if not s.empty else None

    def govt_rate(self, d: pd.Timestamp, titulos: tuple[str, ...], years: float,
                  exact_maturity: pd.Timestamp | None = None) -> float | None:
        t = self.tpf
        if t.empty:
            return None
        day = t[(t["date"] == d) & (t["titulo"].isin(titulos))]
        if day.empty:
            return None
        if exact_maturity is not None and not pd.isna(exact_maturity):
            hit = day[day["vencimento"] == exact_maturity]
            if not hit.empty and hit["taxa_indicativa"].notna().any():
                return float(hit["taxa_indicativa"].iloc[0])
        yrs = (day["vencimento"] - d).dt.days / 365.25
        return interp(years, yrs.tolist(), day["taxa_indicativa"].tolist())


# ---------------------------------------------------------------- bond metrics
def _debenture_rows(info: dict, mkt: Market, rows: pd.DataFrame) -> list[tuple]:
    out = []
    for _, r in rows.iterrows():
        d: pd.Timestamp = r["date"]
        ds = d.strftime("%Y-%m-%d")
        tx = r.get("taxa_indicativa")
        dur = r["duration_du"] / 252 if pd.notna(r.get("duration_du")) else None
        out += [(ds, "price", r.get("pu")), (ds, "pct_par", r.get("pct_pu_par")), (ds, "yield", tx),
                (ds, "duration", dur)]
        kind, _ = anbima_public.parse_indexer(r.get("indice") or info.get("index_text") or "")
        info["index"] = kind
        spread = None
        if tx is not None and tx == tx:
            yrs = dur or ((r["vencimento"] - d).days / 365.25 if pd.notna(r.get("vencimento")) else 3)
            if kind == "DI_SPREAD":
                spread = tx * 100
            elif kind == "DI_PCT":
                cdi = mkt.cdi_at(d)
                spread = (tx / 100 - 1) * cdi * 100 if cdi else None
            elif kind == "IPCA":
                ntnb = mkt.govt_rate(d, ("NTN-B",), yrs, r.get("ref_ntnb"))
                spread = (tx - ntnb) * 100 if ntnb is not None else None
            elif kind == "PRE":
                pre = mkt.govt_rate(d, ("LTN", "NTN-F"), yrs)
                spread = (tx - pre) * 100 if pre is not None else None
        out.append((ds, "spread_bps", spread))
    return out


def collect_debenture(info: dict, mkt: Market) -> int:
    code = info.get("cetip_code")
    if not code or mkt.deb.empty:
        return 0
    rows = mkt.deb[mkt.deb["codigo"] == code]
    if rows.empty:
        info.setdefault("notes", []).append(f"{code} not in ANBIMA daily file (illiquid or not priced)")
        return 0
    last = rows.iloc[-1]
    info["index_text"] = last.get("indice")
    if pd.isna(last.get("taxa_indicativa")):
        note = "ANBIMA publishes PU but no indicative rate (distressed/no consensus) — spread unavailable"
        if note not in info.setdefault("notes", []):
            info["notes"].append(note)
    return db.put_observations(info["isin"], _debenture_rows(info, mkt, rows), "anbima")


def collect_tesouro(info: dict, mkt: Market) -> int:
    t = mkt.tpf
    if t.empty or not info.get("tpf_titulo") or not info.get("maturity"):
        return 0
    rows = t[(t["titulo"] == info["tpf_titulo"]) & (t["vencimento"] == pd.Timestamp(info["maturity"]))]
    obs = []
    for _, r in rows.iterrows():
        ds = r["date"].strftime("%Y-%m-%d")
        obs += [(ds, "price", r["pu"]), (ds, "yield", r["taxa_indicativa"])]
        yrs = (r["vencimento"] - r["date"]).days / 365.25
        obs.append((ds, "duration", yrs))
    return db.put_observations(info["isin"], obs, "anbima")


def collect_cri_cra(info: dict, mkt: Market) -> int:
    if not settings.anbima_api_enabled or not info.get("cetip_code"):
        return 0
    try:
        df = anbima_api.cri_cra()
    except Exception as e:
        log.warning("ANBIMA CRI/CRA API failed: %s", e)
        return 0
    if df.empty or "codigo" not in df.columns:
        return 0
    rows = df[df["codigo"] == info["cetip_code"]]
    obs = []
    for _, r in rows.iterrows():
        ds = str(r.get("date"))[:10]
        tx = r.get("taxa_indicativa")
        obs += [(ds, "price", r.get("pu")), (ds, "yield", tx)]
        if "duration" in r and pd.notna(r["duration"]):
            obs.append((ds, "duration", r["duration"] / 252))
        idx = str(r.get("tipo_remuneracao") or r.get("indexador") or "").upper()
        if tx is not None and "DI" in idx and "%" not in idx:
            obs.append((ds, "spread_bps", tx * 100))
        elif tx is not None and "IPCA" in idx:
            yrs = (r["duration"] / 252) if "duration" in r and pd.notna(r["duration"]) else 3
            ntnb = mkt.govt_rate(pd.Timestamp(ds), ("NTN-B",), yrs)
            if ntnb is not None:
                obs.append((ds, "spread_bps", (tx - ntnb) * 100))
    return db.put_observations(info["isin"], obs, "anbima_api")


def collect_eurobond(info: dict) -> int:
    n = 0
    if settings.tradingview_mcp and tradingview_mcp.available():
        try:
            if not info.get("tv_symbol"):
                # Prefer FINRA/TRACE (real US trades) over German exchange quotes.
                hits = [h for h in tradingview_mcp.search(info["isin"][:11]) if h.get("type") == "bond"]
                hits.sort(key=lambda h: {"FINRA": 0, "FWB": 1}.get(h.get("exchange"), 2))
                if hits:
                    info["tv_symbol"] = hits[0]["symbol"]
            if not info.get("tv_symbol"):
                raise RuntimeError("bond not found on TradingView")
            bars = tradingview_mcp.ohlcv(info["tv_symbol"], "1D", 300)
            n += db.put_observations(info["isin"], [
                (datetime.fromtimestamp(b["t"], tz=timezone.utc).strftime("%Y-%m-%d"), "price", b["c"])
                for b in bars], "tradingview")
        except Exception as e:
            log.warning("TradingView MCP failed for %s: %s", info["isin"], e)
    # Derive yield / spread / duration from whatever prices we have (TradingView or manual input).
    prices = db.series(info["isin"], ["price"])
    if prices.empty or info.get("coupon") is None or not info.get("maturity"):
        return n
    mat = date.fromisoformat(info["maturity"])
    ust = fred.ust_curve()
    obs = []
    for d, p in prices["price"].dropna().items():
        y = ytm(p, info["coupon"], mat, d.date())
        dur = mod_duration(p, info["coupon"], mat, d.date())
        spread = None
        if y is not None and not ust.empty:
            row = ust[ust.index <= d].tail(1)
            if not row.empty:
                yrs = (mat - d.date()).days / 365.25
                bench = interp(yrs, list(row.columns), row.iloc[0].tolist())
                spread = (y - bench) * 100 if bench is not None else None
        ds = d.strftime("%Y-%m-%d")
        obs += [(ds, "yield", y), (ds, "duration", dur), (ds, "spread_bps", spread)]
    return n + db.put_observations(info["isin"], obs, "derived")


# ---------------------------------------------------------------- stock & fundamentals
def collect_stock(info: dict, first_run: bool) -> int:
    t = info.get("stock_ticker")
    if not t:
        return 0
    n = 0
    try:
        hist = brapi.history(t, "1y" if first_run else "5d")
        n += db.put_observations(info["isin"], [(d, "stock_close", c) for d, c, _ in hist], "brapi")
    except Exception as e:
        log.warning("brapi history failed for %s: %s", t, e)
    try:
        f = brapi.fundamentals(t)
        today = date.today().isoformat()
        n += db.put_observations(info["isin"], [(today, f"f_{k}", v) for k, v in f.items()], "brapi")
    except Exception as e:
        log.warning("brapi fundamentals failed for %s: %s", t, e)
    return n


# ---------------------------------------------------------------- events
def collect_events(info: dict, first_run: bool) -> list[dict]:
    new: list[dict] = []
    isin = info["isin"]
    cnpj = info.get("cnpj")
    if cnpj and info.get("kind") != "tesouro":
        years = [date.today().year] + ([date.today().year - 1] if first_run else [])
        for y in years:
            try:
                for f in cvm.financial_statements(cnpj, y):
                    if db.put_event(isin, f["date"], "fundamentals", f["title"], f["url"], "CVM", f["uid"], "high"):
                        new.append({**f, "kind": "fundamentals", "severity": "high"})
                for f in cvm.material_facts(cnpj, y):
                    sev = "high" if "relevante" in f["category"].lower() else "info"
                    kind = "material_fact" if sev == "high" else "filing"
                    if db.put_event(isin, f["date"], kind, f["title"], f["url"], "CVM", f["uid"], sev):
                        new.append({**f, "kind": kind, "severity": sev})
            except Exception as e:
                log.warning("CVM fetch failed (%s, %s): %s", cnpj, y, e)
    q = info.get("news_query")
    if q:
        for it in news.search(q, "30d" if first_run else "3d"):
            if db.put_event(isin, it["ts"], "news", it["title"], it["url"], it["source"], it["uid"], it["severity"]):
                new.append({**it, "kind": "news"})
    return new


# ---------------------------------------------------------------- intraday refresh
BRT = ZoneInfo("America/Sao_Paulo")
NY = ZoneInfo("America/New_York")


def b3_open(now: datetime | None = None) -> bool:
    n = (now or datetime.now(timezone.utc)).astimezone(BRT)
    return n.weekday() < 5 and (10, 0) <= (n.hour, n.minute) <= (18, 10)


def us_bond_hours(now: datetime | None = None) -> bool:
    n = (now or datetime.now(timezone.utc)).astimezone(NY)
    return n.weekday() < 5 and 8 <= n.hour < 17


def refresh_quotes(include_eurobonds: bool = True) -> list[str]:
    """Fast path: live issuer stock prices (one batched brapi call) and eurobond prices (TradingView).
    Today's value is overwritten on every run, so the dashboard and alerts see the latest print."""
    infos = []
    for e in load_watchlist():
        try:
            info = db.get_asset(isin_mod.normalize(e["isin"]))
        except ValueError:
            continue
        if info:
            infos.append(info)
    updated: list[str] = []
    tickers = sorted({i["stock_ticker"] for i in infos if i.get("stock_ticker")})
    if tickers and b3_open():
        try:
            q = brapi.quotes(tickers)
        except Exception as e:
            log.warning("brapi quotes failed: %s", e)
            q = {}
        for info in infos:
            hit = q.get(info.get("stock_ticker"))
            if not hit:
                continue
            d = (pd.Timestamp(hit["time"]).tz_convert(BRT) if hit.get("time") else pd.Timestamp.now(BRT))
            db.put_observations(info["isin"], [(d.strftime("%Y-%m-%d"), "stock_close", hit["price"])], "brapi_live")
            info["stock_last"] = {"price": hit["price"], "time": d.strftime("%d/%m %H:%M"),
                                  "change_pct": hit.get("change_pct")}
            db.save_asset(info)
            updated.append(info["isin"])
    if include_eurobonds and settings.tradingview_mcp and us_bond_hours():
        now = datetime.now(timezone.utc)
        for info in infos:
            last = info.get("tv_last_fetch")
            if info.get("kind") == "eurobond" and (not last or (now - datetime.fromisoformat(last)).total_seconds() >= 300):
                collect_eurobond(info)  # at most every 5 min — TradingView data is delayed ~15 min anyway
                info["tv_last_fetch"] = now.isoformat()
                db.save_asset(info)
                updated.append(info["isin"])
    return sorted(set(updated))


# ---------------------------------------------------------------- orchestration
def collect_asset(entry: dict, mkt: Market) -> dict:
    info = resolve(entry["isin"], entry)
    isin = info["isin"]
    first_run = db.series(isin).empty
    n = 0
    kind = info.get("kind")
    if kind == "debenture":
        n += collect_debenture(info, mkt)
    elif kind == "tesouro":
        n += collect_tesouro(info, mkt)
    elif kind == "cri_cra":
        n += collect_cri_cra(info, mkt)
    elif kind == "eurobond":
        n += collect_eurobond(info)
    n += collect_stock(info, first_run)
    new_events = collect_events(info, first_run)
    info["last_collected"] = db.now_iso()
    db.save_asset(info)
    return {"isin": isin, "observations": n, "new_events": new_events, "first_run": first_run}


def collect_all(backfill_days: int | None = None) -> list[dict]:
    entries = load_watchlist()
    if not entries:
        return []
    def is_new(e: dict) -> bool:
        try:
            return db.series(isin_mod.normalize(e["isin"])).empty
        except ValueError:
            return False

    need_backfill = backfill_days or (settings.backfill_days if any(is_new(e) for e in entries) else 10)
    mkt = Market(need_backfill)
    results = []
    for e in entries:
        try:
            results.append(collect_asset(e, mkt))
        except Exception as ex:
            log.exception("collect failed for %s", e.get("isin"))
            results.append({"isin": e.get("isin"), "error": str(ex)})
    return results


def add_manual_price(isin: str, d: str, price: float) -> None:
    db.put_observations(isin, [(d, "price", price)], "manual")
