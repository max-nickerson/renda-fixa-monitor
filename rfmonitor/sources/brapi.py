"""brapi.dev: B3 stock quotes/history, fundamentals, SELIC and IPCA."""
from __future__ import annotations

from datetime import datetime, timezone

from ..config import settings
from ..http import get

BASE = "https://brapi.dev/api"


def _get(path: str, **params) -> dict:
    headers = {"Authorization": f"Bearer {settings.brapi_token}"} if settings.brapi_token else {}
    r = get(f"{BASE}{path}", params={k: v for k, v in params.items() if v is not None}, headers=headers)
    r.raise_for_status()
    return r.json()


def search(query: str, limit: int = 10) -> list[dict]:
    return _get("/quote/list", search=query, limit=limit, type="stock").get("stocks", [])


def best_ticker(query: str, prefix: str | None = None) -> str | None:
    """Most liquid B3 stock matching an issuer code/name (e.g. 'BRKM' -> 'BRKM5')."""
    try:
        cands = search(query, 20)
    except Exception:
        return None
    if prefix:
        cands = [c for c in cands if c["stock"].startswith(prefix)]
    cands = [c for c in cands if not c["stock"].endswith("F")]  # skip fractional market
    if not cands:
        return None
    return max(cands, key=lambda c: c.get("volume") or 0)["stock"]


def quote(ticker: str, range_: str | None = None, interval: str = "1d", modules: str | None = None) -> dict:
    res = _get(f"/quote/{ticker}", range=range_, interval=interval if range_ else None, modules=modules)
    return (res.get("results") or [{}])[0]


def quotes(tickers: list[str]) -> dict[str, dict]:
    """Latest quotes for many tickers in few calls: {ticker: {price, time, change_pct}}."""
    out: dict[str, dict] = {}
    for i in range(0, len(tickers), 20):
        chunk = tickers[i:i + 20]
        for r in _get(f"/quote/{','.join(chunk)}").get("results") or []:
            if r.get("regularMarketPrice") is not None:
                out[r["symbol"]] = {"price": r["regularMarketPrice"], "time": r.get("regularMarketTime"),
                                    "change_pct": r.get("regularMarketChangePercent")}
    return out


def history(ticker: str, range_: str = "1y") -> list[tuple[str, float, float | None]]:
    """[(YYYY-MM-DD, close, volume)]"""
    q = quote(ticker, range_=range_)
    out = []
    for bar in q.get("historicalDataPrice") or []:
        if bar.get("close") is None:
            continue
        d = datetime.fromtimestamp(bar["date"], tz=timezone.utc).strftime("%Y-%m-%d")
        out.append((d, bar["close"], bar.get("volume")))
    return out


def fundamentals(ticker: str) -> dict:
    """Snapshot of credit-relevant figures from brapi's financialData module."""
    q = quote(ticker, modules="financialData,defaultKeyStatistics")
    fd = q.get("financialData") or {}
    ks = q.get("defaultKeyStatistics") or {}
    debt, cash, ebitda = fd.get("totalDebt"), fd.get("totalCash"), fd.get("ebitda")
    net_lev = (debt - (cash or 0)) / ebitda if debt is not None and ebitda and ebitda > 0 else None
    return {
        "total_debt": debt, "total_cash": cash, "ebitda": ebitda, "net_debt_ebitda": net_lev,
        "debt_to_equity": fd.get("debtToEquity"), "current_ratio": fd.get("currentRatio"),
        "operating_cashflow": fd.get("operatingCashflow"), "free_cashflow": fd.get("freeCashflow"),
        "profit_margins": fd.get("profitMargins"), "enterprise_value": ks.get("enterpriseValue"),
        "market_cap": q.get("marketCap"),
    }


def usdbrl() -> float | None:
    try:
        res = _get("/v2/currency", currency="USD-BRL")
        c = (res.get("currency") or [{}])[0]
        return (float(c["bidPrice"]) + float(c["askPrice"])) / 2
    except Exception:
        return None


def selic_series(historical: bool = True) -> list[tuple[str, float]]:
    res = _get("/v2/prime-rate", country="brazil", historical=str(historical).lower())
    return [(datetime.strptime(x["date"], "%d/%m/%Y").strftime("%Y-%m-%d"), x["value"])
            for x in res.get("prime-rate", [])]


def ipca_series() -> list[tuple[str, float]]:
    res = _get("/v2/inflation", country="brazil", historical="true")
    return [(datetime.strptime(x["date"], "%d/%m/%Y").strftime("%Y-%m-%d"), x["value"])
            for x in res.get("inflation", [])]
