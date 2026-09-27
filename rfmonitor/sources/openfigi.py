"""OpenFIGI: free ISIN -> name/ticker/coupon/maturity lookup (works for eurobonds and BR assets)."""
from __future__ import annotations

from ..bonds import parse_figi_ticker
from ..http import client


def lookup(isin: str) -> dict | None:
    r = client().post("https://api.openfigi.com/v3/mapping", json=[{"idType": "ID_ISIN", "idValue": isin}])
    if r.status_code != 200:
        return None
    data = (r.json() or [{}])[0].get("data") or []
    if not data:
        return None
    d = data[0]
    coupon, maturity = parse_figi_ticker(d.get("ticker", ""))
    return {
        "name": d.get("name"), "figi": d.get("figi"), "ticker": d.get("ticker"),
        "security_type": d.get("securityType"), "market_sector": d.get("marketSector"),
        "description": d.get("securityDescription"), "coupon": coupon, "maturity": maturity,
    }
