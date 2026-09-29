"""Resolve any fixed-income ISIN into everything the collectors need.

ISIN -> kind (debenture / cri_cra / tesouro / eurobond), issuer, CNPJ, CVM code, issuer stock ticker,
CETIP code, maturity, coupon, indexer, TradingView symbol and news query.
Anything can be overridden per asset in watchlist.yaml.
"""
from __future__ import annotations

import logging
from datetime import date, timedelta

import pandas as pd

from . import db, isin as isin_mod
from .sources import brapi, cvm, openfigi, snd

log = logging.getLogger(__name__)

TPF_FROM_FIGI = {"BNTNB": "NTN-B", "BLTN": "LTN", "BNTNF": "NTN-F", "BLFT": "LFT", "BNTNC": "NTN-C"}
OVERRIDABLE = {"name", "issuer", "cnpj", "cvm_code", "stock_ticker", "cetip_code", "maturity", "coupon",
               "index", "index_rate", "kind", "tv_symbol", "news_query", "tpf_titulo", "currency", "notes"}


def _iso(d) -> str | None:
    if d is None or (isinstance(d, float) and d != d) or d is pd.NaT:
        return None
    return d.date().isoformat() if isinstance(d, pd.Timestamp) else d.isoformat() if isinstance(d, date) else str(d)


def resolve(raw_isin: str, overrides: dict | None = None, refresh: bool = False) -> dict:
    overrides = overrides or {}
    code = isin_mod.normalize(raw_isin)
    cached = db.get_asset(code)
    fresh = cached and cached.get("resolved_at", "") >= (date.today() - timedelta(days=7)).isoformat()
    if fresh and not refresh:
        cached.update({k: v for k, v in overrides.items() if k in OVERRIDABLE})
        return cached

    info: dict = {"isin": code, "kind": isin_mod.guess_kind(code), "notes": [], "currency": "BRL",
                  "resolved_at": date.today().isoformat()}
    if cached:  # keep runtime state across re-resolution
        info.update({k: cached[k] for k in ("last_signal", "last_collected", "news_last") if k in cached})

    # 1) Debentures: SND registry has CETIP code, issuer CNPJ, indexer, maturity.
    deb = snd.by_isin(code) if code.startswith("BR") else None
    if deb:
        info.update(kind="debenture", cetip_code=deb["cetip_code"], issuer=deb["issuer"], cnpj=deb["cnpj"],
                    maturity=_iso(deb["maturity"]), index_text=deb.get("index"), status=deb.get("status"),
                    law_12431=deb.get("law_12431"))
        if deb.get("status") and not str(deb["status"]).startswith("Registrado"):
            info["notes"].append(f"SND status: {deb['status']} (may be matured/redeemed)")

    # 2) OpenFIGI: names, coupons, maturities for eurobonds, CRI/CRA and federal bonds.
    figi = None
    try:
        figi = openfigi.lookup(code)
    except Exception as e:  # network / rate limit — not fatal
        log.warning("OpenFIGI failed for %s: %s", code, e)
    if figi:
        info.setdefault("issuer", figi["name"])
        info["name"] = figi.get("description") or figi.get("ticker")
        info["figi_ticker"] = figi.get("ticker")
        if figi.get("coupon") is not None and info["kind"] != "debenture":
            info.setdefault("coupon", figi["coupon"])
        if figi.get("maturity") and not info.get("maturity"):
            info["maturity"] = _iso(figi["maturity"])
        st = (figi.get("security_type") or "").upper()
        if info["kind"] == "other" and "EURO" in st:
            info["kind"] = "eurobond"
        root = (figi.get("ticker") or "").split(" ")[0]
        if root in TPF_FROM_FIGI:
            info.update(kind="tesouro", tpf_titulo=TPF_FROM_FIGI[root], issuer="Tesouro Nacional")

    if info["kind"] == "eurobond":
        info["currency"] = "USD"

    # 3) Issuer -> CVM registry (CNPJ, CVM code) -> B3 stock ticker.
    company = None
    if info.get("cnpj"):
        company = cvm.company_by_cnpj(info["cnpj"])
    if company is None and info.get("issuer") and info["kind"] != "tesouro":
        company = cvm.company_by_name(info["issuer"])
        if company:
            info["notes"].append(f"Parent company matched by name: {company['name']} — override `cnpj` if wrong")
    if company:
        info.update(cvm_code=company["cvm_code"], parent_name=company["name"], sector=company.get("sector"))
        if not info.get("cnpj"):
            info["cnpj"] = company["cnpj"]

    if info["kind"] != "tesouro":
        prefix = isin_mod.b3_issuer_code(code)
        ticker = brapi.best_ticker(prefix, prefix) if prefix else None
        if not ticker and company:
            core = cvm.core_name(company["name"]).split()
            ticker = brapi.best_ticker(core[0]) if core else None
        if ticker:
            info["stock_ticker"] = ticker
        else:
            info["notes"].append("No listed stock found for the issuer (set `stock_ticker` to override)")

    if info.get("cetip_code"):
        info["name"] = f"{info['cetip_code']} · {cvm.core_name(info.get('issuer') or '').title()}"
    info.setdefault("name", code)
    if info["kind"] != "tesouro":  # sovereign bonds: news would be noise
        base_name = cvm.core_name(info.get("parent_name") or info.get("issuer") or "").title()
        info["news_query"] = base_name or None
    else:
        info["name"] = f"{info.get('tpf_titulo', 'Tesouro')} {info.get('maturity', '')}".strip()

    if info["kind"] == "cri_cra" and not info.get("cetip_code"):
        info["notes"].append("CRI/CRA: set `cetip_code` in watchlist.yaml (ANBIMA prices are keyed by it)")

    info.update({k: v for k, v in overrides.items() if k in OVERRIDABLE})
    db.save_asset(info)
    return info
