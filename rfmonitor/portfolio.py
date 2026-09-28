"""Portfolio-manager analytics for the holdings in watchlist.yaml.

Per position: market value (BRL), weight, duration, CDI+ spread, CS01/DV01, carry over CDI, spread moves,
rich/cheap vs own history and vs peers (fair spread, fair price, upside), ML view, and an action hint.
Aggregates: MV-weighted duration & spread, CS01, rate DV01, carry R$/yr, concentration, maturity ladder,
stress tests.
"""
from __future__ import annotations

import math
from datetime import date
from functools import lru_cache

import numpy as np
import pandas as pd

from . import db, isin as isin_mod, strategy
from .bonds import price_for_spread_change
from .config import load_watchlist
from .sources import brapi

LADDER = [(0, 1, "< 1a"), (1, 2, "1–2a"), (2, 3, "2–3a"), (3, 5, "3–5a"), (5, 7, "5–7a"), (7, 100, "> 7a")]


@lru_cache(maxsize=1)
def _fx_cached(day: str) -> float | None:
    return brapi.usdbrl()


def fx_usdbrl() -> float:
    return _fx_cached(date.today().isoformat()) or 5.0


def _last(df: pd.DataFrame, col: str, back: int = 0) -> float | None:
    if col not in df:
        return None
    s = df[col].dropna()
    if len(s) <= back:
        return None
    v = float(s.iloc[-1 - back])
    return v if math.isfinite(v) else None


def _history_view(s: pd.Series) -> dict:
    """Where the current spread sits in its own history."""
    s = s.dropna()
    out = {"n": int(len(s)), "z": None, "pct": None, "min": None, "max": None}
    if len(s) >= 20:
        w = s.tail(250)
        sd = w.std()
        out.update(z=float((w.iloc[-1] - w.mean()) / sd) if sd else None,
                   pct=float((w <= w.iloc[-1]).mean() * 100), min=float(w.min()), max=float(w.max()))
    return out


def _screener_index() -> dict[str, dict]:
    res = db.load_screener()
    return {r["codigo"]: r for r in res["rows"]} if res else {}


def _verdict(resid_z: float | None, hist_z: float | None) -> str:
    """Spread high = bond cheap (price low); spread low = rich (price high)."""
    zs = [z for z in (resid_z, hist_z) if z is not None]
    if not zs:
        return "—"
    z = sum(zs) / len(zs)
    if z >= 1.0:
        return "Barato (spread alto / preço baixo)"
    if z <= -1.0:
        return "Caro (spread baixo / preço alto)"
    return "Justo"


def position_rows() -> tuple[list[dict], list[dict]]:
    """(positions with quantity, monitored-only assets)."""
    fx = fx_usdbrl()
    scr = _screener_index()
    from .ml import live
    model = (live.cached_selection() or {}).get("rows", {})
    held, watched = [], []
    for e in load_watchlist():
        try:
            code = isin_mod.normalize(e["isin"])
        except ValueError:
            continue
        info = db.get_asset(code) or {"isin": code, "name": code, "kind": "…"}
        df = db.series(code)
        price = _last(df, "price")
        dur = _last(df, "duration")
        cdi = _last(df, "cdi_spread_bps")
        spread_main = cdi if cdi is not None else _last(df, "spread_bps")  # eurobonds: vs UST
        spread_col = "cdi_spread_bps" if cdi is not None else "spread_bps"
        chg = {k: (spread_main - v) if spread_main is not None and (v := _last(df, spread_col, n)) is not None else None
               for k, n in (("d1", 1), ("d5", 5), ("d20", 20))}
        hist = _history_view(df[spread_col]) if spread_col in df else _history_view(pd.Series(dtype=float))
        sr = scr.get((info.get("cetip_code") or "").strip())
        fair = resid_z = None
        if sr and sr.get("fair_cdi_bps") is not None and cdi is not None:
            fair, resid_z = sr["fair_cdi_bps"], sr.get("resid_z")
        fair_price = upside = None
        if fair is not None and price and dur:
            fair_price = price_for_spread_change(price, dur, fair - spread_main)
            upside = (fair_price / price - 1) * 100
        sig = strategy.current(code)
        ccy = info.get("currency", "BRL")
        pos = e.get("position") or {}
        qty, cost = pos.get("quantity"), pos.get("avg_price")
        mv = None
        if qty and price is not None:
            mv = qty * price * (0.01 if ccy == "USD" else 1) * (fx if ccy == "USD" else 1)
        kind = info.get("index") or info.get("kind")
        rate_dur = 0.0 if kind in ("DI_SPREAD", "DI_PCT") else (dur or 0)  # floaters carry ~no rate duration
        row = {
            "info": info, "isin": code, "name": info.get("name"), "kind": info.get("kind"),
            "index_text": info.get("index_text") or info.get("tpf_titulo") or "", "currency": ccy,
            "maturity": info.get("maturity"), "price": price, "pct_par": _last(df, "pct_par"),
            "yield": _last(df, "yield"), "duration": dur, "spread": spread_main,
            "spread_label": "CDI+" if cdi is not None else ("UST+" if ccy == "USD" else "—"),
            "chg": chg, "hist": hist, "fair": fair, "resid_z": resid_z, "fair_price": fair_price,
            "upside_pct": upside, "verdict": _verdict(resid_z, hist["z"]), "signal": sig,
            "stock": info.get("stock_ticker"), "stock_close": _last(df, "stock_close"),
            "quantity": qty, "avg_price": cost, "mv_brl": mv,
            "pnl_pct": (price / cost - 1) * 100 if price and cost else None,
            "cs01": mv * dur / 10000 if mv and dur else None,
            "dv01": mv * rate_dur / 10000 if mv and rate_dur else 0.0 if mv else None,
            "carry_brl": mv * spread_main / 10000 if mv and spread_main is not None else None,
            "last_alert": (db.alerts(code, limit=1) or [None])[0],
            "model_pct": (model.get((info.get("cetip_code") or "").strip()) or {}).get("blend_pct"),
            "ml_pred_bps": (model.get((info.get("cetip_code") or "").strip()) or {}).get("ml_pred_bps"),
        }
        row["action"] = _action(row)
        (held if qty else watched).append(row)
    total = sum(r["mv_brl"] or 0 for r in held)
    for r in held:
        r["weight"] = r["mv_brl"] / total * 100 if total and r["mv_brl"] else None
    return held, watched


def _action(r: dict) -> str:
    """Tested rule first (research/run_selection.py): model top 20% → add; out of top 50% → reduce.
    Risk blocks (material facts, negative news, leverage) override an 'add'."""
    pct = r.get("model_pct")
    blocked = bool(r["signal"] and r["signal"].blocked_by)
    if pct is not None:
        if pct > 0.5:
            return f"Reduzir (modelo: top {pct:.0%})"
        if pct <= 0.2:
            return "Revisar risco antes de aumentar" if blocked else f"Aumentar (modelo: top {max(pct, 0.01):.0%})"
        return "Revisar risco" if blocked else f"Manter (modelo: top {pct:.0%})"
    # No recent trade → no model score: fall back to rich/cheap vs peers and the per-asset signal.
    sig = r["signal"].label if r["signal"] else "N/A"
    v = r["verdict"]
    if sig.startswith("SELL") or v.startswith("Caro"):
        return "Reduzir / realizar"
    if blocked:
        return "Revisar risco"
    if v.startswith("Barato"):
        return "Aumentar"
    return "Manter"


def summary(held: list[dict]) -> dict:
    mv = sum(r["mv_brl"] or 0 for r in held)
    if not mv:
        return {"mv": 0}

    def wavg(key, only=None):
        pts = [(r["mv_brl"], r[key]) for r in held
               if r["mv_brl"] and r[key] is not None and (only is None or r["spread_label"] == only)]
        w = sum(p[0] for p in pts)
        return sum(a * b for a, b in pts) / w if w else None

    cs01 = sum(r["cs01"] or 0 for r in held)
    dv01 = sum(r["dv01"] or 0 for r in held)
    by = lambda key: sorted(((k, sum(r["mv_brl"] or 0 for r in held if (r.get(key) or "—") == k) / mv * 100)
                             for k in {r.get(key) or "—" for r in held}), key=lambda x: -x[1])
    for r in held:
        r["issuer_key"] = (r["info"].get("parent_name") or r["info"].get("issuer") or "—").title()
        r["index_key"] = {"DI_SPREAD": "DI+", "DI_PCT": "%DI", "IPCA": "IPCA+", "PRE": "Pré"}.get(
            r["info"].get("index"), r["kind"])
    ladder = []
    for lo, hi, lab in LADDER:
        v = sum(r["mv_brl"] or 0 for r in held if r["duration"] is not None and lo <= r["duration"] < hi)
        ladder.append((lab, v / mv * 100))
    return {
        "mv": mv, "fx": fx_usdbrl(), "n": len(held),
        "duration": wavg("duration"), "spread": wavg("spread", "CDI+"), "spread_ust": wavg("spread", "UST+"),
        "cdi_share": sum(r["mv_brl"] or 0 for r in held if r["spread_label"] == "CDI+") / mv * 100,
        "cs01": cs01, "dv01": dv01, "carry": sum(r["carry_brl"] or 0 for r in held),
        "by_issuer": by("issuer_key"), "by_index": by("index_key"), "ladder": ladder,
        "stress": [("Spreads +50 bps", -cs01 * 50), ("Spreads +100 bps", -cs01 * 100),
                   ("Spreads −50 bps", cs01 * 50), ("Juros +100 bps (Pré/IPCA)", -dv01 * 100)],
        "top_issuer": max((w for _, w in by("issuer_key")), default=0),
    }
