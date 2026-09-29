"""'Meus ativos': live monitor of the researched holdings list (data/ativos/ativos_pesquisa.xlsx, private, git-ignored).

Every minute during market hours: live quotes for every listed proxy (IBKR if IB Gateway/TWS is running, else brapi)
and eurobonds by ISIN (IBKR). On each page load: latest ANBIMA spread / verdict / P4 call per debenture, stock
drawdowns from the paper-trading price history, and alert flags. Nothing here is ever exported or published.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import db
from .config import DATA_DIR

log = logging.getLogger(__name__)
DIR = DATA_DIR / "ativos"
FILE = Path(os.getenv("ATIVOS_FILE", DIR / "ativos_pesquisa.xlsx"))
LIVE = DIR / "live.json"
TICKER = re.compile(r"^[A-Z]{4}\d{1,2}$")
NON_CREDIT = r"Caixa|despesa|Taxa|Derivativo|Futuro|Ajuste|Compromissada|Colateral|interno"


def table() -> pd.DataFrame:
    if not FILE.exists():
        return pd.DataFrame()
    t = pd.read_excel(FILE, sheet_name="Ativos", dtype=str)
    t["proxy_ticker"] = t["proxy_ticker"].fillna("").str.strip().str.upper()
    return t


def refresh_live() -> dict:
    """Quotes for all proxies and eurobonds; IBKR first, brapi for what IBKR did not return."""
    t = table()
    if t.empty:
        return {}
    tick = sorted({x for x in t["proxy_ticker"] if TICKER.match(x)})
    bonds = sorted({i for i in t["isin"].dropna() if re.match(r"^(US|USL|USN)\w{9,10}$", i)})
    out, src = {}, "brapi"
    try:
        asyncio.set_event_loop(asyncio.new_event_loop())  # scheduler thread
        from .sources import ibkr
        ib = ibkr.connect(timeout=3)
        try:
            out = ibkr.quotes(ib, [ibkr.b3_stock(x) for x in tick] + [ibkr.bond_isin(i) for i in bonds])
            src = "ibkr"
        finally:
            ib.disconnect()
    except Exception as e:
        log.info("ativos: IBKR unavailable (%s), using brapi", str(e)[:80])
    missing = [x for x in tick if x not in out]
    if missing:
        try:
            from .sources import brapi
            for k, q in brapi.quotes(missing).items():
                out[k] = {"last": q["price"], "chg_pct": q.get("change_pct"), "time": str(q.get("time")), "source": "brapi"}
        except Exception as e:
            log.warning("ativos: brapi quotes failed: %s", e)
    for q in out.values():
        if q.get("chg_pct") is None and q.get("last") and q.get("close"):
            q["chg_pct"] = (q["last"] / q["close"] - 1) * 100
    DIR.mkdir(parents=True, exist_ok=True)
    LIVE.write_text(json.dumps({"at": datetime.now(timezone.utc).isoformat(), "source": src, "quotes": out}), encoding="utf-8")
    log.info("ativos: %d quotes (%s)", len(out), src)
    return out


def _drawdowns(tickers: list[str]) -> dict:
    """4-week return and distance from the 52-week high, from the paper-trading stock history (paper.db)."""
    try:
        from .paper import store
        e = store.df("SELECT date, ticker, close FROM eq_close WHERE date >= date('now', '-400 day')")
    except Exception:
        return {}
    out = {}
    for tk, g in e[e["ticker"].isin(tickers)].groupby("ticker"):
        s = g.sort_values("date")["close"].astype(float)
        if len(s) > 25:
            out[tk] = {"r4w": s.iloc[-1] / s.iloc[-21] - 1, "dd52": s.iloc[-1] / s.iloc[-252:].max() - 1}
    return out


def view(q: str = "", tipo: str = "", grupo: str = "", setor: str = "", only_alerts: bool = False,
         credit_only: bool = True) -> dict:
    t = table()
    if t.empty:
        return {"rows": [], "missing": True}
    live = json.loads(LIVE.read_text(encoding="utf-8")) if LIVE.exists() else {"quotes": {}}
    quotes = live.get("quotes", {})
    scr = {r["codigo"]: r for r in (db.load_screener() or {"rows": []})["rows"]}
    dd = _drawdowns(sorted({x for x in t["proxy_ticker"] if TICKER.match(x)}))
    rows = []
    for r in t.to_dict("records"):
        r = {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in r.items()}
        s = scr.get(r.get("codigo") or "")
        pq = quotes.get(r["proxy_ticker"]) or {}
        bq = quotes.get(r.get("isin") or "") or {}
        d = dd.get(r["proxy_ticker"], {})
        alerts = []
        if pq.get("chg_pct") is not None and pq["chg_pct"] <= -5:
            alerts.append(f"ação {r['proxy_ticker']} {pq['chg_pct']:+.1f}% hoje")
        if d.get("r4w") is not None and d["r4w"] <= -0.15:
            alerts.append(f"ação {d['r4w']:+.0%} em 4 sem.")
        if s:
            if (s.get("chg_5d_bps") or 0) >= 50:
                alerts.append(f"spread +{s['chg_5d_bps']:.0f} bps em 5d")
            if str(s.get("strategy", "")).startswith(("VENDER", "C3 · não comprar")):
                alerts.append(s["strategy"].replace("C3 · ", ""))
        if re.search(r"recupera[cç][aã]o judicial|suspend|default|inadimpl|ALERTA", r.get("notas") or "", re.I):
            alerts.append("alerta de crédito (pesquisa)")
        bid = bq.get("bid") or bq.get("last")
        rows.append({**r, "px": pq.get("last"), "px_chg": pq.get("chg_pct"), "px_src": pq.get("source"),
                     "r4w": d.get("r4w"), "dd52": d.get("dd52"), "bond_px": bid,
                     "bond_chg": ((bq.get("last") or bid) / bq["close"] - 1) * 100 if bq.get("close") and (bq.get("last") or bid) else None,
                     "spread": s.get("spread_bps") if s else None, "chg5": s.get("chg_5d_bps") if s else None,
                     "veredito": s.get("verdict") if s else r.get("veredito"), "p4": s.get("strategy") if s else None,
                     "alerts": alerts})
    all_rows = rows
    if credit_only:
        rows = [r for r in rows if not re.search(NON_CREDIT, r.get("tipo") or "", re.I)]
    ql = q.strip().lower()
    rows = [r for r in rows if (not tipo or r.get("tipo") == tipo) and (not grupo or r.get("grupo_economico") == grupo)
            and (not setor or r.get("setor") == setor) and (not only_alerts or r["alerts"])
            and (not ql or ql in " ".join(str(r.get(k) or "") for k in ("id_origem", "isin", "codigo", "emissor",
                                                                             "grupo_economico", "proxy_ticker")).lower())]
    rows.sort(key=lambda r: (-len(r["alerts"]), r.get("grupo_economico") or "~", r.get("id_origem") or ""))
    by_group = {}
    for r in all_rows:
        if re.search(NON_CREDIT, r.get("tipo") or "", re.I):
            continue
        g = by_group.setdefault(r.get("grupo_economico") or "(sem grupo)", {"n": 0, "alerts": 0, "proxy": r["proxy_ticker"],
                                                                            "px_chg": r["px_chg"], "r4w": r["r4w"]})
        g["n"] += 1
        g["alerts"] += bool(r["alerts"])
    groups = sorted(by_group.items(), key=lambda kv: (-kv[1]["alerts"], -kv[1]["n"]))
    opts = lambda k: sorted({r.get(k) for r in all_rows if r.get(k)})
    return {"rows": rows, "n_total": len(all_rows), "n_alerts": sum(bool(r["alerts"]) for r in all_rows),
            "live_at": live.get("at"), "live_src": live.get("source"), "n_quotes": len(quotes), "groups": groups[:40],
            "tipos": opts("tipo"), "grupos": opts("grupo_economico"), "setores": opts("setor"),
            "f": {"q": q, "tipo": tipo, "grupo": grupo, "setor": setor, "only_alerts": only_alerts,
                  "credit_only": credit_only}}
