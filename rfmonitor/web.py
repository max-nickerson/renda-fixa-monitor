"""Local dashboard (FastAPI + Jinja2). Binds to 127.0.0.1 by default."""
from __future__ import annotations

import math
import threading
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from . import db, isin as isin_mod, jobs, strategy
from .collect import add_manual_price
from .config import load_watchlist, save_watchlist, settings
from .resolver import resolve

templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))


def _fmt(v, nd=2, suffix=""):
    if not isinstance(v, (int, float)) or (isinstance(v, float) and math.isnan(v)):
        return "—"  # None, NaN or a missing (jinja Undefined) value
    return f"{v:,.{nd}f}{suffix}"


templates.env.filters["fmt"] = _fmt


@asynccontextmanager
async def lifespan(app: FastAPI):
    sched = jobs.start_scheduler() if app.state.scheduler else None
    yield
    if sched:
        sched.shutdown(wait=False)


def create_app(scheduler: bool = True) -> FastAPI:
    app = FastAPI(title="Renda Fixa Monitor", lifespan=lifespan)
    app.state.scheduler = scheduler

    def row(entry: dict) -> dict:
        code = isin_mod.normalize(entry["isin"])
        info = db.get_asset(code) or {"isin": code, "name": entry.get("name") or code, "kind": "…"}
        df = db.series(code)
        last = {}
        for m in ("price", "yield", "spread_bps", "stock_close"):
            last[m] = last[m + "_chg"] = last[m + "_date"] = None
            if m in df and df[m].notna().any():
                s = df[m].dropna()
                last[m] = float(s.iloc[-1])
                last[m + "_chg"] = float(s.iloc[-1] - s.iloc[-2]) if len(s) > 1 else None
                last[m + "_date"] = s.index[-1].strftime("%d/%m")
        last["stock_pct"] = None
        if "stock_close" in last and last.get("stock_close_chg") is not None:
            prev = last["stock_close"] - last["stock_close_chg"]
            last["stock_pct"] = last["stock_close_chg"] / prev * 100 if prev else None
        sig = strategy.current(code)
        al = db.alerts(code, limit=1)
        return {"info": info, "last": last, "signal": sig, "last_alert": al[0] if al else None}

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request):
        rows = []
        for e in load_watchlist():
            try:
                rows.append(row(e))
            except ValueError:
                continue
        return templates.TemplateResponse(request, "index.html", {
            "rows": rows, "alerts": db.alerts(limit=15), "settings": settings})

    @app.post("/add")
    def add(isin: str = Form(...), stock_ticker: str = Form(""), cetip_code: str = Form("")):
        try:
            code = isin_mod.normalize(isin)
        except ValueError as e:
            raise HTTPException(400, str(e))
        wl = load_watchlist()
        if not any(isin_mod.normalize(e["isin"]) == code for e in wl):
            entry = {"isin": code}
            if stock_ticker.strip():
                entry["stock_ticker"] = stock_ticker.strip().upper()
            if cetip_code.strip():
                entry["cetip_code"] = cetip_code.strip().upper()
            wl.append(entry)
            save_watchlist(wl)
            resolve(code, entry, refresh=True)
            threading.Thread(target=jobs.run_cycle, daemon=True).start()
        return RedirectResponse(f"/asset/{code}", status_code=303)

    @app.post("/remove/{isin}")
    def remove(isin: str):
        save_watchlist([e for e in load_watchlist() if isin_mod.normalize(e["isin"]) != isin])
        db.delete_asset(isin)
        return RedirectResponse("/", status_code=303)

    @app.post("/collect")
    def collect_now():
        threading.Thread(target=jobs.run_cycle, daemon=True).start()
        return RedirectResponse("/", status_code=303)

    @app.get("/asset/{isin}", response_class=HTMLResponse)
    def asset(request: Request, isin: str):
        info = db.get_asset(isin)
        if not info:
            raise HTTPException(404, "Unknown ISIN — add it on the dashboard first")
        return templates.TemplateResponse(request, "asset.html", {
            "info": info, "signal": strategy.current(isin), "events": db.events(isin, limit=60),
            "alerts": db.alerts(isin, limit=40), "today": date.today().isoformat()})

    @app.post("/asset/{isin}/price")
    def manual_price(isin: str, d: str = Form(...), price: float = Form(...)):
        add_manual_price(isin, d, price)
        threading.Thread(target=jobs.run_cycle, daemon=True).start()
        return RedirectResponse(f"/asset/{isin}", status_code=303)

    @app.get("/alerts", response_class=HTMLResponse)
    def alerts_page(request: Request):
        return templates.TemplateResponse(request, "alerts.html", {"alerts": db.alerts(limit=500)})

    # ---------------- JSON API
    @app.get("/api/assets")
    def api_assets():
        return [db.get_asset(isin_mod.normalize(e["isin"])) for e in load_watchlist()]

    @app.get("/api/asset/{isin}/series")
    def api_series(isin: str):
        df = db.series(isin)
        if df.empty:
            return {"dates": [], "series": {}}
        df = df.astype(object).where(df.notna(), None)
        return {"dates": [d.strftime("%Y-%m-%d") for d in df.index],
                "series": {c: df[c].tolist() for c in df.columns if not c.startswith("f_")}}

    @app.get("/api/asset/{isin}/signal")
    def api_signal(isin: str):
        return strategy.current(isin).as_dict()

    @app.get("/api/asset/{isin}/backtest")
    def api_backtest(isin: str, cost_bps: float = 30.0, entry: float = strategy.ENTRY, exit: float = strategy.EXIT):
        return JSONResponse(strategy.backtest(isin, cost_bps, entry, exit))

    return app
