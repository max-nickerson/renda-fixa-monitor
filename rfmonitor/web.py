"""Local dashboard (FastAPI + Jinja2). Binds to 127.0.0.1 by default."""
from __future__ import annotations

import hmac
import json
import math
import threading
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from . import db, isin as isin_mod, jobs, portfolio, screener, strategy
from .ml import live
from .collect import add_manual_price
from .config import ROOT, load_watchlist, save_watchlist, settings
from .resolver import resolve

templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))


def _fmt(v, nd=2, suffix=""):
    if not isinstance(v, (int, float)) or (isinstance(v, float) and math.isnan(v)):
        return "—"  # None, NaN or a missing (jinja Undefined) value
    return f"{v:,.{nd}f}{suffix}"


templates.env.filters["fmt"] = _fmt


def _num(s: str) -> float | None:
    """Accepts '1.234,56' or '1234.56'."""
    s = (s or "").strip()
    if not s:
        return None
    if "," in s:
        s = s.replace(".", "").replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    sched = jobs.start_scheduler() if app.state.scheduler else None
    yield
    if sched:
        sched.shutdown(wait=False)


EDIT_COOKIE = "rfm_edit_key"


def create_app(scheduler: bool = True) -> FastAPI:
    app = FastAPI(title="Renda Fixa Monitor", lifespan=lifespan)
    app.state.scheduler = scheduler

    @app.middleware("http")
    async def edit_guard(request: Request, call_next):
        """Viewing is open to anyone. When ADMIN_KEY is set (public hosting), actions that change data
        (add/remove assets, positions, manual prices, manual refresh) need the edit key once per browser."""
        key = settings.admin_key
        given = request.cookies.get(EDIT_COOKIE) or request.headers.get("x-edit-key")
        request.state.can_edit = (not key) or (given is not None and hmac.compare_digest(given, key))
        request.state.locked = bool(key)
        if request.method not in ("GET", "HEAD", "OPTIONS") and not request.state.can_edit \
                and request.url.path not in ("/unlock",):
            return RedirectResponse(f"/unlock?next={request.headers.get('referer', '/')}", status_code=303)
        return await call_next(request)

    @app.get("/healthz")
    def healthz():
        return {"ok": True}

    @app.get("/unlock", response_class=HTMLResponse)
    def unlock_page(request: Request, next: str = "/", error: str = ""):
        return templates.TemplateResponse(request, "unlock.html", {"tab": "", "next": next, "error": error})

    @app.post("/unlock")
    def unlock(request: Request, key: str = Form(...), next: str = Form("/")):
        dest = next if next.startswith("/") or next.startswith(str(request.base_url)) else "/"
        if not settings.admin_key or not hmac.compare_digest(key.strip(), settings.admin_key):
            return RedirectResponse(f"/unlock?error=1&next={dest}", status_code=303)
        resp = RedirectResponse(dest, status_code=303)
        resp.set_cookie(EDIT_COOKIE, key.strip(), max_age=365 * 86400, httponly=True, samesite="lax",
                        secure=request.url.scheme == "https")
        return resp

    @app.get("/lock")
    def lock():
        resp = RedirectResponse("/", status_code=303)
        resp.delete_cookie(EDIT_COOKIE)
        return resp

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
        if last.get("stock_close") is not None and last.get("stock_close_chg") is not None:
            prev = last["stock_close"] - last["stock_close_chg"]
            last["stock_pct"] = last["stock_close_chg"] / prev * 100 if prev else None
        pos = entry.get("position") or {}
        qty, cost, px = pos.get("quantity"), pos.get("avg_price"), last.get("price")
        scale = 0.01 if info.get("currency") == "USD" else 1  # eurobond prices are % of par; qty = nominal
        position = None
        if qty:
            position = {"quantity": qty, "avg_price": cost,
                        "value": qty * px * scale if px is not None else None,
                        "pnl_pct": (px / cost - 1) * 100 if px and cost else None,
                        "currency": info.get("currency", "BRL")}
        sig = strategy.current(code)
        al = db.alerts(code, limit=1)
        return {"info": info, "last": last, "signal": sig, "last_alert": al[0] if al else None,
                "position": position}

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request):
        held, watched = portfolio.position_rows()
        return templates.TemplateResponse(request, "index.html", {
            "tab": "carteira", "held": held, "watched": watched, "s": portfolio.summary(held),
            "regime": live.cached_regime(), "alerts": db.alerts(limit=15), "settings": settings})

    @app.get("/screener", response_class=HTMLResponse)
    def screener_page(request: Request, added: str = ""):
        res = screener.latest()
        if res is None:
            threading.Thread(target=screener.run, daemon=True).start()
        mine = {isin_mod.normalize(e["isin"]) for e in load_watchlist()}
        return templates.TemplateResponse(request, "screener.html", {
            "tab": "screener", "res": res, "mine": mine, "added": added,
            "curves_json": json.dumps(res["curves"]) if res else "{}"})

    @app.get("/research", response_class=HTMLResponse)
    def research_page(request: Request):
        out = ROOT / "research" / "out"
        load = lambda n: json.loads((out / n).read_text()) if (out / n).exists() else None
        return templates.TemplateResponse(request, "research.html", {
            "tab": "research", "timing": load("timing_results.json"), "selection": load("selection_results.json"),
            "lab": load("lab_summary.json"), "lab3": load("lab3_results.json"), "lab3b": load("lab3b_results.json"),
            "coefs": (live.cached_selection() or {}).get("coefs", {})})

    @app.get("/research/img/{name}")
    def research_img(name: str):
        path = ROOT / "research" / "out" / name
        if path.suffix != ".png" or not path.exists() or path.parent != ROOT / "research" / "out":
            raise HTTPException(404)
        return FileResponse(path)

    @app.post("/screener/run")
    def screener_run():
        threading.Thread(target=screener.run, daemon=True).start()
        return RedirectResponse("/screener", status_code=303)

    @app.get("/api/screener")
    def api_screener():
        return screener.latest() or {"rows": []}

    @app.post("/asset/{isin}/position")
    def set_position(isin: str, quantity: str = Form(""), avg_price: str = Form("")):
        wl = load_watchlist()
        for e in wl:
            if isin_mod.normalize(e["isin"]) == isin:
                q, c = _num(quantity), _num(avg_price)
                if q:
                    e["position"] = {"quantity": q, **({"avg_price": c} if c else {})}
                else:
                    e.pop("position", None)
        save_watchlist(wl)
        return RedirectResponse(f"/asset/{isin}", status_code=303)

    @app.post("/add")
    def add(isin: str = Form(...), stock_ticker: str = Form(""), cetip_code: str = Form(""),
            next: str = Form("")):
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
        if next == "screener":
            return RedirectResponse(f"/screener?added={code}", status_code=303)
        return RedirectResponse(f"/asset/{code}", status_code=303)

    @app.post("/add_bulk")
    def add_bulk(items: str = Form(...)):
        from .lookup import resolve_tokens
        found, missing = resolve_tokens(items)
        wl = load_watchlist()
        have = {isin_mod.normalize(e["isin"]) for e in wl}
        new = [c for c in found if c not in have]
        for code in new:
            wl.append({"isin": code})
        if new:
            save_watchlist(wl)

            def work():
                for code in new:
                    try:
                        resolve(code, {"isin": code}, refresh=True)
                    except Exception:
                        pass
                jobs.run_cycle()
            threading.Thread(target=work, daemon=True).start()
        q = f"added={len(new)}" + (f"&missing={','.join(missing)[:300]}" if missing else "")
        return RedirectResponse(f"/?{q}", status_code=303)

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
        entry = next((e for e in load_watchlist() if isin_mod.normalize(e["isin"]) == isin), {})
        return templates.TemplateResponse(request, "asset.html", {
            "tab": "carteira", "info": info, "signal": strategy.current(isin), "events": db.events(isin, limit=60),
            "alerts": db.alerts(isin, limit=40), "today": date.today().isoformat(),
            "position": entry.get("position") or {}})

    @app.post("/asset/{isin}/price")
    def manual_price(isin: str, d: str = Form(...), price: float = Form(...)):
        add_manual_price(isin, d, price)
        threading.Thread(target=jobs.run_cycle, daemon=True).start()
        return RedirectResponse(f"/asset/{isin}", status_code=303)

    @app.get("/alerts", response_class=HTMLResponse)
    def alerts_page(request: Request):
        return templates.TemplateResponse(request, "alerts.html", {"tab": "alerts", "alerts": db.alerts(limit=500)})

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
