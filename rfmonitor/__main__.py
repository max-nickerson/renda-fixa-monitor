"""CLI: python -m rfmonitor <command>"""
from __future__ import annotations

import argparse
import json
import logging

from . import __version__


def main() -> None:
    p = argparse.ArgumentParser(prog="rfmonitor", description="Renda fixa monitor & strategy")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="start dashboard + scheduler (http://127.0.0.1:8000)")
    r.add_argument("--no-scheduler", action="store_true")
    c = sub.add_parser("collect", help="run one collect → alerts → e-mail cycle")
    c.add_argument("--backfill", type=int, help="business days of history to load")
    a = sub.add_parser("add", help="add an ISIN to watchlist.yaml")
    a.add_argument("isin")
    a.add_argument("--stock-ticker")
    a.add_argument("--cetip-code")
    s = sub.add_parser("resolve", help="show what an ISIN resolves to")
    s.add_argument("isin")
    sg = sub.add_parser("signal", help="current strategy signal")
    sg.add_argument("isin")
    b = sub.add_parser("backtest", help="backtest the strategy on stored history")
    b.add_argument("isin")
    b.add_argument("--cost-bps", type=float, default=30)
    ip = sub.add_parser("import-prices", help="import a price history CSV (date,price or t[unix],close)")
    ip.add_argument("isin")
    ip.add_argument("csv")
    ip.add_argument("--source", default="import")
    sc = sub.add_parser("screener", help="rank the whole debenture universe (quant model)")
    sc.add_argument("--top", type=int, default=20)
    sub.add_parser("test-email", help="send a test e-mail")
    sub.add_parser("tv-login", help="sign in to TradingView MCP (optional eurobond prices)")
    sub.add_parser("version")
    args = p.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    from .config import ensure_dirs, load_watchlist, save_watchlist, settings
    ensure_dirs()

    if args.cmd == "run":
        import uvicorn
        from .web import create_app
        print(f"Dashboard: http://{settings.host}:{settings.port}/")
        uvicorn.run(create_app(scheduler=not args.no_scheduler), host=settings.host, port=settings.port)
    elif args.cmd == "collect":
        from .jobs import run_cycle
        print(json.dumps(run_cycle(args.backfill), indent=2, default=str))
    elif args.cmd == "add":
        from .isin import normalize
        from .resolver import resolve
        code = normalize(args.isin)
        wl = load_watchlist()
        if not any(normalize(e["isin"]) == code for e in wl):
            entry = {"isin": code}
            if args.stock_ticker:
                entry["stock_ticker"] = args.stock_ticker
            if args.cetip_code:
                entry["cetip_code"] = args.cetip_code
            wl.append(entry)
            save_watchlist(wl)
        print(json.dumps(resolve(code, refresh=True), indent=2, default=str, ensure_ascii=False))
    elif args.cmd == "resolve":
        from .resolver import resolve
        print(json.dumps(resolve(args.isin, refresh=True), indent=2, default=str, ensure_ascii=False))
    elif args.cmd == "signal":
        from .isin import normalize
        from .strategy import current
        print(json.dumps(current(normalize(args.isin)).as_dict(), indent=2))
    elif args.cmd == "backtest":
        from .isin import normalize
        from .strategy import backtest
        res = backtest(normalize(args.isin), args.cost_bps)
        res.pop("equity_curve", None)
        print(json.dumps(res, indent=2, default=str))
    elif args.cmd == "import-prices":
        import pandas as pd
        from . import db
        from .isin import normalize
        df = pd.read_csv(args.csv)
        cols = [c.lower() for c in df.columns]
        df.columns = cols
        if "t" in cols:
            dates = pd.to_datetime(df["t"], unit="s")
        else:
            dates = pd.to_datetime(df[cols[0]], dayfirst="/" in str(df[cols[0]].iloc[0]))
        px_col = next((c for c in ("close", "price", "preco", "c") if c in cols), cols[1])
        rows = [(d.strftime("%Y-%m-%d"), "price", float(p)) for d, p in zip(dates, df[px_col])]
        print(f"imported {db.put_observations(normalize(args.isin), rows, args.source)} prices")
    elif args.cmd == "screener":
        from . import screener
        res = screener.run()
        if "error" in res:
            raise SystemExit(res["error"])
        print(f"ANBIMA {res['date']}: {res['ranked']} of {res['universe']} ranked")
        for r in res["rows"][: args.top]:
            flags = f"  ⚠ {'; '.join(r['flags'])}" if r["flags"] else ""
            print(f"{r['rank']:>3} {r['codigo']:<8} {r['nome'][:34]:<34} {r['indice']:<16} dur {r['duration']:.1f}"
                  f"  spread {r['spread_bps']:.0f} (fair {r['fair_bps']:.0f})  score {r['score']:.2f}{flags}")
    elif args.cmd == "test-email":
        from . import db
        from .alerts import send_pending_email
        db.put_alert("_TEST", "test", "info", "Teste de e-mail do Renda Fixa Monitor", f"test:{db.now_iso()}")
        n = send_pending_email()
        print("sent" if n else "not sent — check SMTP_* in .env")
    elif args.cmd == "tv-login":
        from .sources import tradingview_mcp
        if not tradingview_mcp.available():
            raise SystemExit('Install the optional dependency first: pip install "mcp>=1.10"')
        tradingview_mcp.login()
    elif args.cmd == "version":
        print(__version__)


if __name__ == "__main__":
    main()
