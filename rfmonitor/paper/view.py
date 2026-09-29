"""Data for the /paper page."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import collect, engine, store


def context(book: str | None = None) -> dict:
    books = store.rows("SELECT * FROM books ORDER BY name DESC")
    book = book if book in {b["name"] for b in books} else (books[0]["name"] if books else None)
    nav = store.df("SELECT * FROM nav ORDER BY date")
    bm = engine.benchmarks() if books else pd.DataFrame()
    cards = []
    for b in books:
        n = nav[nav["book"] == b["name"]]
        if n.empty:
            continue
        last = n.iloc[-1]
        ret = last["nav"] / b["aum"] - 1
        cdi = bm["CDI"].iloc[-1] - 1 if len(bm) else np.nan
        uni = bm["Universo"].iloc[-1] - 1 if len(bm) else np.nan
        o = store.df("SELECT * FROM orders WHERE book = ?", (b["name"],))
        f = o[(o["status"].isin(["filled", "forced"]))]
        fb = f[f["side"] == "buy"]
        dtf = [np.busday_count(a, c) for a, c in zip(fb["created"], fb["fill_date"])] if len(fb) else []
        cards.append({"name": b["name"], "start": b["start"], "nav": last["nav"], "ret": ret * 100,
                      "vs_cdi": (ret - cdi) * 1e4, "vs_uni": (ret - uni) * 1e4,
                      "inv_pct": 100 * last["invested"] / last["nav"] if last["nav"] else 0, "n_pos": int(last["n_pos"]),
                      "filled": len(f), "open": int((o["status"] == "open").sum()),
                      "expired": int((o["status"] == "expired").sum()),
                      "slip_buy": fb["slippage_bps"].mean() if len(fb) else None,
                      "days_to_fill": float(np.median(dtf)) if dtf else None})
    curves = {"dates": [], "series": {}}
    if len(bm):
        curves["dates"] = list(bm.index)
        for b in books:
            s = nav[nav["book"] == b["name"]].set_index("date")["nav"].reindex(bm.index).ffill() / b["aum"]
            curves["series"][b["name"]] = [None if pd.isna(v) else round(float(v) * 100, 4) for v in s]
        for c in ("CDI", "Universo"):
            curves["series"][c] = [round(float(v) * 100, 4) for v in bm[c]]
    names = store.df("SELECT codigo, nome, spread_bps FROM snap_bonds WHERE date = (SELECT max(date) FROM snap_bonds)")
    names = names.set_index("codigo")
    pos = store.df("SELECT * FROM positions WHERE book = ? AND status IN ('open', 'selling') ORDER BY value DESC", (book,))
    tot = pos["value"].sum() if len(pos) else 0
    pos["weight"] = 100 * pos["value"] / (nav[nav["book"] == book]["nav"].iloc[-1] if tot else 1)
    pos = pos.join(names, on="codigo")
    sig_date = store.rows("SELECT max(date) d FROM signals WHERE book = ?", (book,))[0]["d"] if book else None
    sig = store.df("SELECT * FROM signals WHERE book = ? AND date = ? ORDER BY weight DESC, note", (book, sig_date))
    sig = sig.join(names[["nome"]], on="codigo")
    clean = lambda d: d.astype(object).where(d.notna(), None).to_dict("records")
    return {
        "books": cards, "book": book, "curves": curves, "last_day": nav["date"].max() if len(nav) else None,
        "positions": clean(pos), "signals": clean(sig), "sig_date": sig_date,
        "open_orders": store.rows("SELECT * FROM orders WHERE book = ? AND status = 'open' ORDER BY created DESC, value DESC "
                                  "LIMIT 60", (book,)),
        "fills": store.rows("SELECT * FROM orders WHERE book = ? AND status IN ('filled', 'forced') ORDER BY fill_date DESC "
                            "LIMIT 60", (book,)),
        "status": collect.status(),
        "ibkr": ibkr_forward(),
    }


def ibkr_forward() -> list[dict]:
    """IBKR paper forward-validation strategies (research/ibkr_lab/runner): NAV, return, live vs backtest expectation."""
    import json
    import sqlite3
    from ..config import DATA_DIR
    path = DATA_DIR / "ibkr_lab" / "forward.db"
    if not path.exists():
        return []
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    try:
        out = []
        for st in con.execute("SELECT * FROM strategies ORDER BY name"):
            nav = [dict(r) for r in con.execute("SELECT date, nav, gross, n_pos FROM nav WHERE strategy=? ORDER BY date",
                                                (st["name"],))]
            fills = [dict(r) for r in con.execute("SELECT * FROM fills WHERE strategy=?", (st["name"],))]
            spreads = [(f["ask"] - f["bid"]) / ((f["ask"] + f["bid"]) / 2) * 1e4 for f in fills if f["bid"] and f["ask"]]
            exp = json.loads(st["expected"] or "{}")
            last = nav[-1] if nav else {}
            ret = (last.get("nav", st["capital"]) / st["capital"] - 1) * 100 if nav else 0.0
            out.append({"name": st["name"], "description": st["description"], "rebalance": st["rebalance"],
                        "started": st["started"], "days": len(nav), "ret": ret, "n_pos": last.get("n_pos", 0),
                        "gross": last.get("gross", 0), "fills": len(fills),
                        "modes": sorted({f["mode"] for f in fills}),
                        "avg_spread_bps": sum(spreads) / len(spreads) if spreads else None,
                        "exp_ann": exp.get("ann_return"), "exp_sharpe": exp.get("sharpe"),
                        "curve": [round(r["nav"] / st["capital"] * 100, 3) for r in nav]})
        return out
    finally:
        con.close()


EXPORTABLE = {"positions", "orders", "signals", "nav", "snap_bonds", "snd_trades", "eq_close", "rates"}
