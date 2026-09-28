"""Export a self-contained, read-only HTML snapshot of the dashboard (portfolio, P4 opportunities, research).

    python -m rfmonitor export            → data/export/renda-fixa-monitor.html
The page has no server behind it: data is embedded as JSON, charts as data: URIs.
"""
from __future__ import annotations

import base64
import json
import math
import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from . import db, portfolio
from .config import DATA_DIR, ROOT
from .ml import live

OUT = DATA_DIR / "export"


def _clean(v):
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    return v


def _img(name: str) -> str | None:
    p = ROOT / "research" / "out" / name
    return "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode() if p.exists() else None


def _series(isin: str, label: str, n: int = 260) -> dict | None:
    """Last ~year of the asset's spread (or price) for a small chart."""
    col = "cdi_spread_bps" if label == "CDI+" else "spread_bps" if label == "UST+" else "price"
    s = db.series(isin, [col])
    if s.empty or col not in s:
        s = db.series(isin, ["price"])
        col = "price"
        if s.empty:
            return None
    s = s[col].dropna().tail(n)
    return {"kind": "spread" if col != "price" else "price", "d": [d.strftime("%Y-%m-%d") for d in s.index],
            "v": [round(float(v), 2) for v in s.values]} if len(s) > 1 else None


def _lab3(a: dict | None, b: dict | None) -> dict | None:
    """Compact round-4 results (stock, commodities, ratings, regulators, sector news) for the snapshot."""
    if not a:
        return None
    ev = [{"name": k, "n": v["n"], "post": v["post_ret_bps"], "post_t": v["post_ret_t"], "spread": v["post_spread_bps"],
           "spread_t": v["post_spread_t"]} for k, v in a.get("events", {}).items()]
    strat = []
    for src in (a.get("strategies", {}), (b or {}).get("strategies", {})):
        for k, v in src.items():
            if any(s["name"] == k for s in strat):
                continue
            strat.append({"name": k, "vs": v["lag1_25"]["vs_bench_ann_%"], "lag2": v["lag2_25"]["vs_bench_ann_%"],
                          "c50": v["lag1_50"]["vs_bench_ann_%"], "sharpe": v["lag1_25"]["sharpe"],
                          "dd": v["lag1_25"]["max_dd_%"], "n": v["lag1_25"]["avg_bonds"]})
    paired = {k: {"diff": v["diff_ann_%"], "t": v["t"], "p_holm": v.get("p_holm")} for k, v in a.get("paired_vs_p4", {}).items()}
    abl = [{"name": k, "ic": v["IC"], "vs": v["lag1_25"], "lag2": v["lag2_25"], "c50": v["lag1_50"], "sharpe": v["sharpe"]}
           for k, v in a.get("ablation", {}).items()]
    pl = a.get("placebo") or {}
    rp = (b or {}).get("regulator_placebo") or {}
    return {"events": ev, "strategies": strat, "paired": paired, "ablation": abl,
            "com_placebo": {"real_rule": pl.get("real_rule"), "placebo_mean": float(sum(pl.get("placebo_rule", [0])) /
                            max(len(pl.get("placebo_rule", [])), 1)), "pctile": pl.get("rule_pctile"),
                            "real_ic": pl.get("real_ic"), "ic_pctile": pl.get("ic_pctile")},
            "reg_placebo": {"real": rp.get("real"), "shifted_mean": float(sum(rp.get("shifted", [0])) /
                            max(len(rp.get("shifted", [])), 1))},
            "coverage": a.get("coverage", {})}


def build_payload() -> dict:
    held, watched = portfolio.position_rows()
    summ = portfolio.summary(held)

    def prow(r: dict) -> dict:
        return {k: _clean(v) for k, v in {
            "name": r["name"], "isin": r["isin"], "kind": r["kind"], "index": r["index_text"],
            "maturity": r["maturity"], "duration": r["duration"], "spread": r["spread"],
            "label": r["spread_label"], "yield": r["yield"], "pct_par": r["pct_par"], "price": r["price"],
            "d1": r["chg"]["d1"], "d5": r["chg"]["d5"], "d20": r["chg"]["d20"], "hist_pct": r["hist"]["pct"],
            "hist_z": r["hist"]["z"], "hist_n": r["hist"]["n"], "fair": r["fair"], "fair_price": r["fair_price"],
            "upside": r["upside_pct"], "verdict": r["verdict"], "action": r["action"], "stock": r["stock"],
            "stock_close": r["stock_close"], "mv": r["mv_brl"], "weight": r.get("weight"), "pnl": r["pnl_pct"],
            "cs01": r["cs01"], "carry": r["carry_brl"], "quantity": r["quantity"],
            "alert": (r["last_alert"] or {}).get("message"), "alert_high": (r["last_alert"] or {}).get("severity") == "high",
            "blocks": r["signal"].blocked_by if r["signal"] else [],
            "series": _series(r["isin"], r["spread_label"]),
        }.items()}

    scr = db.load_screener() or {"rows": []}
    keep = ["rank", "codigo", "isin", "nome", "indice", "peer", "vencimento", "duration", "taxa_indicativa",
            "spread_bps", "fair_bps", "resid_bps", "chg_5d_bps", "pu", "fair_pu", "upside_pct", "verdict",
            "strategy", "press_neg_30d", "press_titles", "model_pct", "ml_pred_bps", "flags", "incentivada",
            "pct_pu_par", "group"]
    opp = [{k: _clean(r.get(k)) for k in keep} for r in scr["rows"]]
    for r in opp:
        r["nome"] = (r["nome"] or "").replace(" (*)", "").replace(" (**)", "").strip()

    def load(name):
        p = ROOT / "research" / "out" / name
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

    alerts = [{"ts": a["ts"][:16].replace("T", " "), "msg": a["message"], "high": a["severity"] == "high"}
              for a in db.alerts(limit=25)]
    return {
        "generated": datetime.now(ZoneInfo("America/Sao_Paulo")).strftime("%d/%m/%Y %H:%M"),
        "auto": bool(os.getenv("GITHUB_ACTIONS")), "anbima_date": scr.get("date"),
        "regime": live.cached_regime(), "summary": {k: _clean(v) for k, v in summ.items() if not isinstance(v, list)},
        "by_issuer": summ.get("by_issuer", []), "ladder": summ.get("ladder", []), "stress": summ.get("stress", []),
        "held": [prow(r) for r in held], "watched": [prow(r) for r in watched], "alerts": alerts,
        "opp": opp, "universe": scr.get("universe"), "ranked": scr.get("ranked"), "c3_n": scr.get("c3_n"),
        "rich_n": scr.get("rich_n"), "regime_on": scr.get("regime_on"),
        "lab": load("lab_summary.json"), "timing": load("timing_results.json"),
        "selection": load("selection_results.json"), "lab3": _lab3(load("lab3_results.json"), load("lab3b_results.json")),
        "img": {"lab": _img("lab_final_equity_curves.png"), "timing": _img("timing_equity_curves.png"),
                "selection": _img("selection_equity_curves.png"), "ic": _img("selection_feature_ic.png"),
                "events": _img("lab3_event_studies.png")},
    }


def export(path: Path | None = None) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = path or OUT / "renda-fixa-monitor.html"
    tpl = (Path(__file__).parent / "templates" / "snapshot.html").read_text(encoding="utf-8")
    data = json.dumps(build_payload(), ensure_ascii=False, default=str).replace("</", "<\\/")
    path.write_text(tpl.replace("/*__DATA__*/null", data), encoding="utf-8")
    return path
