"""Model state used by the dashboard: credit-regime (Study A) refreshed at most once a day, cached on disk."""
from __future__ import annotations

import json
import logging
from datetime import date

from ..config import DATA_DIR

log = logging.getLogger(__name__)
STATE = DATA_DIR / "models"


def selection(refresh: bool = False) -> dict | None:
    """Daily bond-selection model (Study B): {codigo: {ml_pred_bps, blend_pct, ...}} cached on disk."""
    STATE.mkdir(parents=True, exist_ok=True)
    path = STATE / "selection.json"
    if path.exists() and not refresh:
        cached = json.loads(path.read_text())
        if cached.get("computed") == date.today().isoformat():
            return cached
    try:
        import pandas as pd
        from . import selection as sel
        from ..config import DATA_DIR as D
        panel = sel.build_panel()
        snap_path = D / "history" / "snapshots.pkl"
        snap = pd.read_pickle(snap_path) if snap_path.exists() else None
        # Rebuild the labelled history at most once a month (labels only change when a month completes).
        if snap is None or snap["rebalance"].max() < pd.Timestamp.today() - pd.offsets.BMonthBegin(1):
            snap = sel.snapshots(panel)
            snap.to_pickle(snap_path)
        cur = sel.live_scores(snap, panel)
        out = {"computed": date.today().isoformat(), "n": int(len(cur)), "coefs": cur.attrs.get("coefs", {}),
               "rows": {r.codigo: {"ml_pred_bps": round(float(r.ml_pred) * 1e4, 1),
                                   "blend_pct": None if pd.isna(r.blend_pct) else round(float(r.blend_pct), 3),
                                   "trade_cdi_bps": round(float(r.cdi_bps), 1), "last_trade": str(r.date.date()),
                                   "eligible": bool(r.eligible)}
                        for r in cur.itertuples()}}
        path.write_text(json.dumps(out))
        return out
    except Exception:
        log.exception("selection model failed")
        return json.loads(path.read_text()) if path.exists() else None


def press_now(codes: list[str], refresh: bool = False) -> dict:
    """Negative press in the last 30 days per bond code (Google News, same query as the backtest).
    {codigo: {"brand": .., "n": count, "titles": [...]}} cached for the day."""
    STATE.mkdir(parents=True, exist_ok=True)
    path = STATE / "press_now.json"
    cached = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    if cached.get("computed") != date.today().isoformat() or refresh:
        cached = {"computed": date.today().isoformat(), "brands": {}, "codes": {}}
    import re
    import time
    from datetime import timedelta
    from . import press
    from .selection import reference
    from ..config import ROOT
    # Same universe as the backtest (P4): only the issuers whose press history was collected (top 250 by
    # trading, research/out/press_brands.json). Others are not filtered — exactly as in the test.
    covered_path = ROOT / "research" / "out" / "press_brands.json"
    covered = json.loads(covered_path.read_text(encoding="utf-8")) if covered_path.exists() else {}
    ref = reference().drop_duplicates("codigo").set_index("codigo")
    today = date.today()
    for code in codes:
        if code in cached["codes"] or code not in ref.index:
            continue
        cnpj8 = re.sub(r"\D", "", str(ref.at[code, "cnpj"]))[:8]
        b = covered.get(cnpj8)
        if not b:
            continue
        if b not in cached["brands"]:
            try:
                its = press._fetch(press.neg_query(b), today - timedelta(days=31), today + timedelta(days=1))
                bn = press._norm(b)
                its = [i for i in its if bn in press._norm(i["title"])
                       and i["date"] >= (today - timedelta(days=30)).isoformat()]
                cached["brands"][b] = {"n": len(its), "titles": [i["title"] for i in its[:5]]}
            except Exception as e:
                log.warning("press %s: %s", b, e)
                continue
            time.sleep(0.7)
        cached["codes"][code] = {"brand": b, **cached["brands"][b]}
    path.write_text(json.dumps(cached, ensure_ascii=False), encoding="utf-8")
    return cached["codes"]


def cached_press() -> dict:
    path = STATE / "press_now.json"
    return json.loads(path.read_text(encoding="utf-8")).get("codes", {}) if path.exists() else {}


def cached_selection() -> dict | None:
    path = STATE / "selection.json"
    return json.loads(path.read_text()) if path.exists() else None


def cached_regime() -> dict | None:
    """Read-only for the web request path (the scheduler computes it)."""
    path = STATE / "regime.json"
    return json.loads(path.read_text()) if path.exists() else None


def regime(refresh: bool = False) -> dict | None:
    """Latest credit-timing view: DI credit via 21d momentum rule, IPCA credit via logistic model."""
    STATE.mkdir(parents=True, exist_ok=True)
    path = STATE / "regime.json"
    if path.exists() and not refresh:
        cached = json.loads(path.read_text())
        if cached.get("computed") == date.today().isoformat():
            return cached
    try:
        from . import timing
        out = {"computed": date.today().isoformat(), "series": {}}
        for name, df in timing.excess_series().items():
            res = timing.run_one(name, df, only=("Logistic",))
            is_di = name.startswith("DI")
            model = "Momentum 21d > 0" if is_di else "Logistic"
            st = res.stats.loc[model]
            bh = res.stats.loc["Buy & hold crédito"]
            out["series"][name] = {
                "model": model, "position": res.latest["position"][model], "asof": res.latest["date"],
                "prob": res.latest["prob"].get("Logistic"),
                "mom_21_bps": res.latest["features"]["mom_21"] * 1e4,
                "oos": {"sharpe": round(float(st["sharpe"]), 2), "max_dd": round(float(st["max_dd_%"]), 1),
                        "bh_sharpe": round(float(bh["sharpe"]), 2), "bh_max_dd": round(float(bh["max_dd_%"]), 1)},
            }
        path.write_text(json.dumps(out, indent=2))
        return out
    except Exception:
        log.exception("regime model failed")
        return json.loads(path.read_text()) if path.exists() else None
