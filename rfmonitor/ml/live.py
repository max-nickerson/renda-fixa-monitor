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
