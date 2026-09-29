"""Data for the hedging & overlays study (cached under data/history/nightly/hedging_overlays/).

Point-in-time rules
- B3 DI x Pre curve (TaxaSwap file of day d): published the evening of d -> usable for a hedge traded at the d+1 mark.
- NTN-B real curve (Tesouro Direto closing rates of day d): same (published at/after the close of d).
- EMBI+ Brazil spread (JPM, via Ipeadata JPM366_EMBI366): end-of-day value of d, usable from d+1.
- Stock/index closes (brapi / equity_daily): close of d, usable from d+1 (the harness overlay already lags 1 day).
All hedge P&Ls below are computed with position sizes known at the close before the return they earn.
"""
from __future__ import annotations

import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
CACHE = ROOT / "data" / "history" / "nightly" / "hedging_overlays"
CACHE.mkdir(parents=True, exist_ok=True)
TENORS = [126, 252, 504, 756, 1260, 1764, 2520]


def pre_curve() -> pd.DataFrame:
    """DI x Pre (B3 TaxaSwap 'PRE') at standard tenors (bdays) per date, from the cached B3 daily files."""
    path = CACHE / "pre_curve.pkl"
    if path.exists():
        return pd.read_pickle(path)
    rows = {}
    for f in sorted(glob.glob(str(ROOT / "data" / "history" / "b3" / "*.csv"))):
        try:
            df = pd.read_csv(f)
        except Exception:
            continue
        g = df[df["curve"] == "PRE"].sort_values("du")
        if len(g) < 3:
            continue
        d = pd.Timestamp(os.path.basename(f)[:8])
        rows[d] = {f"PRE_{t}": float(np.interp(t, g["du"], g["rate"])) for t in TENORS}
    out = pd.DataFrame.from_dict(rows, orient="index").sort_index()
    out.to_pickle(path)
    return out


def real_curve() -> pd.DataFrame:
    """NTN-B real yields at standard tenors (Tesouro Direto), columns DIC_<du> (repo convention)."""
    path = CACHE / "real_curve.pkl"
    if path.exists():
        return pd.read_pickle(path)
    from rfmonitor.history import ntnb_panel
    out = ntnb_panel(pd.Timestamp("2020-12-01"), pd.Timestamp("2026-09-30"))
    out.to_pickle(path)
    return out


def embi() -> pd.Series:
    """EMBI+ Brazil sovereign spread (bps), daily, from Ipeadata (JPM366_EMBI366)."""
    path = CACHE / "embi_br.pkl"
    if path.exists():
        return pd.read_pickle(path)
    import httpx
    r = httpx.get("http://www.ipeadata.gov.br/api/odata4/ValoresSerie(SERCODIGO='JPM366_EMBI366')", timeout=120,
                  follow_redirects=True)
    r.raise_for_status()
    v = pd.DataFrame(r.json()["value"])
    s = pd.Series(v["VALVALOR"].astype(float).to_numpy(),
                  index=pd.to_datetime(v["VALDATA"].str[:10])).dropna().sort_index()
    s = s[~s.index.duplicated(keep="last")]
    s.to_pickle(path)
    return s


def stocks() -> pd.DataFrame:
    """Closes (adjusted where available) of hedge candidates: IBOV, SMAL11 (HY-lab cache), bank basket
    (ITUB4, BBDC4, BBAS3, SANB11, BPAC11), USDBRL not needed. brapi used only for tickers missing locally."""
    path = CACHE / "stocks.pkl"
    if path.exists():
        return pd.read_pickle(path)
    out = {}
    hy = pd.read_pickle(ROOT / "research" / "out" / "hy_indices.pkl")
    out["IBOV"] = hy["^BVSP"]
    out["SMAL11"] = hy["SMAL11"]
    eq = pd.read_pickle(ROOT / "data" / "history" / "equity_daily.pkl")
    eq["date"] = pd.to_datetime(eq["date"])
    for t in ("ITUB4", "BPAC11"):
        g = eq[eq["ticker"] == t].sort_values("date").drop_duplicates("date")
        if len(g):
            out[t] = g.set_index("date")["adj_close"].astype(float)
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    from rfmonitor.sources import brapi
    for t in ("BBDC4", "BBAS3", "SANB11"):
        try:
            h = brapi.history(t, "10y")
            out[t] = pd.Series({pd.Timestamp(d): c for d, c, _ in h}).sort_index()
        except Exception as e:  # pragma: no cover
            print(f"[data] {t} unavailable: {type(e).__name__}")
    df = pd.DataFrame(out).sort_index()
    df.to_pickle(path)
    return df


if __name__ == "__main__":
    for nm, f in (("pre", pre_curve), ("real", real_curve), ("embi", embi), ("stocks", stocks)):
        x = f()
        print(nm, x.shape, x.index.min(), x.index.max())
        print(x.tail(2))
