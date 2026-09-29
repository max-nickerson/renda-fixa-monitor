"""PIT liquidity dataset from SND trades (qty x pu_avg, R$), for capacity / liquidity constraints.

Output: data/history/nightly/portfolio_construction/liquidity.pkl  columns
  codigo, day (decision day on the harness panel grid), vol91_brl (R$ traded over the 91 calendar days <= day),
  tdays91 (days with a trade in that window), adv_brl = vol91_brl / 63 (average daily R$ volume per bday).
PIT rule: only SND trade dates <= decision day (the decision is taken at that day's close, trades are published
same evening by SND). No reference/maturity info is used.
"""
from __future__ import annotations
import glob
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "data" / "history" / "nightly" / "portfolio_construction"
OUT.mkdir(parents=True, exist_ok=True)


def build(force: bool = False) -> pd.DataFrame:
    path = OUT / "liquidity.pkl"
    if path.exists() and not force:
        return pd.read_pickle(path)
    fs = sorted(glob.glob(str(ROOT / "data" / "history" / "snd_trades_*.csv.gz")))
    parts = []
    for f in fs:
        d = pd.read_csv(f, usecols=["date", "codigo", "qty", "pu_avg"])
        d["vol"] = d["qty"] * d["pu_avg"]
        parts.append(d.groupby(["codigo", "date"], as_index=False)["vol"].sum())
    T = pd.concat(parts)
    T["date"] = pd.to_datetime(T["date"])
    T = T.groupby(["codigo", "date"], as_index=False)["vol"].sum()
    from research.nightly import harness as H
    dec = sorted(set(H.load_panel("M", holdout=True)["day"]) | set(H.load_panel("W", holdout=True)["day"]))
    dec = pd.DatetimeIndex(dec)
    rows = []
    for c, g in T.groupby("codigo"):
        g = g.set_index("date")["vol"].sort_index()
        cs = g.cumsum()
        cn = pd.Series(np.arange(1, len(g) + 1), index=g.index)
        # value at d = cum up to d minus cum up to d-91
        i1 = cs.index.searchsorted(dec, side="right") - 1
        i0 = cs.index.searchsorted(dec - pd.Timedelta(days=91), side="right") - 1
        v1 = np.where(i1 >= 0, cs.to_numpy()[np.maximum(i1, 0)], 0.0)
        v0 = np.where(i0 >= 0, cs.to_numpy()[np.maximum(i0, 0)], 0.0)
        n1 = np.where(i1 >= 0, cn.to_numpy()[np.maximum(i1, 0)], 0)
        n0 = np.where(i0 >= 0, cn.to_numpy()[np.maximum(i0, 0)], 0)
        ok = (v1 - v0) > 0
        if ok.any():
            rows.append(pd.DataFrame({"codigo": c, "day": dec[ok], "vol91_brl": (v1 - v0)[ok],
                                      "tdays91": (n1 - n0)[ok]}))
    L = pd.concat(rows, ignore_index=True)
    L["adv_brl"] = L["vol91_brl"] / 63.0
    L.to_pickle(path)
    return L


if __name__ == "__main__":
    L = build(force=True)
    print(L.shape, L["day"].min(), L["day"].max())
    print(L["adv_brl"].describe())
