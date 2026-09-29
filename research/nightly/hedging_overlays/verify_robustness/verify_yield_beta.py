"""Check the 'harness over-hedges IPCA+' insight: does the yield beta of IPCA+ debentures to NTN-B depend on the
horizon over which it is measured? (Short trade-to-trade gaps attenuate beta if marks are stale/sticky.)
Pre-2026 trades only. Same yield construction as hedge_ratio.py."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.nightly import harness as H

OUT = Path(__file__).resolve().parent
g = pd.read_pickle(H.HIST / "lab_daily.pkl")
g = g[(g["kind"] == "IPCA") & g["fresh"]][["codigo", "date", "ratio", "dur", "contract", "bench_rate"]]
g = g[g["date"] < H.HOLDOUT].drop_duplicates(["codigo", "date"]).sort_values(["codigo", "date"]).reset_index(drop=True)
g["y"] = ((1 + g["contract"] / 100) * g["ratio"] ** (-1 / g["dur"]) - 1) * 100
print(len(g), g["codigo"].nunique(), flush=True)

res = {}
for hz, (lo, hi) in {"1-14d": (1, 14), "25-35d": (25, 35), "80-100d": (80, 100), "170-190d": (170, 190)}.items():
    rows = []
    for cod, gg in g.groupby("codigo", sort=False):
        d = gg["date"].to_numpy()
        if len(d) < 2:
            continue
        y, bnc = gg["y"].to_numpy(), gg["bench_rate"].to_numpy()
        tgt = d + np.timedelta64(int((lo + hi) / 2), "D")
        j = np.searchsorted(d, tgt)
        for jj in (j,):
            ok = jj < len(d)
            idx = np.where(ok)[0]
            jj2 = jj[idx]
            gap = (d[jj2] - d[idx]).astype("timedelta64[D]").astype(int)
            m = (gap >= lo) & (gap <= hi)
            if lo == 1:   # consecutive trades
                idx = np.arange(len(d) - 1); jj2 = idx + 1
                gap = (d[jj2] - d[idx]).astype("timedelta64[D]").astype(int); m = (gap >= lo) & (gap <= hi)
            # non-overlapping sampling would be better; we keep all and use per-bond cluster SE
            rows.append(pd.DataFrame({"cod": cod, "t": d[idx][m], "dy": (y[jj2] - y[idx])[m], "db": (bnc[jj2] - bnc[idx])[m]}))
    x = pd.concat(rows)
    x = x[x["dy"].abs() < 3 * (1 + (hi > 30)) ]
    x = x.dropna()
    X = np.c_[np.ones(len(x)), x["db"]]
    beta = np.linalg.lstsq(X, x["dy"], rcond=None)[0][1]
    # cluster by month
    out = {"n": len(x), "beta_pooled": float(beta)}
    for per, (a, b_) in {"2021-23": ("2021-01-01", "2024-01-01"), "2024-25": ("2024-01-01", "2026-01-01")}.items():
        w = x[(x["t"] >= a) & (x["t"] < b_)]
        Xw = np.c_[np.ones(len(w)), w["db"]]
        out[f"beta_{per}"] = float(np.linalg.lstsq(Xw, w["dy"], rcond=None)[0][1])
    # robust: median of monthly cross-bond regressions
    res[hz] = out
    print(hz, out, flush=True)
json.dump(res, open(OUT / "verify_yield_beta_results.json", "w"), indent=1)
