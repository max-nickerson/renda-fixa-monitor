"""Check the 'IPCA+ yield beta to NTN-B ~0.45' insight: horizon dependence (stale marks) and pre-2026 only."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.nightly import harness as H

OUT = Path(__file__).resolve().parent
g = pd.read_pickle(H.HIST / "lab_daily.pkl")
g = g[(g["kind"] == "IPCA") & g["fresh"]][["codigo", "date", "ratio", "dur", "contract", "bench_rate"]]
g = g[g["date"] < H.HOLDOUT].drop_duplicates(["codigo", "date"]).sort_values(["codigo", "date"])
g["y"] = ((1 + g["contract"] / 100) * g["ratio"] ** (-1 / g["dur"]) - 1) * 100
R = {}


def ols(y, X):
    X = np.c_[np.ones(len(y)), X]
    c, *_ = np.linalg.lstsq(X, y, rcond=None)
    return c[1:].round(3).tolist()


grp = g.groupby("codigo")
g["dy"], g["db"] = grp["y"].diff(), grp["bench_rate"].diff()
g["gap"] = grp["date"].diff().dt.days
g["db_prev"] = grp["db"].shift(1)
x = g[(g["gap"] <= 14) & g["dy"].abs().lt(1.5) & g["db"].notna()]
R["pooled_tt_gap14"] = {"beta": ols(x["dy"].to_numpy(), x[["db"]].to_numpy()), "n": len(x)}
xx = x.dropna(subset=["db_prev"])
R["pooled_tt_gap14_with_lag"] = {"beta_db_dbprev": ols(xx["dy"].to_numpy(), xx[["db", "db_prev"]].to_numpy()), "n": len(xx)}
# longer horizons: sample each bond's fresh trades at month ends (last trade of month), month-to-month changes
for freq, lab in (("ME", "monthly"), ("QE", "quarterly")):
    s = g.set_index("date").groupby("codigo")[["y", "bench_rate"]].resample(freq).last().dropna()
    d = s.groupby(level=0).diff().dropna()
    d = d[d["y"].abs() < 3]
    R[f"pooled_{lab}_changes"] = {"beta": ols(d["y"].to_numpy(), d[["bench_rate"]].to_numpy()), "n": len(d)}
# by year, trade-to-trade vs monthly
for yr in range(2021, 2026):
    a = x[x["date"].dt.year == yr]
    R[f"tt_{yr}"] = ols(a["dy"].to_numpy(), a[["db"]].to_numpy())
print(json.dumps(R, indent=1))
(OUT / "verify_beta_results.json").write_text(json.dumps(R, indent=1, default=float))
