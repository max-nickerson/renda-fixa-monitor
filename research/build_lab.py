"""Build the weekly point-in-time lab dataset (data/history/lab_weekly.pkl)."""
import sys as _sys
from pathlib import Path as _P
_sys.path.insert(0, str(_P(__file__).resolve().parent.parent))  # run from anywhere

import time
import warnings

from rfmonitor.config import DATA_DIR
from rfmonitor.ml import lab

warnings.filterwarnings("ignore")
t = time.time()
g = lab.build()
g.to_pickle(DATA_DIR / "history" / "lab_weekly.pkl")
print(g.shape, f"{time.time() - t:.0f}s", g["week"].min().date(), g["week"].max().date())
print("fresh marks/week:", int(g[g["fresh"]].groupby("week").size().median()),
      "eligible/week:", int(g[g["eligible"]].groupby("week").size().median()))
print("rows with issuer news coverage (any CVM filing in 90d):",
      f"{(g[[c for c in g.columns if c.endswith('_90d') and c.startswith('n_')]].sum(axis=1) > 0).mean():.0%}")
print(g[["ret", "resid_z", "own_z", "d_spread_4w", "n_distress_90d", "n_fact_30d", "mkt_distress_z"]].describe().round(3).T)
