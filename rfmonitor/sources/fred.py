"""US Treasury constant-maturity yields from FRED (no key needed) — benchmark for eurobond spreads."""
from __future__ import annotations

import io
from functools import lru_cache

import pandas as pd

from ..http import cached_bytes

TENORS = {"DGS1": 1, "DGS2": 2, "DGS3": 3, "DGS5": 5, "DGS7": 7, "DGS10": 10, "DGS20": 20, "DGS30": 30}


@lru_cache(maxsize=1)
def ust_curve() -> pd.DataFrame:
    """Daily UST curve, columns = tenor in years, values in %."""
    frames = []
    for sid, yrs in TENORS.items():
        raw = cached_bytes(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}", max_age_hours=12,
                           name=f"fred_{sid}.csv")
        if not raw:
            continue
        s = pd.read_csv(io.BytesIO(raw))
        s.columns = ["date", "v"]
        s["v"] = pd.to_numeric(s["v"], errors="coerce")
        frames.append(s.set_index(pd.to_datetime(s["date"]))["v"].rename(yrs))
    return pd.concat(frames, axis=1).sort_index().ffill() if frames else pd.DataFrame()
