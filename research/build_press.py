"""Collect point-in-time negative-press history (Google News) for the most-traded issuers + market gauge,
and write weekly features keyed by (cnpj8, week) → data/history/press_weekly.pkl (input to run_lab2.py)."""
import sys as _sys
from pathlib import Path as _P
_sys.path.insert(0, str(_P(__file__).resolve().parent.parent))  # run from anywhere

import time
import warnings
from datetime import date

import pandas as pd

from rfmonitor.config import DATA_DIR
from rfmonitor.ml import press
from rfmonitor.ml.selection import reference

warnings.filterwarnings("ignore")
N = int(_sys.argv[1]) if len(_sys.argv) > 1 else 250
t0 = time.time()
g = pd.read_pickle(DATA_DIR / "history" / "lab_weekly.pkl")
ref = reference().drop_duplicates("codigo").set_index("codigo")
g["issuer"] = g["codigo"].map(ref["issuer"])
top = g[g["eligible"]].groupby("cnpj8").agg(rows=("codigo", "size"), issuer=("issuer", "first"))
top = top.sort_values("rows", ascending=False).head(N)
brands = {c: press.brand(i) for c, i in top["issuer"].items()}
print(f"{N} issuers → {len(set(brands.values()) - {''})} brands; coverage "
      f"{top['rows'].sum() / g[g['eligible']].shape[0]:.0%} of eligible bond-weeks", flush=True)

import json
(_P(__file__).parent / "out" / "press_brands.json").write_text(
    json.dumps({c: b for c, b in brands.items() if b}, ensure_ascii=False, indent=0), encoding="utf-8")
start, end = date(2020, 10, 1), date.today()
items = press.items_frame(brands, start, end, workers=3)
items.to_pickle(DATA_DIR / "history" / "press_items.pkl")
print(f"items {len(items)} ({time.time() - t0:.0f}s)", flush=True)
mkt = press.market_frame(start, end)
mkt.to_pickle(DATA_DIR / "history" / "press_market.pkl")
print(f"market items {len(mkt)} ({time.time() - t0:.0f}s)", flush=True)

keys = g[["cnpj8", "week"]].drop_duplicates().rename(columns={"week": "asof"}).reset_index(drop=True)
f = press.weekly_features(items, keys)
out = pd.concat([keys.rename(columns={"asof": "week"}), f], axis=1)
out["press_covered"] = out["cnpj8"].isin(set(brands))
mw = press.market_weekly(mkt, pd.DatetimeIndex(sorted(g["week"].unique())))
out = out.merge(mw, left_on="week", right_index=True, how="left")
out.to_pickle(DATA_DIR / "history" / "press_weekly.pkl")
print(out.describe().round(2).T.to_string())
print(f"done {time.time() - t0:.0f}s")
