"""Capacity curve (pre-2026): top-25-carry-in-P4+Q made liquidity-feasible at different fund sizes (25% of the bond's
91-day SND R$ volume per position; extend down the carry ranking before relaxing). Diagnostic sweep, not a candidate."""
import json
from pathlib import Path
from research.nightly import harness as H
from research.nightly.portfolio_construction import run as RUN, build_weights as BW

out = {}
bq = H.baseline("P4Q")["daily"]
for aum in (50e6, 100e6, 250e6, 500e6, 1e9, 2e9):
    BW.AUM = aum
    w = RUN.topk_carry(fund=True)
    r = H.backtest(w, freq="M", hold=126, issuer_cap=1.0)
    s = H.stats(r["daily"], bench=bq)
    out[f"{aum/1e6:.0f}m"] = {"exCDI_%": round(s["ann_excess_%"], 3), "vs_P4Q_%": round(s["diff_ann_%"], 3),
                              "t": round(s["diff_t_nw"], 2), "n_avg": round(r["n_avg"], 1)}
    print(aum, out[f"{aum/1e6:.0f}m"], flush=True)
json.dump(out, open(Path(__file__).resolve().parent / "capacity.json", "w"), indent=1)
