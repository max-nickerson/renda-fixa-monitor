"""B3 walk-forward factor combo, long-only top tercile vs BOVA11 (control), monthly.

Reuses research/ibkr_lab/b3_factors/factors.py: universe = top-60 B3 stocks by 63d median traded value (one line per
root, >= 273 days of history), factor z-scores from signals_at() (MOM 12-1, LOWVOL, STREV, QUAL, CREDIT) at the latest
close in the IBKR store, combined with the walk-forward weights of the last rebalance in b3_factors/results.json
(max(0, trailing 36m L/S Sharpe) per factor; re-run factors.py monthly to refresh them). Long the top third, equal
weight, 250k / k per name (k ~ 20 -> ~12.5k each). Benchmark for the verdict: BOVA11 total return over the same days
(key 'bench' in the README's evaluation); BOVA11 is NOT held.
"""
from __future__ import annotations

import json
from datetime import date

from research.ibkr_lab.runner.strategies import _common as C

NAME = "b3_factor_combo"
DESCRIPTION = "B3 walk-forward factor combo (MOM/LOWVOL/STREV/QUAL/CREDIT) long-only top tercile, EW, vs BOVA11"
REBALANCE = "monthly"
EXPECTED = {"ann_return": 10.5, "vol": 16.6, "sharpe": -0.05, "status": "control",
            "note": "b3_factors WF combo LO 2022-26: 10.5%/yr, 16.6% vol, Sharpe over CDI -0.05; vs Ibov/BOVA11 "
                    "active -2.4%/yr, IR -0.30 (t -0.55), beta 0.87. L/S placebo p 0.07. Judge: active return vs BOVA11."}


def signal():
    import pandas as pd
    from research.ibkr_lab.b3_factors import factors as F
    px, ret, adv, bova, _ = F.load_prices()
    fund = F.fundamentals()
    cred = F.credit_panel(fund)
    nhist = ret.notna().cumsum()
    t = ret.index[-1]
    elig = [c for c in ret.columns if nhist.loc[t, c] >= 273 and ret[c].iloc[-5:].notna().all()]
    a = adv.loc[t, elig].dropna().sort_values(ascending=False)
    seen, cols = set(), []
    for c in a.index:
        if c[:4] not in seen:
            seen.add(c[:4]); cols.append(c)
    cols = cols[:F.TOPN]
    z = F.signals_at(t, ret, cols, fund, cred).apply(F.zs)
    res = json.loads((F.OUT / "results.json").read_text(encoding="utf-8"))
    wd, w = list(res["wf_weights"].items())[-1]
    w = pd.Series(w)
    w = w[w > 0]
    zz = z[w.index]
    sig = (zz.fillna(0) * w).sum(axis=1).where(zz.notna().any(axis=1)).dropna()
    k = max(int(round(len(sig) / 3)), 4)
    return list(sig.sort_values(ascending=False).index[:k]), str(t.date()), wd


def targets(ib, asof: date):
    from research.ibkr_lab.setup import lib
    names, t, wd = signal()
    cs = lib.b3_stocks(ib, names)
    names = [n for n in names if n in cs]
    each = min(C.CAPITAL / max(len(names), 1), C.POS_CAP)
    return [{"contract": cs[n], "key": f"B3:{n}", "target_notional_usd": each, "side": "LONG",
             "note": f"rank {i + 1}/{len(names)} data={t} wf_weights={wd}"} for i, n in enumerate(names)]
