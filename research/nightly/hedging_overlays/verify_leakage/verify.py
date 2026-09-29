"""Adversarial leakage verification of hedging_overlays' E_gate_ibov claim (and quick checks of the hedge-ratio insight).
Writes only into this folder. Pre-2026 for all choices; holdout numbers only re-reported (not used for choices)."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from research.nightly import harness as H
from research.nightly.hedging_overlays import data as D

OUT = Path(__file__).resolve().parent
T0 = time.time()
R = {}


def log(*a):
    print(f"[{time.time()-T0:5.0f}s]", *a, flush=True)


dd = H.days()
CDI = H.cdi_daily()
ST = D.stocks()


def on_grid(s, limit=5):
    return s.reindex(s.index.union(dd)).ffill(limit=limit).reindex(dd)


lv = on_grid(ST["IBOV"])
xib = (lv.pct_change().shift(-1) - CDI).fillna(0.0)       # t->t+1 booked at t (as in run.py)


def ibx(win=63, extra_lag=0):
    return xib.shift(1 + extra_lag).rolling(win).sum()


def gate(fn):
    def sig(x):
        d_ = x["day"].iloc[0]
        return np.zeros(len(x), bool) if fn(d_) else x["p4q"].to_numpy()
    return sig


def st(r, b):
    s = H.stats(r["daily"], bench=b["daily"])
    return {k: round(float(s[k]), 3) for k in ("ann_excess_%", "diff_ann_%", "diff_t_nw", "max_dd_%", "worst_month_%",
                                               "vol_%", "diff_h1_%", "diff_h2_%")}


B25, B50 = H.baseline("P4Q", cost_bps=25), H.baseline("P4Q", cost_bps=50)
Pm = H.load_panel("M")
decs = sorted(Pm["day"].unique())
decs = [pd.Timestamp(d) for d in decs if H.START <= pd.Timestamp(d) < H.HOLDOUT]
R["decisions_on_grid"] = bool(all(d in dd for d in decs))
x0 = ibx()
R["base"] = st(B25, B25)

# 1) reproduction
fn0 = lambda d_: x0.get(d_, 0) < -0.10
g25 = H.backtest(gate(fn0), cost_bps=25)
g50 = H.backtest(gate(fn0), cost_bps=50)
R["repro_25"] = st(g25, B25)
R["repro_50"] = st(g50, B50)
R["gated"] = [str(d.date()) for d in decs if fn0(d)]
R["ibx_at_decisions"] = {str(d.date()): round(float(x0.get(d, np.nan)), 4) for d in decs}
log("repro", R["repro_25"], R["gated"])

# 2) timing: extra lag 1 and 5 days; compounded vs summed
for lag in (1, 5):
    xl = ibx(extra_lag=lag)
    f = lambda d_, xl=xl: xl.get(d_, 0) < -0.10
    R[f"extra_lag_{lag}"] = {**st(H.backtest(gate(f)), B25), "gated": [str(d.date()) for d in decs if f(d)]}
    log("lag", lag, R[f"extra_lag_{lag}"])

# 3) parameter sensitivity (the 63d / -10% choice)
grid = {}
for win in (42, 63, 126):
    xw = ibx(win)
    for th in (-0.05, -0.075, -0.10, -0.125, -0.15):
        f = lambda d_, xw=xw, th=th: xw.get(d_, 0) < th
        ng = sum(bool(f(d)) for d in decs)
        if ng == 0:
            grid[f"{win}_{th}"] = {"n_gated": 0}
            continue
        grid[f"{win}_{th}"] = {**st(H.backtest(gate(f)), B25), "n_gated": ng}
        log("grid", win, th, grid[f"{win}_{th}"])
R["grid"] = grid

# 4) bigger placebo: 300 random 4-gate sets; also leave-one-gate-out (which gate drives maxDD)
rng = np.random.default_rng(123)
act = R["repro_25"]
pl = []
for i in range(300):
    pick = set(rng.choice(np.array(decs, dtype=object), size=len(R["gated"]), replace=False))
    s = H.stats(H.backtest(gate(lambda d_, pk=pick: d_ in pk))["daily"], bench=B25["daily"])
    pl.append((s["diff_ann_%"], s["max_dd_%"]))
pl = np.array(pl)
R["placebo300"] = {"share_maxDD_ge_actual": float((pl[:, 1] >= act["max_dd_%"]).mean()),
                   "share_diff_ge_actual": float((pl[:, 0] >= act["diff_ann_%"]).mean()),
                   "maxDD_mean": float(pl[:, 1].mean()), "diff_mean": float(pl[:, 0].mean())}
log("placebo", R["placebo300"])
loo = {}
gs = [pd.Timestamp(g) for g in R["gated"]]
for g in gs:
    keep = set(gs) - {g}
    loo[str(g.date())] = st(H.backtest(gate(lambda d_, k=keep: d_ in k)), B25)
for g in gs:
    loo["only_" + str(g.date())] = st(H.backtest(gate(lambda d_, g=g: d_ == g)), B25)
R["leave_one_out"] = loo
log("loo", loo)

# 5) where is P4Q's maxDD
m = H.monthly(B25["daily"][B25["daily"].index < H.HOLDOUT])
nav = (1 + m).cumprod()
ddm = nav / nav.cummax() - 1
R["p4q_maxdd_trough_month"] = str(ddm.idxmin().date())
R["p4q_maxdd_peak_month"] = str(nav[:ddm.idxmin()].idxmax().date())
mg = H.monthly(g25["daily"][g25["daily"].index < H.HOLDOUT])
navg = (1 + mg).cumprod(); ddg = navg / navg.cummax() - 1
R["gate_maxdd_trough_month"] = str(ddg.idxmin().date())

# 6) holdout: which 2026 decisions gated (holdout already spent by author; recheck only)
Ph = H.load_panel("M", holdout=True)
hd = [pd.Timestamp(d) for d in sorted(Ph["day"].unique()) if pd.Timestamp(d) >= H.HOLDOUT]
R["holdout_decisions_gated"] = [str(d.date()) for d in hd if x0.get(d, 0) < -0.10]
bh = H.baseline("P4Q", holdout=True)
gh = H.backtest(gate(fn0), holdout=True)
s = H.stats(gh["daily"], bench=bh["daily"], holdout="only")
R["holdout_recheck"] = {k: float(s[k]) for k in ("ann_excess_%", "diff_ann_%", "diff_t_nw", "max_dd_%")}
dif = (gh["daily"] - bh["daily"]).reindex(dd).fillna(0)
R["holdout_diff_by_month_bps"] = {str(k.date()): round(float(v) * 1e4, 2) for k, v in
                                  dif[dif.index >= H.HOLDOUT].groupby(pd.Grouper(freq="MS")).sum().items()}
R["pre_diff_by_month_bps_2025H2"] = {str(k.date()): round(float(v) * 1e4, 2) for k, v in
                                     dif[(dif.index >= "2025-06-01") & (dif.index < H.HOLDOUT)].groupby(pd.Grouper(freq="MS")).sum().items()}
log("holdout", R["holdout_decisions_gated"], R["holdout_recheck"], R["holdout_diff_by_month_bps"])

(OUT / "verify_results.json").write_text(json.dumps(R, indent=1, default=float))
log("done")
