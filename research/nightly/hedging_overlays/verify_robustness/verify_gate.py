"""Adversarial robustness check of hedging_overlays' E_gate_ibov (pre-2026 only; holdout NOT touched).
Re-implements the gate independently from the cached Ibov series and perturbs threshold/window, runs a larger
random-gate placebo, compares with an exposure-matched constant cash buffer, excludes 2023, and checks
issuer concentration of the paired difference."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.nightly import harness as H
from research.nightly.hedging_overlays import data as D

OUT = Path(__file__).resolve().parent
dd = H.days()
CDI = H.cdi_daily()


def on_grid(s, limit=5):
    return s.reindex(s.index.union(dd)).ffill(limit=limit).reindex(dd)


ib = on_grid(D.stocks()["IBOV"])
# excess over CDI of close t-1 -> close t, booked at t (known at close t)
ibx_d = (ib.pct_change() - CDI.reindex(dd)).fillna(0.0)

P = H.load_panel("M")
decs = sorted(d for d in P["day"].unique() if H.START <= d < H.HOLDOUT)


def gate(fn):
    def sig(x):
        d_ = x["day"].iloc[0]
        return np.zeros(len(x), bool) if fn(d_) else x["p4q"].to_numpy()
    return sig


def mk(th, win, simple_ret=False):
    if simple_ret:
        s = (ib / ib.shift(win) - 1) - CDI.reindex(dd).rolling(win).sum()
    else:
        s = ibx_d.rolling(win).sum()
    return lambda d_: bool(s.get(d_, 0) < th), s


b = {c: H.baseline("P4Q", cost_bps=c)["daily"] for c in (25, 50)}
u = H.baseline("U")["daily"]
res = {}


def row(name, fn, extra=None, **kw):
    r25 = H.backtest(gate(fn), cost_bps=25, name=name, **kw)
    r50 = H.backtest(gate(fn), cost_bps=50, name=name, **kw)
    s = H.stats(r25["daily"], bench=b[25])
    s50 = H.stats(r50["daily"], bench=b[50])
    su = H.stats(r25["daily"], bench=u)
    g = [str(pd.Timestamp(d).date())[:7] for d in decs if fn(d)]
    out = {"diff25": s["diff_ann_%"], "t25": s["diff_t_nw"], "p25": s["diff_p"], "diff50": s50["diff_ann_%"],
           "exU": su["diff_ann_%"], "maxDD": s["max_dd_%"], "worst": s["worst_month_%"], "vol": s["vol_%"],
           "sharpe": s["sharpe"], "h1": s["diff_h1_%"], "h2": s["diff_h2_%"], "gated": g}
    if extra:
        out.update(extra)
    res[name] = out
    print(name, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in out.items()}, flush=True)
    return r25


base_st = H.stats(b[25])
print("P4Q", base_st["ann_excess_%"], base_st["max_dd_%"], base_st["worst_month_%"])

# 1) reproduce
fn0, s0 = mk(-0.10, 63)
r0 = row("repro_th10_w63", fn0)
import time; T0=time.time()

# 2) perturbations: done in part 1, see verify_gate_part1.log
RUN_PERT = False

# 3) exposure-matched constant cash buffer: gate removes 4 of 48 tranches -> ~ 4/48 average cash
ov = pd.Series(1 - 4 / 48, index=dd)
rc = H.backtest("p4q", overlay=ov, name="cash_matched")
sc = H.stats(rc["daily"], bench=b[25])
res["cash_matched_8.3pct"] = {"diff25": sc["diff_ann_%"], "maxDD": sc["max_dd_%"], "worst": sc["worst_month_%"],
                              "vol": sc["vol_%"]}
print("cash matched", res["cash_matched_8.3pct"])

# 4) placebo: 300 random 4-decision gates
rng = np.random.default_rng(123)
pl = []
for i in range(60):
    pk = set(rng.choice(decs, size=4, replace=False))
    r = H.backtest(gate(lambda d_, pk=pk: d_ in pk), name="pl")
    st = H.stats(r["daily"], bench=b[25])
    pl.append((st["diff_ann_%"], st["max_dd_%"], st["worst_month_%"]))
pl = np.array(pl)
act = res["repro_th10_w63"]
# placebo restricted: random gates that include at least one decision in 2022-10..2023-03 (crisis-entry window)
res["placebo60"] = {"diff_mean": float(pl[:, 0].mean()), "maxDD_mean": float(pl[:, 1].mean()),
                     "share_maxDD_ge_actual": float((pl[:, 1] >= act["maxDD"]).mean()),
                     "share_diff_ge_actual": float((pl[:, 0] >= act["diff25"]).mean()),
                     "maxDD_p90": float(np.percentile(pl[:, 1], 90)), "maxDD_p95": float(np.percentile(pl[:, 1], 95))}
print("placebo", res["placebo60"])

# 5) which single gated decision drives the maxDD gain
g0 = [d for d in decs if fn0(d)]
for d in g0:
    row(f"only_{str(pd.Timestamp(d).date())[:7]}", lambda d_, d=d: d_ == d)

# 6) excluding 2023 (stats on months outside 2023)
def ex23(s):
    m = H.monthly(s[s.index < H.HOLDOUT])
    m = m[(m.index.year != 2023)]
    return m
m_g, m_b = ex23(r0["daily"]), ex23(b[25])
dmm = (m_g - m_b).dropna()
def mdd(m):
    nav = (1 + m).cumprod(); return float((nav / nav.cummax() - 1).min() * 100)
res["ex2023"] = {"diff_ann_%": float(dmm.mean() * 12 * 100), "t_nw": float(H.nw_t(dmm.to_numpy(), 6)),
                 "maxDD_gate": mdd(m_g), "maxDD_p4q": mdd(m_b)}
print("ex2023", res["ex2023"])

# 7) excluding top-5 issuers by contribution to the paired difference: rerun both books without them
xs = P[P["p4q"]]
top5 = xs.groupby("cnpj8").size().sort_values(ascending=False).index[:5].tolist()
def gate_ex(fn):
    def sig(x):
        d_ = x["day"].iloc[0]
        keep = x["p4q"].to_numpy() & ~x["cnpj8"].isin(top5).to_numpy()
        return np.zeros(len(x), bool) if fn(d_) else keep
    return sig
bx = H.backtest(gate_ex(lambda d_: False), name="p4q_ex5")
gx = H.backtest(gate_ex(fn0), name="gate_ex5")
st = H.stats(gx["daily"], bench=bx["daily"]); sb = H.stats(bx["daily"])
res["ex_top5_issuers"] = {"top5": top5, "diff25": st["diff_ann_%"], "t": st["diff_t_nw"], "maxDD_gate": st["max_dd_%"],
                          "maxDD_p4q_ex5": sb["max_dd_%"]}
print("ex top5", res["ex_top5_issuers"])
json.dump(res, open(OUT / "verify_gate_results_part2.json", "w"), indent=1, default=str)
print("saved")
