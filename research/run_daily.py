"""Daily ('live') decisions vs weekly: speed decay and next-day event reactions.
Outputs research/out/daily_results.json, daily_*.png."""
import sys as _sys
from pathlib import Path as _P
_sys.path.insert(0, str(_P(__file__).resolve().parent.parent))  # run from anywhere

import json
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rfmonitor.config import DATA_DIR
from rfmonitor.ml import lab_daily as ld
from rfmonitor.ml.lab_daily import DRule

warnings.filterwarnings("ignore")
OUT = Path(__file__).parent / "out"
CACHE = DATA_DIR / "history" / "lab_daily.pkl"
t0 = time.time()
log = lambda *a: print(f"[{time.time() - t0:5.0f}s]", *a, flush=True)

if CACHE.exists() and "--rebuild" not in _sys.argv:
    g = pd.read_pickle(CACHE)
else:
    g = ld.build()
    g.to_pickle(CACHE)
log(f"daily grid {g.shape}, {g['day'].min().date()}..{g['day'].max().date()}, "
    f"eligible/day {int(g[g['eligible']].groupby('day').size().median())}")

nz = lambda x, d=0.0: d if x is None or x != x else x
rich = lambda r: nz(r["resid_z"]) <= -1.5
p4_in = lambda r: nz(r["cdi_pct"], 1) <= 0.3 and not rich(r) and nz(r["press_neg_30d"]) < 1
event = lambda r: (nz(r["rat_days_since_down"], 9999) <= 2 or nz(r["eq_ret_1w"]) <= -0.15
                   or nz(r["eq_ret_4w"]) <= -0.25 or nz(r["distress_2d"]) >= 1 or nz(r["press_neg_7d"]) >= 3)
recent_risk = lambda r: (nz(r["rat_days_since_down"], 9999) <= 180 or nz(r["eq_ret_4w"]) <= -0.15)
REGIME = (g.groupby("day")["mkt_mom_21"].first() > 0).astype(float)

U0 = DRule("U0", lambda r: True, lambda r: False)
P4w = DRule("P4 semanal", p4_in, rich, decide="weekly")
P4d = DRule("P4 diário", p4_in, rich)
P5d = DRule("P5 diário (bloqueia compra após evento)", lambda r: p4_in(r) and not recent_risk(r), rich)
P4ev = DRule("P4 diário + venda no dia seguinte a evento", p4_in, rich, event_exit=event, reentry_days=90)
P4wev = DRule("P4 semanal + venda no dia seguinte a evento (híbrido)", p4_in, rich, decide="weekly",
              event_exit=event, reentry_days=90)
U0ev = DRule("Universo + venda no dia seguinte a evento", lambda r: True, lambda r: False, event_exit=event,
             reentry_days=90)

res, curves, bench = {}, {}, {}


def go(label, rule, lag=1, cost=25, ecost=None, ov=None):
    key = (lag, cost)
    if key not in bench:
        bench[key] = ld.run(g, U0, lag=lag, cost_bps=cost)["daily"]
    r = ld.run(g, rule, lag=lag, cost_bps=cost, event_cost_bps=ecost, overlay=ov)
    st = ld.stats(r["daily"], bench[key])
    st["avg_bonds"] = r["avg_n"]
    res[label] = st
    curves[label] = r["daily"]
    log(f"{label:<75} vsU0 {st['vs_bench_ann_%']:6.2f}% (t {st['t_vs_bench']:4.1f}) | SR {st['sharpe']:5.2f} "
        f"DD {st['max_dd_%']:6.2f}% | n {st['avg_bonds']:4.0f}")
    return st


# A) decision frequency (same execution rule: next fresh trade, lag 1)
go("P4 decisões semanais", P4w, ov=REGIME)
go("P4 decisões diárias", P4d, ov=REGIME)
# B) speed decay: execution delay in business days
decay = {}
for lag in (1, 2, 5, 10, 20):
    decay[lag] = go(f"P4 diário, execução ≥{lag} d.u. depois", P4d, lag=lag, ov=REGIME)["vs_bench_ann_%"]
# C) immediate reactions to events
go("P5 diário (bloqueia compra após rebaixamento/queda da ação)", P5d, ov=REGIME)
for ec in (25, 50, 100):
    go(f"P4 diário + venda no dia seguinte a evento (custo da venda {ec} bps)", P4ev, ecost=ec, ov=REGIME)
go("Híbrido: P4 semanal + venda diária em evento (50 bps)", P4wev, ecost=50, ov=REGIME)
for ec in (25, 100):
    go(f"Universo + venda no dia seguinte a evento (custo {ec} bps)", U0ev, ecost=ec)

# paired: event-exit variants vs their base
def paired(a, b):
    d = (curves[a] - curves[b].reindex(curves[a].index).fillna(0)).dropna()
    return {"diff_ann_%": d.mean() * 25200, "t": d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))}
pt = {
    "diário vs semanal": paired("P4 decisões diárias", "P4 decisões semanais"),
    "P5 vs P4 diário": paired("P5 diário (bloqueia compra após rebaixamento/queda da ação)", "P4 decisões diárias"),
    "venda em evento 50bps vs P4 diário": paired("P4 diário + venda no dia seguinte a evento (custo da venda 50 bps)",
                                                 "P4 decisões diárias"),
    "híbrido vs P4 semanal": paired("Híbrido: P4 semanal + venda diária em evento (50 bps)", "P4 decisões semanais"),
}
log("paired " + json.dumps({k: {kk: round(vv, 2) for kk, vv in v.items()} for k, v in pt.items()}, ensure_ascii=False))

json.dump({"results": res, "decay": decay, "paired": pt}, open(OUT / "daily_results.json", "w", encoding="utf-8"),
          indent=1, default=float, ensure_ascii=False)
pd.DataFrame(curves).to_pickle(DATA_DIR / "history" / "daily_curves.pkl")

fig, ax = plt.subplots(1, 2, figsize=(13, 4.8), dpi=105)
ax[0].plot(list(decay), list(decay.values()), marker="o", color="#0B6E63")
ax[0].set_xscale("log")
ax[0].set_xticks(list(decay))
ax[0].set_xticklabels([str(k) for k in decay])
ax[0].set_xlabel("atraso de execução (dias úteis, além de esperar um negócio novo)")
ax[0].set_ylabel("% a.a. acima do universo")
ax[0].set_title("Quanto vale reagir rápido? (P4 com decisões diárias)", fontsize=10)
ax[0].grid(alpha=0.3)
for label in ["P4 decisões semanais", "P4 decisões diárias", "P5 diário (bloqueia compra após rebaixamento/queda da ação)",
              "P4 diário + venda no dia seguinte a evento (custo da venda 50 bps)",
              "Híbrido: P4 semanal + venda diária em evento (50 bps)"]:
    eq = (1 + (curves[label] - bench[(1, 25)].reindex(curves[label].index).fillna(0))).cumprod()
    ax[1].plot(eq.index, (eq - 1) * 100, label=label, lw=1.4)
ax[1].set_title("Excesso acumulado vs universo (crédito puro, 2022–26)", fontsize=10)
ax[1].legend(fontsize=7, frameon=False)
ax[1].grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "daily_speed.png")
log("done")
