"""Strategy lab: many strategy styles, with and without point-in-time news. Weekly, 2022-01 → 2026-09 OOS.

Outputs research/out/lab_results.json, lab_equity_*.png.
"""
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
from rfmonitor.ml import lab, lab_models as lm
from rfmonitor.ml.lab import Rule

warnings.filterwarnings("ignore")
OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)
t0 = time.time()
g = lm.prepare(pd.read_pickle(DATA_DIR / "history" / "lab_weekly.pkl"))
g["cdi_pct"] = lm.weekly_pct(g, "cdi_bps")          # 0 = highest spread
weeks = sorted(g["week"].unique())
g["rebal"] = g["week"].isin(weeks[::4])             # monthly decision weeks for ranked strategies

# ML scores (walk-forward, retrained quarterly, purged)
for name, feats in (("nonews", lm.BASE), ("news", lm.BASE + lm.NEWS)):
    g[f"ml_{name}"] = lm.walk_forward_scores(g, feats, kind="rank")
    g[f"ml_{name}_pct"] = lm.weekly_pct(g, f"ml_{name}")
    g[f"blow_{name}"] = lm.walk_forward_scores(g, feats, kind="blowup")
    g[f"blow_{name}_pct"] = lm.weekly_pct(g, f"blow_{name}")  # 0 = highest blow-up risk
    print(f"ML {name} done ({time.time() - t0:.0f}s)")

no_news_now = lambda r: r["n_distress_90d"] == 0 and r["n_fact_30d"] == 0
bad_news = lambda r: r["n_distress_30d"] > 0 or (r["n_fact_30d"] > 0 and (r["d_spread_1w"] or 0) > 25)
spread_blowout = lambda r: (r["d_spread_4w"] or 0) > 100
stop = lambda r, st: r["cdi_bps"] - st["entry_spread"] > 150
nz = lambda x, d=0.0: d if x is None or x != x else x

RULES = [
    # --- hold-everything family ("buy & hold, get out when needed")
    Rule("U0 Universo: compra e segura", lambda r: True, lambda r, st: False, note="benchmark"),
    Rule("U1 Segura + sai se spread abre >100bps/4s", lambda r: True, lambda r, st: spread_blowout(r), 13),
    Rule("U2 Segura + sai com notícia ruim (CVM)", lambda r: True, lambda r, st: bad_news(r), 26),
    Rule("U3 Segura + sai spread OU notícia", lambda r: True, lambda r, st: spread_blowout(r) or bad_news(r), 26),
    Rule("U4 U3 + não compra emissor c/ distress 90d", lambda r: r["n_distress_90d"] == 0,
         lambda r, st: spread_blowout(r) or bad_news(r), 26),
    Rule("U5 Segura + sai se ML(sem notícia) risco top 5%", lambda r: True,
         lambda r, st: nz(r["blow_nonews_pct"], 1) <= 0.05, 13),
    Rule("U6 Segura + sai se ML(com notícia) risco top 5%", lambda r: True,
         lambda r, st: nz(r["blow_news_pct"], 1) <= 0.05, 13),
    # --- carry
    Rule("C0 Carry alto (top 30% CDI+), segura", lambda r: nz(r["cdi_pct"], 1) <= 0.3, lambda r, st: False),
    Rule("C1 Carry alto + sai spread/notícia", lambda r: nz(r["cdi_pct"], 1) <= 0.3 and no_news_now(r),
         lambda r, st: spread_blowout(r) or bad_news(r), 26),
    # --- buy cheap, sell expensive / at fair
    Rule("RV1 Barato vs pares (z≥1.5) → vende no justo", lambda r: nz(r["resid_z"]) >= 1.5,
         lambda r, st: nz(r["resid_z"]) <= 0 or stop(r, st)),
    Rule("RV2 RV1 só sem notícia", lambda r: nz(r["resid_z"]) >= 1.5 and no_news_now(r),
         lambda r, st: nz(r["resid_z"]) <= 0 or stop(r, st) or bad_news(r)),
    Rule("RV3 Barato COM notícia recente (diagnóstico)", lambda r: nz(r["resid_z"]) >= 1.5 and r["news_any_30d"] > 0,
         lambda r, st: nz(r["resid_z"]) <= 0 or stop(r, st)),
    Rule("RV4 Caro vs pares (z≤−1.5) (diagnóstico)", lambda r: nz(r["resid_z"]) <= -1.5,
         lambda r, st: nz(r["resid_z"]) >= 0),
    # --- spread vs own history
    Rule("MR1 Spread alto vs próprio histórico → vende na média", lambda r: nz(r["own_z"]) >= 1.5,
         lambda r, st: nz(r["own_z"]) <= 0 or stop(r, st)),
    Rule("MR2 MR1 só sem notícia", lambda r: nz(r["own_z"]) >= 1.5 and no_news_now(r),
         lambda r, st: nz(r["own_z"]) <= 0 or stop(r, st) or bad_news(r)),
    # --- 'spread not where it should be'
    Rule("D1 Deslocado (pares z≥1 e histórico z≥1) sem notícia",
         lambda r: nz(r["resid_z"]) >= 1 and nz(r["own_z"]) >= 1 and no_news_now(r),
         lambda r, st: nz(r["resid_z"]) <= 0 or nz(r["own_z"]) <= 0 or stop(r, st) or bad_news(r)),
    # --- ML ranking (monthly decisions, buffer 20% in / 50% out)
    Rule("ML1 Ridge sem notícia (top20/sai>50%)", lambda r: r["rebal"] and nz(r["ml_nonews_pct"], 1) <= 0.2,
         lambda r, st: r["rebal"] and nz(r["ml_nonews_pct"], 1) > 0.5),
    Rule("ML2 Ridge com notícia (top20/sai>50%)", lambda r: r["rebal"] and nz(r["ml_news_pct"], 1) <= 0.2,
         lambda r, st: r["rebal"] and nz(r["ml_news_pct"], 1) > 0.5),
    Rule("ML3 Ridge c/ notícia + saída por notícia ruim",
         lambda r: r["rebal"] and nz(r["ml_news_pct"], 1) <= 0.2 and no_news_now(r),
         lambda r, st: (r["rebal"] and nz(r["ml_news_pct"], 1) > 0.5) or bad_news(r), 26),
]

results, curves = {}, {}
bench = None
for rule in RULES:
    t = time.time()
    res = lab.run_rule(g, rule, lag=1, cost_bps=25)
    if bench is None:
        bench = res["weekly"]
    st = lab.stats(res["weekly"], bench)
    st.update(turnover_wk_pct=res["turnover"] * 100, avg_bonds=res["avg_n"])
    results[rule.name] = st
    curves[rule.name] = res["weekly"]
    print(f"{rule.name:<52} exc {st['excess_ann_%']:6.2f}%  vsU0 {st['vs_bench_ann_%']:6.2f}%  "
          f"SR {st['sharpe']:5.2f}  DD {st['max_dd_%']:6.2f}%  n {res['avg_n']:5.0f}  ({time.time() - t:.0f}s)")

# Regime overlay on the best strategies: be in credit only when DI-credit momentum > 0 and the market
# distress pulse (CVM) is not elevated; otherwise CDI.
reg = g.groupby("week").agg(mom=("mkt_mom_21", "first"), dz=("mkt_distress_z", "first"))
overlay = ((reg["mom"] > 0) & (reg["dz"] < 1.5)).astype(float)
overlay_mom_only = (reg["mom"] > 0).astype(float)
for base in ["U0 Universo: compra e segura", "U3 Segura + sai spread OU notícia", "ML3 Ridge c/ notícia + saída por notícia ruim"]:
    rule = next(r for r in RULES if r.name == base)
    for lab_name, ov in (("+regime momentum", overlay_mom_only), ("+regime momentum&notícias", overlay)):
        res = lab.run_rule(g, rule, lag=1, cost_bps=25, overlay=ov)
        nm = f"{base.split(' ')[0]} {lab_name}"
        st = lab.stats(res["weekly"], bench)
        st.update(turnover_wk_pct=res["turnover"] * 100, avg_bonds=res["avg_n"])
        results[nm], curves[nm] = st, res["weekly"]
        print(f"{nm:<52} exc {st['excess_ann_%']:6.2f}%  vsU0 {st['vs_bench_ann_%']:6.2f}%  SR {st['sharpe']:5.2f}  DD {st['max_dd_%']:6.2f}%")

# Robustness: execution lag 2 weeks and 50 bps cost for the top strategies
top = sorted(results, key=lambda k: -np.nan_to_num(results[k]["sharpe"]))[:6]
robust = {}
for nm in top:
    rule = next((r for r in RULES if r.name == nm), None)
    if rule is None:
        continue
    for lag, cost in ((2, 25), (1, 50)):
        res = lab.run_rule(g, rule, lag=lag, cost_bps=cost)
        robust[f"{nm} | lag {lag} | {cost}bps"] = lab.stats(res["weekly"], bench)

pd.DataFrame(results).T.round(3).to_csv(OUT / "lab_results.csv")
json.dump({"main": results, "robust": robust}, open(OUT / "lab_results.json", "w"), indent=1, default=float)
pd.DataFrame(curves).to_pickle(DATA_DIR / "history" / "lab_curves.pkl")
print("\nROBUSTNESS")
for k, v in robust.items():
    print(f"{k:<75} exc {v['excess_ann_%']:6.2f}%  vsU0 {v['vs_bench_ann_%']:6.2f}%  SR {v['sharpe']:5.2f}  DD {v['max_dd_%']:6.2f}%")
print(f"total {time.time() - t0:.0f}s")
