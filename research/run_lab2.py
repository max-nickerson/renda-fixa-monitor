"""Strategy lab — round 2: builds on round-1 findings (avoid rich bonds, contrarian after news, regime).
Every rule is reported at execution lag 1 and lag 2 (robustness to bid-ask bounce) and at 25/50 bps."""
import sys as _sys
from pathlib import Path as _P
_sys.path.insert(0, str(_P(__file__).resolve().parent.parent))  # run from anywhere

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from rfmonitor.config import DATA_DIR
from rfmonitor.ml import lab, lab_models as lm
from rfmonitor.ml.lab import Rule

warnings.filterwarnings("ignore")
OUT = Path(__file__).parent / "out"
import os
extra = _sys.argv[1] if len(_sys.argv) > 1 else None   # optional extra news feature file (GDELT)
RET = os.getenv("RET", "ret")                           # 'ret_hedged' = pure credit (rates hedged)
g = lm.prepare(pd.read_pickle(DATA_DIR / "history" / "lab_weekly.pkl"), RET)
PRESS = ["press_neg_7d", "press_neg_30d", "press_neg_90d", "press_spike", "mkt_rj_z", "mkt_calote_z"]
if extra:
    ext = pd.read_pickle(extra)                          # columns: cnpj8, week, <feature columns>
    g = g.merge(ext, on=["cnpj8", "week"], how="left")
    for c in PRESS[:4]:
        g[c] = g[c].fillna(0.0)
    g["press_covered"] = g["press_covered"].fillna(False)
g["cdi_pct"] = lm.weekly_pct(g, "cdi_bps")
weeks = sorted(g["week"].unique())
g["rebal"] = g["week"].isin(weeks[::4])
g["ml_news"] = lm.walk_forward_scores(g, lm.BASE + lm.NEWS, kind="rank")
g["ml_news_pct"] = lm.weekly_pct(g, "ml_news")

nz = lambda x, d=0.0: d if x is None or x != x else x
rich = lambda r: nz(r["resid_z"]) <= -1.5
news30 = lambda r: r["news_any_30d"] > 0
bad_news = lambda r: r["n_distress_30d"] > 0 or (r["n_fact_30d"] > 0 and nz(r["d_spread_1w"]) > 25)
reg = g.groupby("week").agg(mom=("mkt_mom_21", "first"))
REGIME = (reg["mom"] > 0).astype(float)

RULES = [
    (Rule("U0 Universo: compra e segura", lambda r: True, lambda r, st: False), None),
    (Rule("A1 Segura tudo, vende quando fica CARO (z≤−1.5)", lambda r: not rich(r), lambda r, st: rich(r), 0), None),
    (Rule("A2 A1 + só compra se z≥0 (não compra caro/justo-caro)", lambda r: nz(r["resid_z"]) >= 0,
          lambda r, st: rich(r)), None),
    (Rule("A3 A1 + regime momentum", lambda r: not rich(r), lambda r, st: rich(r)), REGIME),
    (Rule("B1 Contrarian: barato (z≥1.5) COM notícia 30d", lambda r: nz(r["resid_z"]) >= 1.5 and news30(r),
          lambda r, st: nz(r["resid_z"]) <= 0 or rich(r)), None),
    (Rule("B2 Contrarian: spread abriu >50bps em 4s COM notícia, sem distress",
          lambda r: nz(r["d_spread_4w"]) > 50 and news30(r) and r["n_distress_30d"] == 0,
          lambda r, st: nz(r["resid_z"]) <= 0, 0, 26), None),
    (Rule("B3 Contrarian: spread abriu >50bps em 4s SEM notícia",
          lambda r: nz(r["d_spread_4w"]) > 50 and not news30(r),
          lambda r, st: nz(r["resid_z"]) <= 0, 0, 26), None),
    (Rule("C2 Carry alto e não-caro", lambda r: nz(r["cdi_pct"], 1) <= 0.3 and not rich(r),
          lambda r, st: rich(r)), None),
    (Rule("C3 C2 + regime momentum", lambda r: nz(r["cdi_pct"], 1) <= 0.3 and not rich(r),
          lambda r, st: rich(r)), REGIME),
    (Rule("M1 Ridge c/ notícia top20/50, nunca caro", lambda r: r["rebal"] and nz(r["ml_news_pct"], 1) <= 0.2 and not rich(r),
          lambda r, st: (r["rebal"] and nz(r["ml_news_pct"], 1) > 0.5) or rich(r)), None),
    (Rule("M2 M1 + regime momentum", lambda r: r["rebal"] and nz(r["ml_news_pct"], 1) <= 0.2 and not rich(r),
          lambda r, st: (r["rebal"] and nz(r["ml_news_pct"], 1) > 0.5) or rich(r)), REGIME),
    (Rule("M3 M1 + sai com notícia ruim + regime",
          lambda r: r["rebal"] and nz(r["ml_news_pct"], 1) <= 0.2 and not rich(r) and not bad_news(r),
          lambda r, st: (r["rebal"] and nz(r["ml_news_pct"], 1) > 0.5) or rich(r) or bad_news(r), 26), REGIME),
    (Rule("RV4 Só os CAROS (diagnóstico)", lambda r: rich(r), lambda r, st: nz(r["resid_z"]) >= 0), None),
]

if extra:  # --- press (Google News) point-in-time news
    g["ml_press"] = lm.walk_forward_scores(g, lm.BASE + lm.NEWS + PRESS, kind="rank")
    g["ml_press_pct"] = lm.weekly_pct(g, "ml_press")
    press_spike = lambda r: r["press_neg_30d"] >= 3 and r["press_spike"] >= 1
    press_any = lambda r: r["press_neg_30d"] >= 1
    regp = g.groupby("week").agg(mom=("mkt_mom_21", "first"), rj=("mkt_rj_z", "first"), ca=("mkt_calote_z", "first"))
    REGIME_PRESS = ((regp["mom"] > 0) & (regp["rj"].fillna(0) < 1.5) & (regp["ca"].fillna(0) < 1.5)).astype(float)
    REGIME_PRESS_ONLY = ((regp["rj"].fillna(0) < 1.0) & (regp["ca"].fillna(0) < 1.0)).astype(float)
    c3_in = lambda r: nz(r["cdi_pct"], 1) <= 0.3 and not rich(r)
    RULES += [
        (Rule("P1 Universo + sai com pico de notícia ruim (imprensa)", lambda r: True,
              lambda r, st: press_spike(r), 26), None),
        (Rule("P2 Universo + sai imprensa OU CVM", lambda r: True,
              lambda r, st: press_spike(r) or bad_news(r), 26), None),
        (Rule("P3 C3 + sai com pico de imprensa/CVM", c3_in,
              lambda r, st: rich(r) or press_spike(r) or bad_news(r), 26), REGIME),
        (Rule("P4 C3 + não compra c/ imprensa negativa 30d", lambda r: c3_in(r) and not press_any(r),
              lambda r, st: rich(r)), REGIME),
        (Rule("P5 C3 + regime momentum&imprensa", c3_in, lambda r, st: rich(r)), REGIME_PRESS),
        (Rule("P6 Universo + regime só imprensa (índice RJ/calote)", lambda r: True, lambda r, st: False),
         REGIME_PRESS_ONLY),
        (Rule("P7 Ridge c/ CVM+imprensa top20/50, nunca caro",
              lambda r: r["rebal"] and nz(r["ml_press_pct"], 1) <= 0.2 and not rich(r),
              lambda r, st: (r["rebal"] and nz(r["ml_press_pct"], 1) > 0.5) or rich(r)), None),
        (Rule("P8 P7 + regime momentum",
              lambda r: r["rebal"] and nz(r["ml_press_pct"], 1) <= 0.2 and not rich(r),
              lambda r, st: (r["rebal"] and nz(r["ml_press_pct"], 1) > 0.5) or rich(r)), REGIME),
        (Rule("P9 Contrarian: barato c/ imprensa negativa, sem distress CVM",
              lambda r: nz(r["resid_z"]) >= 1.5 and press_any(r) and r["n_distress_90d"] == 0,
              lambda r, st: nz(r["resid_z"]) <= 0 or rich(r)), None),
        (Rule("P10 Só emissores com pico de imprensa (diagnóstico)", lambda r: press_spike(r),
              lambda r, st: not press_any(r), 0, 13), None),
    ]

rows, curves = {}, {}
bench = {}
for rule, ov in RULES:
    for lag, cost in ((1, 25), (2, 25), (1, 50)):
        res = lab.run_rule(g, rule, lag=lag, cost_bps=cost, overlay=ov)
        key = (lag, cost)
        if rule.name.startswith("U0"):
            bench[key] = res["weekly"]
        st = lab.stats(res["weekly"], bench[key])
        st.update(avg_bonds=res["avg_n"], turnover_wk_pct=res["turnover"] * 100)
        rows[f"{rule.name} | lag{lag} {cost}bps"] = st
        if key == (1, 25):
            curves[rule.name] = res["weekly"]
    a, b, c = (rows[f"{rule.name} | lag{l} {k}bps"] for l, k in ((1, 25), (2, 25), (1, 50)))
    print(f"{rule.name:<60} vsU0 lag1 {a['vs_bench_ann_%']:6.2f}% | lag2 {b['vs_bench_ann_%']:6.2f}% | 50bps "
          f"{c['vs_bench_ann_%']:6.2f}%   SR {a['sharpe']:5.2f}  DD {a['max_dd_%']:6.2f}%  n {a['avg_bonds']:4.0f}", flush=True)

tag = ("_gdelt" if extra else "") + ("_hedged" if RET == "ret_hedged" else "")
json.dump(rows, open(OUT / f"lab2_results{tag}.json", "w"), indent=1, default=float)
pd.DataFrame(curves).to_pickle(DATA_DIR / "history" / f"lab2_curves{tag}.pkl")
