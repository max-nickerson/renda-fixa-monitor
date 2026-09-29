"""Adversarial ROBUSTNESS verification of event_driven's P4Q_exNegEvents (pre-2026 only; holdout NOT touched).

  PYTHONPATH=. PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
      .venv/Scripts/python.exe research/nightly/event_driven/verify_robustness/verify.py
"""
from __future__ import annotations

import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from research.nightly import harness as H
from research.nightly.event_driven import run
from research.nightly.event_driven.events import build

OUT = Path(__file__).resolve().parent
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)


res: dict = {}
PM = run.load("M")
b = H.baseline("P4Q")
b50 = H.baseline("P4Q", cost_bps=50)
NEG = {"ma": 180, "rating_down": 180, "agd": 180, "resgate": 126, "rj": 365}


def dstats(r, bench=b):
    s = H.stats(r["daily"], bench=bench["daily"])
    return {k: s[k] for k in ["ann_excess_%", "diff_ann_%", "diff_t_nw", "diff_p", "diff_h1_%", "diff_h2_%"]}


def diff_monthly(r, bench=b):
    s = r["daily"]; s = s[s.index < H.HOLDOUT]
    m = H.monthly(s); mb = H.monthly(bench["daily"].reindex(s.index).fillna(0))
    return (m - mb).dropna()


def mask_windows(x, win):
    m = np.zeros(len(x), bool)
    for et, w in win.items():
        if w:
            m |= run.within(x, et, w)
    return m


def rule_w(win):
    return lambda x: x["p4q"].to_numpy() & ~mask_windows(x, win)


# ---------------------------------------------------------------- 1. reproduce + costs
r0 = H.backtest(rule_w(NEG), panel=PM, name="repro")
res["repro"] = dstats(r0)
res["repro"]["n_avg"] = r0["n_avg"]; res["repro"]["turnover"] = r0["turnover_ann"]
res["repro_50bps"] = dstats(H.backtest(rule_w(NEG), panel=PM, cost_bps=50, name="r50"), b50)
log("repro", res["repro"], res["repro_50bps"])

# ---------------------------------------------------------------- 2. exclude 2023 / per-year
d0 = diff_monthly(r0)
res["diff_by_year_%"] = {str(y): round(float(g.mean() * 1200), 3) for y, g in d0.groupby(d0.index.year)}
ex23 = d0[d0.index.year != 2023]
res["ex2023"] = {"diff_ann_%": float(ex23.mean() * 1200), "t_nw": H.nw_t(ex23, 6), "n_months": len(ex23)}
# concentration in time: share of cumulative diff from the best 3 months
srt = d0.sort_values(ascending=False)
res["top3_months_share_of_total_diff"] = float(srt.iloc[:3].sum() / d0.sum())
res["diff_ex_top3_months_ann_%"] = float(srt.iloc[3:].mean() * 1200)
res["top3_months"] = {str(k.date()): round(float(v) * 100, 3) for k, v in srt.iloc[:3].items()}
log("years", res["diff_by_year_%"], res["ex2023"], res["top3_months_share_of_total_diff"])

# ---------------------------------------------------------------- 3. leave-one-component-out and single components
loo = {}
for et in NEG:
    w = {k: v for k, v in NEG.items() if k != et}
    loo[f"without_{et}"] = dstats(H.backtest(rule_w(w), panel=PM, name=et))
loo["rj_only_365"] = dstats(H.backtest(rule_w({"rj": 365}), panel=PM, name="rj"))
res["leave_one_out"] = loo
log("loo", {k: v["diff_ann_%"] for k, v in loo.items()})

# ---------------------------------------------------------------- 4. window perturbation (garden of forking paths)
grid = []
for f in [0.35, 0.5, 0.7, 1.0, 1.4, 2.0]:
    w = {k: int(round(v * f)) for k, v in NEG.items()}
    s = dstats(H.backtest(rule_w(w), panel=PM, name=f"scale{f}"))
    grid.append({"kind": "scale", "f": f, **w, **s})
rng = np.random.default_rng(7)
opts = [63, 126, 180, 252, 365]
for i in range(24):
    w = {k: int(rng.choice(opts)) for k in NEG}
    s = dstats(H.backtest(rule_w(w), panel=PM, name=f"rand{i}"))
    grid.append({"kind": "random", **w, **s})
G = pd.DataFrame(grid)
res["window_perturbation"] = G.to_dict("records")
rr = G[G["kind"] == "random"]["diff_ann_%"]
res["window_perturbation_summary"] = {"random_mean_%": float(rr.mean()), "random_min_%": float(rr.min()),
                                      "random_max_%": float(rr.max()), "frac_pos": float((rr > 0).mean()),
                                      "frac_t_gt_1.96": float((G[G["kind"] == "random"]["diff_t_nw"] > 1.96).mean())}
log("windows\n", G[["kind", "f", "ma", "rating_down", "agd", "resgate", "rj", "diff_ann_%", "diff_t_nw"]])

# ---------------------------------------------------------------- 5. exclude top contributing issuers (from BOTH books)
X = PM[PM["univ"] & PM["p4q"]].copy()
X["neg"] = mask_windows(X, NEG)
X = X[X["fwd_126"].notna()]
mu = X.groupby("day")["fwd_126"].transform("mean")
nd = X.groupby("day")["cnpj8"].transform("nunique")
X["contrib"] = (mu - X["fwd_126"]) / X.groupby(["day", "cnpj8"])["codigo"].transform("size") / nd
top = X[X["neg"]].groupby("cnpj8")["contrib"].sum().sort_values(ascending=False)
res["top_dropped_issuers_contrib"] = {str(k): round(float(v) * 100, 3) for k, v in top.head(10).items()}
res["n_distinct_dropped_issuers"] = int(X.loc[X["neg"], "cnpj8"].nunique())
for k in (5, 10):
    ex = set(top.index[:k].astype(str))
    base_k = lambda x, ex=ex: x["p4q"].to_numpy() & ~x["cnpj8"].astype(str).isin(ex).to_numpy()
    rule_k = lambda x, ex=ex: base_k(x) & ~mask_windows(x, NEG)
    bk = H.backtest(base_k, panel=PM, name=f"P4Q_ex{k}")
    rk = H.backtest(rule_k, panel=PM, name=f"rule_ex{k}")
    res[f"ex_top{k}_issuers"] = dstats(rk, bk)
log("topK", res["top_dropped_issuers_contrib"], res.get("ex_top5_issuers"), res.get("ex_top10_issuers"))

# ---------------------------------------------------------------- 6. placebos on EVENT TIMING
ref, E = build()
E = E.copy(); E["cnpj8"] = E["cnpj8"].astype(str); E["date"] = pd.to_datetime(E["date"])


def ev_mask_from(Ev, panel, win):
    Xp = panel[["codigo", "day", "cnpj8"]].copy(); Xp["cnpj8"] = Xp["cnpj8"].astype(str)
    Xp["day"] = Xp["day"].astype("datetime64[ns]"); Xp["_i"] = np.arange(len(Xp)); Xp = Xp.sort_values("day")
    m = np.zeros(len(Xp), bool)
    for et, w in win.items():
        e = Ev[Ev["etype"] == et][["cnpj8", "date"]].rename(columns={"date": "_ed"}).sort_values("_ed")
        e["_ed"] = e["_ed"].astype("datetime64[ns]")
        mm = pd.merge_asof(Xp, e, left_on="day", right_on="_ed", by="cnpj8", allow_exact_matches=False)
        d = (mm["day"] - mm["_ed"]).dt.days.to_numpy()
        m |= np.nan_to_num(d, nan=1e9) <= w
    out = np.zeros(len(Xp), bool); out[Xp["_i"].to_numpy()] = m
    return out


def rule_from_col(col):
    return lambda x: x["p4q"].to_numpy() & ~x[col].to_numpy()


# (a) stale events: shift every event +365d (i.e. excludes issuers whose last event >1y ago was within window)
P2 = PM.copy()
Es = E.copy(); Es["date"] = Es["date"] + pd.Timedelta(days=365)
P2["_stale"] = ev_mask_from(Es, P2, NEG)
res["placebo_stale_events_+365d"] = dstats(H.backtest(rule_from_col("_stale"), panel=P2, name="stale"))
# (b) same issuers, random event dates (keep per-issuer counts; uniform over 2020-06..2025-12) -> identity vs timing
lo, hi = pd.Timestamp("2020-06-01").value, pd.Timestamp("2025-12-31").value
negE = E[E["etype"].isin(NEG.keys())]
pl = []
for sd in range(20):
    rg = np.random.default_rng(1000 + sd)
    Er = negE.copy(); Er["date"] = pd.to_datetime(rg.integers(lo, hi, len(Er))).normalize()
    P2["_rnd"] = ev_mask_from(Er, P2, NEG)
    s = dstats(H.backtest(rule_from_col("_rnd"), panel=P2, name=f"rnd{sd}"))
    pl.append(s["diff_ann_%"])
pl = np.array(pl)
res["placebo_random_dates_same_issuers"] = {"mean_%": float(pl.mean()), "p95_%": float(np.percentile(pl, 95)),
                                            "max_%": float(pl.max()), "frac_ge_real": float((pl >= res["repro"]["diff_ann_%"]).mean()),
                                            "draws": pl.round(3).tolist()}
log("placebo random dates", res["placebo_random_dates_same_issuers"])
# (c) static issuer blacklist: drop issuers that EVER have a neg event 2021-2025 (look-ahead identity, upper bound)
ever = set(negE["cnpj8"])
res["placebo_ever_event_issuer_blacklist"] = dstats(H.backtest(
    lambda x: x["p4q"].to_numpy() & ~x["cnpj8"].astype(str).isin(ever).to_numpy(), panel=PM, name="ever"))
# (d) "neutral" events with the same windows: equity raise / new-deb offer / supply (should NOT help if story holds)
neu = {"equity_raise": 180, "ipe_new_deb": 180, "supply_new_issue": 180}
res["placebo_neutral_events_180"] = dstats(H.backtest(rule_w(neu), panel=PM, name="neutral"))
log("placebos", res["placebo_stale_events_+365d"], res["placebo_ever_event_issuer_blacklist"],
    res["placebo_neutral_events_180"])

# ---------------------------------------------------------------- 7. capacity / liquidity of what is dropped vs kept
Z = PM[PM["univ"] & PM["p4q"]].copy(); Z["neg"] = mask_windows(Z, NEG)
res["dropped_share_of_p4q_rows"] = float(Z["neg"].mean())
res["liquidity_trades_30d_median"] = {"dropped": float(Z.loc[Z["neg"], "trades_30d"].median()),
                                      "kept": float(Z.loc[~Z["neg"], "trades_30d"].median())}
res["executed_rate"] = {"dropped": float(Z.loc[Z["neg"], "executed"].mean()),
                        "kept": float(Z.loc[~Z["neg"], "executed"].mean())}
res["cdi_bps_median"] = {"dropped": float(Z.loc[Z["neg"], "cdi_bps"].median()),
                         "kept": float(Z.loc[~Z["neg"], "cdi_bps"].median())}

# ---------------------------------------------------------------- 8. multiple-testing accounting
ps = [res["repro"]["diff_p"]] + [g for g in G["diff_p"]] + [v["diff_p"] for v in loo.values()]
res["effective_variants_note"] = ("author reports 12 variants; the combo is a union of 4 separately tested "
                                  "exclusions + an untested RJ leg with windows chosen after ~14 event studies")
res["bonferroni_p_if_40_specs"] = float(min(1.0, res["repro"]["diff_p"] * 40))
res["runtime_s"] = time.time() - T0


def js(o):
    if isinstance(o, dict):
        return {str(k): js(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [js(v) for v in o]
    if isinstance(o, (float, np.floating)):
        return None if not np.isfinite(o) else round(float(o), 4)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


(OUT / "results.json").write_text(json.dumps(js(res), indent=1), encoding="utf-8")
log("done")
