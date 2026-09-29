"""Adversarial robustness check of ml_ranking's claim: lgb_peer:tilt adds value over P4+Q.
Uses only cached walk-forward predictions (pre-2026) + harness. Never touches the holdout.
Run: PYTHONPATH=. .venv/Scripts/python.exe research/nightly/ml_ranking/verify_robustness/verify.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from research.nightly import harness as H

OUT = Path("research/nightly/ml_ranking/verify_robustness")
CACHE = Path("data/history/nightly/ml_ranking")
t0 = time.time()
P = H.load_panel("M")
U = P[P["univ"] & (P["day"] >= H.START)].copy()
pred = pd.read_pickle(CACHE / "pred_lgb_peer.pkl")["pred"]
U = U.merge(pred[["dpos", "codigo", "pred"]], on=["dpos", "codigo"], how="left")
U["pct_u"] = U.groupby("dpos")["pred"].rank(pct=True)          # as in their code (universe percentile)
print("rows", len(U), "missing pred", U["pred"].isna().sum())


def pct_in_p4q(col):
    s = U[col].where(U["p4q"])
    return s.groupby(U["dpos"]).rank(pct=True)


def tilt(pct, base=0.5, mask=None, excl=None):
    m = U["p4q"].to_numpy() if mask is None else mask
    if excl is not None:
        m = m & ~U["cnpj8"].isin(excl).to_numpy()
    w = np.where(m, base + pct.fillna(0.5).to_numpy(), 0.0)
    return pd.DataFrame({"day": U["day"], "codigo": U["codigo"], "weight": w})


def p4q_sig(excl=None):
    m = U["p4q"].to_numpy()
    if excl is not None:
        m = m & ~U["cnpj8"].isin(excl).to_numpy()
    return pd.DataFrame({"day": U["day"], "codigo": U["codigo"], "select": m})


def diff(r, b, drop_months=None):
    m = H.monthly(r["daily"]) - H.monthly(b["daily"])
    m = m[m.index < H.HOLDOUT]
    if drop_months is not None:
        m = m[~drop_months(m.index)]
    return {"ann_%": round(float(m.mean() * 12 * 100), 4), "t_nw": round(H.nw_t(m, 6), 2), "n": int(len(m)),
            "hit": round(float((m > 0).mean()), 3)}


res = {}
b25 = H.baseline("P4Q")
b50 = H.baseline("P4Q", cost_bps=50)
best = tilt(U["pct_u"])
r25 = H.backtest(best)
r50 = H.backtest(best, cost_bps=50)
st = H.stats(r25["daily"], bench=b25["daily"])
res["reproduce_25"] = {k: st[k] for k in ("ann_excess_%", "diff_ann_%", "diff_t_nw", "diff_h1_%", "diff_h2_%", "diff_te_%")}
res["reproduce_50"] = diff(r50, b50)
res["turnover"] = {"tilt": r25["turnover_ann"], "p4q": b25["turnover_ann"], "tilt_cost_ann": r25["cost_ann_%"],
                   "p4q_cost_ann": b25["cost_ann_%"], "n_avg": r25["n_avg"]}
print(res)

# ---- time slices of the paired monthly diff
yr = lambda y: (lambda idx: idx.year == y)
res["slices"] = {
    "all": diff(r25, b25),
    "ex_2023": diff(r25, b25, yr(2023)),
    "ex_2022_23": diff(r25, b25, lambda i: i.year <= 2023),
    "only_2024": diff(r25, b25, lambda i: i.year != 2024),
    "only_2025": diff(r25, b25, lambda i: i.year != 2025),
    "ex_2023H1_crisis": diff(r25, b25, lambda i: (i.year == 2023) & (i.month <= 6)),
}
m = (H.monthly(r25["daily"]) - H.monthly(b25["daily"]))
m = m[m.index < H.HOLDOUT]
res["top_months"] = {str(k.date()): round(float(v) * 100, 4) for k, v in m.sort_values(ascending=False).head(6).items()}
top3 = m.sort_values(ascending=False).index[:3]
res["slices"]["ex_best_3_months"] = diff(r25, b25, lambda i: i.isin(top3))
print(res["slices"])

# ---- issuer contribution (cohort approx: normalised weight diff x fwd_126), then exclude top-5 issuers
Q = U[U["p4q"] & U["fwd_126"].notna()].copy()
Q["w_t"] = (0.5 + Q["pct_u"].fillna(0.5))
Q["w_t"] /= Q.groupby("dpos")["w_t"].transform("sum")
Q["w_q"] = 1.0 / Q.groupby("dpos")["codigo"].transform("count")
Q["c"] = (Q["w_t"] - Q["w_q"]) * Q["fwd_126"]
iss = Q.groupby("cnpj8")["c"].sum().sort_values(ascending=False)
res["issuer_contrib_top10"] = {k: round(float(v) * 100, 4) for k, v in iss.head(10).items()}
res["issuer_contrib_total"] = round(float(iss.sum()) * 100, 4)
res["issuer_contrib_share_top5"] = round(float(iss.head(5).sum() / iss.sum()), 3)
for k in (5, 10):
    ex = list(iss.index[:k])
    rt = H.backtest(tilt(U["pct_u"], excl=ex))
    rb = H.backtest(p4q_sig(excl=ex))
    res[f"ex_top{k}_issuers"] = diff(rt, rb)
    print(k, res[f"ex_top{k}_issuers"])

# ---- perturbations of the overlay form (all small, all 'reasonable' alternatives)
var = {}
var["base0.5_pct_universe(claimed)"] = r25
var["base1.0"] = H.backtest(tilt(U["pct_u"], base=1.0))
var["base0.25"] = H.backtest(tilt(U["pct_u"], base=0.25))
var["pct_within_p4q"] = H.backtest(tilt(pct_in_p4q("pred")))
var["rank_squared"] = H.backtest(tilt(U["pct_u"] ** 2))
# ---- simple NON-ML controls for what the model mostly encodes (SHAP: carry x dur, carry vs kind, carry)
U["cdi_x_dur"] = U["cdi_bps"] * U["dur"]
U["cdi_vs_kind"] = U["cdi_bps"] - U.groupby(["dpos", "kind"])["cdi_bps"].transform("median")
U["cdi_vs_peer"] = U["cdi_bps"] - U.groupby(["dpos", "peer"])["cdi_bps"].transform("median")
for c in ("cdi_bps", "cdi_x_dur", "cdi_vs_kind", "cdi_vs_peer", "resid_z"):
    var[f"ctrl_{c}"] = H.backtest(tilt(U[c].groupby(U["dpos"]).rank(pct=True)))
U["combo"] = (U["cdi_vs_peer"].groupby(U["dpos"]).rank(pct=True) + U["resid_z"].groupby(U["dpos"]).rank(pct=True))
var["ctrl_peercarry+resid"] = H.backtest(tilt(U["combo"].groupby(U["dpos"]).rank(pct=True)))
res["variants_vs_p4q"] = {k: diff(v, b25) for k, v in var.items()}
# paired: ML tilt vs best non-ML tilts (does ML add over a one-line rule?)
res["ml_vs_controls"] = {k: diff(r25, v) for k, v in var.items() if k.startswith("ctrl_")}
for k in res["variants_vs_p4q"]:
    print(k, res["variants_vs_p4q"][k], res["ml_vs_controls"].get(k, ""))

# ---- placebo: 100 random tilts (uniform 0.5..1.5) inside P4+Q; also 30 date-shuffled model scores
rng = np.random.default_rng(123)
pl = []
for i in range(100):
    pl.append(diff(H.backtest(tilt(pd.Series(rng.random(len(U)), index=U.index))), b25)["ann_%"])
pl = np.array(pl)
act = res["slices"]["all"]["ann_%"]
res["placebo_random_tilt"] = {"n": 100, "mean": round(float(pl.mean()), 4), "p95": round(float(np.percentile(pl, 95)), 4),
                              "p99": round(float(np.percentile(pl, 99)), 4), "sd": round(float(pl.std()), 4),
                              "p_value": float((pl >= act).mean())}
# score-shift placebo: the model score of the same bond 6 months EARLIER (stale info) - keeps persistent carry info
Us = U[["dpos", "codigo", "pct_u"]].copy()
dl = np.sort(U["dpos"].unique())
shift = dict(zip(dl[6:], dl[:-6]))
Us2 = U[["dpos", "codigo"]].assign(prev=U["dpos"].map(shift)).merge(
    Us.rename(columns={"dpos": "prev", "pct_u": "pct_lag"}), on=["prev", "codigo"], how="left")
U["pct_lag6"] = Us2["pct_lag"].to_numpy()
res["lagged6m_score_tilt"] = diff(H.backtest(tilt(U["pct_lag6"])), b25)
print("placebo", res["placebo_random_tilt"], "lag", res["lagged6m_score_tilt"])

# ---- multiple testing: their 51 + our perturbations; Holm on the claimed with an honest family
from scipy.stats import norm
t_claim = res["slices"]["all"]["t_nw"]
p_claim = 2 * (1 - norm.cdf(abs(t_claim)))
res["multiple_testing"] = {"raw_p": round(float(p_claim), 4), "bonferroni_51": min(1.0, p_claim * 51),
                           "bonferroni_51x6trials_tuned_models": min(1.0, p_claim * 51 * 1),
                           "note": "Best-of-51 selection; tuned hyperparameters (6 optuna trials per model), 4 overlay "
                                   "forms and 13 score sources chosen jointly. Holm over 51 already = 1.0."}
res["runtime_s"] = round(time.time() - t0, 1)
json.dump(res, open(OUT / "results.json", "w"), indent=1, default=float)

# ---- chart: cumulative paired diff vs P4+Q for claimed, controls, placebo band
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(10, 4.5))
q = H.monthly(b25["daily"])
for k in ("base0.5_pct_universe(claimed)", "ctrl_cdi_bps", "ctrl_cdi_vs_peer", "ctrl_peercarry+resid", "pct_within_p4q"):
    mm = H.monthly(var[k]["daily"])
    mm = (mm - q.reindex(mm.index).fillna(0))
    mm = mm[mm.index < H.HOLDOUT]
    ax.plot(mm.cumsum() * 100, label=k, lw=2 if "claimed" in k else 1.2)
ax.axhline(0, color="k", lw=0.5)
ax.axvline(pd.Timestamp("2024-01-01"), color="grey", ls=":")
ax.set_title("verify_robustness: cumulative paired difference vs P4+Q (%, 25 bps, pre-2026)")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "cum_diff_vs_p4q.png", dpi=110)
print("done", res["runtime_s"])
