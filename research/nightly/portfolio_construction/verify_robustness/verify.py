"""Adversarial robustness check of portfolio_construction's headline: top-25-by-carry within P4+Q beats P4+Q.

Pre-2026 only (holdout sealed; the holdout numbers already published by the original agent are not re-run here).
Writes verify_results.json + verify_curves.png in this folder.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

from research.nightly import harness as H

OUTD = Path(__file__).resolve().parent
OUTD.mkdir(exist_ok=True, parents=True)
P = H.load_panel("M")
U = P[P["univ"] & (P["day"] >= H.START)].copy()
LIQ = pd.read_pickle(Path(H.__file__).resolve().parents[2] / "data/history/nightly/portfolio_construction/liquidity.pkl")
U = U.merge(LIQ[["codigo", "day", "vol91_brl", "tdays91"]], on=["codigo", "day"], how="left")
BQ = H.baseline("P4Q")["daily"]
RES = {}


def topk(k=25, cap=0.05, mask=None, key="cdi_bps", asc=False, base="p4q", rng=None, drop_iss=()):
    rows = []
    for d, x in U.groupby("day", sort=True):
        q = x[x[base].astype(bool)]
        if mask is not None:
            q = q[mask(q)]
        if drop_iss:
            q = q[~q["cnpj8"].astype(str).isin(drop_iss)]
        if rng is not None:
            q = q.sample(frac=1.0, random_state=int(rng.integers(1e9)))
        else:
            q = q.sort_values(key, ascending=asc)
        s = q.head(k)
        if len(s) < 5:
            continue
        w = H.cap_weights(s["cnpj8"].astype(str).to_numpy(), cap)
        rows.append(pd.DataFrame({"day": d, "codigo": s["codigo"].to_numpy(), "weight": w}))
    return pd.concat(rows, ignore_index=True)


def bt(sig, **kw):
    return H.backtest(sig, freq="M", hold=126, issuer_cap=1.0, **kw)


def st(r, bench=BQ):
    s = H.stats(r["daily"], bench=bench)
    return {k: s.get(k) for k in ("ann_excess_%", "diff_ann_%", "diff_t_nw", "diff_p", "diff_h1_%", "diff_h2_%",
                                  "max_dd_%", "vol_%")}


def diff_monthly(r, bench=BQ):
    a = H.monthly(r["daily"]); b = H.monthly(bench.reindex(r["daily"].index).fillna(0))
    return (a - b).dropna()


def sub_stats(dm, mask):
    x = dm[mask]
    return {"diff_ann_%": float(x.mean() * 1200), "t": float(H.nw_t(x, 6)), "n": int(len(x))}


out = {}
# ---- A. reproduction
w25 = topk(25)
r25 = bt(w25)
RES["top25"] = r25
out["A_repro_top25_cap5"] = st(r25)
try:
    from research.nightly.portfolio_construction import run as RUN
    rr = bt(RUN.topk_carry())
    out["A_repro_original_code"] = st(rr)
except Exception as e:  # pragma: no cover
    out["A_repro_original_code"] = str(e)
print("A", out["A_repro_top25_cap5"], out.get("A_repro_original_code"), flush=True)

# ---- B. 50bps / rec40 / subperiods
b50 = H.baseline("P4Q", cost_bps=50)["daily"]
b40 = H.baseline("P4Q", scenario="rec40")["daily"]
b100 = H.baseline("P4Q", cost_bps=100)["daily"]
out["B_cost50"] = st(bt(w25, cost_bps=50), b50)
out["B_cost100"] = st(bt(w25, cost_bps=100), b100)
out["B_rec40"] = st(bt(w25, scenario="rec40"), b40)
dm = diff_monthly(r25)
idx = dm.index
out["B_sub"] = {
    "all": sub_stats(dm, np.ones(len(dm), bool)),
    "ex_2023": sub_stats(dm, idx.year != 2023),
    "ex_crisis_2022-12_2023-06": sub_stats(dm, ~((idx >= "2022-12-01") & (idx <= "2023-06-30"))),
    "2022": sub_stats(dm, idx.year == 2022), "2023": sub_stats(dm, idx.year == 2023),
    "2024": sub_stats(dm, idx.year == 2024), "2025": sub_stats(dm, idx.year == 2025),
    "hit_rate": float((dm > 0).mean()),
    "ex_best6_months": float(dm.sort_values().iloc[:-6].mean() * 1200),
}
print("B", out["B_cost50"], out["B_rec40"], out["B_sub"], flush=True)

# ---- C. parameter perturbations (k, cap, base set)
C = {}
for k in (10, 15, 20, 30, 40, 50, 75):
    r = bt(topk(k)); RES[f"top{k}"] = r; C[f"top{k}_cap5"] = st(r)
C["top25_cap10"] = st(bt(topk(25, cap=0.10)))
C["top25_cap3"] = st(bt(topk(25, cap=0.03)))
C["bottom25_carry_in_p4q"] = st(bt(topk(25, asc=True)))
C["top25_carry_in_P4_noQ"] = st(bt(topk(25, base="p4")))
C["top25_carry_in_universe"] = st(bt(topk(25, base="univ")))
out["C_perturb"] = C
print("C", json.dumps(C, default=float)[:1500], flush=True)

# ---- D. placebo: 25 random P4+Q names (same cap). Null for "concentration itself"
rng = np.random.default_rng(7)
pl = [H.stats(bt(topk(25, rng=rng))["daily"], bench=BQ)["diff_ann_%"] for _ in range(20)]
out["D_placebo_random25_in_p4q"] = {"mean": float(np.mean(pl)), "p95": float(np.percentile(pl, 95)),
                                     "max": float(np.max(pl)), "actual": out["A_repro_top25_cap5"]["diff_ann_%"]}
print("D", out["D_placebo_random25_in_p4q"], flush=True)

# ---- E. issuer concentration of the edge: contribution by issuer from cohort labels (w_top25 - w_p4q) * fwd_126
X = U[["day", "codigo", "cnpj8", "fwd_126", "p4q"]].copy()
X["cnpj8"] = X["cnpj8"].astype(str)
Xp = X[X["p4q"].astype(bool)].copy()
Xp["wq"] = Xp.groupby("day")["codigo"].transform(lambda s: 1.0 / len(s))
Xm = Xp.merge(w25.rename(columns={"weight": "wt"}), on=["day", "codigo"], how="left").fillna({"wt": 0.0})
Xm["contrib"] = (Xm["wt"] - Xm["wq"]) * Xm["fwd_126"].fillna(0)
ci = Xm.groupby("cnpj8")["contrib"].sum().sort_values(ascending=False)
top5 = list(ci.index[:5]); top10 = list(ci.index[:10])
out["E_issuer_contrib"] = {"top10": {k: float(v) for k, v in ci.head(10).items()},
                           "share_top5_of_total": float(ci.head(5).sum() / ci.sum()), "total": float(ci.sum()),
                           "n_issuers_ever_in_top25": int((Xm.groupby("cnpj8")["wt"].sum() > 0).sum())}
bq_drop5 = H.backtest(lambda x: (x["p4q"] & ~x["cnpj8"].astype(str).isin(top5)).to_numpy(), freq="M", hold=126)["daily"]
bq_drop10 = H.backtest(lambda x: (x["p4q"] & ~x["cnpj8"].astype(str).isin(top10)).to_numpy(), freq="M", hold=126)["daily"]
out["E_drop_top5_issuers_vs_P4Q"] = st(bt(topk(25, drop_iss=tuple(top5))))
out["E_drop_top5_issuers_vs_P4Q_drop5"] = st(bt(topk(25, drop_iss=tuple(top5))), bq_drop5)
out["E_drop_top10_issuers_vs_P4Q_drop10"] = st(bt(topk(25, drop_iss=tuple(top10))), bq_drop10)
print("E", out["E_issuer_contrib"], out["E_drop_top5_issuers_vs_P4Q_drop5"], out["E_drop_top10_issuers_vs_P4Q_drop10"], flush=True)

# ---- F. survivorship / stale-mark segment: covered, listed, liquid-only versions (benchmark restricted the same way)
F = {}
for nm, m in {"covered": lambda q: q["covered"].fillna(False).astype(bool),
              "listed": lambda q: q["listed"].fillna(False).astype(bool),
              "liquid_tdays91>=20": lambda q: q["tdays91"].fillna(0) >= 20,
              "vol91>=R$50m": lambda q: q["vol91_brl"].fillna(0) >= 50e6}.items():
    rk = bt(topk(25, mask=m))
    bm = H.backtest(lambda x, m=m: (x["p4q"].astype(bool) & m(x)).to_numpy(), freq="M", hold=126, panel=P.merge(
        LIQ[["codigo", "day", "vol91_brl", "tdays91"]], on=["codigo", "day"], how="left"))["daily"]
    F[nm] = {"vs_P4Q": st(rk), "vs_P4Q_same_restriction": st(rk, bm),
             "n_avg": rk["n_avg"]}
out["F_segments"] = F
print("F", json.dumps(F, default=float)[:2000], flush=True)

# ---- G. carry vs price decomposition (ex-ante spread of the books) and the stop-trading look-ahead flag (diagnostic)
g = []
for d, x in U.groupby("day"):
    q = x[x["p4q"].astype(bool)]
    s = q.sort_values("cdi_bps", ascending=False).head(25)
    g.append({"day": d, "spread_p4q": q["cdi_bps"].mean(), "spread_top25": s["cdi_bps"].mean(),
              "dur_p4q": q["dur"].mean(), "dur_top25": s["dur"].mean(),
              "stop_p4q": q["dist_stop_LOOKAHEAD"].fillna(False).astype(bool).mean() if "dist_stop_LOOKAHEAD" in q else np.nan,
              "stop_top25": s["dist_stop_LOOKAHEAD"].fillna(False).astype(bool).mean() if "dist_stop_LOOKAHEAD" in s else np.nan,
              "covered_p4q": q["covered"].fillna(False).astype(bool).mean(), "covered_top25": s["covered"].fillna(False).astype(bool).mean(),
              "fresh_frac_top25_tdays91<5": (s["tdays91"].fillna(0) < 5).mean(),
              "fresh_frac_p4q_tdays91<5": (q["tdays91"].fillna(0) < 5).mean()})
G = pd.DataFrame(g)
out["G_profile"] = {c: float(G[c].mean()) for c in G.columns if c != "day"}
out["G_profile"]["carry_gap_bps_per_yr"] = float((G["spread_top25"] - G["spread_p4q"]).mean())
print("G", out["G_profile"], flush=True)

# ---- H. capacity re-check with stricter participation (10%) at R$500m, same original constructor
try:
    from research.nightly.portfolio_construction import run as RUN, build_weights as BW
    capx = {}
    for part, aum in ((0.10, 500e6), (0.25, 500e6), (0.10, 250e6)):
        BW.PART, BW.AUM = part, aum
        r = bt(RUN.topk_carry(fund=True))
        capx[f"part{part}_aum{aum/1e6:.0f}m"] = st(r)
    BW.PART, BW.AUM = 0.25, 500e6
    out["H_capacity"] = capx
    print("H", capx, flush=True)
except Exception as e:
    out["H_capacity"] = str(e)

# ---- Holm across the variants of this verification (paired vs P4+Q)
pv = {}
for sec in ("C_perturb",):
    for k, v in out[sec].items():
        pv[k] = v["diff_p"]
pv["top25"] = out["A_repro_top25_cap5"]["diff_p"]
names = list(pv)
hp = H.holm(np.array([pv[n] for n in names]))
out["holm_within_verify"] = {n: float(h) for n, h in zip(names, hp)}


def clean(o):
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (float, np.floating)):
        return None if not np.isfinite(o) else round(float(o), 4)
    if isinstance(o, (np.integer,)):
        return int(o)
    return o


(OUTD / "verify_results.json").write_text(json.dumps(clean(out), indent=1, default=str))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(14, 5), dpi=110)
for k in ("top10", "top25", "top50", "top75"):
    d = diff_monthly(RES[k])
    ax[0].plot(d.index, d.cumsum() * 100, label=f"{k} carry in P4+Q")
ax[0].set_title("Cumulative monthly excess vs P4+Q (%), pre-2026")
ax[0].axhline(0, color="k", lw=.6); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
kk = [10, 15, 20, 25, 30, 40, 50, 75]
ax[1].plot(kk, [out["C_perturb"].get(f"top{k}_cap5", out["A_repro_top25_cap5"])["diff_ann_%"] for k in kk], "o-")
ax[1].set_xlabel("k (names)"); ax[1].set_ylabel("vs P4+Q %/yr"); ax[1].grid(alpha=.3)
ax[1].set_title("Edge vs concentration k")
fig.tight_layout(); fig.savefig(OUTD / "verify_curves.png")
print("done")
