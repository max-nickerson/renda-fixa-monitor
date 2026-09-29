"""Issuer-curve & cross-indexer relative value (nightly slug: issuer_curve_rv).

Stages (pre-2026 only until the final holdout block):
  1. fair-value quality: which residual predicts 13-week spread convergence best (cross-sectional, and WITHIN issuer)
  2. forward-return ICs (harness ic) + within-issuer ICs
  3. within-issuer pair trades (long cheap / short rich bond of the same issuer) by pair type: same-curve,
     cross-indexer (DI+ vs IPCA+ vs Pre, CDI+-equivalent), incentivada vs taxable; random-pair placebo
  4. market-level cross-indexer / incentivada basis timing diagnostic
  5. harness backtests (monthly decisions, 126-bday tranches, 25 bps): P4+Q with issuer-aware rich filter, within-issuer
     switches holding issuer exposure fixed, carry+RV tilt, pure RV books; compare() with Holm across all variants,
     50 bps, rec40, halves, placebo for the best
  6. sealed holdout (>= 2026-01-01) once, for the pre-chosen best variant
Writes results.json, equity_total_return.png, cum_excess.png, pair_trades.png, signal cache (data/history/nightly/...).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from research.nightly import harness as H  # noqa: E402
from research.nightly.issuer_curve_rv import signals as S  # noqa: E402

OUT = Path(__file__).resolve().parent
S.CACHE.mkdir(parents=True, exist_ok=True)
T0 = time.time()
RES: dict = {"slug": "issuer_curve_rv", "harness_version": H._VERSION}
SIGCOLS = ["resid_bps", "e_iss", "e_grp", "e_isec", "r_sec", "e_slope"]


def log(*a):
    print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)


def nwt(s, lag):
    s = pd.Series(s).dropna()
    return float(H.nw_t(s, lag)) if len(s) > 5 else np.nan


# ------------------------------------------------------------------------------------------------ data + signals
PW = H.load_panel("W")
PW = S.attach(PW)
PM = S.attach(H.load_panel("M"))
log("panels", PW.shape, PM.shape)
PWh = H.load_panel("W", holdout=True)
PMh = H.load_panel("M", holdout=True)
for fq, P in (("W", PWh), ("M", PMh)):
    sg = S.signals(P)
    sg.to_pickle(S.CACHE / f"signals_{fq}.pkl")
del PWh
log("signal caches written")

# ------------------------------------------------------------------------------------------------ 1. convergence
# 13-week-ahead CDI+ spread of the same bond (fresh at t+13w) - labels only, pre-2026 by construction
W = PW[PW["univ"]].copy()
dts = np.sort(PW["day"].unique())
nxt = pd.Series(dts[13:], index=dts[:-13])
W["day13"] = W["day"].map(nxt)
fut = PW.loc[PW["fresh"], ["codigo", "day", "cdi_bps"]].rename(columns={"day": "day13", "cdi_bps": "s13"})
W = W.merge(fut, on=["codigo", "day13"], how="left")
W["ds13"] = W["s13"] - W["cdi_bps"]
W = W[W["day"] >= "2021-06-01"]
conv = {}
C = W.dropna(subset=["ds13", "e_iss", "e_slope"]).copy()
C = C[C["ds13"].abs() < 500]
for scope in ("cross", "within"):
    rows = {}
    for c in SIGCOLS + ["cdi_bps"]:
        x = C[c].clip(-500, 1000)
        y = C["ds13"]
        if scope == "cross":
            key = C["day"]
        else:
            key = C["day"].astype(str) + C["cnpj8"].astype(str)
        xd = x - x.groupby(key).transform("mean")
        yd = y - y.groupby(key).transform("mean")
        per = []
        for d, idx in C.groupby("day").groups.items():
            a, b = xd.loc[idx].to_numpy(), yd.loc[idx].to_numpy()
            if (a ** 2).sum() > 0 and len(a) > 30:
                per.append((d, (a * b).sum() / (a ** 2).sum(), np.corrcoef(a, b)[0, 1] ** 2))
        per = pd.DataFrame(per, columns=["day", "beta", "r2"]).set_index("day")
        pooled_r2 = float(np.corrcoef(xd, yd)[0, 1] ** 2)
        rows[c] = {"beta_mean": round(float(per["beta"].mean()), 3), "beta_t_nw": round(nwt(per["beta"], 13), 2),
                   "r2_mean": round(float(per["r2"].mean()), 4), "r2_pooled": round(pooled_r2, 4)}
    conv[scope] = rows
conv["n_obs"] = int(len(C))
conv["note"] = ("13-week change in the bond's own CDI+ spread regressed on the residual (per date; 'within' = both "
                "demeaned by issuer-date). beta=-1 means full convergence to fair value within 13 weeks. Sample: "
                "universe bonds whose issuer has >=2 fresh bonds, fresh again 13 weeks later.")
RES["convergence_13w"] = conv
log("convergence", json.dumps(conv["within"]["resid_bps"]), json.dumps(conv["cross"]["e_iss"]))

# ------------------------------------------------------------------------------------------------ 2. ICs
ics = {}
for tgt in ("fwd_63", "fwd_126"):
    ics[tgt] = {}
    for c in SIGCOLS + ["cdi_bps"]:
        r = H.ic(PM, c, target=tgt)
        ics[tgt][c] = {"ic": round(r["mean"], 4), "t": round(r["t_nw"], 2), "n": r["n_dates"]}
# common sample (multi-bond issuers) and within-issuer IC
Mu = PM[PM["univ"] & PM["e_iss"].notna()].copy()
for tgt, H_ in (("fwd_63", 63), ("fwd_126", 126)):
    X = Mu[Mu[f"dok_{H_}"] & Mu[tgt].notna()]
    for c in SIGCOLS + ["cdi_bps"]:
        per, perw = {}, {}
        for d, x in X.groupby("day"):
            if len(x) < 30:
                continue
            per[d] = x[c].rank().corr(x[tgt].rank())
            k = x["cnpj8"]
            xr = x[c] - x.groupby(k)[c].transform("mean")
            yr = x[tgt] - x.groupby(k)[tgt].transform("mean")
            m = x.groupby(k)[c].transform("count") >= 2
            perw[d] = xr[m].rank().corr(yr[m].rank())
        s1, s2 = pd.Series(per), pd.Series(perw)
        ics[tgt][c + "|multi"] = {"ic": round(s1.mean(), 4), "t": round(nwt(s1, H_ // 21), 2)}
        ics[tgt][c + "|within_issuer"] = {"ic": round(s2.mean(), 4), "t": round(nwt(s2, H_ // 21), 2)}
RES["ic"] = ics
log("ic done", ics["fwd_126"]["resid_bps"], ics["fwd_126"]["e_iss"], ics["fwd_126"].get("resid_bps|within_issuer"))

# ------------------------------------------------------------------------------------------------ 3. pair trades
def pair_trades(P, rank_col, tgt="fwd_126", min_gap=0.0, rng=None, types=None):
    """Per decision date: every issuer with >=2 universe bonds (executed labels): long the max-`rank_col` bond,
    short the min one. Returns per-date mean return difference (and per pair-type)."""
    H_ = int(tgt.split("_")[1])
    X = P[P["univ"] & P[f"dok_{H_}"] & P[tgt].notna() & P[rank_col].notna()]
    X = X[X.groupby(["day", "cnpj8"])["codigo"].transform("count") >= 2]
    out = []
    for (d, c8), x in X.groupby(["day", "cnpj8"]):
        if rng is not None:
            o = rng.permutation(len(x))
            hi, lo = x.iloc[o[0]], x.iloc[o[1]]
        else:
            hi, lo = x.loc[x[rank_col].idxmax()], x.loc[x[rank_col].idxmin()]
        gap = hi[rank_col] - lo[rank_col]
        if rng is None and gap < min_gap:
            continue
        if hi["kind"] != lo["kind"]:
            ty = "xidx"
        elif hi["incent"] != lo["incent"]:
            ty = "xinc"
        else:
            ty = "curve"
        out.append((d, c8, ty, hi[tgt] - lo[tgt], gap, hi["cdi_bps"] - lo["cdi_bps"], hi["dur"] - lo["dur"]))
    return pd.DataFrame(out, columns=["day", "cnpj8", "type", "dret", "gap", "dcarry_bps", "ddur"])


pairs = {}
PT = {}
for rc in ("resid_bps", "e_slope", "cdi_bps"):
    for tgt in ("fwd_63", "fwd_126"):
        pt = pair_trades(PM, rc, tgt)
        PT[(rc, tgt)] = pt
        H_ = int(tgt.split("_")[1])
        ann = 252 / H_
        rows = {}
        for ty in ("all", "curve", "xidx", "xinc"):
            q = pt if ty == "all" else pt[pt["type"] == ty]
            s = q.groupby("day")["dret"].mean()
            rows[ty] = {"ann_%": round(float(s.mean() * ann * 100), 3), "t_nw": round(nwt(s, H_ // 21), 2),
                        "n_pairs": int(len(q)), "hit": round(float((q["dret"] > 0).mean()), 3),
                        "h1_%": round(float(s[s.index < H.SPLIT].mean() * ann * 100), 3),
                        "h2_%": round(float(s[s.index >= H.SPLIT].mean() * ann * 100), 3),
                        "mean_gap_bps": round(float(q["gap"].mean()), 1),
                        "mean_carry_diff_bps": round(float(q["dcarry_bps"].mean()), 1)}
        pairs[f"{rc}|{tgt}"] = rows
# large-gap pairs only (>= 50 bps on the issuer curve)
for tgt in ("fwd_126",):
    pt = PT[("resid_bps", tgt)]
    q = pt[pt["gap"] >= 50]
    s = q.groupby("day")["dret"].mean()
    pairs[f"resid_bps_gap50|{tgt}"] = {"ann_%": round(float(s.mean() * 2 * 100), 3), "t_nw": round(nwt(s, 6), 2),
                                        "n_pairs": int(len(q))}
# random-pair placebo
rng = np.random.default_rng(7)
pl = []
for i in range(10):
    pt = pair_trades(PM, "resid_bps", "fwd_126", rng=rng)
    pl.append(pt.groupby("day")["dret"].mean().mean() * 2 * 100)
pairs["placebo_random_pair_fwd126"] = {"mean_ann_%": round(float(np.mean(pl)), 3), "p95": round(float(np.percentile(pl, 95)), 3),
                                        "p5": round(float(np.percentile(pl, 5)), 3)}
RES["pair_trades"] = pairs
RES["pair_trades_note"] = ("Gross, unhedged-for-credit (same issuer both legs), rate-hedged returns; the short leg is NOT "
                           "implementable for debentures - read as the value of a SWITCH (sell rich, buy cheap bond of "
                           "the same issuer) per unit swapped, before 2x cost (~0.25-0.5%/switch).")
log("pairs", json.dumps(pairs["resid_bps|fwd_126"]["all"]), json.dumps(pairs["cdi_bps|fwd_126"]["all"]))

# ------------------------------------------------------------------------------------------------ 4. market basis timing
# CDI+-equivalent basis on the same day's fresh marks: IPCA+ minus DI+ (dur 2-7, taxable), incent minus taxable
# (IPCA, dur 3-8). Does a wide basis predict the subsequent 126-bday return spread of the respective sub-universes?
B = PM[PM["univ"]].copy()
bas = {}
for d, x in B.groupby("day"):
    di = x[(x["kind"] == "DI_SPREAD") & x["dur"].between(2, 7)]["cdi_bps"].median()
    ip = x[(x["kind"] == "IPCA") & x["dur"].between(2, 7)]["cdi_bps"].median()
    ii = x[(x["kind"] == "IPCA") & (x["incent"] == 1) & x["dur"].between(3, 8)]["cdi_bps"].median()
    it = x[(x["kind"] == "IPCA") & (x["incent"] == 0) & x["dur"].between(3, 8)]["cdi_bps"].median()
    dok = x["dok_126"].all()
    f = x[x["fwd_126"].notna()] if dok else x.iloc[:0]
    r_ip = f[f["kind"] == "IPCA"]["fwd_126"].mean()
    r_di = f[f["kind"] == "DI_SPREAD"]["fwd_126"].mean()
    r_ii = f[(f["kind"] == "IPCA") & (f["incent"] == 1)]["fwd_126"].mean()
    r_it = f[(f["kind"] == "IPCA") & (f["incent"] == 0)]["fwd_126"].mean()
    bas[d] = (ip - di, ii - it, r_ip - r_di, r_ii - r_it)
bas = pd.DataFrame(bas, index=["basis_ipca_di", "basis_inc_tax", "fwd_ipca_minus_di", "fwd_inc_minus_tax"]).T
bt = {}
for bcol, rcol in (("basis_ipca_di", "fwd_ipca_minus_di"), ("basis_inc_tax", "fwd_inc_minus_tax")):
    z = bas[[bcol, rcol]].dropna()
    z = z[z.index >= "2021-06-01"]
    zz = (z[bcol] - z[bcol].expanding(6).mean()) / z[bcol].expanding(6).std()
    zz = zz.dropna()
    y = z[rcol].reindex(zz.index)
    beta = float(np.polyfit(zz, y, 1)[0]) if len(zz) > 8 else np.nan
    sgn = np.sign(zz) * y
    bt[bcol] = {"basis_first_bps": round(float(bas[bcol].dropna().iloc[0]), 1),
                "basis_last_pre2026_bps": round(float(bas[bcol].dropna().iloc[-1]), 1),
                "basis_min": round(float(bas[bcol].min()), 1), "basis_max": round(float(bas[bcol].max()), 1),
                "corr_expanding_z_vs_fwd126": round(float(np.corrcoef(zz, y)[0, 1]), 3) if len(zz) > 8 else None,
                "sign_trade_ann_%": round(float(sgn.mean() * 2 * 100), 3), "sign_trade_t_nw": round(nwt(sgn, 6), 2),
                "n_months": int(len(zz))}
RES["basis_timing"] = bt
bas.to_csv(OUT / "market_basis.csv")
log("basis", bt)

# ------------------------------------------------------------------------------------------------ 5. backtests
def p4_flags(x, rich):
    nz = lambda s, v: s.fillna(v)
    return (x["univ"] & (nz(x["cdi_pct"], 1) <= 0.3) & ~rich & (nz(x["press_neg_30d"], 0) < 1) & ~x["worstQ"]).to_numpy()


def f_issrich(x):              # rich = issuer+sector-aware residual z <= -1.5 (instead of peer resid_z)
    return p4_flags(x, x["e_isec_z"].fillna(0) <= -1.5)


def f_bothrich(x):
    return p4_flags(x, (x["e_isec_z"].fillna(0) <= -1.5) | (x["resid_z"].fillna(0) <= -1.5))


def f_secrich(x):
    return p4_flags(x, x["r_sec_z"].fillna(0) <= -1.5)


def make_switch(col):
    """P4+Q issuer exposure kept; each issuer's slots re-assigned to its cheapest-`col` universe bonds that pass the
    P4 filters except the carry cut (same number of bonds per issuer)."""
    def f(x):
        base = x["p4q"].to_numpy()
        cand = (x["univ"] & x["p4f"] & ~x["worstQ"]).to_numpy()
        w = np.zeros(len(x))
        xs = x.assign(_base=base, _cand=cand, _i=np.arange(len(x)))
        cnt = xs[xs["_base"]].groupby("cnpj8").size()
        for c8, m in cnt.items():
            g = xs[(xs["cnpj8"] == c8) & (xs["_cand"] | xs["_base"])]
            g = g.assign(_v=g[col].fillna(-1e9))
            pick = g.sort_values("_v", ascending=False).head(int(m))["_i"].to_numpy()
            w[pick] = 1.0
        return w
    return f


def make_tilt(lam, col="e_iss"):
    def f(x):
        sc = x["cdi_bps"] + lam * x[col].fillna(0).clip(-300, 300)
        thr = sc.quantile(0.7)
        nz = lambda s, v: s.fillna(v)
        return ((sc >= thr) & ~(nz(x["resid_z"], 0) <= -1.5) & (nz(x["press_neg_30d"], 0) < 1) & ~x["worstQ"]).to_numpy()
    return f


VARIANTS = {
    "P4Q_issrich": (f_issrich, {}),
    "P4Q_bothrich": (f_bothrich, {}),
    "P4Q_secrich": (f_secrich, {}),
    "P4Q_switch_resid": (make_switch("resid_bps"), {"as_weights": True}),
    "P4Q_switch_eslope": (make_switch("e_slope"), {"as_weights": True}),
    "P4Q_switch_carry": (make_switch("cdi_bps"), {"as_weights": True}),
    "tilt_carry+0.5eiss": (make_tilt(0.5), {}),
    "tilt_carry+1.0eiss": (make_tilt(1.0), {}),
    "RV_top20_e_iss_z": ("e_iss_z", {"top_frac": 0.2}),
    "RV_top20_e_isec_z": ("e_isec_z", {"top_frac": 0.2}),
    "RV_top20_r_sec_z": ("r_sec_z", {"top_frac": 0.2}),
    "RV_top20_resid_z": ("resid_z", {"top_frac": 0.2}),
}
BT = {}
for nm, (sig, kw) in VARIANTS.items():
    BT[nm] = H.backtest(sig, panel=PM, name=nm, **kw)
    log("bt", nm, round(H.stats(BT[nm]["daily"])["ann_excess_%"], 3))
tab = H.compare({**BT, "P4": H.baseline("P4"), "P4Q": H.baseline("P4Q"), "U": H.baseline("U")}, bench="P4Q")
# Holm only across OUR variants (baselines excluded from the family)
fam = list(VARIANTS)
tab.loc[fam, "p_holm"] = H.holm(tab.loc[fam, "p_vs_bench"].astype(float).to_numpy())
tab.loc[["P4", "P4Q", "U"], "p_holm"] = np.nan
RES["backtests_25bps"] = tab.round(4).reset_index().rename(columns={"index": "variant"}).to_dict("records")
RES["n_variants_tried"] = len(fam)
log("\n" + tab[["exCDI_%", "exU_%", "vs_bench_%", "t_vs_bench", "p_holm", "h1_vs_bench", "h2_vs_bench", "turnover", "n_avg"]].round(3).to_string())

# choose the best variant by paired diff vs P4+Q (pre-2026 only)
best = tab.loc[fam, "vs_bench_%"].astype(float).idxmax()
RES["best_variant"] = best
sig, kw = VARIANTS[best]
sens = {}
for lab, extra, bn in (("50bps", {"cost_bps": 50}, {"cost_bps": 50}), ("rec40", {"scenario": "rec40"}, {"scenario": "rec40"}),
                       ("hold63", {"hold": 63}, {"hold": 63})):
    r = H.backtest(sig, panel=PM, **kw, **extra)
    b = H.baseline("P4Q", **bn)
    st = H.stats(r["daily"], bench=b["daily"])
    sens[lab] = {"exCDI_%": round(st["ann_excess_%"], 3), "vs_P4Q_%": round(st["diff_ann_%"], 3),
                 "t": round(st["diff_t_nw"], 2)}
# same sensitivities for the two main economic ideas regardless of which is best
for nm in ("P4Q_switch_resid", "P4Q_issrich"):
    s_, k_ = VARIANTS[nm]
    r = H.backtest(s_, panel=PM, cost_bps=50, **k_)
    st = H.stats(r["daily"], bench=H.baseline("P4Q", cost_bps=50)["daily"])
    sens[f"{nm}_50bps"] = {"vs_P4Q_%": round(st["diff_ann_%"], 3), "t": round(st["diff_t_nw"], 2)}
RES["best_sensitivities"] = sens
log("sens", sens)
pl = H.placebo(sig, n=10, panel=PM, **kw)
RES["best_placebo"] = {"actual": round(pl["actual"], 3), "random_mean": round(pl["mean"], 3), "random_p95": round(pl["p95"], 3)}
log("placebo", RES["best_placebo"])

# ------------------------------------------------------------------------------------------------ charts
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

show = {best: BT[best], "P4Q_switch_resid": BT["P4Q_switch_resid"], "P4Q_issrich": BT["P4Q_issrich"],
        "RV_top20_e_iss_z": BT["RV_top20_e_iss_z"]}
H.plot_curves(show, OUT / "equity_total_return.png", title="Issuer-curve RV vs baselines (pre-2026, 25 bps, 126d tranches)")
u = H.baseline("U")["daily"]
fig, ax = plt.subplots(figsize=(11, 5), dpi=110)
for nm, r in {**show, "P4": H.baseline("P4"), "P4+Q": H.baseline("P4Q")}.items():
    m = H.monthly(r["daily"]) - H.monthly(u).reindex(H.monthly(r["daily"]).index).fillna(0)
    ax.plot(m.index, m.cumsum() * 100, label=nm, lw=2 if nm == best else 1.2,
            ls="--" if nm in ("P4", "P4+Q") else "-")
ax.axhline(0, color="k", lw=.6)
ax.set_title("Cumulative excess vs universe (sum of monthly diffs, %)")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "cum_excess.png")
plt.close(fig)
# pair-trade cumulative (switch value) chart
fig, ax = plt.subplots(figsize=(11, 5), dpi=110)
for (rc, tgt), pt in PT.items():
    if tgt != "fwd_126":
        continue
    for ty in ("curve", "xidx", "xinc"):
        s = pt[pt["type"] == ty].groupby("day")["dret"].mean() * 100
        ax.plot(s.index, s.rolling(3, min_periods=1).mean(), label=f"{rc} {ty}", lw=1)
ax.axhline(0, color="k", lw=.6)
ax.set_title("Within-issuer switch: 126-bday return, cheap minus rich bond (% per pair, 3m avg)")
ax.legend(fontsize=7, ncol=3)
fig.tight_layout()
fig.savefig(OUT / "pair_trades.png")
plt.close(fig)
log("charts")

# ------------------------------------------------------------------------------------------------ 6. sealed holdout (once)
PMh = S.attach(PMh)
ho = {}
for nm in (best, "P4Q_switch_resid", "P4Q_issrich"):
    s_, k_ = VARIANTS[nm]
    rh = H.backtest(s_, panel=PMh, holdout=True, **k_)
    bh = H.baseline("P4Q", holdout=True)
    uh = H.baseline("U", holdout=True)
    st = H.stats(rh["daily"], bench=bh["daily"], holdout="only")
    su = H.stats(rh["daily"], bench=uh["daily"], holdout="only")
    ho[nm] = {"exCDI_%": round(st["ann_excess_%"], 3), "vs_P4Q_%": round(st["diff_ann_%"], 3),
              "t": round(st["diff_t_nw"], 2), "vs_U_%": round(su["diff_ann_%"], 3),
              "months": int(len(H.monthly(rh["daily"])[H.monthly(rh["daily"]).index >= H.HOLDOUT]))}
RES["holdout_2026"] = ho
RES["runtime_min"] = round((time.time() - T0) / 60, 1)
log("holdout", ho)
json.dump(RES, open(OUT / "results.json", "w"), indent=1, default=str)
log("done")
