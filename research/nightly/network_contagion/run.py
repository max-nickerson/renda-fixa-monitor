"""NETWORKS & CONTAGION — main study (pre-2026 only; the sealed holdout is reported once, at the end, with --holdout).

Steps
 1. network features (features.py; PIT) merged on the harness W / M panels
 2. Fama-MacBeth: weekly cross-sections, fwd_21 / fwd_63 (demeaned by date) on each neighbour-stress feature with
    controls (carry, RV, duration, own 4w return, own shock). NW t on the weekly slope series.
 3. event study: issuers hit by a 'hard' shock -> forward excess of NON-shocked group-mates / sector-mates
 4. portfolios: P4+Q minus neighbour-stressed names (avoid filters), sector momentum / reversal tilts,
    monthly 126-bday tranches, 25 bps (and 50 bps), Holm across all variants, halves, rec40, drop-placebo
 5. charts + results.json

Rerun: PYTHONPATH=. PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=2 .venv/Scripts/python.exe research/nightly/network_contagion/run.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import features as FT  # noqa: E402
from research.nightly import harness as H  # noqa: E402

T0 = time.time()
HOLDOUT = "--holdout" in sys.argv
RES = {}


def log(*a):
    print(f"[+{time.time() - T0:6.0f}s]", *a, flush=True)


def jf(x):
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else round(float(x), 4)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, dict):
        return {str(k): jf(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jf(v) for v in x]
    if isinstance(x, pd.Timestamp):
        return str(x.date())
    return x


FEATS = {  # name: (column, sign so that + = more stress)
    "sib_min_r4": ("sib_min_r4", -1), "sib_gap": ("sib_gap", -1),
    "grp_hard": ("i_grp_hard", 1), "grp_r4": ("i_grp_r4", -1),
    "aff_any": ("i_aff_any", 1),
    "sec_hard": ("i_sec_hard", 1), "sec_r4": ("i_sec_r4m", -1), "sec_r13": ("i_sec_r13", -1),
    "exp_any": ("i_exp_any", 1), "exp_r4": ("i_exp_r4", -1),
    "hold_ret": ("i_hold_ret", 1), "hold_any": ("i_hold_any", 1),
    "flow21": ("i_flow21", -1),
}


def load(freq):
    P = H.load_panel(freq, holdout=HOLDOUT)
    P = FT.signals(P)
    for nm, (c, sg) in FEATS.items():
        P["s_" + nm] = sg * P[c]
    return P


# ------------------------------------------------------------------------------------------------------ Fama-MacBeth
def fama_macbeth(P, feat, target="fwd_63", controls=("cdi_bps", "resid_z", "dur", "r4", "own_shock"), sub=None):
    H_ = int(target.split("_")[1])
    X = P[P["univ"] & P[f"dok_{H_}"] & P[target].notna() & P[feat].notna() & (P["day"] >= "2021-06-01")]
    if sub is not None:
        X = X[X[sub]]
    sl = {}
    for d, x in X.groupby("day"):
        if len(x) < 40 or x[feat].nunique() < 3:
            continue
        y = x[target] - x[target].mean()
        f = x[feat].rank(pct=True) - 0.5          # rank-transformed feature: slope = top-vs-bottom spread
        Z = [f.to_numpy()]
        for c in controls:
            z = x[c].astype(float)
            z = z.fillna(z.median())
            Z.append((z.rank(pct=True) - 0.5).to_numpy())
        Z = np.column_stack(Z + [np.ones(len(x))])
        try:
            b = np.linalg.lstsq(Z, y.to_numpy(), rcond=None)[0]
        except Exception:
            continue
        sl[d] = b[0]
    s = pd.Series(sl).sort_index()
    if len(s) < 20:
        return {"n": len(s)}
    lag = max(H_ // 5, 1)
    ann = 252 / H_
    h1 = s[s.index < H.SPLIT]
    h2 = s[s.index >= H.SPLIT]
    return {"slope_ann_%": s.mean() * ann * 100, "t_nw": H.nw_t(s, lag), "n_dates": len(s),
            "h1_ann_%": h1.mean() * ann * 100, "h2_ann_%": h2.mean() * ann * 100,
            "t_h1": H.nw_t(h1, lag) if len(h1) > 20 else np.nan, "t_h2": H.nw_t(h2, lag) if len(h2) > 20 else np.nan}


# ------------------------------------------------------------------------------------------------------ event study
def event_study(P, target="fwd_63"):
    """Non-shocked bonds whose group-mate (or sector-mate) was hit by a hard shock (first in 13 weeks)."""
    H_ = int(target.split("_")[1])
    X = P[P["univ"] & P[f"dok_{H_}"] & P[target].notna() & (P["day"] >= "2021-06-01")].copy()
    X["y"] = X[target] - X.groupby("day")[target].transform("mean")
    X["own"] = X["own_shock"].astype(bool)
    out = {}
    for nm, cond in {
        "group_mate_shocked": (X["i_grp_hard_n"] >= 1) & ~X["own"],
        "group_mate_ret_shocked": (X["i_grp_ret_n"] >= 1) & ~X["own"],
        "sector_hard_top10pct": (X["i_sec_hard"] >= X.groupby("day")["i_sec_hard"].transform(lambda s: s.quantile(.9))) & ~X["own"],
        "sibling_drop_3pct_own_flat": (X["sib_min_r4"] <= -0.03) & (X["r4"] > -0.01),
        "holder_loss_top10pct": (X["i_hold_ret"] >= X.groupby("day")["i_hold_ret"].transform(lambda s: s.quantile(.9))) & ~X["own"],
        "holder_outflow_bottom10pct": (X["i_flow21"] <= X.groupby("day")["i_flow21"].transform(lambda s: s.quantile(.1))),
        "own_shock": X["own"],
    }.items():
        per = X[cond.fillna(False)].groupby("day")["y"].mean()
        n = int(cond.fillna(False).sum())
        if len(per) < 10:
            out[nm] = {"n_obs": n, "n_dates": len(per)}
            continue
        lag = max(H_ // 5, 1)
        out[nm] = {"n_obs": n, "n_dates": len(per), "mean_%": per.mean() * 100,
                   "ann_%": per.mean() * 252 / H_ * 100, "t_nw": H.nw_t(per, lag),
                   "h1_%": per[per.index < H.SPLIT].mean() * 100, "h2_%": per[per.index >= H.SPLIT].mean() * 100}
    return out


# ------------------------------------------------------------------------------------------------------ portfolios
def q_by_day(x, col, q):
    v = x[col]
    if v.notna().sum() < 20:
        return pd.Series(False, index=x.index)
    return v >= v.quantile(q)


def variants():
    """Selection rules on one monthly cross-section x (universe rows). All built on top of p4q."""
    nz = lambda s: s.fillna(0)
    V = {}
    V["avoid_group_shock"] = lambda x: (x["p4q"] & ~(nz(x["i_grp_hard_n"]) >= 1)).to_numpy()
    V["avoid_group_retshock"] = lambda x: (x["p4q"] & ~(nz(x["i_grp_ret_n"]) >= 1)).to_numpy()
    V["avoid_sibling_drop"] = lambda x: (x["p4q"] & ~((x["sib_min_r4"] <= -0.03) & (x["r4"] > -0.01))).to_numpy()
    V["avoid_holder_loss_d10"] = lambda x: (x["p4q"] & ~q_by_day(x, "i_hold_ret", 0.9)).to_numpy()
    V["avoid_holder_outflow_d10"] = lambda x: (x["p4q"] & ~q_by_day(x.assign(o=-x["i_flow21"]), "o", 0.9)).to_numpy()
    V["avoid_sector_stress_q5"] = lambda x: (x["p4q"] & ~q_by_day(x, "i_sec_hard", 0.8)).to_numpy()
    V["avoid_exposure_stress_q5"] = lambda x: (x["p4q"] & ~q_by_day(x, "s_exp_r4", 0.8)).to_numpy()
    V["sector_mom_drop_bottom_q"] = lambda x: (x["p4q"] & ~q_by_day(x, "s_sec_r13", 0.8)).to_numpy()
    V["sector_rev_drop_top_q"] = lambda x: (x["p4q"] & ~q_by_day(x.assign(o=x["i_sec_r13"]), "o", 0.8)).to_numpy()
    V["avoid_any_neighbour"] = lambda x: (x["p4q"] & ~(
        (nz(x["i_grp_hard_n"]) >= 1) | ((x["sib_min_r4"] <= -0.03) & (x["r4"] > -0.01))
        | q_by_day(x, "i_hold_ret", 0.9))).to_numpy()
    return V


def drop_placebo(P, rule, n=20, seed=1, **kw):
    """Null for an avoid filter: from p4q drop the SAME number of names at random each date."""
    rng = np.random.default_rng(seed)
    X = P[P["univ"] & (P["day"] >= H.START)]
    base = X[X["p4q"]]
    keep_n = {}
    for d, x in X.groupby("day"):
        keep_n[d] = int(np.asarray(rule(x)).sum())
    res = []
    for _ in range(n):
        rows = []
        for d, b in base.groupby("day"):
            k = min(keep_n.get(d, len(b)), len(b))
            pick = rng.choice(b["codigo"].to_numpy(), size=k, replace=False)
            rows.append(pd.DataFrame({"day": d, "codigo": pick, "select": True}))
        r = H.backtest(pd.concat(rows), panel=P, **kw)
        res.append(H.stats(r["daily"])["ann_excess_%"])
    return np.array(res)


def main():
    log("building / loading network features")
    FT.build()
    PW = load("W")
    PM = load("M")
    for P in (PW, PM):
        P["own_shock"] = P["i_any"].fillna(False).astype(float)
    Xu = PW[PW["univ"] & (PW["day"] < H.HOLDOUT)]
    RES["coverage"] = {k: float(Xu[c].notna().mean()) for k, (c, _) in FEATS.items()}
    RES["shock_rates_issuer_week"] = {k: float(Xu.drop_duplicates(["day", "cnpj8"])[f"i_{k}"].mean())
                                      for k in ("ret", "down", "press", "cvm", "eq", "any", "hard")}
    RES["in_multi_issuer_group_share"] = float(Xu["i_grp_hard"].notna().mean())
    log("coverage", RES["coverage"])

    # ---- 2) Fama-MacBeth
    fm = {}
    for tgt in ("fwd_21", "fwd_63"):
        for nm in FEATS:
            fm[f"{nm}|{tgt}"] = fama_macbeth(PW, "s_" + nm, tgt)
    for nm in FEATS:
        fm[f"{nm}|fwd_63|within_p4q"] = fama_macbeth(PW, "s_" + nm, "fwd_63", sub="p4q")
    RES["fama_macbeth"] = fm
    log("FM done")
    print(pd.DataFrame(fm).T.round(2).to_string())

    # ---- 3) event study
    RES["event_study"] = {t: event_study(PW, t) for t in ("fwd_21", "fwd_63")}
    print(pd.DataFrame(RES["event_study"]["fwd_63"]).T.round(3).to_string())
    print(pd.DataFrame(RES["event_study"]["fwd_21"]).T.round(3).to_string())

    # ---- 4) portfolios
    V = variants()
    runs = {nm: H.backtest(f, panel=PM, name=nm) for nm, f in V.items()}
    runs["P4"] = H.baseline("P4")
    tab = H.compare(runs, bench="P4Q")
    RES["portfolios_25bps"] = tab.round(4).to_dict(orient="index")
    RES["n_variants_tried"] = len(V)
    print(tab.round(3).to_string())
    runs50 = {nm: H.backtest(f, panel=PM, cost_bps=50, name=nm) for nm, f in V.items()}
    tab50 = H.compare(runs50, bench="P4Q", cost_bps=50)
    RES["portfolios_50bps"] = tab50.round(4).to_dict(orient="index")
    runs40 = {nm: H.backtest(f, panel=PM, scenario="rec40", name=nm) for nm, f in V.items()}
    tab40 = H.compare(runs40, bench=H.baseline("P4Q", scenario="rec40")["daily"],
                      universe=H.baseline("U", scenario="rec40")["daily"])
    RES["portfolios_rec40"] = tab40.round(4).to_dict(orient="index")
    log("portfolios done")

    cand = tab.drop(index="P4")
    best = cand["vs_bench_%"].astype(float).idxmax()
    RES["best_variant"] = best
    RES["best_row"] = cand.loc[best].to_dict()
    RES["best_placebo_drop"] = {}
    pl = drop_placebo(PM, V[best], n=20)
    RES["best_placebo_drop"] = {"actual": float(H.stats(runs[best]["daily"])["ann_excess_%"]),
                                "placebo_mean": float(pl.mean()), "placebo_p95": float(np.percentile(pl, 95)),
                                "p4q": float(H.stats(H.baseline("P4Q")["daily"])["ann_excess_%"]),
                                "share_placebo_ge_actual": float((pl >= H.stats(runs[best]["daily"])["ann_excess_%"]).mean())}
    log("placebo", RES["best_placebo_drop"])
    # names removed per date
    rem = {}
    for nm, f in V.items():
        X = PM[PM["univ"] & (PM["day"] >= H.START)]
        rem[nm] = float(np.mean([x["p4q"].sum() - np.asarray(f(x)).sum() for _, x in X.groupby("day")]))
    RES["avg_names_removed_from_p4q"] = rem

    # ---- 5) charts
    show = {best: runs[best]}
    for extra in ("avoid_any_neighbour", "sector_mom_drop_bottom_q"):
        if extra != best:
            show[extra] = runs[extra]
    H.plot_curves(show, HERE / "equity_total_return.png",
                  title="Network / contagion filters on P4+Q (monthly 126d tranches, 25 bps, pre-2026)")
    plot_cum_excess(runs, HERE / "cum_excess.png")
    log("charts done")

    # ---- holdout (only when asked, after every choice is frozen)
    if HOLDOUT:
        rh = H.backtest(V[best], panel=PM, holdout=True)
        bh = H.baseline("P4Q", holdout=True)
        uh = H.baseline("U", holdout=True)
        RES["holdout"] = {"best": best, "vs_p4q": H.stats(rh["daily"], bench=bh["daily"], holdout="only"),
                          "vs_u": H.stats(rh["daily"], bench=uh["daily"], holdout="only")}
        print(RES["holdout"])
    out = HERE / ("results_holdout.json" if HOLDOUT else "results.json")
    out.write_text(json.dumps(jf(RES), indent=1))
    log("wrote", out)


def plot_cum_excess(runs, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    u = H.monthly(H.baseline("U")["daily"])
    b = H.monthly(H.baseline("P4Q")["daily"])
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    for nm, r in runs.items():
        m = H.monthly(r["daily"])
        lw = 2.2 if nm in ("avoid_any_neighbour",) else 1.0
        ax[0].plot((m - u).cumsum() * 100, label=nm, lw=lw)
        ax[1].plot((m - b).cumsum() * 100, label=nm, lw=lw)
    ax[0].plot((b - u).cumsum() * 100, "k--", lw=2, label="P4+Q")
    ax[0].set_title("cumulative excess vs universe (%)")
    ax[1].axhline(0, color="k", lw=1)
    ax[1].set_title("cumulative difference vs P4+Q (%)")
    ax[0].legend(fontsize=7)
    for a in ax:
        a.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)


if __name__ == "__main__":
    main()
