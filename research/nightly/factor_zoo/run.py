"""Factor zoo: factor-by-factor ICs, decay, clustering, survivor composite, and backtests vs P4+Q.

Rerun (pre-2026 only; the sealed holdout is read ONLY with --holdout, once, after everything is frozen):
  PYTHONPATH=. PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
      .venv/Scripts/python.exe research/nightly/factor_zoo/run.py [--rebuild] [--holdout]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy import stats as sst
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from research.nightly import harness as H
from research.nightly.factor_zoo import features as FZ

OUT = Path(__file__).parent
T0 = time.time()
SEL_END = pd.Timestamp("2023-07-01")     # factor selection: decisions < 2023-07 (fwd_126 labels end before 2024)
VAL_START = pd.Timestamp("2024-01-01")   # validation of the ICs: decisions 2024-01 .. 2025-06 (labels pre-2026)


def log(*a):
    print(f"[run +{time.time() - T0:6.1f}s]", *a, flush=True)


CONTROLS = ["cdi_bps", "resid_z", "dur", "is_ipca", "incent"]
NEW = ["ds_5", "ds_21", "ds_63", "ds_126", "ds_252", "dres_21", "drz_21", "dres_63", "drz_63", "svol_63", "svol_126",
       "s_from_peak_252", "s_from_trough_252", "s_ownz_252", "rz_ma5", "rz_ma21", "rz_ma63", "rb_ma21",
       "rz_minus_ma63", "ratio_ch_21", "ratio_ch_63", "ratio_from_hi_252", "par_dist", "rmom_21", "rmom_63",
       "rmom_126", "rmom_252", "rmom_252_21", "rvol_126", "rdd_252", "rmin_126", "ntr_21", "ntr_63", "tdays_63",
       "log_vol_63", "ticket_63", "vol_trend", "amihud_126", "days_since_trade", "iss_resid_bps",
       "bond_minus_iss_resid", "iss_n_bonds", "iss_resid_disp", "iss_ds_63", "iss_rmom_126", "cs_minus_iss_cs",
       "log_issue_size", "guar_real", "guar_quiro", "is_476", "call_clause", "amortizing", "coupon_freq_m",
       "emission_no", "bond_age_y", "ttm_orig_y", "life_frac", "y_to_amort", "iss_n_issues",
       "is_ipca", "log_eq_adtv", "eq_idio63", "eq_mom_12_1", "eq_vol_trend", "rat_down_365", "ds_63_vs_peer",
       "sec_ds_63", "sec_resid", "cs_x_dur"]
ZOO = list(dict.fromkeys(NEW + [c for c in FZ.BASE_FEATS if c != "eq_adtv"]))


# ---------------------------------------------------------------------------------------------------------------------
def nscore(s: pd.Series) -> pd.Series:
    r = s.rank(method="average")
    n = s.notna().sum()
    return pd.Series(sst.norm.ppf((r - 0.5) / n), index=s.index)


def per_date_ics(P: pd.DataFrame, feats, target: str, mode: str, subset: str = "univ") -> pd.DataFrame:
    """Per-date IC of each feature. mode: raw (Spearman), peer (ranks within day x peer), partial (normal scores
    residualised on the control normal scores, both feature and target)."""
    Hh = int(target.split("_")[1])
    X = P[P[subset] & P[target].notna() & P[f"dok_{Hh}"]]
    out = {}
    for d, x in X.groupby("day"):
        if len(x) < 30:
            continue
        row = {}
        if mode == "raw":
            y = x[target].rank()
            for f in feats:
                v = x[f]
                ok = v.notna()
                if ok.sum() >= 30 and v[ok].nunique() > 2:
                    row[f] = np.corrcoef(v[ok].rank(), y[ok].rank())[0, 1]
        elif mode == "peer":
            g = x.groupby("peer")
            y = g[target].rank(pct=True) - 0.5
            for f in feats:
                v = g[f].rank(pct=True) - 0.5
                ok = v.notna() & y.notna()
                if ok.sum() >= 30 and x.loc[ok, f].nunique() > 2:
                    row[f] = np.corrcoef(v[ok], y[ok])[0, 1]
        else:  # partial
            Z = np.column_stack([np.ones(len(x))] + [nscore(x[c]).fillna(0).to_numpy() if x[c].nunique() > 2 else
                                                     x[c].fillna(0).to_numpy(float) for c in CONTROLS])
            y = nscore(x[target]).to_numpy()
            for f in feats:
                if f in CONTROLS:
                    continue
                v = x[f]
                ok = v.notna().to_numpy()
                if ok.sum() < 30 or v[ok].nunique() < 2:
                    continue
                fv = nscore(v[ok]).to_numpy() if v[ok].nunique() > 2 else v[ok].to_numpy(float)
                Zo = Z[ok]
                bf = np.linalg.lstsq(Zo, fv, rcond=None)[0]
                by = np.linalg.lstsq(Zo, y[ok], rcond=None)[0]
                ef, ey = fv - Zo @ bf, y[ok] - Zo @ by
                if ef.std() > 1e-9:
                    row[f] = np.corrcoef(ef, ey)[0, 1]
        out[d] = row
    return pd.DataFrame(out).T.sort_index()


def summarize(S: pd.DataFrame, H_: int, lo=None, hi=None) -> pd.DataFrame:
    X = S
    if lo is not None:
        X = X[X.index >= lo]
    if hi is not None:
        X = X[X.index < hi]
    lag = max(H_ // 21, 1)
    return pd.DataFrame({"ic": X.mean(), "t": X.apply(lambda s: H.nw_t(s, lag)), "n": X.count()})


def corr_clusters(P, feats, thr=0.5):
    U = P[P["univ"] & (P["day"] < H.HOLDOUT)]
    mats = []
    for d, x in U.groupby("day"):
        mats.append(x[feats].rank().corr(min_periods=100).to_numpy())
    Cm = np.nanmean(np.stack(mats), axis=0)
    Cm = pd.DataFrame(Cm, index=feats, columns=feats).fillna(0)
    D = 1 - Cm.abs().to_numpy()
    np.fill_diagonal(D, 0)
    D = (D + D.T) / 2
    L = linkage(squareform(np.clip(D, 0, None), checks=False), "average")
    cl = fcluster(L, t=1 - thr, criterion="distance")
    return Cm, pd.Series(cl, index=feats), L


def composite(P, spec: dict, col="zoo"):
    U = P["univ"]
    parts = []
    for f, s in spec.items():
        z = P[U].groupby("day")[f].transform(nscore)
        parts.append(z * s)
    P[col] = pd.concat(parts, axis=1).mean(axis=1, skipna=True).reindex(P.index)
    return P


def screen_fn(col, q=0.2, base="p4q"):
    """base names minus the worst q of `col` among the base names (NaN kept)."""
    def f(x):
        b = x[base].to_numpy()
        v = x[col].where(x[base])
        thr = v.quantile(q)
        bad = (v <= thr).to_numpy() & v.notna().to_numpy()
        return b & ~bad
    return f


def tilt_fn(col, lam=0.5):
    """same count as P4, chosen among p4f & ~worstQ by nscore(cdi_bps) + lam * composite."""
    def f(x):
        k = int(x["p4"].sum())
        cand = (x["p4f"] & ~x["worstQ"]).to_numpy()
        sc = nscore(x["cdi_bps"]).to_numpy() + lam * x[col].fillna(0).to_numpy()
        sc = np.where(cand, sc, -np.inf)
        sel = np.zeros(len(x), bool)
        if k > 0:
            sel[np.argsort(-sc)[:k]] = True
        return sel & cand
    return f


# ---------------------------------------------------------------------------------------------------------------------
def main(rebuild=False, do_holdout=False):
    FZ.build_features(force=rebuild)
    P = FZ.attach(H.load_panel("M"))
    feats = [f for f in ZOO if f in P.columns]
    log(f"panel {P.shape}; {len(feats)} candidate features")
    res = {"n_features": len(feats), "features": feats}

    # ---------- 1) ICs
    tabs = {}
    ser = {}
    for tgt in ("fwd_21", "fwd_63", "fwd_126", "fwd_252"):
        Hh = int(tgt.split("_")[1])
        for mode in (("raw", "partial") if tgt != "fwd_126" else ("raw", "peer", "partial")):
            S = per_date_ics(P, feats, tgt, mode)
            ser[(mode, tgt)] = S
            tabs[(mode, tgt, "all")] = summarize(S, Hh)
        log(f"ICs {tgt} done")
    Sq = per_date_ics(P, [f for f in feats if f != "cdi_bps"], "fwd_126", "partial", subset="p4q")
    ser[("p4q_partial", "fwd_126")] = Sq
    S = ser[("partial", "fwd_126")]
    T = pd.DataFrame({
        "raw126": tabs[("raw", "fwd_126", "all")]["ic"], "raw126_t": tabs[("raw", "fwd_126", "all")]["t"],
        "peer126": tabs[("peer", "fwd_126", "all")]["ic"], "peer126_t": tabs[("peer", "fwd_126", "all")]["t"],
        "part21": tabs[("partial", "fwd_21", "all")]["ic"], "part63": tabs[("partial", "fwd_63", "all")]["ic"],
        "part126": tabs[("partial", "fwd_126", "all")]["ic"], "part126_t": tabs[("partial", "fwd_126", "all")]["t"],
        "part252": tabs[("partial", "fwd_252", "all")]["ic"],
        "part126_sel": summarize(S, 126, hi=SEL_END)["ic"], "part126_sel_t": summarize(S, 126, hi=SEL_END)["t"],
        "part63_sel_t": summarize(ser[("partial", "fwd_63")], 63, hi=SEL_END)["t"],
        "part126_val": summarize(S, 126, lo=VAL_START)["ic"], "part126_val_t": summarize(S, 126, lo=VAL_START)["t"],
        "p4q126": summarize(Sq, 126)["ic"], "p4q126_t": summarize(Sq, 126)["t"],
        "p4q126_sel": summarize(Sq, 126, hi=SEL_END)["ic"], "p4q126_val": summarize(Sq, 126, lo=VAL_START)["ic"],
        "coverage": P[P["univ"]][feats].notna().mean(),
    })
    T = T.loc[feats]
    # ---------- 2) clusters
    Cm, cl, L = corr_clusters(P, feats)
    T["cluster"] = cl
    # ---------- 3) survivors (selection period only; controls excluded)
    elig = T.drop(index=[c for c in CONTROLS if c in T.index] + ["resid_bps", "cs_x_dur", "rb_ma21", "rz_ma5",
                                                                 "rz_ma21", "rz_ma63", "carry_per_dur"],
                  errors="ignore")
    elig = elig[elig["coverage"] >= 0.3]

    def pick(E, tcol, t2col, p4col, tmin=2.0):
        ok = (E[tcol].abs() >= tmin) & (np.sign(E[t2col]) == np.sign(E[tcol])) & \
             (np.sign(E[p4col]) == np.sign(E[tcol]))
        C_ = E[ok].copy()
        C_["abs_t"] = C_[tcol].abs()
        best = C_.sort_values("abs_t", ascending=False).groupby("cluster").head(1)
        best = best.sort_values("abs_t", ascending=False).head(8)
        return {f: int(np.sign(best.loc[f, tcol])) for f in best.index}

    spec_sel = pick(elig, "part126_sel_t", "part63_sel_t", "p4q126_sel")
    # in-sample variant (full pre-2026 window), for comparison only
    elig2 = elig.copy()
    elig2["part63_t_all"] = tabs[("partial", "fwd_63", "all")]["t"].reindex(elig2.index)
    spec_full = pick(elig2, "part126_t", "part63_t_all", "p4q126")
    log("survivors (selection period):", spec_sel)
    log("survivors (full pre-2026, in-sample):", spec_full)
    res["composite_spec"] = spec_sel
    res["composite_spec_insample"] = spec_full
    P = composite(P, spec_sel, "zoo")
    P = composite(P, spec_full, "zoo_is")
    # composite IC
    for c in ("zoo", "zoo_is"):
        s1 = per_date_ics(P, [c], "fwd_126", "partial")
        s2 = per_date_ics(P, [c], "fwd_126", "partial", subset="p4q")
        s3 = per_date_ics(P, [c], "fwd_126", "raw")
        res[f"ic_{c}"] = {
            "raw126": summarize(s3, 126).loc[c].to_dict(),
            "partial126_all": summarize(s1, 126).loc[c].to_dict(),
            "partial126_sel": summarize(s1, 126, hi=SEL_END).loc[c].to_dict(),
            "partial126_val": summarize(s1, 126, lo=VAL_START).loc[c].to_dict(),
            "p4q126_all": summarize(s2, 126).loc[c].to_dict(),
            "p4q126_val": summarize(s2, 126, lo=VAL_START).loc[c].to_dict()}
    log("composite ICs", json.dumps(res["ic_zoo"], default=float)[:400])

    # ---------- 4) backtests (monthly decisions, 126-bday tranches, 25 bps) vs P4+Q
    bt = lambda sig, **kw: H.backtest(sig, freq="M", hold=126, panel=P, **kw)
    top3 = list(spec_sel)[:3]
    for f in top3:
        P[f"s_{f}"] = P[f] * spec_sel[f]
    # factor-group sub-composites (diagnostic: which group carries the increment); counted in Holm
    groups = {"eq": [f for f in spec_sel if f.startswith("eq_")],
              "markdyn": [f for f in spec_sel if f in ("ds_5", "dres_21", "ds_21", "drz_21", "ratio_ch_21")]}
    groups["other"] = [f for f in spec_sel if f not in groups["eq"] + groups["markdyn"]]
    for gname, fl in groups.items():
        if fl:
            P = composite(P, {f: spec_sel[f] for f in fl}, f"zoo_{gname}")
    res["groups"] = groups
    variants = {
        "zoo_screen20": screen_fn("zoo", 0.2),
        "zoo_screen33": screen_fn("zoo", 1 / 3),
        "zoo_tilt": tilt_fn("zoo", 0.5),
        "zoo_is_screen20 (in-sample)": screen_fn("zoo_is", 0.2),
        **{f"single_{f}_screen20": screen_fn(f"s_{f}", 0.2) for f in top3},
        "p4_zoo_screen20 (vs P4 w/o quality)": screen_fn("zoo", 0.2, base="p4"),
        **{f"group_{g}_screen20": screen_fn(f"zoo_{g}", 0.2) for g, fl in groups.items() if fl},
    }
    R = {nm: bt(fn, name=nm) for nm, fn in variants.items()}
    R["P4"] = H.baseline("P4")
    tab = H.compare(R, bench="P4Q")
    log("\n" + tab.round(3).to_string())
    res["n_variants_tried"] = len(variants)
    res["compare_25bps"] = tab.round(4).to_dict(orient="index")
    # best = highest paired t among the pre-registered out-of-sample variants (not in-sample / not single)
    cand = [v for v in ("zoo_screen20", "zoo_screen33", "zoo_tilt")]
    best = max(cand, key=lambda v: tab.loc[v, "t_vs_bench"])
    res["best_variant"] = best
    # costs, rec40, weekly, cohort
    rb = {}
    rb["cost50"] = H.stats(bt(variants[best], cost_bps=50)["daily"], bench=H.baseline("P4Q", cost_bps=50)["daily"])
    rb["rec40"] = H.stats(bt(variants[best], scenario="rec40")["daily"],
                          bench=H.baseline("P4Q", scenario="rec40")["daily"])
    rb["hold63"] = H.stats(H.backtest(variants[best], freq="M", hold=63, panel=P)["daily"],
                           bench=H.baseline("P4Q", hold=63)["daily"])
    PW = FZ.attach(H.load_panel("W"))
    PW = composite(PW, spec_sel, "zoo")
    rb["weekly"] = H.stats(H.backtest(variants[best], freq="W", hold=126, panel=PW)["daily"],
                           bench=H.baseline("P4Q", freq="W")["daily"])
    rb["vs_U_25"] = H.stats(R[best]["daily"], bench=H.baseline("U")["daily"])
    res["robustness_best"] = rb
    log("robustness", json.dumps({k: {kk: v.get(kk) for kk in ("ann_excess_%", "diff_ann_%", "diff_t_nw")}
                                  for k, v in rb.items()}, default=float))
    # placebo: drop the same fraction of P4+Q names at random
    frac = 0.2 if best != "zoo_screen33" else 1 / 3
    rng = np.random.default_rng(7)
    pl = []
    for i in range(30):
        P["_rnd"] = rng.random(len(P))
        r = bt(screen_fn("_rnd", frac) if best != "zoo_tilt" else tilt_fn("_rnd", 0.5))
        pl.append(H.stats(r["daily"], bench=H.baseline("P4Q")["daily"])["diff_ann_%"])
    pl = np.array(pl)
    act = tab.loc[best, "vs_bench_%"]
    res["placebo_best"] = {"null_mean": float(pl.mean()), "null_p95": float(np.percentile(pl, 95)),
                           "actual": float(act), "p_emp": float((pl >= act).mean())}
    log("placebo", res["placebo_best"])

    # ---------- 5) signals cache (frozen spec; PIT features; covers all dates incl. 2026 — spec chosen pre-2026)
    PA = FZ.attach(H.load_panel("M", holdout=True))
    PA = composite(PA, spec_sel, "zoo")
    PWA = FZ.attach(H.load_panel("W", holdout=True))
    PWA = composite(PWA, spec_sel, "zoo")
    sig = pd.concat([PA[PA["univ"]][["day", "codigo", "cnpj8", "zoo"]].assign(freq="M"),
                     PWA[PWA["univ"]][["day", "codigo", "cnpj8", "zoo"]].assign(freq="W")])
    sig = sig.rename(columns={"zoo": "value"}).dropna(subset=["value"])
    sig.to_pickle(FZ.CACHE / "zoo_composite.pkl")
    res["signal_cache"] = str(FZ.CACHE / "zoo_composite.pkl")

    # ---------- 6) charts
    plots(P, R, best, T, ser, Cm, L, feats)

    # ---------- 7) tables to json
    res["factor_table"] = T.round(4).to_dict(orient="index")
    res["decay"] = {f: {h: float(tabs[("partial", f"fwd_{h}", "all")]["ic"].get(f, np.nan)) for h in (21, 63, 126, 252)}
                    for f in list(spec_sel) + list(spec_full)}
    res["clusters"] = {int(k): list(v.index) for k, v in cl.groupby(cl)}

    # ---------- 8) holdout (ONCE, frozen)
    if do_holdout:
        rh = H.backtest(variants[best], freq="M", hold=126, holdout=True, panel=PA)
        bh = H.baseline("P4Q", holdout=True)
        uh = H.baseline("U", holdout=True)
        res["holdout"] = {"vs_P4Q": H.stats(rh["daily"], bench=bh["daily"], holdout="only"),
                          "vs_U": H.stats(rh["daily"], bench=uh["daily"], holdout="only"),
                          "note": "months >= 2026-01; decisions from 2026 only enter tranches from 2026; frozen spec"}
        log("HOLDOUT", json.dumps(res["holdout"], default=float)[:600])
    (OUT / "results.json").write_text(json.dumps(res, indent=1, default=lambda o: float(o) if isinstance(
        o, (np.floating, np.integer)) else str(o)))
    log("done")
    return res


def plots(P, R, best, T, ser, Cm, L, feats):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    dd = H.days()
    refs = {"CDI": None, "Universe": H.baseline("U")["daily"], "P4": H.baseline("P4")["daily"],
            "P4+Q": H.baseline("P4Q")["daily"]}
    for nm in ("IDADI", "IBOV"):
        try:
            refs[{"IDADI": "IDA-DI", "IBOV": "Ibovespa"}[nm]] = H.index_excess(nm)
        except Exception:
            pass
    mine = {best: R[best]["daily"], "zoo_is_screen20 (in-sample)": R["zoo_is_screen20 (in-sample)"]["daily"]}
    if "group_eq_screen20" in R:
        mine["group_eq_screen20 (equity factors only)"] = R["group_eq_screen20"]["daily"]
    idx = R[best]["daily"].index
    sty = {"CDI": ("k", ":"), "Universe": ("#7f7f7f", "--"), "P4": ("#1f77b4", "--"), "P4+Q": ("#17becf", "--"),
           "IDA-DI": ("#bcbd22", "-."), "Ibovespa": ("#c7c7c7", "-.")}
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=110)
    for nm, s in refs.items():
        s = pd.Series(0.0, index=idx) if s is None else s.reindex(idx).fillna(0)
        c, ls = sty[nm]
        ax.plot(idx, H.total_return_curve(s), ls, color=c, lw=1.1, label=nm)
    for nm, s in mine.items():
        ax.plot(idx, H.total_return_curve(s.reindex(idx).fillna(0)), lw=1.8, label=nm)
    ax.set_yscale("log")
    ax.grid(alpha=.25)
    ax.legend(fontsize=8, frameon=False)
    ax.set_title("Total return CDI x (1+excess), base 100 — factor-zoo screen vs references (pre-2026, 25 bps)")
    fig.tight_layout()
    fig.savefig(OUT / "equity_total_return.png")
    plt.close(fig)
    uni = refs["Universe"].reindex(idx).fillna(0)
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=110)
    for nm in ("P4", "P4+Q", "IDA-DI"):
        if nm in refs:
            s = refs[nm].reindex(idx).fillna(0)
            ax.plot(idx, ((1 + s - uni).cumprod() - 1) * 100, sty[nm][1], color=sty[nm][0], lw=1.1, label=nm)
    for nm, s in mine.items():
        s = s.reindex(idx).fillna(0)
        ax.plot(idx, ((1 + s - uni).cumprod() - 1) * 100, lw=1.8, label=nm)
    ax.axhline(0, color="k", lw=.6)
    ax.grid(alpha=.25)
    ax.legend(fontsize=8, frameon=False)
    ax.set_title("Cumulative excess vs universe (%)")
    fig.tight_layout()
    fig.savefig(OUT / "cum_excess.png")
    plt.close(fig)
    # IC bar chart: partial IC selection vs validation, and within-P4Q
    top = T.drop(index=[c for c in CONTROLS if c in T.index]).reindex(
        T["part126_t"].abs().sort_values(ascending=False).index).dropna(subset=["part126"]).head(30)
    fig, ax = plt.subplots(figsize=(10, 9), dpi=110)
    y = np.arange(len(top))
    ax.barh(y - 0.27, top["part126_sel"], 0.27, label="partial IC 6m, selection 2021-03..2023-06")
    ax.barh(y, top["part126_val"], 0.27, label="partial IC 6m, validation 2024-01..2025-06")
    ax.barh(y + 0.27, top["p4q126"], 0.27, label="partial IC 6m inside P4+Q (all pre-2026)")
    ax.set_yticks(y)
    ax.set_yticklabels(top.index, fontsize=8)
    ax.invert_yaxis()
    ax.axvline(0, color="k", lw=.6)
    ax.legend(fontsize=8)
    ax.set_title("Top-30 factors by |partial IC t| (controls: carry, RV, dur, IPCA, incentivada)")
    fig.tight_layout()
    fig.savefig(OUT / "factor_ics.png")
    plt.close(fig)
    # decay
    fig, ax = plt.subplots(figsize=(8, 5), dpi=110)
    for f in list(T["part126_t"].abs().drop(index=[c for c in CONTROLS if c in T.index]).sort_values(
            ascending=False).index[:10]):
        ax.plot([21, 63, 126, 252], [T.loc[f, "part21"], T.loc[f, "part63"], T.loc[f, "part126"], T.loc[f, "part252"]],
                marker="o", label=f)
    ax.axhline(0, color="k", lw=.6)
    ax.set_xlabel("horizon (bdays)")
    ax.set_ylabel("partial IC")
    ax.legend(fontsize=7)
    ax.set_title("Factor decay (partial IC by horizon), top-10 factors")
    fig.tight_layout()
    fig.savefig(OUT / "factor_decay.png")
    plt.close(fig)


if __name__ == "__main__":
    main(rebuild="--rebuild" in sys.argv, do_holdout="--holdout" in sys.argv)
