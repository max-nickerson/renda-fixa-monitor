"""w2_unlisted_issuer_health: extend P7's equity-health (EQH) screen to unlisted debenture issuers.

  PYTHONPATH=. PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/Scripts/python.exe research/nightly/w2_unlisted_issuer_health/build_map.py
  PYTHONPATH=. PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/Scripts/python.exe research/nightly/w2_unlisted_issuer_health/run.py            # pre-2026
  PYTHONPATH=. PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/Scripts/python.exe research/nightly/w2_unlisted_issuer_health/run.py --holdout  # once

PRE-REGISTERED (before any backtest here): the screen intensity is P7's (bottom 20%), carry construction and liquidity
caps are P7's, nothing is tuned.  Tiers: own harness equity -> PIT FRE parent equity -> static name-token parent ->
proxy (sector-peer equity composite / fundamentals f_quality / bond-mark health / their mean = cascade).
Equity tiers are pooled and ranked together (same composite as P7).  Proxy-tier names are ranked among the day's
universe proxy-tier rows and the bottom 20% dropped.  Selection of the 'best' variant = highest NW t vs P4+Q pre-2026.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from research.nightly import harness as H
from research.nightly.combined import p7
from research.nightly.combined import run as CR
from research.nightly.w2_unlisted_issuer_health import features as F

OUT = Path("research/nightly/w2_unlisted_issuer_health")
CACHE = F.CACHE
T0 = time.time()
HOLD = "--holdout" in sys.argv
RES: dict = {}


def log(*a):
    print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)


rnd = CR.rnd
KW = dict(as_weights=True, issuer_cap=1.0)


# ------------------------------------------------------------------------------------------------ screens
def eq_score(x, tiers):
    ok = x["eq_tier"].isin(tiers).to_numpy() & x[list(F.XQ)].notna().all(axis=1).to_numpy()
    out = np.full(len(x), np.nan)
    if ok.sum() < 10:
        return out
    r = np.zeros(ok.sum())
    for c, s in zip(F.XQ, p7.EQ_SIGN):
        r += (s * x.loc[ok, c]).rank(pct=True).to_numpy()
    out[ok] = r / 3
    return out


def bottom(s, q):
    ok = np.isfinite(s)
    if ok.sum() < 10:
        return np.zeros(len(s), bool)
    return ok & (s <= np.quantile(s[ok], q))


def proxy_col(x, proxy):
    if proxy == "cascade":
        return x[["px_sector", "px_fund", "px_bond"]].mean(axis=1, skipna=True).to_numpy(float)
    return x[proxy].to_numpy(float)


def flags(x, tiers=("own",), proxy=None, q=p7.Q_SCREEN):
    drop = bottom(eq_score(x, tiers), q)
    if proxy is not None:
        rest = ~(x["eq_tier"].isin(tiers).to_numpy() & x[list(F.XQ)].notna().all(axis=1).to_numpy())
        v = np.where(rest, proxy_col(x, proxy), np.nan)
        drop |= bottom(v, q)
    return drop


def make_signal(tiers=("own",), proxy=None, base="p4q", construct="carry", rand_seed=None, rand_ref=None):
    """rand_seed: placebo - instead of the proxy screen, drop as many unscreened ('rest') base names as `rand_ref`
    (tiers, proxy) would drop from them, chosen at random."""
    def fn(x):
        sel = x[base].to_numpy().astype(bool)
        if rand_seed is None:
            drop = flags(x, tiers, proxy)
        else:
            drop = flags(x, tiers, None)
            ref = flags(x, *rand_ref)
            extra = sel & ref & ~drop
            rest = sel & ~drop & ~(x["eq_tier"].isin(rand_ref[0]).to_numpy() & x[list(F.XQ)].notna().all(axis=1).to_numpy())
            idx = np.flatnonzero(rest)
            rng = np.random.default_rng(rand_seed * 7919 + int(x["day"].iloc[0].strftime("%Y%m")))
            k = int(extra.sum())
            if k and len(idx):
                drop = drop.copy()
                drop[rng.choice(idx, size=min(k, len(idx)), replace=False)] = True
        sel = sel & ~drop
        if construct is None:
            w = np.zeros(len(x))
            if sel.sum() >= H.MIN_NAMES:
                w[sel] = H.cap_weights(x.loc[sel, "cnpj8"].to_numpy(), 0.10)
            return w
        return p7.carry_weights(x, sel, mode=construct)
    return fn


def bt(fn, P, **kw):
    return H.backtest(fn, panel=P, **KW, **kw)


VARIANTS = {
    "P7": dict(tiers=("own",)),
    "X1_parent_fre_name": dict(tiers=("own", "fre", "name")),
    "X3_sector_proxy": dict(tiers=("own", "fre", "name"), proxy="px_sector"),
    "X4_fund_proxy": dict(tiers=("own", "fre", "name"), proxy="px_fund"),
    "X5_bond_proxy": dict(tiers=("own", "fre", "name"), proxy="px_bond"),
    "X6_cascade": dict(tiers=("own", "fre", "name"), proxy="cascade"),
}


# ------------------------------------------------------------------------------------------------ diagnostics
def coverage(P):
    U = P[P["univ"] & (P["day"] >= H.START)]
    out = {}
    wP7 = p7.weights_frame(make_signal(), P)
    wP7 = wP7.merge(P[["day", "codigo", "eq_tier", "px_sector", "px_fund", "px_bond"]], on=["day", "codigo"], how="left")
    for nm, x, w in (("universe_rows", U, None), ("p4q_rows", U[U["p4q"]], None), ("P7_weight", wP7, "weight")):
        ww = x[w] if w else pd.Series(1.0, index=x.index)
        tot = ww.sum()
        d = {t: float(ww[x["eq_tier"] == t].sum() / tot) for t in ("own", "fre", "name", "none")}
        rest = x["eq_tier"] == "none"
        for c in ("px_sector", "px_fund", "px_bond"):
            d[f"none_with_{c}"] = float(ww[rest & x[c].notna()].sum() / tot)
        d["none_with_any_proxy"] = float(ww[rest & x[["px_sector", "px_fund", "px_bond"]].notna().any(axis=1)].sum() / tot)
        d["none_no_proxy"] = float(ww[rest & x[["px_sector", "px_fund", "px_bond"]].isna().all(axis=1)].sum() / tot)
        if "eq_map_type" in x:
            d["harness_direct"] = float(ww[x["eq_map_type"] == "direct"].sum() / tot)
            d["harness_parent"] = float(ww[x["eq_map_type"] == "parent"].sum() / tot)
        out[nm] = d
    # who are the unlisted names inside P4+Q
    names = pd.read_csv(H.RDATA / "issuer_sectors.csv", dtype=str).drop_duplicates("cnpj8").set_index("cnpj8")["issuer_name"]
    top = U[U["p4q"] & (U["eq_tier"] == "none")].groupby("cnpj8").size().sort_values(ascending=False).head(25)
    out["top_unlisted_p4q_issuers"] = {f"{k} {names.get(k, '')[:40]}": int(v) for k, v in top.items()}
    out["n_unlisted_issuers_U"] = int(U.loc[U["eq_tier"] == "none", "cnpj8"].nunique())
    out["n_fre_mapped_issuers_U"] = int(U.loc[U["eq_tier"] == "fre", "cnpj8"].nunique())
    out["n_name_mapped_issuers_U"] = int(U.loc[U["eq_tier"] == "name", "cnpj8"].nunique())
    return out


def ic_block(P):
    """Per-date Spearman ICs vs fwd_126 on subsets (universe rows, dok dates)."""
    Q = P.copy()
    Q["eq_comp_own"] = np.nan
    Q["eq_comp_ext"] = np.nan
    for d, x in Q[Q["univ"]].groupby("day"):
        Q.loc[x.index, "eq_comp_own"] = eq_score(x, ("own",))
        Q.loc[x.index, "eq_comp_ext"] = eq_score(x, ("own", "fre", "name"))
    Q["px_cascade"] = Q[["px_sector", "px_fund", "px_bond"]].mean(axis=1, skipna=True).where(Q["univ"])
    out = {}
    sub = {"listed_own": Q["eq_tier"] == "own", "unlisted_none": Q["eq_tier"] == "none", "fre_parent": Q["eq_tier"] == "fre",
           "all": Q["univ"]}
    for sn, m in sub.items():
        for f in ("eq_comp_own", "eq_comp_ext", "px_sector", "px_fund", "px_bond", "px_cascade", "cdi_bps"):
            if sn == "unlisted_none" and f.startswith("eq_comp"):
                continue
            if sn == "fre_parent" and f != "eq_comp_ext":
                continue
            try:
                r = H.ic(Q[m], f, "fwd_126", min_n=15 if sn == "fre_parent" else 30)
                out[f"{sn}:{f}"] = {"mean": r["mean"], "t_nw": r["t_nw"], "n_dates": r["n_dates"]}
            except Exception as e:  # pragma: no cover
                out[f"{sn}:{f}"] = str(e)
    # proxy validation on listed rows: correlation of each proxy with the true equity composite
    L = Q[Q["univ"] & (Q["eq_tier"] == "own")]
    out["proxy_vs_true_eqh_spearman_listed"] = {c: float(L.groupby("day").apply(lambda g: g[c].corr(g["eq_comp_own"], method="spearman")).mean())
                                                for c in ("px_sector", "px_fund", "px_bond", "px_cascade")}
    # bottom-quintile hit: do proxy-bottom-20% listed names overlap the true EQH bottom-20%?
    hit = {}
    for c in ("px_sector", "px_fund", "px_bond", "px_cascade"):
        a, b = [], []
        for d, g in L.groupby("day"):
            g = g[g[c].notna() & g["eq_comp_own"].notna()]
            if len(g) < 30:
                continue
            fb = g[c] <= g[c].quantile(0.2)
            tb = g["eq_comp_own"] <= g["eq_comp_own"].quantile(0.2)
            a.append((fb & tb).sum() / max(fb.sum(), 1))
        hit[c] = float(np.mean(a))
    out["proxy_bottom20_precision_vs_true_eqh_listed(random=0.20)"] = hit
    return out


# ------------------------------------------------------------------------------------------------ unlisted / orphan
def orphan_block(P):
    """Is unlisted / orphan outperformance a stale-mark / survivorship artefact?  Same assumptions on both sides."""
    ff = pd.read_pickle(H.HIST / "nightly" / "fund_flows" / "bond_signals_M.pkl")[["day", "codigo", "nh"]]
    Q = P.merge(ff, on=["day", "codigo"], how="left")
    Q["orphan"] = Q["nh"].fillna(-1) == 0
    Q["held"] = Q["nh"].fillna(-1) > 0
    listed = lambda x: (x["eq_tier"] == "own").to_numpy()
    books = {
        "U_listed": lambda x: x["univ"].to_numpy() & listed(x),
        "U_unlisted": lambda x: x["univ"].to_numpy() & ~listed(x),
        "P4Q_listed": lambda x: x["p4q"].to_numpy() & listed(x),
        "P4Q_unlisted": lambda x: x["p4q"].to_numpy() & ~listed(x),
        "U_orphan": lambda x: x["univ"].to_numpy() & x["orphan"].to_numpy(),
        "U_held": lambda x: x["univ"].to_numpy() & x["held"].to_numpy(),
        "P4Q_orphan": lambda x: x["p4q"].to_numpy() & x["orphan"].to_numpy(),
        "P4Q_held": lambda x: x["p4q"].to_numpy() & x["held"].to_numpy(),
    }
    cb = CR.cost_vector(Q)
    Rh, nh = CR.harsh_R()
    scen = {"base": {}, "harsh_surv": dict(R=Rh), "liq_costs": dict(cost_b=cb), "harsh+liq": dict(R=Rh, cost_b=cb)}
    out = {"n_bonds_harsh": nh}
    series = {}
    for sn, kw in scen.items():
        lv = {}
        for bn, fn in books.items():
            r = CR.engine_run(fn, Q, as_w=False, icap=0.10, **kw)
            series[(sn, bn)] = r["daily"]
            lv[bn] = H.stats(r["daily"])["ann_excess_%"]
        pairs = {}
        for a, b in (("U_unlisted", "U_listed"), ("P4Q_unlisted", "P4Q_listed"), ("U_orphan", "U_held"), ("P4Q_orphan", "P4Q_held")):
            s = H.stats(series[(sn, a)], bench=series[(sn, b)])
            pairs[f"{a}-{b}"] = {"diff": s["diff_ann_%"], "t": s["diff_t_nw"], "h1": s["diff_h1_%"], "h2": s["diff_h2_%"]}
        out[sn] = {"levels_exCDI": lv, "pairs": pairs}
        log("orphan", sn, rnd(pairs, 3))
    # stale-mark diagnostics: trade frequency, share that goes silent (harsh set), smoothing (AR1 of monthly excess)
    C = H._core()
    last = C["last"]
    harsh_codes = set(last[(last["last_day"] <= H.HOLDOUT - pd.Timedelta(days=90)) &
                           (last["maturity"].isna() | (last["maturity"] > last["last_day"] + pd.Timedelta(days=60))) &
                           (last["last_ratio"] < 0.98)].index)
    U = Q[Q["univ"] & (Q["day"] >= H.START)]
    diag = {}
    for nm, m in (("listed", U["eq_tier"] == "own"), ("unlisted", U["eq_tier"] != "own"), ("orphan", U["orphan"]), ("held", U["held"])):
        x = U[m]
        diag[nm] = {"trades_30d_med": float(x["trades_30d"].median()), "share_rows_later_harsh": float(x["codigo"].isin(harsh_codes).mean()),
                    "cdi_bps_med": float(x["cdi_bps"].median()), "dur_med": float(x["dur"].median()),
                    "fwd126_mean_%": float(x["fwd_126"].mean() * 100), "n_rows": int(len(x))}
    for bn in ("U_listed", "U_unlisted", "U_orphan", "U_held"):
        m = H.monthly(series[("base", bn)])
        diag.setdefault("ar1_monthly", {})[bn] = float(m.autocorr(1))
    out["diagnostics"] = diag
    # carry-matched: within-P4Q unlisted-listed cohort, controlling for carry decile
    out["cohort_carry_neutral"] = carry_neutral(U)
    return out, series


def carry_neutral(U):
    """Per-date mean fwd_126 of unlisted minus listed inside carry deciles (P4Q-free, universe), NW t."""
    x = U[U["fwd_126"].notna() & U["dok_126"]].copy()
    x["dec"] = x.groupby("day")["cdi_bps"].transform(lambda s: pd.qcut(s.rank(method="first"), 10, labels=False))
    x["unl"] = x["eq_tier"] != "own"
    g = x.groupby(["day", "dec", "unl"])["fwd_126"].mean().unstack()
    d = (g[True] - g[False]).groupby(level=0).mean().dropna()
    return {"mean_6m_%": float(d.mean() * 100), "ann_%": float(d.mean() * 2 * 100), "t_nw": float(H.nw_t(d.to_numpy(), 6)),
            "n_dates": int(len(d))}


# ------------------------------------------------------------------------------------------------ main
def main():
    Pall = pd.read_pickle(CACHE / "panel_M_ext_all.pkl") if (CACHE / "panel_M_ext_all.pkl").exists() else None
    if Pall is None:
        Pall = F.attach(p7.load(holdout=True))
        Pall.to_pickle(CACHE / "panel_M_ext_all.pkl")
    P = Pall[Pall["day"] < H.HOLDOUT].reset_index(drop=True)
    log("panel", P.shape)
    RES["coverage"] = rnd(coverage(P))
    log("coverage", json.dumps(RES["coverage"], indent=0)[:1500])
    RES["ic"] = rnd(ic_block(P))
    log("ic", RES["ic"])

    R = {k: bt(make_signal(**v), P) for k, v in VARIANTS.items()}
    ref = bt(p7.make_signal(), P)
    RES["p7_repro_maxabs"] = float((ref["daily"] - R["P7"]["daily"]).abs().max())
    log("P7 repro maxabs", RES["p7_repro_maxabs"])
    comp = H.compare({**R, "P4": H.baseline("P4")}, bench="P4Q")
    comp.to_csv(OUT / "compare_pre2026.csv")
    log("\n" + comp.to_string())
    RES["compare_vs_P4Q"] = rnd(comp.reset_index().to_dict(orient="records"))
    # paired vs P7 (the question: does the extension add to P7?)
    vsP7 = {}
    for k in VARIANTS:
        if k == "P7":
            continue
        s = H.stats(R[k]["daily"], bench=R["P7"]["daily"])
        vsP7[k] = {"diff": s["diff_ann_%"], "t": s["diff_t_nw"], "p": s["diff_p"], "h1": s["diff_h1_%"], "h2": s["diff_h2_%"]}
    ps = [v["p"] for v in vsP7.values()]
    for (k, v), hp in zip(vsP7.items(), H.holm(ps)):
        v["holm_p"] = float(hp)
    RES["vs_P7"] = rnd(vsP7)
    log("vs P7", RES["vs_P7"])
    # costs 50 bps, rec40
    sens = {}
    b50 = H.baseline("P4Q", cost_bps=50)
    b40 = H.baseline("P4Q", scenario="rec40")
    for k, v in VARIANTS.items():
        r50 = bt(make_signal(**v), P, cost_bps=50)
        r40 = bt(make_signal(**v), P, scenario="rec40")
        s50 = H.stats(r50["daily"], bench=b50["daily"])
        s40 = H.stats(r40["daily"], bench=b40["daily"])
        sens[k] = {"net50_vs_P4Q": s50["diff_ann_%"], "net50_t": s50["diff_t_nw"], "rec40_vs_P4Q": s40["diff_ann_%"], "rec40_t": s40["diff_t_nw"]}
    RES["sensitivity"] = rnd(sens)
    log("sens", RES["sensitivity"])
    # realistic (liq costs + harsh survivorship), vs P4Q and vs P7, same assumptions
    cb = CR.cost_vector(P)
    Rh, nh = CR.harsh_R()
    real = {}
    base_mask = lambda x: x["p4q"].to_numpy()
    for tag, kw in {"liq_costs": dict(cost_b=cb), "harsh+liq": dict(cost_b=cb, R=Rh)}.items():
        eq = CR.engine_run(base_mask, P, as_w=False, icap=0.10, **kw)
        e7 = CR.engine_run(make_signal(), P, **kw)
        d = {}
        for k in ("X1_parent_fre_name", "X3_sector_proxy", "X4_fund_proxy", "X5_bond_proxy", "X6_cascade"):
            ek = CR.engine_run(make_signal(**VARIANTS[k]), P, **kw)
            a = H.stats(ek["daily"], bench=eq["daily"])
            b = H.stats(ek["daily"], bench=e7["daily"])
            d[k] = {"vs_P4Q": a["diff_ann_%"], "t": a["diff_t_nw"], "vs_P7": b["diff_ann_%"], "t_vs_P7": b["diff_t_nw"]}
        d["P7_vs_P4Q"] = H.stats(e7["daily"], bench=eq["daily"])["diff_ann_%"]
        real[tag] = d
    RES["realistic"] = rnd(real)
    log("realistic", RES["realistic"])
    # placebo: random drop of the same number of unscreened P4Q names each date (vs the cascade and the best proxy)
    best = max((k for k in VARIANTS if k != "P7"), key=lambda k: comp.loc[k, "t_vs_bench"] if "t_vs_bench" in comp.columns else 0)
    pl = {}
    for ref_k in ("X6_cascade", best) if best != "X6_cascade" else ("X6_cascade",):
        ref_v = VARIANTS[ref_k]
        vals = []
        for sd in range(20):
            r = bt(make_signal(tiers=ref_v["tiers"], rand_seed=sd + 1, rand_ref=(ref_v["tiers"], ref_v.get("proxy"))), P)
            vals.append(H.stats(r["daily"], bench=R["P7"]["daily"])["diff_ann_%"])
        act = H.stats(R[ref_k]["daily"], bench=R["P7"]["daily"])["diff_ann_%"]
        pl[ref_k] = {"actual_vs_P7": act, "placebo_mean": float(np.mean(vals)), "placebo_p95": float(np.percentile(vals, 95)),
                     "placebo_p05": float(np.percentile(vals, 5)), "p_value": float((np.array(vals) >= act).mean())}
    RES["placebo_vs_P7"] = rnd(pl)
    RES["best_variant_rule"] = "highest NW t vs P4+Q pre-2026 among X variants"
    RES["best_variant"] = best
    log("placebo", RES["placebo_vs_P7"], "best", best)
    # by year and top-month dependence of best vs P7
    RES["best_vs_P7_by_year"] = rnd(CR.by_year(R[best]["daily"], R["P7"]["daily"]))
    RES["best_vs_P7_excl_top3_months"] = rnd(CR.excl_top_months(R[best]["daily"], R["P7"]["daily"]))
    RES["best_vs_P4Q_excl_top3_months"] = rnd(CR.excl_top_months(R[best]["daily"], H.baseline("P4Q")["daily"]))
    # unlisted / orphan block
    ob, _ = orphan_block(P)
    RES["unlisted_orphan"] = rnd(ob)
    # charts
    H.plot_curves({"P7": R["P7"], best: R[best], "X6_cascade": R["X6_cascade"]}, OUT / "equity_total_return.png",
                  title="Unlisted issuer health: P7 vs extended screens (pre-2026)")
    cum_chart(R, OUT / "cum_vs_P7.png")
    RES["n_variants_tried"] = len(VARIANTS) - 1
    RES["runtime_s"] = time.time() - T0
    json.dump(rnd(RES), open(OUT / "results.json", "w"), indent=1, default=str)
    log("done")


def cum_chart(R, path):
    fig, ax = plt.subplots(1, 1, figsize=(9, 4.5))
    base = H.monthly(R["P7"]["daily"])
    for k in VARIANTS:
        if k == "P7":
            continue
        d = (H.monthly(R[k]["daily"]) - base).cumsum() * 100
        ax.plot(d.index, d.values, label=k)
    b = (H.monthly(H.baseline("P4Q")["daily"]) - base).cumsum() * 100
    ax.plot(b.index, b.values, "k--", label="P4+Q")
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_title("Cumulative monthly excess vs P7 (pp), pre-2026")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def holdout():
    Pall = pd.read_pickle(CACHE / "panel_M_ext_all.pkl")
    res = json.load(open(OUT / "results.json"))
    best = res["best_variant"]
    out = {}
    bq = H.baseline("P4Q", holdout=True)
    r7 = H.backtest(make_signal(), panel=Pall, holdout=True, **KW)
    for k in sorted({best, "X6_cascade"}):
        rk = H.backtest(make_signal(**VARIANTS[k]), panel=Pall, holdout=True, **KW)
        a = H.stats(rk["daily"], bench=bq["daily"], holdout="only")
        b = H.stats(rk["daily"], bench=r7["daily"], holdout="only")
        out[k] = {"exCDI": a["ann_excess_%"], "vs_P4Q": a["diff_ann_%"], "t": a["diff_t_nw"], "vs_P7": b["diff_ann_%"], "t_vs_P7": b["diff_t_nw"]}
    s7 = H.stats(r7["daily"], bench=bq["daily"], holdout="only")
    out["P7"] = {"exCDI": s7["ann_excess_%"], "vs_P4Q": s7["diff_ann_%"], "t": s7["diff_t_nw"]}
    out["P4Q_exCDI"] = H.stats(bq["daily"], holdout="only")["ann_excess_%"]
    # unlisted vs listed inside P4Q in the holdout (base engine)
    lst = lambda x: x["p4q"].to_numpy() & (x["eq_tier"] == "own").to_numpy()
    unl = lambda x: x["p4q"].to_numpy() & (x["eq_tier"] != "own").to_numpy()
    rl = H.backtest(lst, panel=Pall, holdout=True)
    ru = H.backtest(unl, panel=Pall, holdout=True)
    s = H.stats(ru["daily"], bench=rl["daily"], holdout="only")
    out["P4Q_unlisted_minus_listed"] = {"diff": s["diff_ann_%"], "t": s["diff_t_nw"]}
    res["holdout_2026"] = rnd(out)
    json.dump(res, open(OUT / "results.json", "w"), indent=1, default=str)
    H.plot_curves({"P7": r7, best: H.backtest(make_signal(**VARIANTS[best]), panel=Pall, holdout=True, **KW)},
                  OUT / "equity_total_return_with_holdout.png", title="P7 vs best extension incl. 2026 holdout (shaded)",
                  holdout=True)
    log("holdout", out)


if __name__ == "__main__":
    if HOLD:
        holdout()
    else:
        main()
