"""Adversarial verification of P7 (research/nightly/combined).  See lib.py for the pre-declared rules.

  PYTHONPATH=. PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
      .venv/Scripts/python.exe research/nightly/p7_adversarial_verification/verify.py [stage ...]

Stages (each cached in data/history/nightly/p7_adversarial_verification/stage_<name>.pkl):
  repro placebo boot decomp mult flagvote sens honest holdout ; then 'report' writes results.json + charts.
"""
from __future__ import annotations

import glob
import json
import pickle
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as SS

from research.nightly import harness as H
from research.nightly.combined import p7
from research.nightly.combined import run as CR
from research.nightly.p7_adversarial_verification import lib as L

OUT = Path("research/nightly/p7_adversarial_verification")
CACHE = H.HIST / "nightly" / "p7_adversarial_verification"
CACHE.mkdir(parents=True, exist_ok=True)
T0 = time.time()
NDRAW = 40


def log(*a):
    print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)


_P = {}


def PAN():
    if "P" not in _P:
        _P["P"] = p7.load(holdout=False)
    return _P["P"]


def BQ():
    return H.baseline("P4Q")


def summ(daily, bench=None):
    s = H.stats(daily, bench=bench)
    keep = ["ann_excess_%", "vol_%", "sharpe", "max_dd_%", "t_nw", "h1_2022_23_%", "h2_2024_25_%"]
    o = {k: s[k] for k in keep}
    if bench is not None:
        o.update({k: s[k] for k in ["diff_ann_%", "diff_t_nw", "diff_p", "diff_h1_%", "diff_h2_%", "diff_te_%"]})
    return o


def cached(name, fn):
    p = CACHE / f"stage_{name}.pkl"
    if p.exists():
        return pickle.load(open(p, "rb"))
    t = time.time()
    r = fn()
    r["_runtime_s"] = round(time.time() - t, 1)
    pickle.dump(r, open(p, "wb"))
    log("stage", name, "done in", r["_runtime_s"], "s")
    return r


F7 = p7.make_signal()


# =================================================================================================== (1) reproduce
def st_repro():
    P = PAN()
    b = BQ()
    r7 = L.bt(F7, P)
    r7b = L.bt(L.mk(drop=L.d_eqh()), P)       # my generic re-implementation must match
    r7_50 = L.bt(F7, P, cost_bps=50)
    b50 = H.baseline("P4Q", cost_bps=50)
    out = {"P7_vs_P4Q": summ(r7["daily"], b["daily"]), "P7_vs_U": summ(r7["daily"], H.baseline("U")["daily"]),
           "P7_50bps_vs_P4Q_50": summ(r7_50["daily"], b50["daily"]),
           "generic_reimpl_maxabs_daily_diff": float((r7["daily"] - r7b["daily"]).abs().max()),
           "P4Q": summ(b["daily"]), "U": summ(H.baseline("U")["daily"]),
           "m_P7": L.mser(r7["daily"]), "m_P4Q": L.mser(b["daily"]), "m_U": L.mser(H.baseline("U")["daily"]),
           "d_P7": L.mser(r7["daily"], b["daily"]), "daily_P7": r7["daily"], "turnover_P7": r7["turnover_ann"],
           "cost_P7": r7["cost_ann_%"], "n_avg_P7": r7["n_avg"]}
    log("repro", out["P7_vs_P4Q"]["diff_ann_%"], out["P7_vs_P4Q"]["diff_t_nw"], "reimpl", out["generic_reimpl_maxabs_daily_diff"])
    return out


# =================================================================================================== (2) placebos
def st_placebo():
    P = PAN()
    b = BQ()
    r7 = L.bt(F7, P)
    fams = {
        "A_listed_random_carry": lambda s: p7.make_signal(screen=(f"rand:{s}",)),
        "B_strat_bond_carry": lambda s: L.mk(drop=L.drop_strat_bond(s)),
        "C_strat_issuer_carry": lambda s: L.mk(drop=L.drop_strat_issuer(s)),
        "B_strat_bond_EW": lambda s: L.mk(drop=L.drop_strat_bond(s), construct=None),
    }
    out = {}
    for fam, mkf in fams.items():
        for k in L.STATS:
            L.STATS[k] = 0
        n = NDRAW if fam != "B_strat_bond_EW" else 30
        ms, dd = [], []
        for s in range(n):
            r = L.bt(mkf(s), P)
            ms.append(L.mser(r["daily"]))
            dd.append(r["daily"])
        M = pd.concat(ms, axis=1)
        mb = L.mser(b["daily"])
        diffs = M.sub(mb, axis=0)
        per = diffs.mean() * 1200
        out[fam] = {"draws_diff_vs_P4Q": per.to_numpy(), "mean": float(per.mean()), "sd": float(per.std()),
                    "p95": float(np.percentile(per, 95)), "p5": float(np.percentile(per, 5)),
                    "m_mean_book": M.mean(axis=1), "match_stats": dict(L.STATS)}
        log("placebo", fam, round(per.mean(), 3), round(np.percentile(per, 95), 3), L.STATS)
    # actual comparisons
    d7 = L.mser(r7["daily"], b["daily"])
    v1 = L.bt(p7.make_signal(screen=("eqh",), construct=None, icap=0.10), P)
    dv1 = L.mser(v1["daily"], b["daily"])
    m7 = L.mser(r7["daily"])
    mv1 = L.mser(v1["daily"])
    for fam, o in out.items():
        actual_m = mv1 if fam.endswith("_EW") else m7
        act_d = dv1 if fam.endswith("_EW") else d7
        rel = (actual_m - o["m_mean_book"]).dropna()
        o["actual_diff_vs_P4Q"] = float(L.ann(act_d))
        o["honest_alpha"] = float(L.ann(rel))
        o["honest_alpha_t_nw"] = float(H.nw_t(rel.to_numpy(), 6))
        o["honest_alpha_h1"] = float(L.ann(rel[rel.index < H.SPLIT]))
        o["honest_alpha_h2"] = float(L.ann(rel[rel.index >= H.SPLIT]))
        o["p_value_share_ge_actual"] = float((o["draws_diff_vs_P4Q"] >= o["actual_diff_vs_P4Q"]).mean())
        o["z_vs_draws"] = float((o["actual_diff_vs_P4Q"] - o["mean"]) / o["sd"]) if o["sd"] > 0 else None
        o["m_rel"] = rel
    return out


def _diag_drop(dropf, target=None, pool_listed=True):
    """Overlap of a placebo's dropped set with the real flagged set, and carry of dropped names (seed 0 only)."""
    P = PAN()
    target = target or L.d_eqh()
    ov, cp, cf, npd, nfl = [], [], [], [], []
    for d, x in P[P["univ"] & (P["day"] >= H.START)].groupby("day"):
        sel = x["p4q"].to_numpy().astype(bool)
        D = sel & np.asarray(target(x, sel), bool)
        Dp = np.asarray(dropf(x, sel), bool) & sel
        if Dp.sum() == 0 or D.sum() == 0:
            continue
        ov.append((Dp & D).sum() / Dp.sum())
        c = x["cdi_bps"].to_numpy(float)
        cp.append(c[Dp].mean())
        cf.append(c[D].mean())
        npd.append(Dp.sum())
        nfl.append(D.sum())
    return {"overlap_share_of_placebo_drops_that_are_flagged": float(np.mean(ov)), "carry_placebo_drops": float(np.mean(cp)),
            "carry_flagged_drops": float(np.mean(cf)), "n_placebo_drops": float(np.mean(npd)), "n_flagged": float(np.mean(nfl))}


def _run_family(mkf, n, actual_daily, bench_daily, P=None):
    P = PAN() if P is None else P
    ms = []
    for s in range(n):
        ms.append(L.mser(L.bt(mkf(s), P)["daily"]))
    M = pd.concat(ms, axis=1)
    mb = L.mser(bench_daily)
    per = M.sub(mb, axis=0).mean() * 1200
    act = L.ann(L.mser(actual_daily, bench_daily))
    rel = (L.mser(actual_daily) - M.mean(axis=1)).dropna()
    return {"draws_diff_vs_P4Q": per.to_numpy(), "mean": float(per.mean()), "sd": float(per.std()),
            "p5": float(np.percentile(per, 5)), "p95": float(np.percentile(per, 95)), "actual_diff_vs_P4Q": act,
            "p_value_share_ge_actual": float((per >= act).mean()), "honest_alpha": L.ann(rel),
            "honest_alpha_t_nw": float(H.nw_t(rel.to_numpy(), 6)), "honest_alpha_h1": L.ann(rel[rel.index < H.SPLIT]),
            "honest_alpha_h2": L.ann(rel[rel.index >= H.SPLIT]), "m_mean_book": M.mean(axis=1), "m_rel": rel}


def st_placebo2():
    """Second placebo pass (added after stage 'placebo' showed that fine carry-decile x sector strata drawn from ALL
    listed P4+Q names re-select the flagged names ~95% of the time, i.e. those placebos largely replicate P7).
    Controls here are drawn only from NON-flagged names (zero overlap) = the mechanical effect of dropping the same
    number of comparable listed names.  G is a coarse permutation null (flagged names may be drawn)."""
    P = PAN()
    b = BQ()
    r7 = L.bt(F7, P)
    v1 = L.bt(L.mk(drop=L.d_eqh(), construct=None), P)
    fams = {
        "A_listed_random_carry (P7's own placebo)": (lambda s: p7.make_signal(screen=(f"rand:{s}",)), None, r7),
        "D_ctrl_bond_dec10xsector_carry": (lambda s: L.mk(drop=L.drop_strat_bond(s, exclude=True)),
                                           L.drop_strat_bond(0, exclude=True), r7),
        "E_ctrl_bond_tercile_carry": (lambda s: L.mk(drop=L.drop_strat_bond(s, exclude=True, ndec=3, sector=False)),
                                      L.drop_strat_bond(0, exclude=True, ndec=3, sector=False), r7),
        "F_ctrl_issuer_q5xsector_carry": (lambda s: L.mk(drop=L.drop_strat_issuer(s, exclude=True)),
                                          L.drop_strat_issuer(0, exclude=True), r7),
        "G_perm_bond_tercile_carry": (lambda s: L.mk(drop=L.drop_strat_bond(s, ndec=3, sector=False)),
                                      L.drop_strat_bond(0, ndec=3, sector=False), r7),
        "H_ctrl_bond_tercile_EW (vs screen-only V1)": (
            lambda s: L.mk(drop=L.drop_strat_bond(s, exclude=True, ndec=3, sector=False), construct=None),
            L.drop_strat_bond(0, exclude=True, ndec=3, sector=False), v1),
    }
    out = {}
    for fam, (mkf, dropf, act) in fams.items():
        for k in L.STATS:
            L.STATS[k] = 0
        o = _run_family(mkf, NDRAW if not fam.startswith("H") else 30, act["daily"], b["daily"])
        o["match_stats"] = dict(L.STATS)
        if dropf is not None:
            o["diag"] = _diag_drop(dropf)
        out[fam] = o
        log("placebo2", fam, round(o["mean"], 3), round(o["p95"], 3), "alpha", round(o["honest_alpha"], 3),
            round(o["honest_alpha_t_nw"], 2), o.get("diag"))
    # the fine-strata placebos of stage 'placebo' - diagnostic of their contamination
    out["_contamination_of_stage_placebo"] = {"B_strat_bond": _diag_drop(L.drop_strat_bond(0)),
                                              "C_strat_issuer": _diag_drop(L.drop_strat_issuer(0))}
    log("contamination", out["_contamination_of_stage_placebo"])
    # flag-vote FV2 with zero-overlap controls (pool = all non-flagged P4+Q, carry decile x sector)
    fv2 = lambda x, sel: L.votes(x) >= 2
    rfv = L.bt(L.mk(drop=fv2), P)
    o = _run_family(lambda s: L.mk(drop=L.drop_strat_bond(s, pool_listed=False, target=fv2, exclude=True)), 20,
                    rfv["daily"], b["daily"])
    o["diag"] = _diag_drop(L.drop_strat_bond(0, pool_listed=False, target=fv2, exclude=True), target=fv2)
    o["diag_stage_flagvote_placebo"] = _diag_drop(L.drop_strat_bond(0, pool_listed=False, target=fv2), target=fv2)
    out["FV2_carry_ctrl"] = o
    log("FV2 ctrl", round(o["mean"], 3), round(o["honest_alpha"], 3), round(o["honest_alpha_t_nw"], 2), o["diag"])
    return out


# =================================================================================================== (3) bootstrap / concentration
def st_boot():
    P = PAN()
    b = BQ()
    r7 = L.bt(F7, P)
    d = L.mser(r7["daily"], b["daily"])
    x = d.to_numpy()
    n = len(x)
    rng = np.random.default_rng(7)
    B = 20000
    # circular 6-month block bootstrap of the monthly diff
    bl = 6
    nb = int(np.ceil(n / bl))
    starts = rng.integers(0, n, size=(B, nb))
    idx = (starts[:, :, None] + np.arange(bl)[None, None, :]).reshape(B, -1)[:, :n] % n
    bm = x[idx].mean(axis=1) * 1200
    block = {"mean": float(x.mean() * 1200), "ci90": [float(np.percentile(bm, 5)), float(np.percentile(bm, 95))],
             "ci95": [float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))],
             "p_one_sided_le0": float((bm <= 0).mean()), "draws": bm}
    # 3-month block too (sensitivity)
    bl3 = 3
    starts = rng.integers(0, n, size=(B, int(np.ceil(n / bl3))))
    idx3 = (starts[:, :, None] + np.arange(bl3)[None, None, :]).reshape(B, -1)[:, :n] % n
    bm3 = x[idx3].mean(axis=1) * 1200
    block["block3_p_le0"] = float((bm3 <= 0).mean())
    # leave-one-month-out and top months
    lomo = pd.Series({m: float(np.delete(x, i).mean() * 1200) for i, m in enumerate(d.index)})
    srt = d.sort_values(ascending=False)
    excl = {}
    for k in (1, 2, 3, 5):
        r = d.drop(srt.index[:k])
        excl[f"excl_top{k}_months"] = {"diff_ann_%": float(r.mean() * 1200), "t_nw": float(H.nw_t(r.to_numpy(), 6)),
                                       "months": [str(t.date()) for t in srt.index[:k]]}
    # symmetric trim: drop the best k AND worst k months
    for k in (1, 3):
        r = d.drop(list(srt.index[:k]) + list(srt.index[-k:]))
        excl[f"trim_best_and_worst_{k}"] = {"diff_ann_%": float(r.mean() * 1200), "t_nw": float(H.nw_t(r.to_numpy(), 6))}
    # issuer attribution (cohort, 126d labels complete)
    Mc = L.cohort_matrix(F7, L.p4q_ew, P)
    tot_by_iss = Mc.sum().sort_values(ascending=False)
    nd = len(Mc)
    cohort_ann = float(Mc.sum(axis=1).mean() * 2 * 100)   # 6m relative -> %/yr
    tot = tot_by_iss.sum()
    share = {f"top{k}": float(tot_by_iss.head(k).sum() / tot) for k in (1, 3, 5, 10)}
    # issuer-clustered bootstrap of the cohort diff (resample issuers), and two-way (issuers x 6m date blocks)
    A = Mc.to_numpy()
    ni = A.shape[1]
    colsum = A.sum(axis=0)
    bi = np.array([colsum[rng.integers(0, ni, ni)].sum() for _ in range(B)]) / nd * 2 * 100
    two = []
    nbd = int(np.ceil(nd / 6))
    for _ in range(5000):
        ci = rng.integers(0, ni, ni)
        st = rng.integers(0, nd, nbd)
        ri = ((st[:, None] + np.arange(6)[None, :]).ravel()[:nd]) % nd
        two.append(A[np.ix_(ri, ci)].sum() / nd * 2 * 100)
    two = np.array(two)
    iss_boot = {"cohort_diff_ann_%": cohort_ann, "n_dates": nd, "n_issuers": ni,
                "issuer_boot_ci90": [float(np.percentile(bi, 5)), float(np.percentile(bi, 95))],
                "issuer_boot_p_le0": float((bi <= 0).mean()),
                "twoway_ci90": [float(np.percentile(two, 5)), float(np.percentile(two, 95))],
                "twoway_p_le0": float((two <= 0).mean()), "draws_issuer": bi}
    # leave-one-issuer-out (cohort approximation for all issuers)
    loio_c = pd.Series((tot - tot_by_iss) / nd * 2 * 100, index=tot_by_iss.index)
    # real backtests: drop top-k contributing issuers from BOTH books, and LOIO for the top-15 +/- bottom-5
    def drop_run(ex):
        Pk = P.copy()
        Pk["univ"] = Pk["univ"] & ~Pk["cnpj8"].astype(str).isin(set(ex))
        rk = L.bt(F7, Pk)
        bk = H.backtest("p4q", panel=Pk)
        s = H.stats(rk["daily"], bench=bk["daily"])
        return {"diff_ann_%": s["diff_ann_%"], "t": s["diff_t_nw"], "h1": s["diff_h1_%"], "h2": s["diff_h2_%"]}
    dropk = {k: drop_run(list(tot_by_iss.index[:k])) for k in (1, 3, 5, 10)}
    loio_bt = {}
    for i in list(tot_by_iss.index[:15]) + list(tot_by_iss.index[-5:]):
        loio_bt[i] = drop_run([i])
    log("boot", block["p_one_sided_le0"], share, dropk)
    names = {}
    try:
        names = pd.read_csv("research/nightly/bias_audit/distress_events_pit.csv", dtype={"cnpj8": str}).set_index("cnpj8")["name"].dropna().to_dict()
    except Exception:
        pass
    sect = P.drop_duplicates("cnpj8").assign(c=lambda z: z["cnpj8"].astype(str)).set_index("c")["sector"].astype(str).to_dict()
    top_tab = pd.DataFrame({"contrib_sum_6m_%": tot_by_iss * 100, "ann_%_of_diff": tot_by_iss / nd * 2 * 100})
    top_tab["name"] = [names.get(i, "") for i in top_tab.index]
    top_tab["sector"] = [sect.get(i, "") for i in top_tab.index]
    return {"d": d, "block": block, "lomo": lomo, "excl_months": excl, "issuer": iss_boot, "share_of_gain": share,
            "top_issuers": top_tab, "loio_cohort": loio_c, "drop_topk_bt": dropk, "loio_bt": loio_bt}


# =================================================================================================== (4) decomposition
def st_decomp():
    P = PAN()
    b = BQ()
    R = {"A_P4Q": b,
         "B_construct_only": L.bt(L.mk(drop=None), P),
         "C_screen_EW": L.bt(L.mk(drop=L.d_eqh(), construct=None), P),
         "D_P7": L.bt(F7, P)}
    m = {k: L.mser(v["daily"]) for k, v in R.items()}
    eff = {"construction (B-A)": m["B_construct_only"] - m["A_P4Q"], "screen on EW (C-A)": m["C_screen_EW"] - m["A_P4Q"],
           "screen on carry book (D-B)": m["D_P7"] - m["B_construct_only"],
           "interaction (D-B-C+A)": m["D_P7"] - m["B_construct_only"] - m["C_screen_EW"] + m["A_P4Q"],
           "total (D-A)": m["D_P7"] - m["A_P4Q"]}
    effs = {k: {"ann_%": L.ann(v), "t_nw": float(H.nw_t(v.to_numpy(), 6)), "h1": L.ann(v[v.index < H.SPLIT]),
                "h2": L.ann(v[v.index >= H.SPLIT])} for k, v in eff.items()}
    # portfolio_construction books
    from research.nightly.portfolio_construction import run as PCR
    pth = CACHE / "pc_top25_fund500m_pre.pkl"
    if pth.exists():
        W25 = pd.read_pickle(pth)
    else:
        W25 = PCR.topk_carry(fund=True)
        W25.to_pickle(pth)
    Wpre, _ = pd.read_pickle(H.HIST / "nightly" / "portfolio_construction" / "weights_pre.pkl")
    pcs = {"PC_top25_carry_fund500m": W25[["day", "codigo", "weight"]],
           "PC_mv_emp_g20_fund500m": PCR.wsig(Wpre, "mv_emp_g20_fund500m"),
           "PC_carry_prop_nocap": PCR.wsig(Wpre, "carry_prop"),
           "PC_EW_fund500m": PCR.wsig(Wpre, "EW_fund500m")}
    for k, W in pcs.items():
        R[k] = H.backtest(W, freq="M", hold=126, issuer_cap=1.0, name=k)
        m[k] = L.mser(R[k]["daily"])
    # EQH screen applied on top of the PC top-25 fund book (drop flagged, renormalise; constraints approx.)
    flags = []
    for d, x in P[P["univ"] & (P["day"] >= H.START)].groupby("day"):
        f = L.flag_eqh_g(x)
        flags.append(pd.DataFrame({"day": d, "codigo": x["codigo"].to_numpy()[f]}))
    FL = pd.concat(flags).assign(fl=True)
    W25s = W25.merge(FL, on=["day", "codigo"], how="left")
    W25s = W25s[W25s["fl"].isna()][["day", "codigo", "weight"]]
    W25s["weight"] = W25s["weight"] / W25s.groupby("day")["weight"].transform("sum")
    R["PC_top25_fund500m+EQH"] = H.backtest(W25s, freq="M", hold=126, issuer_cap=1.0)
    m["PC_top25_fund500m+EQH"] = L.mser(R["PC_top25_fund500m+EQH"]["daily"])
    vsq = {k: summ(v["daily"], b["daily"]) for k, v in R.items() if k != "A_P4Q"}
    p7vs = {k: summ(R["D_P7"]["daily"], R[k]["daily"]) for k in pcs}
    p7vs["PC_top25_fund500m+EQH"] = summ(R["D_P7"]["daily"], R["PC_top25_fund500m+EQH"]["daily"])
    # regression of P7-P4Q on construction-book excess (spanning test)
    import statsmodels.api as sm
    y = (m["D_P7"] - m["A_P4Q"]).dropna()
    reg = {}
    for lab, cols in {"on_PC_top25_500m": ["PC_top25_carry_fund500m"], "on_PC_mv500m": ["PC_mv_emp_g20_fund500m"],
                      "on_carry_prop_nocap": ["PC_carry_prop_nocap"], "on_own_construct_B": ["B_construct_only"],
                      "on_B_and_screenEW_C": ["B_construct_only", "C_screen_EW"]}.items():
        X = pd.concat([(m[c] - m["A_P4Q"]).rename(c) for c in cols], axis=1).reindex(y.index)
        f = sm.OLS(y, sm.add_constant(X)).fit(cov_type="HAC", cov_kwds={"maxlags": 6})
        reg[lab] = {"alpha_ann_%": float(f.params["const"] * 1200), "alpha_t": float(f.tvalues["const"]),
                    "betas": {c: float(f.params[c]) for c in cols}, "r2": float(f.rsquared)}
    # profile: carry and listed share
    prof = {}
    for k, fn in {"P4Q": L.p4q_ew, "B_construct_only": L.mk(drop=None), "P7": F7}.items():
        rows = []
        for d, x in P[P["univ"] & (P["day"] >= H.START)].groupby("day"):
            w = fn(x)
            if w.sum() <= 0:
                continue
            w = w / w.sum()
            rows.append({"carry": float((w * x["cdi_bps"]).sum()), "listed": float(w[L.listed(x)].sum()),
                         "n": int((w > 0).sum()), "effn": float(1 / (w ** 2).sum())})
        prof[k] = pd.DataFrame(rows).mean().to_dict()
    log("decomp", {k: round(v["ann_%"], 3) for k, v in effs.items()})
    log("p7 vs pc", {k: round(v["diff_ann_%"], 3) for k, v in p7vs.items()})
    return {"effects": effs, "vs_P4Q": vsq, "P7_vs_PC": p7vs, "spanning_regressions": reg, "profile": prof,
            "m": m}


# =================================================================================================== (5) multiplicity
T_KEYS = re.compile(r"(diff_t_nw|t_vs_p4q|t_vs_P4Q|vs_p4q_t|t_diff|diff_t)$", re.I)


def _collect_t(o, acc):
    if isinstance(o, dict):
        for k, v in o.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool) and T_KEYS.search(str(k)) and np.isfinite(v):
                acc.append(float(v))
            else:
                _collect_t(v, acc)
    elif isinstance(o, list):
        for v in o:
            _collect_t(v, acc)


def st_mult():
    counts, tvals = {}, {}
    for f in sorted(glob.glob("research/nightly/*/results.json")):
        slug = Path(f).parent.name
        if slug == OUT.name:
            continue
        try:
            d = json.load(open(f))
        except Exception:
            continue
        n, key = None, None
        for k in ("n_variants_tried", "n_variants_total", "n_variants_holm", "n_variants", "n_candidates"):
            if isinstance(d.get(k), (int, float)):
                n, key = int(d[k]), k
                break
        if n is None and isinstance(d.get("variants_tried"), list):
            n, key = len(d["variants_tried"]), "len(variants_tried)"
        if n is None and isinstance(d.get("holm_all_tried"), dict) and "n" in d["holm_all_tried"]:
            n, key = int(d["holm_all_tried"]["n"]), "holm_all_tried.n"
        if n is None:
            s = json.dumps(d)
            mm = re.findall(r'"n_[a-z_]*variant[a-z_]*": ?(\d+)', s)
            if mm:
                n, key = max(int(v) for v in mm), "max nested n_*variant*"
        counts[slug] = {"n": n, "key": key}
        acc = []
        _collect_t(d, acc)
        tvals[slug] = acc
    N_all = int(sum(v["n"] or 0 for v in counts.values()))
    N_search = int(sum(v["n"] or 0 for k, v in counts.items() if k != "bias_audit"))
    allt = np.array([t for v in tvals.values() for t in v])
    out = {"counts": counts, "N_all": N_all, "N_excl_bias_audit": N_search, "n_tstats_found": int(len(allt)),
           "tstat_sd": float(allt.std()) if len(allt) > 5 else None, "tstat_mean": float(allt.mean()) if len(allt) else None}
    # minimum t for Bonferroni/Holm-first-step at 5% two-sided, several effective N
    tmin = {}
    for N in (1, 12, 20, 50, 100, 200, 300, N_search, N_all):
        tmin[str(N)] = float(SS.norm.ppf(1 - 0.025 / N))
    out["min_t_bonferroni_5pct"] = tmin
    # P7's own stats
    rp = pickle.load(open(CACHE / "stage_repro.pkl", "rb"))
    d = rp["d_P7"].to_numpy()
    T = len(d)
    t_nw = float(H.nw_t(d, 6))
    t_iid = float(d.mean() / d.std(ddof=1) * np.sqrt(T))
    p_nw = float(2 * (1 - SS.norm.cdf(abs(t_nw))))
    out["P7_t_nw"], out["P7_t_iid"], out["P7_p_nw"] = t_nw, t_iid, p_nw
    out["holm_bonf_p"] = {str(N): float(min(1.0, N * p_nw)) for N in (12, 50, 100, 300, N_search, N_all)}
    # Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014) on the monthly paired diff
    sr = d.mean() / d.std(ddof=1)
    g3 = float(SS.skew(d))
    g4 = float(SS.kurtosis(d, fisher=False))
    emc = 0.5772156649
    dsr = {}
    for N in (50, 100, 300, N_search, N_all):
        for lab, vt in {"V_t=1(null)": 1.0, "V_t=empirical": (allt.var() if len(allt) > 5 else 1.0)}.items():
            ez = (1 - emc) * SS.norm.ppf(1 - 1 / N) + emc * SS.norm.ppf(1 - 1 / (N * np.e))
            sr0 = np.sqrt(vt / T) * ez               # SR std under null ~ sqrt(V_t / T)
            z = (sr - sr0) * np.sqrt(T - 1) / np.sqrt(1 - g3 * sr + (g4 - 1) / 4 * sr ** 2)
            dsr[f"N={N}|{lab}"] = {"sr0_monthly": float(sr0), "dsr": float(SS.norm.cdf(z)), "expected_max_t": float(np.sqrt(vt) * ez)}
    out["P7_sr_monthly"], out["skew"], out["kurt"] = float(sr), g3, g4
    out["dsr"] = dsr
    # GLM de-smoothing of the diff
    try:
        from statsmodels.tsa.arima.model import ARIMA
        fit = ARIMA(d, order=(0, 0, 2)).fit()
        psi = fit.maparams
        th = np.r_[1.0, psi] / (1 + psi.sum())
        out["glm_diff"] = {"theta": th.tolist(), "sharpe_factor": float(np.sqrt((th ** 2).sum()))}
    except Exception as e:  # noqa
        out["glm_diff"] = str(e)[:100]
    log("mult", N_search, N_all, tmin, out["holm_bonf_p"])
    return out


# =================================================================================================== (6) flag vote
def st_flagvote():
    P = PAN()
    b = BQ()
    fv = lambda k: (lambda x, sel: L.votes(x) >= k)
    V = {"FV2_carry (primary)": L.mk(drop=fv(2)), "FV3_carry": L.mk(drop=fv(3)),
         "FV2_EW": L.mk(drop=fv(2), construct=None), "FV3_EW": L.mk(drop=fv(3), construct=None)}
    R = {k: L.bt(f, P) for k, f in V.items()}
    r7 = L.bt(F7, P)
    out = {"vs_P4Q": {k: summ(v["daily"], b["daily"]) for k, v in R.items()},
           "vs_P7": {k: summ(v["daily"], r7["daily"]) for k, v in R.items()},
           "vs_P4Q_50bps": {k: summ(L.bt(V[k], P, cost_bps=50)["daily"], H.baseline("P4Q", cost_bps=50)["daily"])["diff_ann_%"] for k in V}}
    # drop counts
    cnt = []
    for d, x in P[P["univ"] & (P["day"] >= H.START)].groupby("day"):
        s = x["p4q"].to_numpy()
        v = L.votes(x)[s]
        cnt.append({"n_p4q": int(s.sum()), "ge2": int((v >= 2).sum()), "ge3": int((v >= 3).sum()),
                    "eqh": int(p7.flag_eqh(x)[s].sum())})
    out["drop_counts_avg"] = pd.DataFrame(cnt).mean().to_dict()
    # matched placebo for the primary: same count, carry decile x sector, pool = all P4+Q names
    pl = []
    for s_ in range(20):
        r = L.bt(L.mk(drop=L.drop_strat_bond(s_, pool_listed=False, target=fv(2))), P)
        pl.append(H.stats(r["daily"], bench=b["daily"])["diff_ann_%"])
    pl = np.array(pl)
    out["placebo_FV2_carry"] = {"mean": float(pl.mean()), "p95": float(np.percentile(pl, 95)),
                                "actual": out["vs_P4Q"]["FV2_carry (primary)"]["diff_ann_%"],
                                "share_ge_actual": float((pl >= out["vs_P4Q"]["FV2_carry (primary)"]["diff_ann_%"]).mean())}
    out["daily"] = {k: v["daily"] for k, v in R.items()}
    log("flagvote", {k: round(v["diff_ann_%"], 3) for k, v in out["vs_P4Q"].items()}, out["placebo_FV2_carry"])
    return out


# =================================================================================================== (7) sensitivity
def st_sens():
    P = PAN().copy()
    b = BQ()
    out = {}
    grid = {"q=0.10": L.mk(drop=L.d_eqh(0.10)), "q=0.20 (P7)": L.mk(drop=L.d_eqh(0.20)), "q=0.30": L.mk(drop=L.d_eqh(0.30)),
            "only eq_dd252": L.mk(drop=L.d_eqh(0.2, ("eq_dd252",))), "only eq_vol63": L.mk(drop=L.d_eqh(0.2, ("eq_vol63",))),
            "only eq_r126": L.mk(drop=L.d_eqh(0.2, ("eq_r126",)))}
    for k, f in grid.items():
        out[k] = summ(L.bt(f, P)["daily"], b["daily"])
        log("sens", k, round(out[k]["diff_ann_%"], 3), round(out[k]["diff_t_nw"], 2))
    # fundamentals lag for Q (P7 and the P4+Q benchmark both use the lagged Q)
    lag = {}
    for lg in (0, 60, 90, 120):
        wq = L.worstq_lag(P, lg)
        col = f"p4q_l{lg}"
        P[col] = P["p4"].to_numpy() & ~wq
        match = float((P.loc[P["univ"], col] == P.loc[P["univ"], "p4q"]).mean())
        r7 = L.bt(L.mk(drop=L.d_eqh(), base=col), P)
        bq = H.backtest(col, panel=P)
        lag[f"lag{lg}"] = {"match_with_panel_p4q": match, "P7L_vs_P4QL": summ(r7["daily"], bq["daily"]),
                           "P7L_vs_P4Q": summ(r7["daily"], b["daily"])["diff_ann_%"],
                           "P4QL_vs_P4Q": summ(bq["daily"], b["daily"])["diff_ann_%"]}
        log("lag", lg, match, round(lag[f"lag{lg}"]["P7L_vs_P4QL"]["diff_ann_%"], 3))
    out["fund_lag"] = lag
    return out


# =================================================================================================== honest engine
def st_honest():
    """bias_audit-style realistic assumptions, same on both sides: liquidity-bucket costs, harsh survivorship,
    holiday-accrual fix, fundamentals lag 90d for Q; with a matched (issuer-stratified) placebo on the same engine."""
    from research.nightly.bias_audit import build as BB
    P = PAN().copy()
    P["p4q_l90"] = P["p4"].to_numpy() & ~L.worstq_lag(P, 90)
    cb = CR.cost_vector(P)
    Rh, nh = CR.harsh_R()
    Rhh = BB.R_hol(Rh)
    maskq = lambda c: (lambda x: x[c].to_numpy())
    scen = {"liq_costs": dict(cost_b=cb), "harsh_surv": dict(R=Rh), "liq+harsh": dict(cost_b=cb, R=Rh),
            "liq+harsh+holiday": dict(cost_b=cb, R=Rhh)}
    out = {"n_bonds_harsh": nh}
    for tag, kw in scen.items():
        e7 = CR.engine_run(F7, P, **kw)
        eq = CR.engine_run(maskq("p4q"), P, as_w=False, icap=0.10, **kw)
        out[tag] = summ(e7["daily"], eq["daily"]) | {"P4Q_exCDI": H.stats(eq["daily"])["ann_excess_%"]}
        log("honest", tag, round(out[tag]["diff_ann_%"], 3))
    # fully honest = liq+harsh+holiday + lag-90 Q on both sides, with matched placebos
    kw = scen["liq+harsh+holiday"]
    f7h = L.mk(drop=L.d_eqh(), base="p4q_l90")
    e7 = CR.engine_run(f7h, P, **kw)
    eq = CR.engine_run(maskq("p4q_l90"), P, as_w=False, icap=0.10, **kw)
    eu = CR.engine_run(lambda x: x["univ"].to_numpy(), P, as_w=False, icap=0.10, **kw)
    full = summ(e7["daily"], eq["daily"]) | {"P4QL90_exCDI": H.stats(eq["daily"])["ann_excess_%"],
                                              "U_exCDI": H.stats(eu["daily"])["ann_excess_%"]}
    m7 = L.mser(e7["daily"])
    mq = L.mser(eq["daily"])
    pls = {}
    for fam, mkf in {"E_ctrl_bond_tercile_carry": lambda s: L.mk(drop=L.drop_strat_bond(s, exclude=True, ndec=3, sector=False), base="p4q_l90"),
                     "F_ctrl_issuer_q5xsector_carry": lambda s: L.mk(drop=L.drop_strat_issuer(s, exclude=True), base="p4q_l90"),
                     "A_listed_random_carry": lambda s: p7.make_signal(screen=(f"rand:{s}",), base="p4q_l90")}.items():
        ms = []
        for s in range(20):
            ms.append(L.mser(CR.engine_run(mkf(s), P, **kw)["daily"]))
        M = pd.concat(ms, axis=1)
        per = M.sub(mq, axis=0).mean() * 1200
        rel = (m7 - M.mean(axis=1)).dropna()
        pls[fam] = {"mean": float(per.mean()), "p95": float(np.percentile(per, 95)),
                    "share_ge_actual": float((per >= full["diff_ann_%"]).mean()),
                    "honest_alpha": L.ann(rel), "honest_alpha_t": float(H.nw_t(rel.to_numpy(), 6)),
                    "h1": L.ann(rel[rel.index < H.SPLIT]), "h2": L.ann(rel[rel.index >= H.SPLIT])}
        log("honest placebo", fam, pls[fam])
    out["fully_honest"] = full
    out["fully_honest_placebos"] = pls
    out["daily"] = {"P7_honest": e7["daily"], "P4QL90_honest": eq["daily"], "U_honest": eu["daily"]}
    return out


# =================================================================================================== holdout (recompute only)
def st_holdout():
    Ph = p7.load(holdout=True)
    r7 = L.bt(F7, Ph, holdout=True)
    bq = H.baseline("P4Q", holdout=True)
    s = H.stats(r7["daily"], bench=bq["daily"], holdout="only")
    out = {"P7_vs_P4Q_2026": {"diff_ann_%": s["diff_ann_%"], "t": s["diff_t_nw"], "exCDI": s["ann_excess_%"],
                              "months": s["n_months"]},
           "published": 0.68, "daily_P7": r7["daily"], "daily_P4Q": bq["daily"], "daily_U": H.baseline("U", holdout=True)["daily"]}
    log("holdout (recompute only)", out["P7_vs_P4Q_2026"])
    return out


STAGES = {"repro": st_repro, "placebo": st_placebo, "placebo2": st_placebo2,"boot": st_boot, "decomp": st_decomp, "mult": st_mult,
          "flagvote": st_flagvote, "sens": st_sens, "honest": st_honest, "holdout": st_holdout}

if __name__ == "__main__":
    todo = [a for a in sys.argv[1:] if a in STAGES] or list(STAGES)
    for s in todo:
        cached(s, STAGES[s])
    log("all requested stages done")
