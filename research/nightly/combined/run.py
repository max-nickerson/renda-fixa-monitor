"""Combined strategy P7 (nightly slug `combined`).  See p7.py for the pre-registered spec.

  PYTHONPATH=. PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/Scripts/python.exe research/nightly/combined/run.py
  PYTHONPATH=. PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/Scripts/python.exe research/nightly/combined/run.py --holdout   # ONCE, at the end

Stage 1 (default) is pre-2026 only.  Stage 2 (--holdout) reads the sealed 2026 months for U, P4, P4+Q and the frozen P7.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from research.nightly import harness as H
from research.nightly.bias_audit import engine as E
from research.nightly.combined import p7

OUT = Path("research/nightly/combined")
CACHE = H.HIST / "nightly" / "combined"
CACHE.mkdir(parents=True, exist_ok=True)
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)


def rnd(d, k=4):
    if isinstance(d, dict):
        return {a: rnd(b, k) for a, b in d.items()}
    if isinstance(d, (list, tuple)):
        return [rnd(b, k) for b in d]
    if isinstance(d, (float, np.floating)):
        return None if not np.isfinite(d) else round(float(d), k)
    if isinstance(d, (np.integer,)):
        return int(d)
    if isinstance(d, (np.bool_,)):
        return bool(d)
    return d


KW = dict(as_weights=True, issuer_cap=1.0)


def bt(fn, P, **kw):
    return H.backtest(fn, panel=P, **KW, **kw)


# ---------------------------------------------------------------------------------------------------- gates
def gates():
    ibx = H.index_excess("IBOV").shift(1).rolling(63).sum()          # hedging_overlays E_gate_ibov
    return {"ibov": lambda d: bool(ibx.get(d, 0) < -0.10)}


def mom63_gate(P):
    s = P.drop_duplicates("day").set_index("day")["idadi_x63"]
    return lambda d: bool(s.get(d, 1.0) <= 0)


# ---------------------------------------------------------------------------------------------------- variants
def variants(P):
    G = gates()
    return {
        "V1_screen_only": p7.make_signal(screen=("eqh",), construct=None, icap=0.10),
        "V2_construct_only": p7.make_signal(screen=(), construct="carry"),
        "P7": p7.make_signal(),
        "V4_P7_no_liq_cap": p7.make_signal(aum=None),
        "V5_P7_zoo_screen": p7.make_signal(screen=("zoo",)),
        "V6_P7_multiflag": p7.make_signal(screen=("eqh", "events", "supply")),
        "V7_P7_top25": p7.make_signal(construct="top25"),
        "V8_P7_gate_mom63": p7.make_signal(gate=mom63_gate(P)),
        "V9_P7_gate_ibov": p7.make_signal(gate=G["ibov"]),
        "V10_P7_ew_liq": p7.make_signal(construct="ew"),
        "V11_P7_on_P4": p7.make_signal(base="p4"),
    }


# ---------------------------------------------------------------------------------------------------- realistic stress
def cost_vector(P):
    """bias_audit 'institutional Roll' round-trip costs by indexer group x ADV quintile (DI 65..45, IPCA/PRE 169..136)."""
    C = H._core()
    codes = C["codes"]
    L = pd.read_pickle(p7.LIQ_PATH)
    L = L[L["day"] < H.HOLDOUT].groupby("codigo")["adv_brl"].median()
    kind = P.drop_duplicates("codigo").set_index("codigo")["kind"]
    df = pd.DataFrame(index=codes)
    df["adv"] = L.reindex(codes)
    df["q"] = np.ceil(df["adv"].rank(pct=True) * 5).clip(1, 5).fillna(1).astype(int)
    df["grp"] = np.where(kind.reindex(codes).fillna("IPCA") == "DI_SPREAD", "DI", "IPCA_PRE")
    ml = {"DI": {1: 65, 2: 60, 3: 55, 4: 50, 5: 45}, "IPCA_PRE": {1: 169, 2: 161, 3: 152, 4: 144, 5: 136}}
    return np.array([ml[g][q] for g, q in zip(df["grp"], df["q"])], float)


def harsh_R(holdout=False):
    """bias_audit 'harsh' survivorship: every bond that stops trading > 60d before maturity (last print before the
    window end - 90d) with a last mark < 0.98 jumps to 40% of par at its last grid row."""
    C = H._core()
    R = H._Rmat("base").copy()
    dd = C["days"]
    end = dd[-1] if holdout else H.HOLDOUT
    t_end = len(dd) if holdout else H._hpos()
    last = C["last"]
    m = (last["last_day"] <= end - pd.Timedelta(days=90)) & \
        (last["maturity"].isna() | (last["maturity"] > last["last_day"] + pd.Timedelta(days=60))) & \
        (last["last_ratio"] < 0.98) & (last["last_ratio"] > 0.40)
    n = 0
    for c, row in last[m].iterrows():
        b = C["codes"].get_loc(c)
        p = int(dd.searchsorted(row["last_day"]))
        p = min(p, len(dd) - 1)
        if p >= t_end:
            continue
        R[p, b] = (1 + R[p, b]) * (0.40 / row["last_ratio"]) - 1
        n += 1
    return R, n


def targets_of(fn, P, as_w=True, icap=1.0, holdout=False):
    t_end = len(H.days()) if holdout else H._hpos()
    Pf = P[(P["day"] >= H.START) & (P["dpos"] < t_end)]
    return H._targets(fn, Pf, 0.2, icap, as_w, H.MIN_NAMES, "univ"), t_end


def engine_run(fn, P, as_w=True, icap=1.0, R=None, cost_b=None, holdout=False):
    T, t_end = targets_of(fn, P, as_w, icap, holdout)
    return E.tranche_book(T, 126, 25.0, R=R, cost_b=cost_b, t_end=t_end)


# ---------------------------------------------------------------------------------------------------- diagnostics
def profile(fn, P, as_w=True, icap=1.0):
    """Average characteristics of the target book per decision (weighted)."""
    U = P[P["univ"] & (P["day"] >= H.START)]
    rows = []
    for d, x in U.groupby("day"):
        if as_w:
            w = fn(x)
        else:
            s = fn(x)
            w = np.zeros(len(x))
            if s.sum() >= 5:
                w[s] = H.cap_weights(x.loc[s, "cnpj8"].to_numpy(), icap)
        if w.sum() <= 0:
            continue
        w = w / w.sum()
        cap500 = 0.25 * x["vol91_brl"].fillna(0).to_numpy() / 500e6
        rows.append({"day": d, "n": int((w > 0).sum()), "carry_bps": float((w * x["cdi_bps"]).sum()),
                     "dur": float((w * x["dur"]).sum()), "di_share": float(w[(x["kind"] == "DI_SPREAD").to_numpy()].sum()),
                     "listed_share": float(w[x[list(p7.EQ_COLS)].notna().all(axis=1).to_numpy()].sum()),
                     "covered_share": float(w[x["covered"].fillna(False).to_numpy().astype(bool)].sum()),
                     "max_w": float(w.max()), "eff_n": float(1 / (w ** 2).sum()),
                     "liq_breach_500m": float(np.clip(w - cap500, 0, None).sum()),
                     "fwd126_w": float(np.nansum(w * x["fwd_126"].to_numpy()) if x["dok_126"].all() else np.nan)})
    return pd.DataFrame(rows)


def issuer_contrib(fnA, fnB_mask, P):
    """Cohort attribution: sum over decisions of (w_A - w_B) x fwd_126 per issuer (fwd NaN -> 0 = cash)."""
    U = P[P["univ"] & (P["day"] >= H.START)]
    acc = {}
    for d, x in U.groupby("day"):
        wa = fnA(x)
        s = fnB_mask(x)
        wb = np.zeros(len(x))
        if s.sum() >= 5:
            wb[s] = H.cap_weights(x.loc[s, "cnpj8"].to_numpy(), 0.10)
        f = np.nan_to_num(x["fwd_126"].to_numpy(float))
        c = pd.Series((wa - wb) * f).groupby(x["cnpj8"].to_numpy()).sum()
        for k, v in c.items():
            acc[k] = acc.get(k, 0.0) + v
    return pd.Series(acc).sort_values(ascending=False)


def flag_overlap(P):
    """Which P4+Q names each wave-1 avoid flag drops, their overlap and the cohort fwd_126 of dropped vs kept."""
    U = P[P["univ"] & (P["day"] >= H.START) & P["p4q"]]
    fl = {"eqh": [], "zoo": [], "events": [], "supply": []}
    idx = []
    for d, x in P[P["univ"] & (P["day"] >= H.START)].groupby("day"):
        m = x["p4q"].to_numpy()
        fl["eqh"].append(p7.flag_eqh(x)[m]); fl["zoo"].append(p7.flag_zoo(x)[m])
        fl["events"].append(p7.flag_events(x)[m]); fl["supply"].append(p7.flag_supply(x)[m])
        idx.append(x.index[m])
    F = pd.DataFrame({k: np.concatenate(v) for k, v in fl.items()}, index=np.concatenate(idx))
    X = P.loc[F.index, ["day", "cnpj8", "fwd_126", "dok_126", "cdi_bps"]].join(F)
    X = X[X["dok_126"]]
    out = {"n_p4q_rows": int(len(X)), "share_flagged": {k: float(X[k].mean()) for k in fl}}
    jac = {}
    ks = list(fl)
    for i, a in enumerate(ks):
        for b in ks[i + 1:]:
            inter = (X[a] & X[b]).sum()
            uni = (X[a] | X[b]).sum()
            jac[f"{a}|{b}"] = float(inter / uni) if uni else np.nan
    out["jaccard"] = jac
    # per-date demeaned fwd_126 (within P4+Q) of flagged rows, 6m horizon, in % (NW-ish: mean over dates)
    X["rel"] = X["fwd_126"] - X.groupby("day")["fwd_126"].transform("mean")
    rel = {}
    for k in ks:
        g = X[X[k]].groupby("day")["rel"].mean()
        rel[k] = {"mean_6m_rel_%": float(g.mean() * 100), "t": float(g.mean() / (g.std(ddof=1) / np.sqrt(len(g)) + 1e-12) / np.sqrt(6)),
                  "n_dates": int(len(g))}
    X["nflags"] = X[ks].sum(axis=1)
    for n in range(0, 4):
        g = X[X["nflags"] == n].groupby("day")["rel"].mean()
        rel[f"nflags=={n}"] = {"mean_6m_rel_%": float(g.mean() * 100) if len(g) else None, "share_rows": float((X["nflags"] == n).mean())}
    out["dropped_rel_fwd126"] = rel
    # only-flag contributions: rows flagged by exactly one flag
    only = {}
    for k in ks:
        m = X[k] & (X["nflags"] == 1)
        g = X[m].groupby("day")["rel"].mean()
        only[k] = {"share_rows": float(m.mean()), "mean_6m_rel_%": float(g.mean() * 100) if len(g) else None}
    out["unique_flag_rel"] = only
    return out


def by_year(daily, bench):
    m = H.monthly(daily) - H.monthly(bench)
    return {str(y): float(v.mean() * 1200) for y, v in m.groupby(m.index.year)}


def excl_top_months(daily, bench, k=3):
    m = (H.monthly(daily) - H.monthly(bench)).dropna()
    m = m[m.index < H.HOLDOUT]
    top = m.sort_values(ascending=False).index[:k]
    r = m.drop(top)
    return {"top_months": [str(t.date()) for t in top], "diff_ann_%": float(r.mean() * 1200),
            "t_nw": float(H.nw_t(r.to_numpy(), 6))}


# ---------------------------------------------------------------------------------------------------- plotting
def plot(curves: dict, path, title, shade_from=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cdi = H.cdi_daily()
    fig, ax = plt.subplots(2, 1, figsize=(11, 8.5), sharex=True, gridspec_kw={"height_ratios": [3, 2]})
    style = {"P7": ("#c0392b", 2.4), "P4+Q": ("#2c3e50", 1.6), "P4": ("#7f8c8d", 1.2), "Universe": ("#95a5a6", 1.0),
             "CDI": ("#000000", 1.0), "IDA-DI": ("#2980b9", 1.1), "Ibovespa": ("#27ae60", 0.9)}
    U = curves["Universe"]
    for k, s in curves.items():
        s = s.dropna()
        tr = (1 + cdi.reindex(s.index).fillna(0) + s).cumprod() * 100 if k != "CDI" else (1 + cdi.reindex(s.index).fillna(0)).cumprod() * 100
        c, lw = style.get(k, ("#8e44ad", 1.2))
        ax[0].plot(tr.index, tr.values, label=k, color=c, lw=lw, ls="--" if k in ("CDI",) else "-")
        if k in ("P7", "P4+Q", "P4", "IDA-DI"):
            u = U.reindex(s.index).fillna(0)
            ce = ((1 + s).cumprod() / (1 + u).cumprod() - 1) * 100
            ax[1].plot(ce.index, ce.values, label=f"{k} vs universe", color=c, lw=lw)
    ax[0].set_yscale("log")
    ax[0].set_ylabel("total return (base 100, log)")
    ax[0].legend(loc="upper left", fontsize=8, ncol=2)
    ax[0].set_title(title, fontsize=11)
    ax[1].axhline(0, color="k", lw=0.6)
    ax[1].set_ylabel("cumulative excess vs universe, %")
    ax[1].legend(loc="upper left", fontsize=8)
    for a in ax:
        a.grid(alpha=0.3)
        if shade_from is not None:
            a.axvspan(pd.Timestamp(shade_from), a.get_xlim()[1] if False else s.index.max(), color="#f1c40f", alpha=0.15,
                      label=None)
    if shade_from is not None:
        ax[0].text(pd.Timestamp(shade_from), ax[0].get_ylim()[1] * 0.98, " sealed holdout 2026", va="top", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------------------------------- stage 1
def stage1():
    RES = {"slug": "combined", "harness_version": H._VERSION, "spec": p7.__doc__}
    P = p7.load(holdout=False)
    b = H.baseline("P4Q")
    V = variants(P)
    R = {"U": H.baseline("U"), "P4": H.baseline("P4"), "P4Q": b}
    for k, fn in V.items():
        R[k] = bt(fn, P, name=k)
        log(k, rnd(H.stats(R[k]["daily"], bench=b["daily"]), 3)["diff_ann_%"], "t",
            rnd(H.stats(R[k]["daily"], bench=b["daily"]), 2)["diff_t_nw"])
    tab = H.compare({k: v for k, v in R.items() if k not in ("U", "P4Q")}, bench="P4Q")
    tab.to_csv(OUT / "compare_pre2026.csv")
    log("\n" + tab.to_string())
    RES["compare_25bps"] = rnd(tab.reset_index().to_dict(orient="records"))
    RES["n_variants_tried"] = len(V)
    # references
    refs = {"CDI": H.baseline("CDI")["daily"], "IDADI": H.index_excess("IDADI"), "IBOV": H.index_excess("IBOV")}
    RES["references"] = {k: rnd(H.stats(v.loc[(v.index >= H.START) & (v.index < H.HOLDOUT)])) for k, v in refs.items()}
    RES["levels"] = {k: rnd(H.stats(R[k]["daily"], bench=b["daily"])) for k in ["U", "P4", "P4Q", "P7"]}
    RES["turnover"] = {k: rnd({"turnover_ann": R[k]["turnover_ann"], "cost_ann_%": R[k]["cost_ann_%"], "n_avg": R[k]["n_avg"]}) for k in R}

    # 50 bps
    b50 = H.baseline("P4Q", cost_bps=50)
    t50 = {k: rnd(H.stats(bt(V[k], P, cost_bps=50)["daily"], bench=b50["daily"])) for k in V}
    RES["vs_P4Q_50bps"] = {k: {"diff_ann_%": v["diff_ann_%"], "t": v["diff_t_nw"], "h1": v["diff_h1_%"], "h2": v["diff_h2_%"]} for k, v in t50.items()}
    log("50bps", {k: v["diff_ann_%"] for k, v in RES["vs_P4Q_50bps"].items()})

    # ---- robustness of P7
    rob = {}
    f = V["P7"]
    for nm, kw in {"rec40": dict(scenario="rec40"), "hold63": dict(hold=63), "hold252": dict(hold=252)}.items():
        bb = H.baseline("P4Q", **kw)
        rob[nm] = rnd(H.stats(bt(f, P, **kw)["daily"], bench=bb["daily"]))
    rob["by_year_vs_P4Q"] = rnd(by_year(R["P7"]["daily"], b["daily"]))
    rob["excl_top3_months"] = rnd(excl_top_months(R["P7"]["daily"], b["daily"]))
    # capacity
    cap = {}
    for aum in (100e6, 250e6, 1e9, 2e9):
        r = bt(p7.make_signal(aum=aum), P)
        s = H.stats(r["daily"], bench=b["daily"])
        cap[f"R${aum/1e6:.0f}m"] = rnd({"diff_ann_%": s["diff_ann_%"], "t": s["diff_t_nw"], "exCDI": s["ann_excess_%"], "n_avg": r["n_avg"]})
    cap["R$500m"] = rnd({"diff_ann_%": RES["levels"]["P7"]["diff_ann_%"], "t": RES["levels"]["P7"]["diff_t_nw"],
                         "exCDI": RES["levels"]["P7"]["ann_excess_%"], "n_avg": R["P7"]["n_avg"]})
    rob["capacity_vs_P4Q"] = cap
    log("capacity", cap)
    # realistic costs + harsh survivorship (bias_audit engine; same assumptions on both sides)
    cb = cost_vector(P)
    Rh, nh = harsh_R()
    base_mask = lambda x: x["p4q"].to_numpy()
    u_mask = lambda x: x["univ"].to_numpy()
    real = {}
    for tag, kw in {"liq_costs": dict(cost_b=cb), "harsh_surv": dict(R=Rh), "liq_costs+harsh_surv": dict(cost_b=cb, R=Rh)}.items():
        e7 = engine_run(f, P, **kw)
        eq = engine_run(base_mask, P, as_w=False, icap=0.10, **kw)
        eu = engine_run(u_mask, P, as_w=False, icap=0.10, **kw)
        e2 = engine_run(V["V1_screen_only"], P, **kw)
        s7 = H.stats(e7["daily"], bench=eq["daily"])
        real[tag] = rnd({"P7_exCDI": s7["ann_excess_%"], "P4Q_exCDI": H.stats(eq["daily"])["ann_excess_%"],
                         "U_exCDI": H.stats(eu["daily"])["ann_excess_%"],
                         "P7_vs_P4Q": s7["diff_ann_%"], "t": s7["diff_t_nw"], "h1": s7["diff_h1_%"], "h2": s7["diff_h2_%"],
                         "P7_vs_U": H.stats(e7["daily"], bench=eu["daily"])["diff_ann_%"],
                         "screen_only_vs_P4Q": H.stats(e2["daily"], bench=eq["daily"])["diff_ann_%"],
                         "P7_cost_ann_%": e7["cost_ann_%"], "P4Q_cost_ann_%": eq["cost_ann_%"]})
        log("realistic", tag, real[tag])
    real["n_bonds_harsh"] = nh
    rob["realistic"] = real
    # engine check: bias engine with flat 25 bps reproduces the harness
    chk = engine_run(f, P)
    rob["engine_repro_maxabs"] = float((chk["daily"] - R["P7"]["daily"]).abs().max())
    # issuer concentration: drop the top-5 / top-10 issuers by P7-vs-P4Q cohort contribution from BOTH books
    con = issuer_contrib(f, base_mask, P)
    rob["top_issuer_contrib"] = rnd(con.head(10).to_dict())
    rob["bottom_issuer_contrib"] = rnd(con.tail(5).to_dict())
    for k in (5, 10):
        ex = set(con.head(k).index)
        Pk = P.copy()
        Pk["univ"] = Pk["univ"] & ~Pk["cnpj8"].astype(str).isin(ex)
        rk = bt(f, Pk)
        bk = H.backtest("p4q", panel=Pk)
        s = H.stats(rk["daily"], bench=bk["daily"])
        rob[f"drop_top{k}_issuers"] = rnd({"diff_ann_%": s["diff_ann_%"], "t": s["diff_t_nw"], "h1": s["diff_h1_%"], "h2": s["diff_h2_%"]})
    log("issuer drops", rob["drop_top5_issuers"], rob["drop_top10_issuers"])
    # placebo: random screen of the same size among listed P4+Q names, same carry/liquidity construction
    pl = []
    for s_ in range(20):
        r = bt(p7.make_signal(screen=(f"rand:{s_}",)), P)
        pl.append(H.stats(r["daily"], bench=b["daily"])["diff_ann_%"])
    pl = np.array(pl)
    rob["placebo_random_screen"] = rnd({"mean": pl.mean(), "p95": np.percentile(pl, 95), "max": pl.max(),
                                        "actual_P7": RES["levels"]["P7"]["diff_ann_%"], "share_ge_actual": float((pl >= RES["levels"]["P7"]["diff_ann_%"]).mean())})
    log("placebo", rob["placebo_random_screen"])
    # harness random-universe placebo on P7's weights (levels)
    try:
        W7 = p7.weights_frame(f, P)
        ph = H.placebo(W7, n=10, panel=P, issuer_cap=1.0)
        rob["placebo_random_universe_names"] = rnd({"mean_exCDI": ph["mean"], "p95_exCDI": ph["p95"], "actual_exCDI": ph["actual"]})
    except Exception as e:  # noqa
        rob["placebo_random_universe_names"] = str(e)[:200]
    RES["robustness_P7"] = rob

    # ---- ablation (drop each component from P7), 25 bps, paired vs P7 and vs P4+Q
    abl = {}
    for nm, k in {"- screen (=V2)": "V2_construct_only", "- carry/liquidity construction (=V1)": "V1_screen_only",
                  "- carry tilt, keep liq cap (=V10)": "V10_P7_ew_liq", "- Q filter (=V11, P4 base)": "V11_P7_on_P4",
                  "- liquidity cap (=V4, not investable)": "V4_P7_no_liq_cap", "+ timing gate mom63 (=V8)": "V8_P7_gate_mom63",
                  "+ hedge gate Ibov (=V9)": "V9_P7_gate_ibov", "+ event/supply flags (=V6)": "V6_P7_multiflag"}.items():
        s = H.stats(R[k]["daily"], bench=R["P7"]["daily"])
        s2 = H.stats(R[k]["daily"], bench=b["daily"])
        abl[nm] = rnd({"vs_P7": s["diff_ann_%"], "t_vs_P7": s["diff_t_nw"], "vs_P4Q": s2["diff_ann_%"],
                       "maxDD": s2["max_dd_%"], "vol": s2["vol_%"]})
    RES["ablation"] = abl

    # ---- profiles and flag overlap
    prof = {}
    for k, fn, aw, ic in [("P4Q", base_mask, False, 0.10), ("P7", f, True, 1.0), ("V1_screen_only", V["V1_screen_only"], True, 1.0),
                          ("V4_P7_no_liq_cap", V["V4_P7_no_liq_cap"], True, 1.0)]:
        pr = profile(fn, P, aw, ic)
        prof[k] = rnd(pr.drop(columns="day").mean().to_dict())
    RES["book_profile"] = prof
    log("profile", prof)
    RES["flag_overlap"] = rnd(flag_overlap(P))
    log("overlap", RES["flag_overlap"])

    # ---- charts (pre-2026)
    curves = {"P7": R["P7"]["daily"], "P4+Q": b["daily"], "P4": R["P4"]["daily"], "Universe": R["U"]["daily"],
              "CDI": H.baseline("CDI")["daily"],
              "IDA-DI": refs["IDADI"].loc[(refs["IDADI"].index > R["U"]["daily"].index[0]) & (refs["IDADI"].index < H.HOLDOUT)],
              "Ibovespa": refs["IBOV"].loc[(refs["IBOV"].index > R["U"]["daily"].index[0]) & (refs["IBOV"].index < H.HOLDOUT)]}
    plot(curves, OUT / "equity_pre2026.png", "P7 vs baselines, pre-2026 (monthly decisions, 126d tranches, 25 bps)")
    # export P7 weights (pre-2026)
    p7.weights_frame(f, P).to_pickle(CACHE / "p7_weights_pre2026.pkl")
    json.dump(rnd(RES), open(OUT / "results.json", "w"), indent=1, default=str)
    log("stage 1 done")


# ---------------------------------------------------------------------------------------------------- stage 2
def stage2():
    RES = json.load(open(OUT / "results.json"))
    P = p7.load(holdout=True)
    f = p7.make_signal()
    kw = dict(holdout=True)
    r7 = bt(f, P, **kw)
    bq = H.baseline("P4Q", **kw)
    b4 = H.baseline("P4", **kw)
    bu = H.baseline("U", **kw)
    ho = {}
    for k, s in {"P7": r7["daily"], "P4": b4["daily"], "U": bu["daily"]}.items():
        st = H.stats(s, bench=bq["daily"], holdout="only")
        ho[k] = rnd({"exCDI_%": st["ann_excess_%"], "vs_P4Q_%": st["diff_ann_%"], "t_vs_P4Q": st["diff_t_nw"],
                     "cum_%": st["cum_%"], "max_dd_%": st["max_dd_%"], "months": st["n_months"],
                     "vs_U_%": H.stats(s, bench=bu["daily"], holdout="only")["diff_ann_%"]})
    ho["P4Q"] = rnd({"exCDI_%": H.stats(bq["daily"], holdout="only")["ann_excess_%"],
                     "vs_U_%": H.stats(bq["daily"], bench=bu["daily"], holdout="only")["diff_ann_%"],
                     "cum_%": H.stats(bq["daily"], holdout="only")["cum_%"]})
    for k, v in {"IDADI": H.index_excess("IDADI"), "IBOV": H.index_excess("IBOV")}.items():
        ho[k] = rnd({"exCDI_%": H.stats(v.loc[v.index >= H.HOLDOUT], holdout="only")["ann_excess_%"]})
    # holdout at 50 bps
    r50 = bt(f, P, cost_bps=50, **kw)
    s50 = H.stats(r50["daily"], bench=H.baseline("P4Q", cost_bps=50, **kw)["daily"], holdout="only")
    ho["P7_50bps_vs_P4Q_%"] = rnd(s50["diff_ann_%"])
    # holdout under realistic costs (liq buckets) for P7 vs P4Q
    cb = cost_vector(P)
    e7 = engine_run(f, P, cost_b=cb, holdout=True)
    eq = engine_run(lambda x: x["p4q"].to_numpy(), P, as_w=False, icap=0.10, cost_b=cb, holdout=True)
    s = H.stats(e7["daily"], bench=eq["daily"], holdout="only")
    ho["P7_liqcost_vs_P4Q_%"] = rnd(s["diff_ann_%"])
    ho["monthly_P7_minus_P4Q_%"] = rnd({str(k.date()): v * 100 for k, v in
                                        (H.monthly(r7["daily"]) - H.monthly(bq["daily"])).loc[lambda m: m.index >= H.HOLDOUT].items()})
    RES["holdout_2026"] = ho
    log("holdout", ho)
    # full-period charts with the holdout shaded
    ida, ibv = H.index_excess("IDADI"), H.index_excess("IBOV")
    i0 = bu["daily"].index[0]
    curves = {"P7": r7["daily"], "P4+Q": bq["daily"], "P4": b4["daily"], "Universe": bu["daily"],
              "CDI": H.baseline("CDI", holdout=True)["daily"], "IDA-DI": ida.loc[ida.index > i0], "Ibovespa": ibv.loc[ibv.index > i0]}
    plot(curves, OUT / "equity_total_return.png",
         "P7 vs P4+Q / P4 / universe / CDI / IDA-DI / Ibov, 2022-01..2026-09 (2026 = sealed holdout, shaded)",
         shade_from=H.HOLDOUT)
    # cumulative excess vs P4+Q
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 4))
    for k, s_, c in [("P7 - P4+Q", r7["daily"], "#c0392b"), ("P4 - P4+Q", b4["daily"], "#7f8c8d")]:
        d = (1 + s_).cumprod() / (1 + bq["daily"].reindex(s_.index).fillna(0)).cumprod() - 1
        ax.plot(d.index, d.values * 100, label=k, color=c)
    ax.axhline(0, color="k", lw=0.6)
    ax.axvspan(H.HOLDOUT, r7["daily"].index.max(), color="#f1c40f", alpha=0.15)
    ax.set_ylabel("cumulative relative return, %")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_title("Cumulative excess vs P4+Q (2026 = sealed holdout, shaded)")
    fig.tight_layout()
    fig.savefig(OUT / "cum_excess.png", dpi=120)
    plt.close(fig)
    p7.weights_frame(f, P).to_pickle(CACHE / "p7_weights_all.pkl")
    json.dump(rnd(RES), open(OUT / "results.json", "w"), indent=1, default=str)
    log("stage 2 done")


if __name__ == "__main__":
    if "--holdout" in sys.argv:
        stage2()
    else:
        stage1()
