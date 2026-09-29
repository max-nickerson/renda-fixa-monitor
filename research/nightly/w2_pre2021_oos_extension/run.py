"""Pre-2021 out-of-sample test of the FROZEN P4 / P4+Q / P7 rules (+ factor_zoo composite screen, macro_cycle mom63
gate) on the rebuilt 2014-2021 grid (build.py). No parameter is re-tuned: rules are the harness flags, p7.make_signal()
defaults and the factor_zoo results.json composite_spec.

Steps
 1. In-sample (2021-03 .. 2025-12, the published harness, holdout sealed) series of the same books, for splicing curves.
 2. Overlap validation: rebuilt grid vs harness in 2021-03..06 (spreads, universe, P4 flags, daily returns).
 3. Inject the rebuilt core into the harness engine (harness._MEM) and run the books with panel=<rebuilt panel>.
 4. Stats on 2015-01..2020-12 (+ 2015-16, 2017-19, 2020), paired vs P4+Q with NW t and Holm, 25/50 bps, rec40,
    placebos, IDA-DI, drawdowns; charts; results.json.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from research.nightly import harness as H  # noqa: E402
from research.nightly.combined import p7  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = ROOT / "data" / "history" / "nightly" / "pre2021_oos_extension"
OOS0, OOS1 = pd.Timestamp("2015-01-01"), pd.Timestamp("2020-12-31")
SPLICE = pd.Timestamp("2021-03-01")
PERIODS = {"2015_20": ("2015-01-01", "2020-12-31"), "2015_16": ("2015-01-01", "2016-12-31"),
           "2017_19": ("2017-01-01", "2019-12-31"), "2020": ("2020-01-01", "2020-12-31")}
T0 = time.time()
KW7 = dict(as_weights=True, issuer_cap=1.0)


def log(*a):
    print(f"[run +{time.time() - T0:6.1f}s]", *a, flush=True)


def zoo_composite(P: pd.DataFrame) -> pd.Series:
    """factor_zoo.features.signals(): mean of per-date normal scores within the eligible universe, frozen spec."""
    from scipy.stats import norm
    spec = json.loads((ROOT / "research/nightly/factor_zoo/results.json").read_text())["composite_spec"]
    U = P[P["univ"]]
    parts = []
    for f, s in spec.items():
        r = U.groupby("day")[f].rank(pct=True)
        n = U.groupby("day")[f].transform("count")
        parts.append(pd.Series(norm.ppf((r * n - 0.5) / n), index=U.index) * s)
    return pd.concat(parts, axis=1).mean(axis=1, skipna=True).reindex(P.index)


def zoo_screen(x):
    """factor_zoo 'zoo_screen20': P4+Q minus the worst 20% of the composite among P4+Q names (NaN kept)."""
    b = x["p4q"].to_numpy()
    v = x["zoo"].where(x["p4q"])
    bad = (v <= v.quantile(0.2)).to_numpy() & v.notna().to_numpy()
    return b & ~bad


def mom63_overlay() -> pd.Series:
    """macro_cycle ov_mom63_inout: IDA-DI 63d excess over CDI (sum of daily excess) > 0, Friday close, held."""
    ix = H.index_levels()["IDADI"]
    cdi = H._core()["cdi_daily_raw"]
    rx = ix.pct_change() - cdi.reindex(ix.index).fillna(0)
    e = (rx.rolling(63).sum() > 0).astype(float)
    wk = e.index.to_period("W-FRI")
    last = pd.Series(e.index, index=e.index).groupby(wk).transform("max") == e.index
    return e.where(last).ffill()


def books(P, start, **kw):
    """All frozen books on one panel/engine configuration."""
    P = P.copy()
    out = {"U": H.backtest("univ", panel=P, start=start, name="U", **kw),
           "P4": H.backtest("p4", panel=P, start=start, name="P4", **kw),
           "P4Q": H.backtest("p4q", panel=P, start=start, name="P4Q", **kw),
           "P7": H.backtest(p7.make_signal(), panel=P, start=start, name="P7", **KW7, **kw),
           "P7_noliq": H.backtest(p7.make_signal(aum=None), panel=P, start=start, name="P7_noliq", **KW7, **kw),
           "ZOO": H.backtest(zoo_screen, panel=P, start=start, name="ZOO", **kw),
           "P4Q_mom63": H.backtest("p4q", panel=P, start=start, overlay=mom63_overlay(), name="P4Q_mom63", **kw)}
    return out


# ------------------------------------------------------------------------------------------------ stats
def mstats(d: pd.Series, b: pd.Series | None, a, z) -> dict:
    s = d[(d.index >= a) & (d.index <= z)]
    m = H.monthly(s)
    o = {"ann_%": m.mean() * 1200, "vol_%": m.std() * np.sqrt(12) * 100, "t": H.nw_t(m, 6),
         "sharpe": m.mean() * 12 / (m.std() * np.sqrt(12)) if m.std() > 0 else np.nan,
         "maxdd_m_%": H._mdd(m) * 100, "n_m": len(m)}
    eq = (1 + s).cumprod()
    o["maxdd_d_%"] = float((eq / eq.cummax() - 1).min() * 100) if len(s) else np.nan
    if b is not None:
        mb = H.monthly(b.reindex(s.index).fillna(0))
        dd = (m - mb).dropna()
        t = H.nw_t(dd, 6)
        from scipy.stats import norm
        o.update({"diff_%": dd.mean() * 1200, "diff_t": t, "diff_p": float(2 * (1 - norm.cdf(abs(t)))) if t == t else np.nan,
                  "diff_hit": float((dd > 0).mean())})
    return {k: (round(float(v), 4) if isinstance(v, (float, np.floating, int, np.integer)) else v) for k, v in o.items()}


def table(R: dict, bench="P4Q", uni="U") -> dict:
    out = {}
    for nm, r in R.items():
        out[nm] = {}
        for pk, (a, z) in PERIODS.items():
            st = mstats(r["daily"], R[bench]["daily"] if nm != bench else None, a, z)
            su = mstats(r["daily"], R[uni]["daily"], a, z) if nm != uni else {}
            st["exU_%"] = su.get("diff_%")
            st["exU_t"] = su.get("diff_t")
            out[nm][pk] = st
        out[nm]["turnover"] = r.get("turnover_ann")
        out[nm]["n_avg"] = r.get("n_avg")
    vs = [k for k in out if k != bench and k != uni]
    ps = [out[k]["2015_20"].get("diff_p", np.nan) for k in vs]
    for k, h in zip(vs, H.holm(ps)):
        out[k]["2015_20"]["holm_p"] = round(float(h), 4)
    return out


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else round(float(o), 4)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, pd.Timestamp):
        return str(o.date())
    return o


# ------------------------------------------------------------------------------------------------ main
def main():
    res = {}
    # ---------- 1) in-sample (published harness core), 2021-03 .. 2025-12, holdout sealed
    PI = p7.load(holdout=False)
    PI["zoo"] = PI["zoo"]  # frozen factor_zoo composite column from its cache
    ins = books(PI, SPLICE)
    ins_series = {k: v["daily"] for k, v in ins.items()}
    ins_series["IDADI"] = H.index_excess("IDADI")
    ins_series["CDI_daily"] = H.cdi_daily()
    pd.to_pickle(ins_series, OUT / "insample_series.pkl")
    log("in-sample books done")
    ins_tab = {}
    for nm, r in ins.items():
        s = H.stats(r["daily"][r["daily"].index >= "2022-01-01"], bench=ins["P4Q"]["daily"])
        ins_tab[nm] = {k: s.get(k) for k in ("ann_excess_%", "diff_ann_%", "diff_t_nw", "max_dd_%")}
    res["insample_2022_25_reference"] = ins_tab

    # ---------- 2) overlap validation 2021-03..06 vs the harness panel/core
    PM = H.load_panel("M")
    PM = PM[(PM["day"] >= "2021-03-01") & (PM["day"] <= "2021-03-31")]
    Cref = H._core()
    ridx = Cref["days"]
    Pn = pd.read_pickle(OUT / "panel_M.pkl")
    C = pd.read_pickle(OUT / "core.pkl")
    Pn_ov = Pn[(Pn["day"] >= "2021-03-01") & (Pn["day"] <= "2021-03-31")]
    j = PM.merge(Pn_ov, on=["codigo", "day"], suffixes=("_h", "_n"))
    val = {"n_bonds_harness_2021_03": int(PM["univ"].sum()), "n_bonds_rebuilt_2021_03": int(Pn_ov["univ"].sum()),
           "common_rows": int(len(j)),
           "corr_cdi_bps": float(j[["cdi_bps_h", "cdi_bps_n"]].corr().iloc[0, 1]),
           "median_abs_diff_cdi_bps": float((j["cdi_bps_h"] - j["cdi_bps_n"]).abs().median()),
           "corr_resid_z": float(j[["resid_z_h", "resid_z_n"]].corr().iloc[0, 1]),
           "univ_agree": float((j["univ_h"] == j["univ_n"]).mean()),
           "p4_agree_within_univ": float((j.loc[j["univ_h"] & j["univ_n"], "p4_h"] ==
                                          j.loc[j["univ_h"] & j["univ_n"], "p4_n"]).mean()),
           "worstQ_agree_within_univ": float((j.loc[j["univ_h"] & j["univ_n"], "worstQ_h"] ==
                                              j.loc[j["univ_h"] & j["univ_n"], "worstQ_n"]).mean()),
           "listed_agree": float((j["listed_h"] == j["listed_n"]).mean())}
    # daily returns on common bonds / days
    cb = C["codes"].intersection(Cref["codes"])
    cd = C["days"][(C["days"] >= "2021-02-01") & (C["days"] <= "2021-05-31")].intersection(ridx)
    Rn = pd.DataFrame(C["R"][C["days"].get_indexer(cd)][:, C["codes"].get_indexer(cb)], index=cd, columns=cb)
    Rh = pd.DataFrame(Cref["R"][ridx.get_indexer(cd)][:, Cref["codes"].get_indexer(cb)], index=cd, columns=cb)
    nz = (Rn != 0) | (Rh != 0)
    val["daily_R_corr_common"] = float(np.corrcoef(Rn.values[nz.values], Rh.values[nz.values])[0, 1])
    val["cum_R_sum_rebuilt_vs_harness_bps"] = [float(Rn.values.sum() / len(cb) * 1e4), float(Rh.values.sum() / len(cb) * 1e4)]
    val["n_common_bonds"] = int(len(cb))
    pb = ((Rn - Rh).sum() * 1e4).abs()
    val["per_bond_cum_abs_diff_bps_median"] = float(pb.median())
    val["share_bonds_within_10bps"] = float((pb < 10).mean())
    val["note"] = ("large per-bond differences are bonds whose first harness row is in 2021 (harness grid starts "
                   "2021-01-25) while the rebuilt grid carries their 2020 history / gap moves; harness 'listed' is "
                   "False for all of 2021-Q1 because its equity window has < 150 prices then")
    res["overlap_validation_2021"] = val
    log("validation", val)

    # ---------- 3) inject the rebuilt core
    H._MEM.clear()
    H._MEM["core"] = C
    H._MEM["idx"] = pd.read_pickle(H.CACHE / "indices.pkl")
    P = Pn.copy()
    P["zoo"] = zoo_composite(P)
    P = P[P["day"] <= OOS1]
    # coverage
    U = P[P["univ"]]
    cov = pd.DataFrame({"univ": U.groupby("day").size(), "p4": P[P["p4"]].groupby("day").size(),
                        "p4q": P[P["p4q"]].groupby("day").size(),
                        "issuers": U.groupby("day")["cnpj8"].nunique(),
                        "covered_share": U.groupby("day")["covered"].mean(),
                        "listed_share": U.groupby("day")["listed"].mean(),
                        "grid_bonds": P.groupby("day").size()})
    cov.to_csv(HERE / "coverage_by_month.csv")
    covy = cov.groupby(cov.index.year).mean().round(2)
    res["coverage_by_year_mean"] = covy.to_dict(orient="index")
    log("coverage\n" + covy.to_string())

    R = books(P, OOS0)
    R50 = books(P, OOS0, cost_bps=50)
    R40 = books(P, OOS0, scenario="rec40")
    res["oos_25bps"] = table(R)
    res["oos_50bps"] = table(R50)
    res["oos_rec40"] = table(R40)
    res["n_variants_vs_P4Q"] = 5
    for nm, r in R.items():
        res["oos_25bps"][nm]["cost_ann_%"] = r.get("cost_ann_%")
    log("books done")

    # P4Q vs P4 (the Q screen alone)
    res["P4Q_minus_P4"] = {pk: mstats(R["P4Q"]["daily"], R["P4"]["daily"], a, z) for pk, (a, z) in PERIODS.items()}
    res["P7_minus_P4"] = {pk: mstats(R["P7"]["daily"], R["P4"]["daily"], a, z) for pk, (a, z) in PERIODS.items()}

    # IDA-DI
    ida = H.index_excess("IDADI")
    res["IDADI"] = {pk: mstats(ida, None, a, z) for pk, (a, z) in PERIODS.items()}
    res["drawdowns_daily"] = {}
    for nm, s in list({k: v["daily"] for k, v in R.items()}.items()) + [("IDADI", ida)]:
        res["drawdowns_daily"][nm] = {
            "2015_16": mstats(s, None, "2015-01-01", "2016-12-31")["maxdd_d_%"],
            "2020": mstats(s, None, "2020-01-01", "2020-12-31")["maxdd_d_%"],
            "2020_03_05_cum_%": float(((1 + s[(s.index >= "2020-03-01") & (s.index <= "2020-05-31")]).prod() - 1) * 100)}

    # placebos
    pl = H.placebo("p4q", n=20, panel=P, start=OOS0)
    res["placebo_P4Q_random_same_size"] = {"actual_2015_20": pl["actual"], "mean": pl["mean"], "p95": pl["p95"],
                                           "note": "H.stats window: all months of the rebuilt grid <= 2020"}
    rs = []
    for sd in range(10):
        r = H.backtest(p7.make_signal(screen=(f"rand:{sd}",)), panel=P, start=OOS0, **KW7)
        rs.append(mstats(r["daily"], R["P4Q"]["daily"], OOS0, OOS1)["diff_%"])
    res["placebo_P7_random_screen_vs_P4Q"] = {"values": rs, "mean": float(np.mean(rs)),
                                              "p95": float(np.percentile(rs, 95)),
                                              "actual": res["oos_25bps"]["P7"]["2015_20"]["diff_%"]}
    # year-by-year diffs vs P4Q
    yy = {}
    for nm in ("P4", "P7", "P7_noliq", "ZOO", "P4Q_mom63", "U"):
        d = (H.monthly(R[nm]["daily"]) - H.monthly(R["P4Q"]["daily"])).dropna()
        yy[nm] = (d.groupby(d.index.year).mean() * 1200).round(3).to_dict()
    yy["P4Q_vs_U"] = ((H.monthly(R["P4Q"]["daily"]) - H.monthly(R["U"]["daily"])).dropna()
                      .pipe(lambda d: d.groupby(d.index.year).mean() * 1200).round(3).to_dict())
    res["yearly_diff_vs_P4Q_%"] = yy
    # IC of the Q composite and the equity-health score in the OOS panel (6m forward, cohort-level diagnostics)
    icq = []
    for d, x in P[P["univ"] & P["fwd_126"].notna()].groupby("day"):
        if x["f_quality"].notna().sum() >= 20:
            icq.append(x[["f_quality", "fwd_126"]].corr("spearman").iloc[0, 1])
    res["ic_fquality_fwd126"] = {"mean": float(np.nanmean(icq)), "n": len(icq),
                                 "t_naive": float(np.nanmean(icq) / (np.nanstd(icq) / np.sqrt(len(icq))))}
    # carry-neutral: inside P4 names only, and partial IC after ranking out cdi_bps (univ)
    ics = {"fq_in_P4": [], "eqh_in_P4": [], "zoo_in_P4": [], "fq_carry_neutral_univ": [], "carry_in_P4": []}
    for d, x in P[P["univ"] & P["fwd_126"].notna()].groupby("day"):
        x = x.copy()
        x["eqh"] = p7.eqh_score(x)
        y = x[x["p4"]]
        for k, c in (("fq_in_P4", "f_quality"), ("eqh_in_P4", "eqh"), ("zoo_in_P4", "zoo"), ("carry_in_P4", "cdi_bps")):
            if y[c].notna().sum() >= 10:
                ics[k].append(y[[c, "fwd_126"]].corr("spearman").iloc[0, 1])
        z = x[x["f_quality"].notna()]
        if len(z) >= 20:
            rq, rc, rf = z["f_quality"].rank(), z["cdi_bps"].rank(), z["fwd_126"].rank()
            eq_ = rq - np.polyval(np.polyfit(rc, rq, 1), rc)
            ef = rf - np.polyval(np.polyfit(rc, rf, 1), rc)
            ics["fq_carry_neutral_univ"].append(np.corrcoef(eq_, ef)[0, 1])
    res["ic_carry_neutral"] = {k: {"mean": float(np.nanmean(v)), "n": len(v),
                                   "t_nw6": float(H.nw_t(pd.Series(v), 6))} for k, v in ics.items()}

    # save series
    oos = {k: v["daily"] for k, v in R.items()}
    oos["IDADI"] = ida
    oos["CDI_daily"] = H.cdi_daily()
    pd.to_pickle(oos, OUT / "oos_series.pkl")
    (HERE / "results.json").write_text(json.dumps(jsonable(res), indent=1))
    log("saved results.json")
    plots(oos, ins_series)


def plots(oos, ins):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cols = {"U": "#888888", "P4": "#1f77b4", "P4Q": "#2ca02c", "P7": "#d62728", "ZOO": "#9467bd",
            "P4Q_mom63": "#ff7f0e", "IDADI": "#000000"}
    names = ["U", "P4", "P4Q", "P7", "ZOO", "P4Q_mom63", "IDADI"]
    cdi = pd.concat([oos["CDI_daily"][(oos["CDI_daily"].index >= OOS0) & (oos["CDI_daily"].index < SPLICE)],
                     ins["CDI_daily"][(ins["CDI_daily"].index >= SPLICE) & (ins["CDI_daily"].index < H.HOLDOUT)]])
    cdi = cdi[~cdi.index.duplicated()]

    def spl(nm):
        a = oos[nm][(oos[nm].index >= OOS0) & (oos[nm].index < SPLICE)]
        b = ins[nm][(ins[nm].index >= SPLICE) & (ins[nm].index < H.HOLDOUT)]
        s = pd.concat([a, b])
        return s[~s.index.duplicated()].reindex(cdi.index).fillna(0)
    S = {nm: spl(nm) for nm in names}
    fig, ax = plt.subplots(2, 1, figsize=(13, 9), dpi=110, sharex=True)
    tr_cdi = (1 + cdi).cumprod() * 100
    ax[0].plot(tr_cdi.index, tr_cdi, color="#bbbbbb", ls="--", lw=1.2, label="CDI")
    for nm in names:
        tr = (1 + cdi + S[nm]).cumprod() * 100
        ax[0].plot(tr.index, tr, color=cols[nm], lw=2 if nm in ("P4Q", "P7") else 1.2, label=nm)
    ax[0].set_yscale("log")
    ax[0].set_title("Total return, CDI x (1 + excess), base 100 (2015-01 .. 2025-12; holdout 2026 not shown)")
    for nm in names:
        if nm == "U":
            continue
        ce = ((1 + S[nm]).cumprod() / (1 + S["U"]).cumprod() - 1) * 100
        ax[1].plot(ce.index, ce, color=cols[nm], lw=2 if nm in ("P4Q", "P7") else 1.2, label=nm)
    ax[1].axhline(0, color="k", lw=0.6)
    ax[1].set_title("Cumulative excess vs universe U (%)")
    for a in ax:
        a.axvspan(SPLICE, pd.Timestamp("2025-12-31"), color="#f3e9c6", alpha=0.6, lw=0)
        a.axvspan(pd.Timestamp("2015-01-01"), pd.Timestamp("2016-12-31"), color="#e6eef7", alpha=0.5, lw=0)
        a.axvspan(pd.Timestamp("2020-02-15"), pd.Timestamp("2020-06-30"), color="#f7e6e6", alpha=0.6, lw=0)
        a.grid(alpha=0.3)
    ax[0].text(SPLICE + pd.Timedelta(days=20), ax[0].get_ylim()[1] * 0.97, "in-sample (harness, 2021-03+)",
               va="top", fontsize=9)
    ax[0].text(pd.Timestamp("2015-02-01"), ax[0].get_ylim()[1] * 0.97, "OUT-OF-SAMPLE rebuilt grid", va="top", fontsize=9)
    ax[0].legend(ncol=4, fontsize=8, loc="lower right")
    ax[1].legend(ncol=4, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(HERE / "equity_curves_2015_2025.png")
    plt.close(fig)
    # OOS-only zoom: cumulative excess vs P4Q
    fig, ax = plt.subplots(figsize=(12, 4.5), dpi=110)
    for nm in ("P4", "P7", "ZOO", "P4Q_mom63", "U"):
        a = oos[nm][(oos[nm].index >= OOS0) & (oos[nm].index <= OOS1)]
        b = oos["P4Q"].reindex(a.index).fillna(0)
        ce = ((1 + a).cumprod() / (1 + b).cumprod() - 1) * 100
        ax.plot(ce.index, ce, color=cols[nm], label=f"{nm} vs P4+Q")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_title("Out-of-sample 2015-2020: cumulative excess vs P4+Q (%)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(HERE / "oos_vs_p4q.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
