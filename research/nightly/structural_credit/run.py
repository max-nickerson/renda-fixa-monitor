"""Structural / equity-implied credit lab (nightly slug: structural_credit).

Steps (all pre-2026 unless --holdout):
  1. features: Merton DD, naive DD, CreditGrades spread, market leverage, their changes, bond-vs-equity divergence
  2. IC table (monthly panel, fwd_63 / fwd_126), plus partial ICs controlling for carry
  3. books: P4+Q with structural avoid rules, divergence buy rules (monthly decisions, 126d tranches, 25 bps)
  4. robustness for the best: 50 bps, rec40, halves, random-exclusion placebo
  5. lead-lag: weekly bond spread changes vs issuer equity returns (time FE, issuer-clustered SE)
  6. big-loss early warning: AUC / recall of DD & friends for fwd_126 < -5% / -10%
  7. charts; 8. sealed holdout (only with --holdout, once, for the frozen best variant)

Run:  PYTHONPATH=. PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
      .venv/Scripts/python.exe research/nightly/structural_credit/run.py [--holdout]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats as sst
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from research.nightly import harness as H  # noqa: E402
from research.nightly.structural_credit.signals import signals, load_daily  # noqa: E402

OUT = ROOT / "research" / "nightly" / "structural_credit"
CACHE = ROOT / "data" / "history" / "nightly" / "structural_credit"
RES: dict = {}
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else round(float(o), 5)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, pd.DataFrame):
        return {str(i): jsonable(r.to_dict()) for i, r in o.iterrows()}
    return o


# ------------------------------------------------------------------------------------------------ helpers
def worst_q(x, col, q=0.2, low_bad=True):
    """bool mask: rows in the worst quintile of `col` among rows of x where col is known (unknown -> False)."""
    v = x[col]
    ok = v.notna()
    if ok.sum() < 10:
        return np.zeros(len(x), bool)
    thr = v[ok].quantile(q if low_bad else 1 - q)
    return (ok & ((v <= thr) if low_bad else (v >= thr))).to_numpy()


def in_book_worst(x, col, q=0.2):
    """worst quintile of `col` computed among the listed P4+Q names of the date only."""
    b = x["p4q"].to_numpy() & x[col].notna().to_numpy()
    out = np.zeros(len(x), bool)
    if b.sum() >= 10:
        v = x[col].to_numpy()
        out = b & (v <= np.nanquantile(v[b], q))
    return out


def partial_ic(P, f, ctrl="cdi_bps", tgt="fwd_126"):
    """per-date Spearman IC of feature residualised (rank-OLS) on the control's rank."""
    Hh = int(tgt.split("_")[1])
    X = P[P["univ"] & P[f"dok_{Hh}"] & P[tgt].notna() & P[f].notna() & P[ctrl].notna()]
    per = {}
    for d, x in X.groupby("day"):
        if len(x) < 30:
            continue
        a = x[f].rank().to_numpy()
        c = x[ctrl].rank().to_numpy()
        b = np.polyfit(c, a, 1)
        per[d] = sst.spearmanr(a - np.polyval(b, c), x[tgt])[0]
    s = pd.Series(per).sort_index()
    return {"mean": float(s.mean()), "t_nw": H.nw_t(s, max(Hh // 21, 1)), "n_dates": len(s)}


def main(do_holdout: bool = False):
    log("loading panels + signals")
    PM = signals(H.load_panel("M"))
    PM.to_pickle(CACHE / "panel_M_sc.pkl")
    U = PM[PM["univ"] & (PM["day"] >= H.START)]
    RES["coverage"] = {"univ_rows_with_dd": float(U["sc_dd"].notna().mean()),
                       "p4q_rows_with_dd": float(U.loc[U["p4q"], "sc_dd"].notna().mean()),
                       "univ_rows_listed": float(U["listed"].mean()),
                       "n_tickers": int(U.loc[U["sc_dd"].notna(), "eq_ticker"].nunique()),
                       "n_issuers": int(U.loc[U["sc_dd"].notna(), "cnpj8"].nunique())}
    RES["spearman_corr_univ"] = jsonable(U[["sc_dd", "sc_cg", "sc_div", "sc_div_res", "sc_dd_chg63", "cdi_bps",
                                            "f_quality", "eq_r63", "eq_dd252", "eq_vol63"]].corr("spearman").round(3))
    log("coverage", RES["coverage"])

    # ---------------------------------------------------------------- 2. IC
    feats = ["sc_dd", "sc_dd_naive", "sc_lcg", "sc_mlev", "sc_dd_chg21", "sc_dd_chg63", "sc_lcg_chg63", "sc_div",
             "sc_div_res", "eq_r63", "eq_vol63", "eq_dd252", "cdi_bps", "f_quality"]
    L = PM[PM["sc_dd"].notna()]   # listed-with-structural subset (IC comparable across features)
    ict = {}
    for f in feats:
        row = {}
        for tgt in ("fwd_63", "fwd_126"):
            a = H.ic(L, f, tgt)
            row[f"ic_{tgt}"] = a["mean"]
            row[f"t_{tgt}"] = a["t_nw"]
        pc = partial_ic(L, f, "cdi_bps", "fwd_126") if f != "cdi_bps" else {"mean": np.nan, "t_nw": np.nan}
        row["pic_carry_fwd_126"] = pc["mean"]
        row["pt_carry_fwd_126"] = pc["t_nw"]
        if f not in ("eq_r63", "cdi_bps"):
            pe = partial_ic(L, f, "eq_r63", "fwd_126")
            row["pic_eqr63_fwd_126"] = pe["mean"]
            row["pt_eqr63_fwd_126"] = pe["t_nw"]
        ict[f] = row
    ICT = pd.DataFrame(ict).T
    RES["ic_listed_subset"] = jsonable(ICT.round(4))
    log("IC table\n" + ICT.round(3).to_string())

    # ---------------------------------------------------------------- 3. books
    V = {
        "A1_P4Q_avoid_DDworstQ": lambda x: x["p4q"].to_numpy() & ~worst_q(x, "sc_dd"),
        "A2_P4Q_avoid_DDlt2": lambda x: x["p4q"].to_numpy() & ~(x["sc_dd"] < 2).to_numpy(),
        "A3_P4Q_avoid_CGworstQ": lambda x: x["p4q"].to_numpy() & ~worst_q(x, "sc_cg", low_bad=False),
        "A4_P4Q_avoid_dDD63worstQ": lambda x: x["p4q"].to_numpy() & ~worst_q(x, "sc_dd_chg63"),
        "A5_P4Q_avoid_divworstQ": lambda x: x["p4q"].to_numpy() & ~worst_q(x, "sc_div_res"),
        "A6_P4Q_avoid_DD_or_dDD": lambda x: x["p4q"].to_numpy() & ~worst_q(x, "sc_dd") & ~worst_q(x, "sc_dd_chg63"),
        "A7_P4_avoid_DDworstQ_noQ": lambda x: x["p4"].to_numpy() & ~worst_q(x, "sc_dd"),
        # divergence measured INSIDE the P4+Q book (A5's universe quintile almost never hits high-carry names)
        "A8_P4Q_avoid_divworstQ_inbook": lambda x: x["p4q"].to_numpy() & ~in_book_worst(x, "sc_div_res"),
        # skeptic controls: plain equity-market risk measures, no balance sheet / structural model
        "C1_P4Q_avoid_eqvol63worstQ": lambda x: x["p4q"].to_numpy() & ~worst_q(x, "eq_vol63", low_bad=False),
        "C2_P4Q_avoid_eqdd252worstQ": lambda x: x["p4q"].to_numpy() & ~worst_q(x, "eq_dd252"),
        # buy side: divergence as a carry-per-equity-risk score among P4 filters (unlisted keep carry rank)
        "B1_P4Q_plus_divbuy": lambda x: x["p4q"].to_numpy() | (x["univ"] & x["p4f"] & ~x["worstQ"]).to_numpy()
                                        & (x["sc_div_res"] >= x["sc_div_res"].quantile(0.8)).to_numpy(),
        "B2_listed_div_top30": lambda x: (x["p4f"] & ~x["worstQ"]).to_numpy()
                                          & (x["sc_div_res"].rank(pct=True) >= 0.7).to_numpy(),
    }
    RES["variants_tried"] = list(V)
    books = {}
    for k, f in V.items():
        books[k] = H.backtest(f, panel=PM, name=k)
        log(k, round(H.stats(books[k]["daily"])["ann_excess_%"], 3))
    tab = H.compare({**books, "P4": H.baseline("P4"), "U": H.baseline("U")}, bench="P4Q")
    RES["books_25bps"] = jsonable(tab.round(4))
    log("books vs P4Q\n" + tab.round(3).to_string())

    # 50 bps + rec40 for every variant (same Holm family)
    b50 = {k: H.backtest(f, panel=PM, cost_bps=50) for k, f in V.items()}
    t50 = H.compare(b50, bench="P4Q", cost_bps=50)
    RES["books_50bps"] = jsonable(t50.round(4))
    brec = {k: H.backtest(f, panel=PM, scenario="rec40") for k, f in V.items()}
    trec = H.compare(brec, bench=H.baseline("P4Q", scenario="rec40")["daily"],
                     universe=H.baseline("U", scenario="rec40")["daily"])
    RES["books_rec40"] = jsonable(trec.round(4))
    log("50bps\n" + t50[["exCDI_%", "vs_bench_%", "t_vs_bench", "p_holm"]].round(3).to_string())
    log("rec40\n" + trec[["exCDI_%", "vs_bench_%", "t_vs_bench", "p_holm"]].round(3).to_string())

    # cohort-level (no daily book) 6m / 12m
    coh = {}
    for k in ("A1_P4Q_avoid_DDworstQ", "A3_P4Q_avoid_CGworstQ", "A8_P4Q_avoid_divworstQ_inbook",
              "C1_P4Q_avoid_eqvol63worstQ", "B2_listed_div_top30"):
        row = {}
        for Hh in (126, 252):
            c = H.cohort_excess(V[k], H=Hh, panel=PM)
            cb = H.cohort_excess(H.BASELINES["P4Q"], H=Hh, panel=PM)
            d = (c - cb).dropna()
            row[f"vsP4Q_{Hh}_ann_%"] = d.mean() / (Hh / 252) * 100
            row[f"t_{Hh}"] = H.nw_t(d, Hh // 21)
        coh[k] = row
    RES["cohort_vs_P4Q"] = jsonable(pd.DataFrame(coh).T.round(4))
    log("cohort\n" + pd.DataFrame(coh).T.round(3).to_string())

    # what the avoid rule removes: realised fwd_126 of removed vs kept P4Q names
    rem = {}
    X = PM[PM["univ"] & PM["dok_126"] & (PM["day"] >= H.START)]
    for k in ("A1_P4Q_avoid_DDworstQ", "A3_P4Q_avoid_CGworstQ", "A4_P4Q_avoid_dDD63worstQ", "A5_P4Q_avoid_divworstQ",
              "A8_P4Q_avoid_divworstQ_inbook", "C1_P4Q_avoid_eqvol63worstQ", "C2_P4Q_avoid_eqdd252worstQ"):
        kept, dropped = [], []
        for d, x in X.groupby("day"):
            m = V[k](x)
            p = x["p4q"].to_numpy()
            kept.append(x.loc[p & m, "fwd_126"].fillna(0).mean())
            dropped.append(x.loc[p & ~m, "fwd_126"].fillna(0).mean())
        kd = pd.Series(kept) - pd.Series(dropped)
        rem[k] = {"kept_minus_dropped_6m_%": kd.mean() * 100, "t": H.nw_t(kd.dropna(), 6),
                  "dropped_mean_6m_%": np.nanmean(dropped) * 100, "kept_mean_6m_%": np.nanmean(kept) * 100,
                  "avg_dropped_names": float(np.mean([(x["p4q"].to_numpy() & ~V[k](x)).sum()
                                                      for _, x in X.groupby("day")]))}
    RES["removed_vs_kept"] = jsonable(pd.DataFrame(rem).T.round(4))
    log("removed\n" + pd.DataFrame(rem).T.round(3).to_string())

    # ---------------------------------------------------------------- 4. best variant + placebo
    tt = tab.loc[list(V)]
    best = tt["vs_bench_%"].astype(float).idxmax()
    RES["best_variant"] = best
    log("best by vs-P4Q", best)
    # random-exclusion placebo for avoid rules: drop the same number of LISTED P4Q names at random
    rng = np.random.default_rng(7)
    Pp = PM[(PM["day"] >= H.START) & PM["univ"]]
    pl = []
    fb = V[best]
    for it in range(20):
        rows = []
        for d, x in Pp.groupby("day"):
            m = np.asarray(fb(x), bool)
            p4q = x["p4q"].to_numpy()
            if best.startswith("A"):
                n_drop = int((p4q & ~m).sum())
                cand = np.where(p4q & x["sc_dd"].notna().to_numpy())[0]
                drop = rng.choice(cand, size=min(n_drop, len(cand)), replace=False) if n_drop else []
                sel = p4q.copy()
                sel[drop] = False
            else:
                n_add = int((m & ~p4q).sum())
                cand = np.where(~p4q & (x["p4f"] & ~x["worstQ"]).to_numpy() & x["sc_dd"].notna().to_numpy())[0]
                add = rng.choice(cand, size=min(n_add, len(cand)), replace=False) if n_add else []
                sel = p4q.copy() if best.startswith("B1") else np.zeros(len(x), bool)
                if not best.startswith("B1"):
                    n = int(m.sum())
                    cand = np.where((x["p4f"] & ~x["worstQ"]).to_numpy() & x["sc_dd"].notna().to_numpy())[0]
                    add = rng.choice(cand, size=min(n, len(cand)), replace=False)
                sel[add] = True
            rows.append(pd.DataFrame({"day": d, "codigo": x["codigo"].to_numpy()[sel], "select": True}))
        r = H.backtest(pd.concat(rows), panel=PM)
        pl.append(H.stats(r["daily"], bench=H.baseline("P4Q")["daily"])["diff_ann_%"])
    pl = np.array(pl)
    act = float(tab.loc[best, "vs_bench_%"])
    RES["placebo_best"] = {"actual_vs_P4Q_%": act, "placebo_mean_%": float(pl.mean()),
                           "placebo_p95_%": float(np.percentile(pl, 95)), "placebo_p5_%": float(np.percentile(pl, 5)),
                           "share_placebo_ge_actual": float((pl >= act).mean()), "n": 20}
    log("placebo", RES["placebo_best"])

    # ---------------------------------------------------------------- 5. lead-lag (weekly)
    log("lead-lag")
    PW = signals(H.load_panel("W"))
    PW = PW[PW["day"] < H.HOLDOUT].sort_values(["codigo", "day"])
    wk = {d: i for i, d in enumerate(sorted(PW["day"].unique()))}
    PW["wk"] = PW["day"].map(wk)
    g = PW.groupby("codigo")
    for c in ("cdi_bps", "fresh", "wk", "eq_r5", "sc_dd", "eligible"):
        PW[c + "_n1"] = g[c].shift(-1)       # next weekly row
        PW[c + "_p1"] = g[c].shift(1)        # previous weekly row
    PW["eq_r5_n1"] = g["eq_r5"].shift(-1)
    for L_ in (1, 2, 3, 4):
        PW[f"eq_r5_l{L_}"] = g["eq_r5"].shift(L_)
        PW[f"wk_l{L_}"] = g["wk"].shift(L_)
    ok = (PW["wk_n1"] == PW["wk"] + 1) & (PW["wk_p1"] == PW["wk"] - 1) & PW["fresh"] & PW["fresh_n1"].astype(bool) \
        & PW["fresh_p1"].astype(bool) & PW["eligible"] & PW["eq_r5"].notna() & PW["cdi_bps"].between(-100, 3000)
    ok &= (PW["wk_l4"] == PW["wk"] - 4)
    W = PW[ok].copy()
    W["ds_fwd"] = (W["cdi_bps_n1"] - W["cdi_bps"]).clip(-300, 300)     # spread change t -> t+1 (bps)
    W["ds_now"] = (W["cdi_bps"] - W["cdi_bps_p1"]).clip(-300, 300)     # spread change t-1 -> t
    W["hy"] = W["cdi_bps"] >= 250
    ll = {}

    def fe_ols(df, y, xs, name):
        d = df[[y] + xs + ["day", "cnpj8"]].dropna()
        for c in [y] + xs:                              # week fixed effects (demean by week)
            d[c] = d[c] - d.groupby("day")[c].transform("mean")
        mdl = sm.OLS(d[y], d[xs]).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d["cnpj8"])[0]})
        ll[name] = {"n": int(len(d)), "n_issuers": int(d["cnpj8"].nunique()),
                    **{f"b_{c}": float(mdl.params[c]) for c in xs}, **{f"t_{c}": float(mdl.tvalues[c]) for c in xs}}
    lags = ["eq_r5", "eq_r5_l1", "eq_r5_l2", "eq_r5_l3"]
    fe_ols(W, "ds_now", lags + ["eq_r5_n1"], "contemporaneous_and_lead: ds(t-1,t) on eq r(t), lags, r(t+1)")
    fe_ols(W, "ds_fwd", lags + ["ds_now"], "predictive: ds(t,t+1) on eq r(t..t-3) + own ds")
    fe_ols(W[W["hy"]], "ds_fwd", lags + ["ds_now"], "predictive HY (cdi_bps>=250)")
    fe_ols(W[~W["hy"]], "ds_fwd", lags + ["ds_now"], "predictive IG (<250)")
    fe_ols(W[W["eq_map_type"] == "direct"], "ds_fwd", lags + ["ds_now"], "predictive direct listing")
    fe_ols(W[W["eq_map_type"] == "parent"], "ds_fwd", lags + ["ds_now"], "predictive parent listing")
    fe_ols(W, "eq_r5_n1", ["ds_now", "eq_r5"], "reverse: eq r(t+1) on ds(t-1,t)")
    # 4-week cumulative predictive response: spread change t -> t+4 on the past 4w equity return
    PW["cdi_n4"] = g["cdi_bps"].shift(-4)
    PW["wk_n4"] = g["wk"].shift(-4)
    PW["fresh_n4"] = g["fresh"].shift(-4)
    W4 = PW[(PW["wk_n4"] == PW["wk"] + 4) & PW["fresh"] & PW["fresh_n4"].astype(bool) & PW["eligible"]
            & PW["eq_r21"].notna() & PW["cdi_bps"].between(-100, 3000)].copy()
    W4["ds4_fwd"] = (W4["cdi_n4"] - W4["cdi_bps"]).clip(-500, 500)
    W4["d_spread_4w"] = W4["d_spread_4w"].clip(-500, 500)
    fe_ols(W4, "ds4_fwd", ["eq_r21", "d_spread_4w"], "4w predictive: ds(t,t+4) on eq r21(t) + own 4w ds")
    fe_ols(W4, "ds4_fwd", ["sc_dd_chg21", "d_spread_4w"], "4w predictive: ds(t,t+4) on dDD21(t) + own 4w ds")
    RES["lead_lag"] = jsonable(ll)
    log("lead-lag\n" + pd.DataFrame(ll).T.round(3).to_string())

    # ---------------------------------------------------------------- 6. big-loss early warning
    log("big losses")
    X = PM[PM["univ"] & PM["dok_126"] & (PM["day"] >= H.START) & PM["fwd_126"].notna()].copy()
    XL = X[X["sc_dd"].notna()].copy()
    bl = {}
    for thr in (-0.05, -0.10):
        y = (XL["fwd_126"] < thr).astype(int)
        row = {"n_rows": int(len(XL)), "n_events": int(y.sum()),
               "n_event_issuers": int(XL.loc[y == 1, "cnpj8"].nunique()),
               "share_events_listed_in_univ": float((X["fwd_126"] < thr)[X["sc_dd"].notna()].sum()
                                                    / max((X["fwd_126"] < thr).sum(), 1))}
        for nm, s in {"-sc_dd": -XL["sc_dd"], "sc_cg": XL["sc_cg"], "-sc_dd_chg63": -XL["sc_dd_chg63"],
                      "-sc_div_res": -XL["sc_div_res"], "cdi_bps": XL["cdi_bps"], "-eq_r63": -XL["eq_r63"],
                      "-eq_dd252": -XL["eq_dd252"], "eq_vol63": XL["eq_vol63"], "-f_quality": -XL["f_quality"],
                      "sc_mlev": XL["sc_mlev"]}.items():
            m = s.notna()
            row[f"auc_{nm}"] = float(roc_auc_score(y[m], s[m])) if y[m].nunique() == 2 else np.nan
        # recall/precision of the DD worst quintile (per date) and the carry top-30% baseline
        wq = XL.groupby("day")["sc_dd"].transform(lambda v: v <= v.quantile(0.2))
        cq = XL["cdi_pct"] <= 0.3
        row["recall_DDworstQ"] = float(wq[y == 1].mean())
        row["precision_DDworstQ"] = float(y[wq].mean())
        row["base_rate"] = float(y.mean())
        row["recall_carryTop30"] = float(cq[y == 1].mean())
        row["precision_carryTop30"] = float(y[cq].mean())
        # inside what P4+Q actually holds
        ins = XL["p4q"]
        yi = y[ins]
        for nm, s in {"-sc_dd": -XL["sc_dd"], "-sc_div_res": -XL["sc_div_res"], "cdi_bps": XL["cdi_bps"],
                      "-eq_r63": -XL["eq_r63"], "-f_quality": -XL["f_quality"]}.items():
            m = s[ins].notna()
            row[f"auc_inP4Q_{nm}"] = float(roc_auc_score(yi[m], s[ins][m])) if yi[m].nunique() == 2 else np.nan
        row["recall_inP4Q_DDworstQ"] = float(wq[ins & (y == 1)].mean()) if (ins & (y == 1)).any() else np.nan
        row["n_events_inP4Q"] = int((ins & (y == 1)).sum())
        # incremental: logit event ~ carry + DD (+ equity momentum) — standardized coefs, clustered by issuer
        Z = XL[["cdi_bps", "sc_dd", "eq_r63", "f_quality"]].copy()
        Z["f_quality"] = Z["f_quality"].fillna(0.5)
        Z = Z.dropna()
        Z = (Z - Z.mean()) / Z.std()
        try:
            lg = sm.Logit(y.loc[Z.index], sm.add_constant(Z)).fit(disp=0, cov_type="cluster",
                                                                   cov_kwds={"groups": pd.factorize(XL.loc[Z.index, "cnpj8"])[0]})
            row["logit_z"] = {c: float(lg.tvalues[c]) for c in Z.columns}
            row["logit_b"] = {c: float(lg.params[c]) for c in Z.columns}
        except Exception as e:  # pragma: no cover
            row["logit_err"] = str(e)
        bl[f"fwd126_lt_{int(-thr * 100)}pct"] = row
    # issuer-level: worst events and the DD percentile at decision
    ev = XL[XL["fwd_126"] < -0.10].copy()
    ev["dd_pct"] = XL.groupby("day")["sc_dd"].rank(pct=True).loc[ev.index]
    first = ev.sort_values("day").drop_duplicates("cnpj8")
    bl["issuer_first_event_dd_pct"] = {"n_issuers": int(len(first)),
                                       "median_dd_pct": float(first["dd_pct"].median()),
                                       "share_in_worst_quintile": float((first["dd_pct"] <= 0.2).mean()),
                                       "share_in_worst_40pct": float((first["dd_pct"] <= 0.4).mean()),
                                       "examples": first.sort_values("fwd_126")[["day", "eq_ticker", "cnpj8", "fwd_126",
                                                                                 "sc_dd", "dd_pct", "cdi_bps"]]
                                       .head(15).astype(str).to_dict("records")}
    # unlisted share of all big-loss events (blind spot)
    Xa = X.copy()
    ya = Xa["fwd_126"] < -0.10
    bl["blind_spot"] = {"events_lt10_total": int(ya.sum()),
                        "share_with_dd": float(Xa.loc[ya, "sc_dd"].notna().mean()),
                        "share_listed": float(Xa.loc[ya, "listed"].mean())}
    RES["big_losses"] = jsonable(bl)
    log("big losses", json.dumps(jsonable({k: {kk: vv for kk, vv in v.items() if not isinstance(vv, (dict, list))}
                                            for k, v in bl.items()}), indent=1)[:4000])

    # ---------------------------------------------------------------- 7. charts
    show = {best: books[best]}
    for k in ("A1_P4Q_avoid_DDworstQ", "B2_listed_div_top30"):
        if k != best:
            show[k] = books[k]
    H.plot_curves(show, OUT / "equity_total_return.png", title="Structural credit: equity-implied rules vs baselines")
    _cum_excess_png(show, OUT / "cum_excess.png")
    _dd_timeline_png(OUT / "dd_big_losses.png")

    # ---------------------------------------------------------------- export signal cache
    sig = PM[["codigo", "cnpj8", "day", "eq_ticker"] + [c for c in PM.columns if c.startswith("sc_")]]
    sig.to_pickle(CACHE / "signals_M.pkl")
    sigw = PW[["codigo", "cnpj8", "day", "eq_ticker"] + [c for c in PW.columns if c.startswith("sc_")]]
    sigw.to_pickle(CACHE / "signals_W.pkl")

    # ---------------------------------------------------------------- 8. holdout (once)
    if do_holdout:
        log("HOLDOUT (sealed) for frozen best:", best)
        PH = signals(H.load_panel("M", holdout=True))
        rh = H.backtest(V[best], panel=PH, holdout=True)
        bh = H.baseline("P4Q", holdout=True)
        uh = H.baseline("U", holdout=True)
        RES["holdout_2026"] = {"variant": best,
                               "vs_P4Q": H.stats(rh["daily"], bench=bh["daily"], holdout="only"),
                               "vs_U": H.stats(rh["daily"], bench=uh["daily"], holdout="only")}
        log("holdout", RES["holdout_2026"])
    RES["runtime_s"] = time.time() - T0
    json.dump(jsonable(RES), open(OUT / "results.json", "w"), indent=1, default=str)
    log("done")


def _cum_excess_png(show, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    u = H.baseline("U")["daily"]
    fig, ax = plt.subplots(figsize=(10, 5))
    refs = {"P4": H.baseline("P4")["daily"], "P4+Q": H.baseline("P4Q")["daily"]}
    for nm, s in {**refs, **{k: v["daily"] for k, v in show.items()}}.items():
        m = H.monthly(s) - H.monthly(u.reindex(s.index).fillna(0))
        ax.plot((1 + m).cumprod().index, ((1 + m).cumprod() - 1) * 100, label=nm,
                lw=2.2 if nm in show else 1.3, ls="-" if nm in show else "--")
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_ylabel("cumulative excess vs universe, %")
    ax.set_title("Cumulative excess vs eligible universe (monthly, 25 bps, pre-2026)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _dd_timeline_png(path):
    """DD of a few issuers that had big bond losses, vs universe-median DD."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    D = load_daily()
    D = D[(D["date"] >= "2021-06-01") & (D["date"] < H.HOLDOUT)]
    med = D.groupby("date")["dd"].median()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(med.index, med, color="black", lw=2, label="median DD (all tickers)")
    for t in ("AMER3", "CSNA3", "BHIA3", "AMBP3", "MOVI3", "SIMH3", "DASA3", "PCAR3"):
        s = D[D["ticker"] == t].set_index("date")["dd"]
        if len(s):
            ax.plot(s.index, s.rolling(5).mean(), lw=1, label=t)
    ax.axhline(2, color="red", ls=":", lw=1)
    ax.set_ylim(-2, 12)
    ax.set_ylabel("Merton distance to default (1y)")
    ax.legend(fontsize=7, ncol=3)
    ax.grid(alpha=0.3)
    ax.set_title("Equity-implied distance to default: stressed issuers vs median")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main(do_holdout="--holdout" in sys.argv)
