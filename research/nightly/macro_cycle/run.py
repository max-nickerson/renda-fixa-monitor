"""macro_cycle: macro & credit-cycle timing / sizing overlays on the P4+Q debenture book.

Pipeline
  1. data.build()  -> PIT daily macro panel 2009-> (see data.py)
  2. walk-forward timing models trained on a LONG target: IDA-DI excess over CDI, forward 21 bdays starting at t+2
     (overlay known at close t trades at t+1, earns t+1->t+2 onward -> exactly the harness overlay convention).
     Monthly refits from 2012, training rows whose label ended before the refit date, 2010-> (2015-16 recession,
     2020 Covid are in the training data).
  3. exposures: in/out (0/1) and continuous (0..1.5, percentile of the model's own past OOS predictions),
     updated weekly (Friday close, held for the next week) to limit switching.
  4. evaluation: (a) on IDA-DI itself 2012-2021 (long OOS, pre-harness sample) and 2022-2025;
     (b) as an overlay replacing the IDA 21d regime on the P4+Q tranche book (harness semantics: 1-day lag,
     switch cost |d exposure| x exposure x cost/2, levered part pays CDI + FUND_BPS);
     (c) implementable 'entry' variant: the exposure only scales NEW monthly tranches (no forced selling).
  5. stats via harness.compare (Holm across all variants), 50 bps, halves, placebo (circular-shift of exposure),
     extra-lag robustness, sealed holdout once at the end for the pre-declared primary + best.
Run:  PYTHONPATH=. PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=2 .venv/Scripts/python.exe research/nightly/macro_cycle/run.py
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from research.nightly import harness as H  # noqa: E402
from research.nightly.macro_cycle import data as D  # noqa: E402

OUT = Path(__file__).resolve().parent
CACHE = D.CACHE
FUND_BPS = 50.0          # levered part funded at CDI + 50 bps/yr
HOLDOUT = pd.Timestamp("2026-01-01")
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)


LONG_FEATS = ["IDADI_x5", "IDADI_x21", "IDADI_x63", "IDADI_x126", "IDADI_x252", "IDADI_dd252", "IDADI_vol63",
              "IDAIPCA_x21", "IDAIPCA_x63", "IDAGERAL_x63",
              "ibov_x21", "ibov_x63", "ibov_dd252", "ibov_vol21", "usdbrl_r21", "usdbrl_r63", "usdbrl_vol21",
              "selic", "selic_ch126", "selic_ch252", "pre1y_m_selic", "slope_5y1y", "slope_5y1y_ch63", "pre5y_ch63",
              "real5y", "breakeven5y", "ipca_surp_3m", "focus_ipca12_ch63", "real_policy",
              "vix", "vix_ch21", "baa10y", "baa10y_ch63", "ust2_ch63", "brent_r63", "hyg_ief_63", "emb_ief_63",
              "ewz_r63", "br_country_proxy", "br_country_proxy_ch63", "iss_yoy"]


# ------------------------------------------------------------------------------------------------ models
def make_label(F):
    rx = F["IDADI_rx1"]
    # y_t = sum rx[t+2 .. t+22]
    y = rx.rolling(21).sum().shift(-22)
    return y


def hmm_filtered(train_X, all_X, seed=0):
    """2-state Gaussian HMM fit on train_X; FILTERED (forward-only) state probs for all_X rows (no smoothing)."""
    from hmmlearn.hmm import GaussianHMM
    m = GaussianHMM(n_components=2, covariance_type="diag", n_iter=200, random_state=seed)
    m.fit(train_X)
    bad = int(np.argmin(m.means_[:, 0]))              # state with lower IDA-DI weekly excess
    from scipy.stats import multivariate_normal
    ll = np.column_stack([multivariate_normal(m.means_[k], np.diag(m.covars_[k].diagonal()
                                                                    if m.covars_[k].ndim == 2 else m.covars_[k])).logpdf(all_X)
                          for k in range(2)])
    A = m.transmat_
    a = m.startprob_.copy()
    out = np.zeros(len(all_X))
    for i in range(len(all_X)):
        if i > 0:
            a = a @ A
        l = np.exp(ll[i] - ll[i].max())
        a = a * l
        a = a / a.sum()
        out[i] = a[bad]
    return out


def walk_forward(F, y):
    """-> DataFrame of daily OOS predictions (higher = more credit exposure) per model, from 2012-01."""
    import lightgbm as lgb
    from sklearn.linear_model import LogisticRegression, Ridge
    idx = F.index
    X = F[LONG_FEATS]
    fits = pd.date_range("2012-01-01", idx.max(), freq="BMS")
    preds = {k: pd.Series(np.nan, index=idx) for k in ("logit", "ridge", "gbm", "hmm")}
    pos = pd.Series(np.arange(len(idx)), index=idx)
    importances = []
    Z_hmm = pd.DataFrame({"a": F["IDADI_x5"], "b": F["IDADI_vol63"], "c": F["ibov_x21"], "d": F["vix"]})
    for i, f in enumerate(fits):
        f = idx[idx.searchsorted(f)] if idx.searchsorted(f) < len(idx) else None
        if f is None:
            break
        nxt = fits[i + 1] if i + 1 < len(fits) else idx.max() + pd.Timedelta(days=1)
        pf = pos[f]
        tr_mask = (idx >= "2010-01-01") & (np.arange(len(idx)) <= pf - 23) & y.notna().to_numpy()
        tr_idx = idx[tr_mask][::5]
        te_idx = idx[(idx >= f) & (idx < nxt)]
        Xtr, ytr = X.loc[tr_idx], y.loc[tr_idx]
        med = Xtr.median()
        mu, sd = Xtr.fillna(med).mean(), Xtr.fillna(med).std().replace(0, 1)
        Ztr = ((Xtr.fillna(med) - mu) / sd).clip(-5, 5)
        Zte = ((X.loc[te_idx].fillna(med) - mu) / sd).clip(-5, 5)
        lg = LogisticRegression(C=0.1, max_iter=2000).fit(Ztr, (ytr > 0).astype(int))
        preds["logit"].loc[te_idx] = lg.predict_proba(Zte)[:, 1] - 0.5
        rg = Ridge(alpha=50.0).fit(Ztr, ytr)
        preds["ridge"].loc[te_idx] = rg.predict(Zte)
        gb = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.03, num_leaves=7, min_child_samples=40,
                               subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0,
                               n_jobs=2, verbose=-1, random_state=0).fit(Ztr, ytr)
        preds["gbm"].loc[te_idx] = gb.predict(Zte)
        if f.month in (1, 7) or i == 0:
            importances.append(pd.Series(gb.feature_importances_, index=LONG_FEATS, name=str(f.date())))
            coefs = pd.Series(rg.coef_, index=LONG_FEATS)
        # HMM: refit every 6 months on weekly rows, filtered probs for all days up to te end
        if f.month in (1, 7) or i == 0:
            zh = Z_hmm.loc[(idx >= "2010-01-01") & (idx <= f)].dropna()
            hm_mu, hm_sd = zh.iloc[::5].mean(), zh.iloc[::5].std()
            ztr = ((zh.iloc[::5] - hm_mu) / hm_sd).to_numpy()
            hmm_state = (ztr, hm_mu, hm_sd)
        ztr, hm_mu, hm_sd = hmm_state
        zall = Z_hmm.loc[(idx >= "2010-01-01") & (idx < nxt)].ffill().dropna()
        try:
            pb = hmm_filtered(ztr, ((zall - hm_mu) / hm_sd).to_numpy()[::1][-min(len(zall), 1500):])
            pbs = pd.Series(pb, index=zall.index[-len(pb):])
            preds["hmm"].loc[te_idx] = 0.5 - pbs.reindex(te_idx).to_numpy()
        except Exception as e:  # pragma: no cover
            log("hmm fail", f, e)
        if i % 24 == 0:
            log("fit", f.date(), "train rows", len(tr_idx))
    P = pd.DataFrame(preds)
    P["mom21"] = F["IDADI_x21"]
    P["mom63"] = F["IDADI_x63"]
    vote = ((F["vix"] > 25).astype(int) + (F["usdbrl_r21"] > 0.05).astype(int) +
            (F["IDADI_x21"] < 0).astype(int) + (F["ibov_dd252"] < -0.15).astype(int))
    P["vote"] = 1.5 - vote                                  # >0 when fewer than 2 stress flags
    imp = pd.concat(importances, axis=1).mean(axis=1).sort_values(ascending=False) if importances else None
    return P[P.index >= "2012-01-01"], imp, coefs


def pct_rank_expanding(s: pd.Series, min_n=250):
    """percentile of today's value among the model's PAST predictions (strictly before today)."""
    v = s.to_numpy()
    out = np.full(len(v), np.nan)
    hist = []
    import bisect
    for i, x in enumerate(v):
        if np.isfinite(x):
            if len(hist) >= min_n:
                out[i] = bisect.bisect_left(hist, x) / len(hist)
            bisect.insort(hist, x)
    return pd.Series(out, index=s.index)


def weekly_hold(e: pd.Series):
    """value at each Friday close (last business day of the week), held until the next one (no look-ahead)."""
    wk = e.index.to_period("W-FRI")
    last = pd.Series(e.index, index=e.index).groupby(wk).transform("max") == e.index
    return e.where(last).ffill()


def exposures(P: pd.DataFrame):
    E = {}
    members = ["logit", "ridge", "gbm", "hmm"]
    pct = {k: pct_rank_expanding(P[k]) for k in members + ["mom21"]}
    for k in ["mom63", "vote", "logit", "ridge", "gbm", "hmm"]:
        E[f"{k}_inout"] = weekly_hold((P[k] > 0).astype(float).where(P[k].notna()))
    for k in members:
        E[f"{k}_size"] = weekly_hold(((2.5 * pct[k] - 0.25).clip(0, 1.5) * 4).round() / 4)
    ens_vote = sum((P[k] > 0).astype(float) for k in members + ["mom21"]) / 5
    E["ens_inout"] = weekly_hold((ens_vote >= 0.5).astype(float))
    ens_pct = pd.concat([pct[k] for k in members + ["mom21"]], axis=1).mean(axis=1)
    E["ens_size"] = weekly_hold(((2.5 * ens_pct - 0.25).clip(0, 1.5) * 4).round() / 4)
    return pd.DataFrame(E), pct


# ------------------------------------------------------------------------------------------------ evaluation
def apply_overlay(base: dict, ov: pd.Series, cost_bps=25.0, fund_bps=FUND_BPS, extra_lag=0, name=None):
    """Harness overlay semantics on a base book (gross/cost/exposure), plus funding on the levered part."""
    dd = H.days()
    o = ov.reindex(ov.index.union(dd)).ffill().reindex(dd).fillna(1.0)
    o = o.shift(1 + extra_lag).fillna(float(o.iloc[0]))
    ix = base["gross"].index
    g, c, e = base["gross"], base["cost"], base["exposure"]
    sw = o.diff().abs().fillna(0)
    oo = o.reindex(ix)
    cost = c * oo + sw.reindex(ix) * e * cost_bps / 2 / 1e4
    fund = (oo - 1).clip(lower=0) * e * (H.cdi_daily().reindex(ix).fillna(0) * 0 + fund_bps / 1e4 / 252)
    net = g * oo - cost - fund
    return {"daily": net.rename(name or "ov"), "exposure": e * oo, "turnover_ann": base.get("turnover_ann"),
            "n_avg": base.get("n_avg"), "avg_ov": float(oo[oo.index < HOLDOUT].mean())}


def entry_sized(signal, ov: pd.Series, cost_bps=25.0, fund_bps=FUND_BPS, holdout=False, name=None):
    """Implementable: scale each NEW monthly P4+Q tranche by the exposure known at the decision close."""
    C = H._core()
    dd, ND = C["days"], len(C["days"])
    P = H.load_panel("M", holdout=holdout)
    t_end = ND if holdout else H._hpos()
    P = P[(P["day"] >= pd.Timestamp(H.START)) & (P["dpos"] < t_end)]
    T = H._targets(signal, P, 0.2, H.ISSUER_CAP, False, H.MIN_NAMES, "univ")
    o = ov.reindex(ov.index.union(dd)).ffill().reindex(dd).fillna(1.0)
    T = {p: w * float(o.iloc[p]) for p, w in T.items()}
    pnl, cost, expo, turn, _ = H._tranche_book(T, 126, cost_bps, "base", None, t_end)
    first = min(T) + 1
    idx = dd[first:t_end]
    fund = np.clip(expo - 1, 0, None) * fund_bps / 1e4 / 252
    net = pd.Series((pnl - cost - fund)[first:t_end], index=idx, name=name or "entry")
    yrs = len(idx) / 252
    return {"daily": net, "exposure": pd.Series(expo[first:t_end], index=idx),
            "turnover_ann": float(turn[first:t_end].sum() / yrs), "n_avg": np.nan,
            "avg_ov": float(pd.Series(expo[first:t_end], index=idx)[lambda s: s.index < HOLDOUT].mean())}


def ida_eval(E: pd.DataFrame, F: pd.DataFrame, cost_bps=25.0):
    """Timed IDA-DI excess: exposure decided at close t earns rx[t+2]."""
    rx = F["IDADI_rx1"]
    out = {}
    for k in ["BUYHOLD", "mom21_daily"] + list(E.columns):
        if k == "BUYHOLD":
            e = pd.Series(1.0, index=F.index)
        elif k == "mom21_daily":
            e = (F["IDADI_x21"] > 0).astype(float)
        else:
            e = E[k].reindex(F.index)
        e = e.fillna(1.0)
        el = e.shift(2).fillna(1.0)
        r = el * rx - el.diff().abs().fillna(0) * cost_bps / 2 / 1e4 - (el - 1).clip(lower=0) * FUND_BPS / 1e4 / 252
        res = {}
        for lab, a, b in (("2012_2021", "2012-01-01", "2021-12-31"), ("2022_2025", "2022-01-01", "2025-12-31")):
            s = r[a:b]
            m = (1 + s).groupby([s.index.year, s.index.month]).prod() - 1
            eq = (1 + m).cumprod()
            res[lab] = {"ann_%": round(m.mean() * 1200, 3), "vol_%": round(m.std() * np.sqrt(12) * 100, 3),
                        "sharpe": round(m.mean() * 12 / (m.std() * np.sqrt(12)), 2),
                        "maxdd_%": round(float((eq / eq.cummax() - 1).min() * 100), 2),
                        "avg_expo": round(float(el[a:b].mean()), 3),
                        "switches_yr": round(float((el[a:b].diff().abs() > 0).sum() / ((len(s)) / 252)), 1)}
        out[k] = res
    return out


def main():
    log("building macro panel")
    F = D.build()
    F = F[F.index >= "2009-06-01"]
    cov = {c: str(F[c].first_valid_index().date()) if F[c].first_valid_index() is not None else None for c in F.columns}
    log("macro panel", F.shape)
    y = make_label(F)
    cp = CACHE / "preds.pkl"
    if cp.exists() and time.time() - cp.stat().st_mtime < 12 * 3600:
        P, imp, coefs = pd.read_pickle(cp)
    else:
        P, imp, coefs = walk_forward(F, y)
        pd.to_pickle((P, imp, coefs), cp)
    log("preds", P.shape)
    E, pct = exposures(P)
    # ---- univariate predictive power of each feature for IDA-DI fwd21 (pre-2026), 2 samples
    uni = {}
    for c in LONG_FEATS:
        for lab, a, b in (("2010_2021", "2010-01-01", "2021-12-31"), ("2022_2025", "2022-01-01", "2025-11-15")):
            x, yy = F[c][a:b], y[a:b]
            ok = x.notna() & yy.notna()
            if ok.sum() > 100:
                xs, ys = x[ok].iloc[::21], yy[ok].iloc[::21]           # non-overlapping monthly
                from scipy.stats import spearmanr
                uni.setdefault(c, {})[lab] = round(float(spearmanr(xs, ys)[0]), 3)
    # ---- (a) IDA-DI long-sample OOS
    ida_res = ida_eval(E, F)
    log("IDA eval done")
    # ---- (b) overlays on P4+Q
    base25 = H.baseline("P4Q")
    base50 = H.baseline("P4Q", cost_bps=50)
    ida_ov = H.ida_regime()
    chk = H.backtest("p4q", overlay="ida")
    mine = apply_overlay(base25, ida_ov, fund_bps=0)
    log("overlay replication check (harness vs mine, ann diff %):",
        round(float((chk["daily"] - mine["daily"]).mean() * 25200), 5))
    res = {"P4Q+IDA21 (current regime)": mine}
    for k in E.columns:
        res[f"ov_{k}"] = apply_overlay(base25, E[k], name=k)
    res["static_1.25x"] = apply_overlay(base25, pd.Series(1.25, index=H.days()))
    res["static_1.5x"] = apply_overlay(base25, pd.Series(1.5, index=H.days()))
    # spread-level sizing (2021+ only, from the harness panel: median CDI+ spread of the universe, expanding z)
    PM = H.load_panel("M")
    sp = PM[PM["univ"]].groupby("day")["cdi_bps"].median()
    z = (sp - sp.expanding(6).mean().shift(1)) / sp.expanding(6).std().shift(1)
    sp_ov = (((1 + 0.5 * z).clip(0, 1.5) * 4).round() / 4).dropna()
    res["ov_spread_size"] = apply_overlay(base25, sp_ov, name="spread")
    res["entry_spread_size"] = entry_sized("p4q", sp_ov, name="entry_spread")
    res["entry_ens_size"] = entry_sized("p4q", E["ens_size"], name="entry_ens")
    res["entry_ens_inout"] = entry_sized("p4q", E["ens_inout"], name="entry_ens_io")
    res["entry_gbm_size"] = entry_sized("p4q", E["gbm_size"], name="entry_gbm")
    res["P4Q"] = base25
    tab = H.compare(res, bench="P4Q")
    tab["avg_expo_ov"] = [res[k].get("avg_ov", 1.0) if isinstance(res[k], dict) else 1.0 for k in tab.index]
    n_var = len(res) - 1
    log("\n" + tab.round(3).to_string())
    # timing skill vs exposure-matched static book: strategy - avg_ov x P4Q (ex-post matched; diagnostic)
    b = base25["daily"]
    timing = {}
    for k, r in res.items():
        if k == "P4Q":
            continue
        a = r.get("avg_ov", 1.0)
        matched = apply_overlay(base25, pd.Series(a, index=H.days()))["daily"]
        st = H.stats(r["daily"], bench=matched)
        timing[k] = {"avg_expo": round(a, 3), "timing_alpha_%": st["diff_ann_%"], "t": st["diff_t_nw"],
                     "h1": st["diff_h1_%"], "h2": st["diff_h2_%"]}
    # 50 bps + extra-lag robustness for the key variants
    key = ["P4Q+IDA21 (current regime)", "ov_ens_size", "ov_ens_inout", "ov_gbm_size", "ov_logit_inout",
           "ov_hmm_inout", "ov_spread_size", "static_1.25x"]
    rob = {}
    for k in key:
        ov = ida_ov if k.startswith("P4Q+IDA") else (sp_ov if "spread" in k else
                                                    pd.Series(1.25, index=H.days()) if "static" in k else E[k[3:]])
        fb = 0 if k.startswith("P4Q+IDA") else FUND_BPS
        r50 = apply_overlay(base50, ov, cost_bps=50, fund_bps=fb)
        s50 = H.stats(r50["daily"], bench=base50["daily"])
        lag5 = H.stats(apply_overlay(base25, ov, extra_lag=5, fund_bps=fb)["daily"], bench=b)
        lag10 = H.stats(apply_overlay(base25, ov, extra_lag=10, fund_bps=fb)["daily"], bench=b)
        f100 = H.stats(apply_overlay(base25, ov, fund_bps=100 if fb else 0)["daily"], bench=b)
        rob[k] = {"diff50_%": s50["diff_ann_%"], "t50": s50["diff_t_nw"], "diff_lag5_%": lag5["diff_ann_%"],
                  "diff_lag10_%": lag10["diff_ann_%"], "diff_fund100_%": f100["diff_ann_%"]}
    log("robustness", json.dumps(rob, indent=0))
    # placebo: circular shifts of the exposure path (same avg exposure, same #switches)
    rng = np.random.default_rng(0)
    plc = {}
    for k in ["ov_ens_size", "ov_ens_inout", "ov_gbm_size", "P4Q+IDA21 (current regime)"]:
        ov = ida_ov if k.startswith("P4Q+IDA") else E[k[3:]]
        fb = 0 if k.startswith("P4Q+IDA") else FUND_BPS
        g = H.days()
        o = ov.reindex(ov.index.union(g)).ffill().reindex(g)
        win = o[(o.index >= "2021-06-01") & (o.index < HOLDOUT)].dropna()
        bb = b[b.index < HOLDOUT]
        mkey = [bb.index.year, bb.index.month]

        def fast(net):   # ann. mean of monthly summed daily differences (placebo only)
            d = (net.reindex(bb.index) - bb)
            return float(d.groupby(mkey).sum().mean() * 1200)
        act = fast(res[k]["daily"])
        vals = []
        log("placebo", k)
        for _ in range(200):
            sh = pd.Series(np.roll(win.to_numpy(), rng.integers(60, len(win) - 60)), index=win.index)
            vals.append(fast(apply_overlay(base25, sh, fund_bps=fb)["daily"]))
        vals = np.array(vals)
        plc[k] = {"actual_diff_%": act, "placebo_mean_%": round(float(vals.mean()), 3),
                  "placebo_p95_%": round(float(np.quantile(vals, 0.95)), 3), "p_placebo": round(float((vals >= act).mean()), 3)}
    log("placebo", plc)
    # ---- choose best (pre-2026 only) among non-static rows
    cand = tab.drop(index=["P4Q", "static_1.25x", "static_1.5x"])
    best = cand["vs_bench_%"].astype(float).idxmax()
    primary = "ov_ens_size"                              # pre-declared before running
    # ---- curves (pre-2026)
    curves = {"P4+Q + IDA21 regime": res["P4Q+IDA21 (current regime)"], "P4+Q x ens_size overlay": res["ov_ens_size"],
              "P4+Q entry-sized ens": res["entry_ens_size"], f"best: {best}": res[best]}
    H.plot_curves(curves, OUT / "equity_total_return.png",
                  title="macro_cycle overlays on P4+Q (monthly 126d tranches, 25 bps, pre-2026)")
    plot_cum_excess(res, b, OUT / "cum_excess.png", [primary, best, "P4Q+IDA21 (current regime)", "static_1.25x",
                                                     "entry_ens_size", "ov_spread_size"])
    plot_ida(E, F, OUT / "ida_long_sample.png")
    plot_exposure(E, sp_ov, ida_ov, OUT / "exposures.png")
    # ---- sealed holdout, ONCE, for primary + best (+ P4Q and IDA overlay for context)
    bh = H.baseline("P4Q", holdout=True)
    hold = {}
    for k in dict.fromkeys([primary, best, "P4Q+IDA21 (current regime)"]):
        if k.startswith("entry_"):
            src = {"entry_ens_size": E["ens_size"], "entry_ens_inout": E["ens_inout"], "entry_gbm_size": E["gbm_size"],
                   "entry_spread_size": sp_ov}[k]
            rh = entry_sized("p4q", src, holdout=True)
        else:
            ov = ida_ov if k.startswith("P4Q+IDA") else (sp_ov if "spread" in k else E[k[3:]])
            rh = apply_overlay(bh, ov, fund_bps=0 if k.startswith("P4Q+IDA") else FUND_BPS)
        hold[k] = H.stats(rh["daily"], bench=bh["daily"], holdout="only")
    hold["P4Q"] = H.stats(bh["daily"], holdout="only")
    log("holdout", json.dumps(hold, indent=0, default=str))
    # ---- signal export: daily PIT exposure series (date, value), known at close of date
    sig = pd.concat({"ens_size": E["ens_size"], "ens_inout": E["ens_inout"], "gbm_size": E["gbm_size"],
                     "hmm_inout": E["hmm_inout"], "logit_inout": E["logit_inout"], "spread_size": sp_ov,
                     "ens_pct": pd.concat([pct[k] for k in ["logit", "ridge", "gbm", "hmm", "mom21"]], axis=1).mean(axis=1)},
                    axis=1)
    sig.index.name = "date"
    sig.to_pickle(CACHE / "macro_overlay_signals.pkl")
    F.to_pickle(CACHE / "macro_daily_used.pkl")
    out = {"n_variants": n_var, "table_pre2026": tab.round(4).astype(object).where(tab.notna(), None).to_dict(orient="index"),
           "timing_vs_exposure_matched": timing, "robustness": rob, "placebo": plc,
           "ida_long_sample": ida_res, "univariate_spearman_fwd21_IDADI": uni,
           "gbm_importance": None if imp is None else imp.round(1).head(20).to_dict(),
           "ridge_coefs_last": coefs.round(5).sort_values().to_dict(),
           "best_variant": best, "primary_predeclared": primary, "holdout_2026": hold,
           "coverage_first_date": cov, "fund_bps": FUND_BPS, "runtime_s": round(time.time() - T0)}
    (OUT / "results.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    log("done")


def plot_cum_excess(res, bench, path, keys):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    u = H.baseline("U")["daily"]
    fig, axs = plt.subplots(1, 2, figsize=(15, 5.5), dpi=110)
    for k in dict.fromkeys(keys + ["P4Q"]):
        s = res[k]["daily"]
        s = s[s.index < HOLDOUT]
        axs[0].plot(s.index, ((1 + s - u.reindex(s.index).fillna(0)).cumprod() - 1) * 100, lw=1.6 if k != "P4Q" else 2.2,
                    label=k, ls="--" if k == "P4Q" else "-")
        if k != "P4Q":
            axs[1].plot(s.index, ((1 + s - bench.reindex(s.index).fillna(0)).cumprod() - 1) * 100, lw=1.5, label=k)
    axs[0].set_title("Cumulative excess vs universe (%)")
    axs[1].set_title("Cumulative difference vs P4+Q (%)")
    for a in axs:
        a.grid(alpha=.25)
        a.axhline(0, color="k", lw=.6)
        a.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_ida(E, F, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rx = F["IDADI_rx1"]["2012-01-01":"2025-12-31"]
    fig, ax = plt.subplots(figsize=(12, 5), dpi=110)
    ax.plot(rx.index, ((1 + rx).cumprod() - 1) * 100, "k", lw=2, label="IDA-DI buy & hold (excess over CDI)")
    for k, e in (("mom21 daily", (F["IDADI_x21"] > 0).astype(float)), ("ens_size", E["ens_size"]),
                 ("ens_inout", E["ens_inout"]), ("mom63_inout", E["mom63_inout"]), ("gbm_size", E["gbm_size"]), ("hmm_inout", E["hmm_inout"])):
        el = e.reindex(F.index).fillna(1).shift(2).fillna(1)
        r = (el * F["IDADI_rx1"] - el.diff().abs().fillna(0) * 25 / 2 / 1e4
             - (el - 1).clip(lower=0) * FUND_BPS / 1e4 / 252)["2012-01-01":"2025-12-31"]
        ax.plot(r.index, ((1 + r).cumprod() - 1) * 100, lw=1.2, label=k)
    ax.axvline(pd.Timestamp("2022-01-01"), color="grey", ls=":")
    ax.set_title("Timing IDA-DI (walk-forward OOS, 25 bps per switch): cumulative excess over CDI, %")
    ax.grid(alpha=.25)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_exposure(E, sp_ov, ida_ov, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 4), dpi=110)
    for k, s in (("IDA21 regime", ida_ov), ("ens_size", E["ens_size"]), ("ens_inout", E["ens_inout"]), ("spread_size", sp_ov)):
        s = s["2021-06-01":"2025-12-31"]
        ax.step(s.index, s, where="post", lw=1.1, label=k)
    ax.set_title("Exposure paths (pre-2026)")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


if __name__ == "__main__":
    main()
