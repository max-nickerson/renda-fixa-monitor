"""Selection lab: do point-in-time issuer FUNDAMENTALS improve picking bonds with the best forward
3/6/12-month rate-hedged credit returns, beyond carry (cdi_bps) and relative value vs peers (resid_bps)?

Outputs research/out/sellab_results.json, sellab_ic.png, sellab_equity.png, SELLAB_README.md (written by hand
from the json). Log: data/sellab.log. Caches: data/history/sellab_*.pkl.

Run:  PYTHONPATH=. PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe research/run_selection_lab.py [--rebuild]
"""
import sys as _sys
from pathlib import Path as _P
_sys.path.insert(0, str(_P(__file__).resolve().parent.parent))

import json
import time
import warnings
from datetime import date

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sst
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge

from rfmonitor.config import DATA_DIR
from rfmonitor.ml.selection import _accruals

warnings.filterwarnings("ignore")
OUT = _P(__file__).parent / "out"
OUT.mkdir(exist_ok=True)
HIST = DATA_DIR / "history"
LOGF = DATA_DIR / "sellab.log"
REBUILD = "--rebuild" in _sys.argv
QUICK = "--quick" in _sys.argv
t0 = time.time()
_logf = open(LOGF, "a", encoding="utf-8")


def log(*a):
    msg = f"[{time.strftime('%H:%M:%S')} +{time.time() - t0:6.0f}s] " + " ".join(str(x) for x in a)
    print(msg, flush=True)
    _logf.write(msg + "\n")
    _logf.flush()


HZ = (63, 126, 252)
ENTRY_MAX = 20           # pending buys expire after 20 business days (as in lab_daily.run)
ISSUER_CAP = 0.10
FUND_STALE_DAYS = 460    # ignore a fundamentals row whose period_end is older than this
SPLIT = pd.Timestamp("2024-01-01")

# ----------------------------------------------------------------------------------------------------------------
# 1) patched daily returns + decision panel with forward returns (cached)
# ----------------------------------------------------------------------------------------------------------------
PANEL_CACHE = HIST / "sellab_panel.pkl"
RET_CACHE = HIST / "sellab_returns.pkl"


def build_returns(g: pd.DataFrame):
    """Rate-hedged excess return per grid row (mark d -> next grid row), generalised to multi-day gaps.
    lab_daily's `ret` is NaN across gaps (>14 days without trades) and when |ret| > 20% — which silently
    drops default-type losses. Here: isolated spikes in the trade series are removed (a trade that moves >10%
    and the NEXT trade returns to within 3% of the prior level), and every other move is kept."""
    g = g.sort_values(["codigo", "day"]).reset_index(drop=True)
    tr = g.drop_duplicates(["codigo", "date"])[["codigo", "date", "ratio"]].sort_values(["codigo", "date"]).copy()
    pv = tr.groupby("codigo")["ratio"].shift(1)
    nx = tr.groupby("codigo")["ratio"].shift(-1)
    spike = ((tr["ratio"] / pv - 1).abs() > 0.10) & ((nx / pv - 1).abs() < 0.03)
    tr["ratio_c"] = np.where(spike, pv, tr["ratio"])
    n_spike = int(spike.sum())
    g = g.merge(tr[["codigo", "date", "ratio_c"]], on=["codigo", "date"], how="left")
    C, I = _accruals(date(2020, 12, 1))
    days = pd.DatetimeIndex(sorted(g["day"].unique()))
    Cd = C.reindex(days, method="ffill")
    Id = I.reindex(days, method="ffill")
    nday = g.groupby("codigo")["day"].shift(-1)
    has = nday.notna()
    d0 = g["day"].values.astype("datetime64[D]")
    d1 = nday.fillna(g["day"]).values.astype("datetime64[D]")
    nbd = np.busday_count(d0, d1)
    cg = Cd.reindex(nday.fillna(g["day"])).to_numpy() / Cd.reindex(g["day"]).to_numpy()
    ig = Id.reindex(nday.fillna(g["day"])).to_numpy() / Id.reindex(g["day"]).to_numpy()
    c, tau = g["contract"].to_numpy() / 100, nbd / 252
    gpar = np.select([g["kind"].eq("DI_SPREAD"), g["kind"].eq("IPCA")], [cg * (1 + c) ** tau, ig * (1 + c) ** tau],
                     (1 + c) ** tau)
    rn = g.groupby("codigo")["ratio_c"].shift(-1)
    r = gpar * rn / g["ratio_c"] - cg
    db = (g.groupby("codigo")["bench_rate"].shift(-1) - g["bench_rate"]) / 100
    r = r + np.where(g["kind"].isin(["IPCA", "PRE"]), (g["dur"] * db).fillna(0), 0)
    r = np.where(has, r, 0.0)
    r = np.clip(np.nan_to_num(r, nan=0.0), -0.95, 1.0)
    g["r_patch"] = r
    g["gap"] = has & (nbd > 1)
    g["pos"] = days.get_indexer(g["day"])
    diag = {
        "n_spikes_removed": n_spike,
        "n_gap_segments": int(g["gap"].sum()),
        "gap_segments_below_-10%": int(((g["r_patch"] < -0.10) & g["gap"]).sum()),
        "rows_ret_nan_but_contiguous": int((g["ret"].isna() & has & (nbd == 1)).sum()),
        "rows_|r_patch|>20%": int((np.abs(g["r_patch"]) > 0.2).sum()),
        "corr_ret_vs_patch_contiguous": float(np.corrcoef(g.loc[g["ret"].notna() & (nbd == 1), "ret"],
                                                          g.loc[g["ret"].notna() & (nbd == 1), "r_patch"])[0, 1]),
        "sum_log_ret_orig": float(np.log1p(g["ret"].fillna(0)).sum()),
        "sum_log_ret_patch": float(np.log1p(g["r_patch"]).sum()),
    }
    return g, days, diag


def fundamentals(strict: bool) -> pd.DataFrame:
    f = pd.read_pickle(HIST / "fundamentals_pit.pkl").copy()
    col = "available_date_strict" if strict else "available_date"
    f["avail"] = f[col]
    f = f.sort_values(["cnpj8", "avail", "period_end"])
    # the as-of row must never go backwards in period_end (re-apply the README rule under the chosen date)
    f["_mx"] = f.groupby("cnpj8")["period_end"].cummax().groupby(f["cnpj8"]).shift(1)
    f = f[f["_mx"].isna() | (f["period_end"] > f["_mx"])]
    f = f.drop_duplicates(["cnpj8", "avail"], keep="last")
    e = f["ebitda_ltm"]
    lev = np.where(e > 0, f["net_debt"] / e, 15.0)
    lev = np.where(f["net_debt"].isna() | e.isna(), np.nan, lev)
    fe = f["fin_exp_ltm"]
    cov = np.where(fe > 0, e / fe, np.where(fe.notna() & e.notna(), 30.0, np.nan))
    out = pd.DataFrame({
        "cnpj8": f["cnpj8"], "avail": f["avail"], "period_end": f["period_end"],
        "f_lev": np.clip(lev, -3, 15),
        "f_cov": np.clip(cov, -5, 30),
        "f_cash_st": np.log1p(f["cash_to_st_debt"].clip(0, 50)),
        "f_eq_ratio": f["equity_ratio"].clip(-1, 1),
        "f_gde": np.where(f["equity"] <= 0, 10.0, f["gross_debt_equity"].clip(0, 10)),
        "f_margin": f["ebitda_margin"].clip(-1, 1),
        "f_rev_g": f["revenue_growth_yoy"].clip(-0.9, 3),
        "f_d_lev": f["d_net_debt_ebitda_4q"].clip(-10, 10),
        "f_d_cov": f["d_interest_coverage_4q"].clip(-10, 10),
        "f_size": np.log(f["total_assets"].where(f["total_assets"] > 0)),
        "f_is_parent": f["is_parent"].astype(float),
    })
    return out.sort_values("avail")


FCOLS = ["f_lev", "f_cov", "f_cash_st", "f_eq_ratio", "f_gde", "f_margin", "f_rev_g", "f_d_lev", "f_d_cov", "f_size"]
# orientation so that + = "better credit quality" (used only for the composite and screens)
QSIGN = {"f_lev": -1, "f_cov": 1, "f_cash_st": 1, "f_eq_ratio": 1, "f_d_lev": -1}


def attach_fund(D: pd.DataFrame, strict: bool) -> pd.DataFrame:
    D = D.drop(columns=[c for c in D.columns if c.startswith("f_") or c in ("covered", "f_quality")], errors="ignore")
    fu = fundamentals(strict)
    m = pd.merge_asof(D[["_i", "day", "cnpj8"]].sort_values("day"), fu, left_on="day", right_on="avail",
                      by="cnpj8", direction="backward")
    stale = (m["day"] - m["period_end"]).dt.days > FUND_STALE_DAYS
    m.loc[stale, FCOLS + ["f_is_parent"]] = np.nan
    m = m.set_index("_i").reindex(D["_i"])
    for c in FCOLS + ["f_is_parent"]:
        D[c] = m[c].to_numpy()
    D["covered"] = D["f_lev"].notna() & D["f_cov"].notna()
    for c in FCOLS:
        D.loc[~D["covered"], c] = np.nan
    rk = pd.DataFrame({c: D.groupby("day")[c].rank(pct=True) * s for c, s in QSIGN.items()})
    D["f_quality"] = rk.mean(axis=1, skipna=True).where(D["covered"])
    return D


def build_panel():
    log("loading lab_daily.pkl")
    cols = ["codigo", "day", "date", "ratio", "cdi_bps", "dur", "kind", "contract", "cnpj8", "incent", "bench_rate",
            "ret", "peer", "resid_bps", "resid_z", "eligible", "cdi_pct", "press_neg_30d", "rat_days_since_down",
            "eq_ret_4w", "age"]
    g = pd.read_pickle(HIST / "lab_daily.pkl")[cols]
    g, days, diag = build_returns(g)
    log("patched returns", json.dumps(diag))
    codes = pd.Index(sorted(g["codigo"].unique()))
    g["b"] = codes.get_indexer(g["codigo"])
    nd, nb = len(days), len(codes)
    L = {}
    for name, col in (("patch", "r_patch"), ("orig", "ret")):
        M = np.zeros((nd + 1, nb))
        M[g["pos"].to_numpy() + 1, g["b"].to_numpy()] = np.log1p(g[col].fillna(0).clip(-0.95, 1.0).to_numpy())
        L[name] = np.cumsum(M, axis=0)  # L[p] = sum of log-returns of rows with pos < p
    # decision dates: first grid day of each month
    s = pd.Series(days, index=days)
    dec = s.groupby([days.year, days.month]).min()
    dec = [d for d in dec if d >= pd.Timestamp("2022-01-01")]
    by_code = {k: (v["pos"].to_numpy(), v["date"].to_numpy()) for k, v in g.groupby("b")}
    rows = []
    for d in dec:
        p = days.get_loc(d)
        x = g[(g["pos"] == p) & g["eligible"] & g["cdi_bps"].notna()].copy()
        ent = np.full(len(x), -1)
        for j, b in enumerate(x["b"].to_numpy()):
            ps, dts = by_code[b]
            k = np.searchsorted(ps, p + 1)
            while k < len(ps) and ps[k] - p <= ENTRY_MAX:
                if dts[k] > np.datetime64(d):
                    ent[j] = ps[k]
                    break
                k += 1
        x["entry_pos"] = ent
        for H in HZ:
            ok = (ent >= 0) & (ent + H <= nd)
            for name in ("patch", "orig"):
                v = np.full(len(x), np.nan)
                e, bb = ent[ok], x["b"].to_numpy()[ok]
                v[ok] = np.expm1(L[name][e + H, bb] - L[name][e, bb])
                x[f"y{H}" if name == "patch" else f"y{H}_orig"] = v
            x[f"done{H}"] = (ent >= 0) & (ent + H <= nd)
            x[f"lab_end{H}"] = np.where(ent >= 0, ent + H, p + ENTRY_MAX + H)  # position when label is known
        x["dpos"] = p
        rows.append(x)
    D = pd.concat(rows, ignore_index=True)
    D["executed"] = D["entry_pos"] >= 0
    D["_i"] = np.arange(len(D))
    # maturity (for survivorship diagnostics) from the monthly snapshots
    snap = pd.read_pickle(HIST / "snapshots.pkl")[["codigo", "maturity"]].drop_duplicates("codigo")
    last = g.groupby("codigo").agg(last_day=("day", "max"), last_ratio=("ratio_c", "last"))
    last = last.join(snap.set_index("codigo"))
    info = {"days": days, "diag": diag, "last": last}
    return D, g[["codigo", "b", "pos", "r_patch", "ret"]], codes, info


if PANEL_CACHE.exists() and RET_CACHE.exists() and not REBUILD:
    D, info = pd.read_pickle(PANEL_CACHE)
    G, codes = pd.read_pickle(RET_CACHE)
    log(f"panel from cache {D.shape}")
else:
    D, G, codes, info = build_panel()
    pd.to_pickle((D, info), PANEL_CACHE)
    pd.to_pickle((G, codes), RET_CACHE)
    log(f"panel built {D.shape}, dates {D['day'].nunique()}")
days = info["days"]
ND = len(days)
for _H in HZ:  # decision dates on which every bond's label (entry <= +20 bdays, then H bdays) is complete
    D[f"dok{_H}"] = D["dpos"] + ENTRY_MAX + _H <= ND

# survivorship diagnostics: bonds whose trading stopped well before maturity and the data end
last = info["last"]
stopped = last[(last["last_day"] < days[-1] - pd.Timedelta(days=30))
               & (last["maturity"].isna() | (last["maturity"] > last["last_day"] + pd.Timedelta(days=60)))]
surv = {"bonds_stop_trading_before_maturity": int(len(stopped)),
        "of_which_last_ratio_below_0.8": int((stopped["last_ratio"] < 0.8).sum()),
        "of_which_last_ratio_below_0.9": int((stopped["last_ratio"] < 0.9).sum()),
        "note": "After a bond's last grid day its return is 0 (cash at CDI): losses up to the last trade are kept, "
                "anything after it (e.g. a default with no further prints) is unobserved.",
        **info["diag"]}
log("survivorship", json.dumps(surv, default=float))

# ----------------------------------------------------------------------------------------------------------------
# helpers: Newey-West t, IC, portfolios
# ----------------------------------------------------------------------------------------------------------------


def nw_t(x, lags):
    x = np.asarray(pd.Series(x).dropna(), float)
    n = len(x)
    if n < 4:
        return np.nan
    e = x - x.mean()
    v = e @ e / n
    for l in range(1, min(lags, n - 1) + 1):
        v += 2 * (1 - l / (lags + 1)) * (e[l:] @ e[:-l]) / n
    return float(x.mean() / np.sqrt(v / n)) if v > 0 else np.nan


def summarize(series: pd.Series, H: int) -> dict:
    s = series.dropna()
    lag = H // 21
    step = max(lag, 1)
    no = s.iloc[::step]
    t_no = float(no.mean() / (no.std(ddof=1) / np.sqrt(len(no)))) if len(no) > 2 and no.std() > 0 else np.nan
    return {"mean": float(s.mean()), "t_nw": nw_t(s, lag), "t_nonoverlap": t_no, "n_dates": int(len(s))}


def _ctrl_matrix(x: pd.DataFrame) -> np.ndarray:
    cols = [x["cdi_bps"].rank(pct=True), x["resid_bps"].rank(pct=True).fillna(0.5), x["resid_bps"].isna().astype(float),
            x["dur"].rank(pct=True)]
    peers = pd.get_dummies(x["peer"], drop_first=True, dtype=float)
    return np.column_stack(cols + [peers.to_numpy(), np.ones(len(x))])


def _resid(y, X):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ beta


def ic_table(D: pd.DataFrame, feats, H, mode="raw", min_n=30):
    """mode: raw (Spearman), partial (rank residualised on carry+RV+dur+peer), peer (within-peer Spearman)."""
    y = f"y{H}"
    out = {}
    per = {f: {} for f in feats}
    for d, x in D[D[y].notna() & D[f"dok{H}"]].groupby("day"):
        if mode == "partial":
            X = _ctrl_matrix(x)
            ry = _resid(x[y].rank(pct=True).to_numpy(), X)
        for f in feats:
            ok = x[f].notna().to_numpy()
            if ok.sum() < min_n:
                continue
            if mode == "raw":
                per[f][d] = sst.spearmanr(x.loc[ok, f], x.loc[ok, y])[0]
            elif mode == "partial":
                rf = x[f].rank(pct=True).to_numpy()
                Xo = X[ok]
                a, b = _resid(rf[ok], Xo), _resid(x[y].rank(pct=True).to_numpy()[ok], Xo)
                per[f][d] = np.corrcoef(a, b)[0, 1] if a.std() > 0 and b.std() > 0 else np.nan
            else:
                vals, wts = [], []
                for _, xp in x[ok].groupby("peer"):
                    if len(xp) >= 15 and xp[f].nunique() > 2:
                        vals.append(sst.spearmanr(xp[f], xp[y])[0])
                        wts.append(len(xp))
                if vals:
                    per[f][d] = np.average(vals, weights=wts)
    for f in feats:
        out[f] = summarize(pd.Series(per[f]).sort_index(), H)
    return out, per


def cap_weights(issuer: np.ndarray, cap=ISSUER_CAP) -> np.ndarray:
    n = len(issuer)
    if n == 0:
        return np.array([])
    w = np.full(n, 1.0 / n)
    iss = pd.Series(issuer)
    for _ in range(20):
        tot = pd.Series(w).groupby(iss).transform("sum").to_numpy()
        over = tot > cap + 1e-12
        if not over.any():
            break
        w[over] *= cap / tot[over]
        free = ~over
        if not free.any():
            break
        w[free] *= (1 - w[over].sum()) / w[free].sum()
    return w / w.sum()


def portfolio(D: pd.DataFrame, select, H: int, name: str, keep_weights=False):
    """select(x) -> boolean mask (or float score, top quintile taken) on the decision cross-section x.
    Returns per-cohort gross forward returns of the tranche (unexecuted weight earns 0 = CDI)."""
    y = f"y{H}"
    res, W = {}, {}
    for d, x in D[D[f"dok{H}"]].groupby("day"):
        if x[y].notna().sum() < 30:
            continue
        s = select(x)
        if s is None:
            continue
        s = pd.Series(s, index=x.index)
        if s.dtype != bool:
            s = s.notna() & (s >= s.quantile(0.8))
        xs = x[s.to_numpy()]
        if len(xs) < 5:
            continue
        w = cap_weights(xs["cnpj8"].to_numpy())
        res[d] = float(np.sum(w * xs[y].fillna(0).to_numpy()))
        if keep_weights:
            W[d] = pd.Series(w, index=xs["codigo"].to_numpy()).groupby(level=0).sum()
    return pd.Series(res, name=name).sort_index(), W


def book_cost(W: dict, H: int, cost_bps: float) -> float:
    """Annual cost of the tranche book: book weight = average of the last H/21 cohorts' weights;
    cost = Σ|Δw| × (cost_bps/2) per month (one-way cost = half the round trip, as in lab_daily)."""
    M = max(H // 21, 1)
    ds = sorted(W)
    prev, tot = None, []
    for i, d in enumerate(ds):
        live = [W[x] for x in ds[max(0, i - M + 1): i + 1]]
        book = pd.concat(live, axis=1).fillna(0).sum(axis=1) / M
        if prev is not None:
            tot.append(book.sub(prev, fill_value=0).abs().sum())
        prev = book
    return float(np.mean(tot) * 12 * cost_bps / 2 / 1e4) if tot else np.nan


# ----------------------------------------------------------------------------------------------------------------
# selection rules
# ----------------------------------------------------------------------------------------------------------------
nz = lambda s, v: s.fillna(v)


def p4_mask(x):
    return (nz(x["cdi_pct"], 1) <= 0.3) & ~(nz(x["resid_z"], 0) <= -1.5) & (nz(x["press_neg_30d"], 0) < 1)


def screen(feat, worst_is_high: bool, keep_uncovered=True, q=0.2):
    def f(x):
        m = p4_mask(x)
        v = x[feat]
        thr = v.quantile(1 - q) if worst_is_high else v.quantile(q)
        bad = (v >= thr) if worst_is_high else (v <= thr)
        bad = bad & v.notna()
        unc = ~x["covered"] if keep_uncovered else pd.Series(False, index=x.index)
        return (m & ~bad & (x["covered"] | unc)).to_numpy()
    return f


# ---------------- expected-return score (d): carry - EL + convergence x duration ----------------
KAPPA = 0.25  # fraction of the peer residual assumed to converge per 6 months (fixed a priori, not fitted)
PD_FEATS = ["f_lev", "f_cov", "f_cash_st", "f_eq_ratio", "f_d_lev"]


def fit_pd(D: pd.DataFrame):
    """Walk-forward PD per decision date: logistic on 'fwd 6m hedged return < -10%' using covered rows whose
    label was fully known at the decision date. Uncovered bonds get the historical event rate of uncovered bonds."""
    pd_hat = pd.Series(np.nan, index=D.index)
    lgd_used = {}
    dpos = D.groupby("day")["dpos"].first()
    for d, p in dpos.items():
        tr = D[(D["lab_end126"] <= p) & D["y126"].notna()]
        x = D[D["day"] == d]
        if len(tr) == 0:
            continue
        ev = (tr["y126"] < -0.10)
        trc = tr[tr["covered"]]
        evc = ev[tr["covered"]]
        unc_rate = ev[~tr["covered"]].mean() if (~tr["covered"]).sum() > 50 else ev.mean()
        lgd = -tr.loc[ev, "y126"].mean() if ev.sum() >= 10 else 0.3
        lgd_used[d] = lgd
        if evc.sum() < 30:
            continue
        Xtr = trc[PD_FEATS].fillna(trc[PD_FEATS].median())
        mu, sd = Xtr.mean(), Xtr.std().replace(0, 1)
        clf = LogisticRegression(C=0.5, max_iter=500).fit((Xtr - mu) / sd, evc)
        xc = x[x["covered"]]
        if len(xc):
            Xp = xc[PD_FEATS].fillna(Xtr.median())
            pd_hat.loc[xc.index] = clf.predict_proba((Xp - mu) / sd)[:, 1]
        pd_hat.loc[x.index[~x["covered"].to_numpy()]] = unc_rate
        D.loc[x.index, "_lgd"] = lgd
    return pd_hat


def er_score(H, use_el=True):
    def f(x):
        carry = x["cdi_bps"] / 1e4 * H / 252
        conv = KAPPA * (H / 126) * x["resid_bps"].fillna(0) / 1e4 * x["dur"].clip(upper=8)
        if use_el:
            if x["pd6"].notna().sum() == 0:
                return None  # PD model not yet trainable -> skip date (same dates for the ablation below)
            el = x["pd6"].fillna(x["pd6"].median()) * (H / 126) * x["_lgd"].fillna(0.3)
            return carry - el + conv
        if x["pd6"].notna().sum() == 0:
            return None
        return carry + conv
    return f


# ---------------- walk-forward ML (e) ----------------
BASE = ["cdi_bps", "resid_bps", "resid_z", "dur", "press_neg_30d", "incent"]
FUNDF = FCOLS + ["f_quality"]


def design(x: pd.DataFrame, feats):
    cols = {}
    for f in feats:
        if f in ("incent",):
            cols[f] = x[f].astype(float)
        elif f == "press_neg_30d":
            cols[f] = (x[f].fillna(0) >= 1).astype(float)
        else:
            r = x.groupby("day")[f].rank(pct=True)
            cols[f] = r.fillna(0.5)
            if x[f].isna().any():
                cols[f + "_na"] = x[f].isna().astype(float)
    cols["kind_ipca"] = (x["kind"] == "IPCA").astype(float)
    cols["kind_pre"] = (x["kind"] == "PRE").astype(float)
    return pd.DataFrame(cols, index=x.index)


def walk_forward(D: pd.DataFrame, H: int, feats, model="ridge", retrain_every=1, min_train_dates=6):
    y = f"y{H}"
    Xall = design(D, feats)
    tgt = D.groupby("day")[y].rank(pct=True) - 0.5
    pred = pd.Series(np.nan, index=D.index)
    dates = sorted(D["day"].unique())
    dpos = D.groupby("day")["dpos"].first()
    m, last_fit = None, -99
    for i, d in enumerate(dates):
        p = dpos[d]
        tr = (D["lab_end%d" % H] <= p) & D[y].notna()
        if D.loc[tr, "day"].nunique() < min_train_dates:
            continue
        if m is None or i - last_fit >= retrain_every:
            Xt, yt = Xall[tr], tgt[tr]
            if model == "ridge":
                m = Ridge(alpha=10.0).fit(Xt, yt)
            else:
                m = HistGradientBoostingRegressor(max_iter=150, learning_rate=0.05, max_leaf_nodes=15,
                                                  min_samples_leaf=200, l2_regularization=1.0,
                                                  random_state=0).fit(Xt, yt)
            last_fit = i
        te = D["day"] == d
        pred[te] = m.predict(Xall[te])
    return pred


# ----------------------------------------------------------------------------------------------------------------
# run everything for one fundamentals-availability setting
# ----------------------------------------------------------------------------------------------------------------
SIG = ["cdi_bps", "resid_bps", "resid_z", "dur"]


def run_all(strict: bool, full: bool):
    tag = "strict" if strict else "nonstrict"
    X = attach_fund(D.copy(), strict)
    X = X.set_index("_i", drop=False)
    X.index.name = None
    cov_share = float(X["covered"].mean())
    log(f"[{tag}] covered share of decision bond-rows {cov_share:.3f}")
    R = {"coverage": {"share_rows_covered": cov_share,
                      "share_rows_covered_by_year": X.groupby(X["day"].dt.year)["covered"].mean().round(3).to_dict()}}

    # ---- IC ----
    ic = {}
    ic_series = {}
    for H in HZ:
        C = X[X["covered"]]
        raw_all, _ = ic_table(X, SIG, H, "raw")
        raw_cov, per_raw = ic_table(C, SIG + FUNDF, H, "raw")
        part, per_part = ic_table(C, FUNDF, H, "partial")
        peer, _ = ic_table(C, FUNDF, H, "peer")
        ic[H] = {"signals_full_universe": raw_all, "raw_covered": raw_cov, "partial_vs_carry_rv_peer": part,
                 "within_peer": peer}
        ic_series[H] = {"raw": per_raw, "partial": per_part}
        log(f"[{tag}] IC H={H}: " + ", ".join(f"{f} raw {raw_cov[f]['mean']:+.3f}({raw_cov[f]['t_nw']:+.1f}) "
                                                  f"part {part[f]['mean']:+.3f}({part[f]['t_nw']:+.1f})"
                                                  for f in FUNDF if f in part))
    R["ic"] = ic

    # ---- PD model for ER score ----
    X["pd6"] = fit_pd(X)
    if "_lgd" not in X:
        X["_lgd"] = np.nan

    # ---- portfolios ----
    variants = {
        "U  universo": lambda H: (lambda x: np.ones(len(x), bool)),
        "P4": lambda H: (lambda x: p4_mask(x).to_numpy()),
        "P4 + excl. pior quintil ND/EBITDA": lambda H: screen("f_lev", True),
        "P4 + excl. pior quintil cobertura": lambda H: screen("f_cov", False),
        "P4 + excl. alavancagem subindo (top quintil dND/EBITDA 4t)": lambda H: screen("f_d_lev", True),
        "P4 + excl. pior quintil qualidade composta": lambda H: screen("f_quality", False),
        "P4 + excl. pior quintil qualidade, exclui não cobertos": lambda H: screen("f_quality", False, False),
        "P4 só cobertos (controle)": lambda H: (lambda x: (p4_mask(x) & x["covered"]).to_numpy()),
        "Score ER: carry - EL + convergência": lambda H: er_score(H, True),
        "Score ER sem EL (ablação)": lambda H: er_score(H, False),
    }
    ml_specs = {}
    if full:
        for H in HZ:
            for model in ("ridge", "gbm"):
                for lab, feats in (("base", BASE), ("fund", BASE + FUNDF)):
                    k = f"{model}_{lab}"
                    col = f"pred_{k}_{H}"
                    X[col] = walk_forward(X, H, feats, model, retrain_every=1 if model == "ridge" else 3)
                    ml_specs.setdefault(k, {})[H] = col
                    log(f"[{tag}] walk-forward {k} H={H} done")
        names = {"ridge_base": "Ridge carry+RV (sem fund.)", "ridge_fund": "Ridge carry+RV+fund.",
                 "gbm_base": "GBM carry+RV (sem fund.)", "gbm_fund": "GBM carry+RV+fund."}
        for k, nm in names.items():
            variants[nm] = (lambda kk: (lambda H: (lambda x: x[ml_specs[kk][H]] if x[ml_specs[kk][H]].notna().any()
                                                    else None)))(k)

    port, cohorts, weights = {}, {}, {}
    for H in HZ:
        uni, Wu = portfolio(X, variants["U  universo"](H), H, "U", keep_weights=True)
        port[H] = {}
        for nm, mk in variants.items():
            s, W = portfolio(X, mk(H), H, nm, keep_weights=True)
            cohorts[(H, nm)] = s
            weights[(H, nm)] = W
            ex = (s - uni.reindex(s.index))
            yrs = H / 252
            c25, c50 = book_cost(W, H, 25), book_cost(W, H, 50)
            u25, u50 = book_cost(Wu, H, 25), book_cost(Wu, H, 50)
            sm = summarize(ex, H)
            port[H][nm] = {
                "n_cohorts": int(len(s)), "first": str(s.index.min().date()) if len(s) else None,
                "avg_bonds": float(np.mean([len(w) for w in W.values()])) if W else np.nan,
                "gross_ann_%": float(s.mean() / yrs * 100),
                "excess_vs_U_ann_%_gross": float(ex.mean() / yrs * 100), "t_nw": sm["t_nw"],
                "t_nonoverlap": sm["t_nonoverlap"],
                "book_cost_ann_%_25": c25 * 100, "book_cost_ann_%_50": c50 * 100,
                "excess_vs_U_ann_%_net25": float((ex.mean() / yrs - (c25 - u25)) * 100),
                "excess_vs_U_ann_%_net50": float((ex.mean() / yrs - (c50 - u50)) * 100),
                "hit_rate_vs_U": float((ex > 0).mean()),
                "excess_2022_23_ann_%": float(ex[ex.index < SPLIT].mean() / yrs * 100),
                "excess_2024_26_ann_%": float(ex[ex.index >= SPLIT].mean() / yrs * 100),
            }
            q = port[H][nm]
            log(f"[{tag}] H={H} {nm:<62} n={q['n_cohorts']:3d} bonds {q['avg_bonds']:5.0f} gross {q['gross_ann_%']:6.2f}"
                f" exU {q['excess_vs_U_ann_%_gross']:+6.2f} (t {q['t_nw']:+5.2f}) net25 {q['excess_vs_U_ann_%_net25']:+6.2f}")
    R["portfolios"] = port

    # ---- incremental (paired) tests: fundamental variant minus its no-fundamentals base ----
    pairs = [("P4 + excl. pior quintil ND/EBITDA", "P4"), ("P4 + excl. pior quintil cobertura", "P4"),
             ("P4 + excl. alavancagem subindo (top quintil dND/EBITDA 4t)", "P4"),
             ("P4 + excl. pior quintil qualidade composta", "P4"),
             ("P4 + excl. pior quintil qualidade, exclui não cobertos", "P4 só cobertos (controle)"),
             ("Score ER: carry - EL + convergência", "Score ER sem EL (ablação)")]
    if full:
        pairs += [("Ridge carry+RV+fund.", "Ridge carry+RV (sem fund.)"), ("GBM carry+RV+fund.", "GBM carry+RV (sem fund.)")]
    inc = {}
    for H in HZ:
        for a, b in pairs:
            sa, sb = cohorts[(H, a)], cohorts[(H, b)]
            dd = (sa - sb.reindex(sa.index)).dropna()
            if len(dd) < 4:
                continue
            t = nw_t(dd, H // 21)
            cost_diff = book_cost(weights[(H, a)], H, 25) - book_cost(weights[(H, b)], H, 25)
            inc[f"{a} vs {b} | H={H}"] = {
                "diff_ann_%_gross": float(dd.mean() / (H / 252) * 100), "t_nw": t,
                "p_two_sided": float(2 * (1 - sst.norm.cdf(abs(t)))) if t == t else np.nan,
                "diff_ann_%_net25": float((dd.mean() / (H / 252) - cost_diff) * 100),
                "diff_2022_23": float(dd[dd.index < SPLIT].mean() / (H / 252) * 100),
                "diff_2024_26": float(dd[dd.index >= SPLIT].mean() / (H / 252) * 100),
                "n": int(len(dd)), "variant": a, "base": b, "H": H}
    # Holm across all incremental tests
    keys = [k for k in inc if inc[k]["p_two_sided"] == inc[k]["p_two_sided"]]
    ps = np.array([inc[k]["p_two_sided"] for k in keys])
    order = np.argsort(ps)
    m = len(ps)
    adj = np.empty(m)
    run = 0.0
    for r_, i in enumerate(order):
        run = max(run, min(1.0, (m - r_) * ps[i]))
        adj[i] = run
    for k, a in zip(keys, adj):
        inc[k]["p_holm"] = float(a)
    R["incremental"] = inc
    for k, v in inc.items():
        log(f"[{tag}] INC {k:<110} {v['diff_ann_%_gross']:+6.2f}%/a t {v['t_nw']:+5.2f} holm {v['p_holm']:.3f}")

    # ---- covered vs uncovered ----
    cu = {}
    for H in HZ:
        y = f"y{H}"
        Z = X[X[y].notna()].copy()
        Z["dm"] = Z[y] - Z.groupby("day")[y].transform("mean")
        per = Z.groupby(["day", "covered"])["dm"].mean().unstack()
        diff = (per.get(False) - per.get(True)).dropna()
        s = summarize(diff, H)
        # partial: uncovered dummy after carry/RV/peer controls
        coefs = {}
        for d, x in Z.groupby("day"):
            Xc = np.column_stack([_ctrl_matrix(x), (~x["covered"]).astype(float)])
            beta, *_ = np.linalg.lstsq(Xc, x[y].rank(pct=True).to_numpy(), rcond=None)
            coefs[d] = beta[-1]
        cu[H] = {"uncovered_minus_covered_ann_%": s["mean"] / (H / 252) * 100, "t_nw": s["t_nw"],
                 "uncovered_rank_coef_after_controls": summarize(pd.Series(coefs).sort_index(), H),
                 "share_uncovered_in_P4": float((~X.loc[p4_mask(X), "covered"]).mean()),
                 "share_uncovered_in_universe": float((~X["covered"]).mean()),
                 "mean_cdi_bps_uncovered": float(X.loc[~X["covered"], "cdi_bps"].median()),
                 "mean_cdi_bps_covered": float(X.loc[X["covered"], "cdi_bps"].median())}
    R["covered_vs_uncovered"] = cu

    # ---- indexer (DI+ vs IPCA+ vs Pré, hedged) ----
    ix = {}
    for H in HZ:
        y = f"y{H}"
        Z = X[X[y].notna()]
        u = Z.groupby("day")[y].mean()
        k = Z.groupby(["day", "kind"])[y].mean().unstack()
        ix[H] = {kd: {**summarize(k[kd] - u, H), "ann_%": float((k[kd] - u).mean() / (H / 252) * 100)}
                 for kd in k.columns}
        # P4 ranking carry within kind instead of across
        def p4_within(x):
            pct = x.groupby("kind")["cdi_bps"].rank(pct=True, ascending=False)
            return ((pct <= 0.3) & ~(nz(x["resid_z"], 0) <= -1.5) & (nz(x["press_neg_30d"], 0) < 1)).to_numpy()
        s, _ = portfolio(X, p4_within, H, "p4w")
        b = cohorts[(H, "P4")]
        dd = (s - b.reindex(s.index)).dropna()
        ix[H]["P4 carry rankeado dentro do indexador vs P4"] = {"diff_ann_%": float(dd.mean() / (H / 252) * 100),
                                                               "t_nw": nw_t(dd, H // 21)}
        ix[H]["share_kind_in_P4"] = X.loc[p4_mask(X), "kind"].value_counts(normalize=True).round(3).to_dict()
        ix[H]["share_kind_in_universe"] = X["kind"].value_counts(normalize=True).round(3).to_dict()
    R["indexer"] = ix
    return R, X, cohorts, weights, ic_series


# ----------------------------------------------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------------------------------------------
results = {"survivorship": surv, "design": {
    "decision_dates": f"first grid day of each month, {D['day'].min().date()}..{D['day'].max().date()}",
    "horizons_bdays": list(HZ), "entry": "first trade dated after decision day, >=1 bday later, within 20 bdays; "
                                         "otherwise the weight stays in cash (0 excess)",
    "target": "compounded rate-hedged excess return over CDI (patched daily chain incl. gap moves), from entry",
    "issuer_cap": ISSUER_CAP, "fund_stale_days": FUND_STALE_DAYS, "kappa_convergence_per_6m": KAPPA}}

R_s, XS, COH, WTS, ICS = run_all(strict=True, full=not QUICK)
results["main_strict"] = R_s
R_n, _, COH_n, _, _ = run_all(strict=False, full=False)
results["check_nonstrict"] = {"ic": R_n["ic"], "incremental": R_n["incremental"],
                              "portfolios": R_n["portfolios"]}

# ---- original-ret (unpatched) check for main portfolios ----
orig = {}
for H in HZ:
    XO = XS.copy()
    XO[f"y{H}"] = XO[f"y{H}_orig"]
    u, _ = portfolio(XO, lambda x: np.ones(len(x), bool), H, "U")
    for nm, f in (("P4", lambda x: p4_mask(x).to_numpy()),
                  ("P4 + excl. pior quintil qualidade composta", screen("f_quality", False))):
        s, _ = portfolio(XO, f, H, nm)
        orig[f"{nm} | H={H}"] = float((s - u.reindex(s.index)).mean() / (H / 252) * 100)
results["check_unpatched_ret_excess_vs_U_ann_%"] = orig
log("unpatched check", json.dumps(orig))

# ---- placebo: shuffle fundamentals across covered issuers within each date ----
inc = R_s["incremental"]
sel_fn = {"P4 + excl. pior quintil ND/EBITDA": lambda: screen("f_lev", True),
          "P4 + excl. pior quintil cobertura": lambda: screen("f_cov", False),
          "P4 + excl. alavancagem subindo (top quintil dND/EBITDA 4t)": lambda: screen("f_d_lev", True),
          "P4 + excl. pior quintil qualidade composta": lambda: screen("f_quality", False),
          "P4 + excl. pior quintil qualidade, exclui não cobertos": lambda: screen("f_quality", False, False)}
cands = {k: v for k, v in inc.items() if v["variant"] in sel_fn}
best_key = max(cands, key=lambda k: cands[k]["t_nw"] if cands[k]["t_nw"] == cands[k]["t_nw"] else -9)
targets = [best_key] + [k for k in cands if k != best_key and (
    k.startswith("P4 + excl. pior quintil qualidade composta vs") or
    (k.startswith("P4 + excl. pior quintil qualidade, exclui") and cands[k]["H"] == 126))]
rng = np.random.default_rng(7)
NPERM = 10 if QUICK else 100
fundcols = FCOLS + ["f_quality"]
cov_rows = XS[XS["covered"]]
groups = [(x.index, x["cnpj8"].to_numpy()) for _, x in cov_rows.groupby("day")]
FV0 = XS[fundcols].to_numpy(dtype=float).copy()
results["placebo"] = {}
for key in targets:
    bv = cands[key]
    H, var, base = bv["H"], bv["variant"], bv["base"]
    base_s = COH[(H, base)]
    null = []
    for it in range(NPERM):
        XP = XS.copy()
        vals = FV0.copy()
        pos = {k: i for i, k in enumerate(XP.index)}
        for idx, iss in groups:
            u = np.unique(iss)
            perm = dict(zip(u, rng.permutation(u)))
            first = {}
            for ii, c in zip(idx, iss):
                first.setdefault(c, pos[ii])
            src = np.array([first[perm[c]] for c in iss])
            vals[[pos[ii] for ii in idx]] = FV0[src]
        XP[fundcols] = vals
        s, _ = portfolio(XP, sel_fn[var](), H, "perm")
        null.append(float((s - base_s.reindex(s.index)).mean() / (H / 252) * 100))
    results["placebo"][key] = {"actual_diff_ann_%": bv["diff_ann_%_gross"], "null_mean": float(np.mean(null)),
                               "null_p95": float(np.percentile(null, 95)),
                               "actual_minus_null_mean": bv["diff_ann_%_gross"] - float(np.mean(null)),
                               "p_value_one_sided": float((np.array(null) >= bv["diff_ann_%_gross"]).mean()),
                               "n_perm": NPERM}
    log("placebo", key, json.dumps(results["placebo"][key], default=float))

# ---- plots ----
fig, axs = plt.subplots(1, 2, figsize=(14, 5.2), dpi=110)
for ax, kind in zip(axs, ("raw_covered", "partial_vs_carry_rv_peer")):
    feats = (SIG + FUNDF) if kind == "raw_covered" else FUNDF
    w = 0.27
    xs = np.arange(len(feats))
    for j, (H, col) in enumerate(zip(HZ, ("#9ecae1", "#4292c6", "#08519c"))):
        vals = [R_s["ic"][H][kind][f]["mean"] for f in feats]
        ts = [R_s["ic"][H][kind][f]["t_nw"] for f in feats]
        bars = ax.bar(xs + (j - 1) * w, vals, w, color=col, label=f"{H} d.u. (~{H // 21}m)")
        for b_, t in zip(bars, ts):
            if t == t and abs(t) >= 2:
                ax.text(b_.get_x() + b_.get_width() / 2, b_.get_height(), "*", ha="center",
                        va="bottom" if b_.get_height() >= 0 else "top", fontsize=11)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(xs)
    ax.set_xticklabels(feats, rotation=45, ha="right", fontsize=8)
    ax.set_title("IC bruto (Spearman), emissores cobertos" if kind == "raw_covered"
                 else "IC parcial: controlando carry, RV, duration e peer", fontsize=10)
    ax.grid(alpha=0.25, axis="y")
axs[0].legend(fontsize=8, frameon=False)
fig.suptitle("Fundamentos vs retorno futuro (hedged) — * = |t Newey-West| ≥ 2", fontsize=11)
fig.tight_layout()
fig.savefig(OUT / "sellab_ic.png")

# equity curves at 6m: daily tranche book (average of the last 6 monthly cohorts), book-level cost 25 bps
H = 126
G_ = G.copy()
Rmat = np.zeros((ND, len(codes)))
Rmat[G_["pos"].to_numpy(), G_["b"].to_numpy()] = G_["r_patch"].to_numpy()
code_idx = {c: i for i, c in enumerate(codes)}
ent = XS.set_index(["day", "codigo"])["entry_pos"]
ent = ent[~ent.index.duplicated()]


def daily_book(W: dict) -> pd.Series:
    M = H // 21
    port = np.zeros(ND)
    for d, w in W.items():
        for c, wi in w.items():
            k = ent.get((d, c), -1)
            if k < 0:
                continue
            e = min(k + H, ND)
            port[k:e] += wi * Rmat[k:e, code_idx[c]] / M
    ds = sorted(W)
    prev = None
    for i, d in enumerate(ds):
        live = [W[x] for x in ds[max(0, i - M + 1): i + 1]]
        book = pd.concat(live, axis=1).fillna(0).sum(axis=1) / M
        if prev is not None:
            port[days.get_loc(d)] -= book.sub(prev, fill_value=0).abs().sum() * 25 / 2 / 1e4
        prev = book
    return pd.Series(port, index=days)


main_vars = ["P4", "P4 + excl. pior quintil qualidade composta", "Score ER: carry - EL + convergência",
             "Score ER sem EL (ablação)"]
if not QUICK:
    main_vars += ["Ridge carry+RV (sem fund.)", "Ridge carry+RV+fund.", "GBM carry+RV (sem fund.)", "GBM carry+RV+fund."]
ub = daily_book(WTS[(H, "U  universo")])
start = pd.Timestamp("2022-01-01")
end = XS.loc[XS["done126"], "day"].max() + pd.Timedelta(days=190)
fig, ax = plt.subplots(figsize=(12, 5.5), dpi=110)
curves = {}
for nm in main_vars:
    s = daily_book(WTS[(H, nm)])
    ex = (s - ub)[(days >= start) & (days <= end)]
    eq = (1 + ex).cumprod() - 1
    curves[nm] = float(eq.iloc[-1] * 100)
    ls = "--" if ("sem" in nm or nm == "P4") else "-"
    ax.plot(eq.index, eq * 100, ls, lw=1.5, label=nm)
ax.axhline(0, color="k", lw=0.6)
ax.set_ylabel("% acumulado acima do universo")
ax.set_title("Excesso acumulado vs universo — horizonte 6m (tranches mensais sobrepostas), custo 25 bps\n"
             "tracejado = sem fundamentos; contínuo = com fundamentos", fontsize=10)
ax.legend(fontsize=8, frameon=False)
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(OUT / "sellab_equity.png")
results["equity_6m_final_cum_excess_%"] = curves


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if o != o else round(float(o), 5)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (pd.Timestamp,)):
        return str(o.date())
    return o


json.dump(clean(results), open(OUT / "sellab_results.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
log("done")
