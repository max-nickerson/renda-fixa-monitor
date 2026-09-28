"""Deep lab: does DEEP LEARNING beat the simple P4 rule (and P4 + quality screen) at picking debentures?

Same data / targets / execution / costs / evaluation as research/run_selection_lab.py (its cached panels
data/history/sellab_panel.pkl + sellab_returns.pkl are reused; helper functions are copied verbatim because that
script runs everything at import time). Models (walk-forward, expanding window, trained only on labels complete at
the decision date = lab_end{H} <= decision position):
  MLP       tabular features, cross-sectionally rank-standardised per date, target = per-date rank, 5-seed ensemble
  ListNet   same MLP, listwise softmax cross-entropy within date
  GRU       26 weekly steps (spread change, resid, resid_z, weekly return, trades, present) + static tabular features
  placebo   MLP trained on targets shuffled across bonds within date
Baselines: P4, P4 + excl. worst quintile composite quality, GBM carry+RV+fund (selection lab spec), and a GBM on the
same tabular features as the MLP (control).

Outputs research/out/deep_results.json, deep_equity.png; README written by hand. Log data/deep.log.
Run: PYTHONPATH=. PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe research/run_deep_lab.py [--quick]
"""
import sys as _sys
from pathlib import Path as _P
_sys.path.insert(0, str(_P(__file__).resolve().parent.parent))

import json
import time
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sst
from sklearn.ensemble import HistGradientBoostingRegressor

from rfmonitor.config import DATA_DIR

warnings.filterwarnings("ignore")
try:
    import torch
    import torch.nn as nn
    HAVE_TORCH = True
    torch.set_num_threads(8)
except Exception:  # pragma: no cover
    HAVE_TORCH = False

OUT = _P(__file__).parent / "out"
HIST = DATA_DIR / "history"
LOGF = DATA_DIR / "deep.log"
QUICK = "--quick" in _sys.argv
t0 = time.time()
_logf = open(LOGF, "a", encoding="utf-8")


def log(*a):
    msg = f"[{time.strftime('%Y-%m-%d %H:%M:%S')} +{time.time() - t0:6.0f}s] " + " ".join(str(x) for x in a)
    print(msg, flush=True)
    _logf.write(msg + "\n")
    _logf.flush()


HZ = (63, 126, 252)
ENTRY_MAX = 20
ISSUER_CAP = 0.10
FUND_STALE_DAYS = 460
SPLIT = pd.Timestamp("2024-01-01")
RETRAIN = 3           # retrain every 3 monthly decision dates (MLP / ListNet / placebo / GBM)
RETRAIN_SEQ = 6       # GRU: every 6 months (cost)
MIN_TRAIN_DATES = 6
NSEED = 2 if QUICK else 5
NSEED_SEQ = 2 if QUICK else 3
NSEED_PLAC = 1 if QUICK else 3
SEQ_T = 26

# ---------------------------------------------------------------------------------------------------------------
# copied helpers from run_selection_lab.py (unchanged logic)
# ---------------------------------------------------------------------------------------------------------------


def fundamentals(strict: bool) -> pd.DataFrame:
    f = pd.read_pickle(HIST / "fundamentals_pit.pkl").copy()
    col = "available_date_strict" if strict else "available_date"
    f["avail"] = f[col]
    f = f.sort_values(["cnpj8", "avail", "period_end"])
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
        "f_lev": np.clip(lev, -3, 15), "f_cov": np.clip(cov, -5, 30),
        "f_cash_st": np.log1p(f["cash_to_st_debt"].clip(0, 50)), "f_eq_ratio": f["equity_ratio"].clip(-1, 1),
        "f_gde": np.where(f["equity"] <= 0, 10.0, f["gross_debt_equity"].clip(0, 10)),
        "f_margin": f["ebitda_margin"].clip(-1, 1), "f_rev_g": f["revenue_growth_yoy"].clip(-0.9, 3),
        "f_d_lev": f["d_net_debt_ebitda_4q"].clip(-10, 10), "f_d_cov": f["d_interest_coverage_4q"].clip(-10, 10),
        "f_size": np.log(f["total_assets"].where(f["total_assets"] > 0)), "f_is_parent": f["is_parent"].astype(float),
    })
    return out.sort_values("avail")


FCOLS = ["f_lev", "f_cov", "f_cash_st", "f_eq_ratio", "f_gde", "f_margin", "f_rev_g", "f_d_lev", "f_d_cov", "f_size"]
QSIGN = {"f_lev": -1, "f_cov": 1, "f_cash_st": 1, "f_eq_ratio": 1, "f_d_lev": -1}


def attach_fund(D: pd.DataFrame, strict: bool) -> pd.DataFrame:
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
    no = s.iloc[::max(lag, 1)]
    t_no = float(no.mean() / (no.std(ddof=1) / np.sqrt(len(no)))) if len(no) > 2 and no.std() > 0 else np.nan
    return {"mean": float(s.mean()), "t_nw": nw_t(s, lag), "t_nonoverlap": t_no, "n_dates": int(len(s))}


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


def portfolio(D: pd.DataFrame, select, H: int, name: str, keep_weights=True):
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
    return pd.Series(res, name=name, dtype=float).sort_index(), W


def book_cost(W: dict, H: int, cost_bps: float) -> float:
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


nz = lambda s, v: s.fillna(v)


def p4_mask(x):
    return (nz(x["cdi_pct"], 1) <= 0.3) & ~(nz(x["resid_z"], 0) <= -1.5) & (nz(x["press_neg_30d"], 0) < 1)


def screen(feat, worst_is_high: bool, keep_uncovered=True, q=0.2):
    def f(x):
        m = p4_mask(x)
        v = x[feat]
        thr = v.quantile(1 - q) if worst_is_high else v.quantile(q)
        bad = ((v >= thr) if worst_is_high else (v <= thr)) & v.notna()
        unc = ~x["covered"] if keep_uncovered else pd.Series(False, index=x.index)
        return (m & ~bad & (x["covered"] | unc)).to_numpy()
    return f


BASE = ["cdi_bps", "resid_bps", "resid_z", "dur", "press_neg_30d", "incent"]
FUNDF = FCOLS + ["f_quality"]


def design_lab(x: pd.DataFrame, feats):
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


# ---------------------------------------------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------------------------------------------
D, info = pd.read_pickle(HIST / "sellab_panel.pkl")
G, codes = pd.read_pickle(HIST / "sellab_returns.pkl")
days = info["days"]
ND, NB = len(days), len(codes)
for _H in HZ:
    D[f"dok{_H}"] = D["dpos"] + ENTRY_MAX + _H <= ND
log(f"panel {D.shape}, {D['day'].nunique()} decision dates, torch={HAVE_TORCH}")

LD = pd.read_pickle(HIST / "lab_daily.pkl")[["codigo", "day", "date", "cdi_bps", "resid_bps", "resid_z", "distress_2d",
                                             "fact_2d", "press_neg_7d", "eq_ret_1w", "mkt_mom_21"]]
extra = LD[["codigo", "day", "distress_2d", "fact_2d", "press_neg_7d", "eq_ret_1w", "mkt_mom_21"]]
D = D.merge(extra, on=["codigo", "day"], how="left")
D = attach_fund(D, strict=True)
D = D.set_index("_i", drop=False)
D.index.name = None
log(f"features merged; covered share {D['covered'].mean():.3f}")

# ---- tabular design (per-date cross-sectional rank-gauss standardisation + missing indicators) ----
CONT = ["cdi_bps", "resid_bps", "resid_z", "dur", "age", "contract", "press_neg_30d", "press_neg_7d", "distress_2d",
        "fact_2d", "eq_ret_1w", "eq_ret_4w", "rat_log"] + FCOLS + ["f_quality"]
D["rat_log"] = np.log1p(D["rat_days_since_down"].clip(upper=3650))


def design_deep(x: pd.DataFrame) -> pd.DataFrame:
    cols = {}
    for f in CONT:
        r = x.groupby("day")[f].rank(pct=True)
        cols[f] = ((r - 0.5) * np.sqrt(12)).fillna(0.0)
        if x[f].isna().any():
            cols[f + "_na"] = x[f].isna().astype(float)
    cols["incent"] = x["incent"].astype(float)
    cols["press_any"] = (x["press_neg_30d"].fillna(0) >= 1).astype(float)
    cols["distress_any"] = (x["distress_2d"].fillna(0) > 0).astype(float)
    cols["rat_down_1y"] = (x["rat_days_since_down"] < 365).astype(float)
    cols["covered"] = x["covered"].astype(float)
    cols["f_is_parent"] = x["f_is_parent"].fillna(0).astype(float)
    cols["mkt_mom_21"] = (x["mkt_mom_21"].fillna(0) * 300).clip(-3, 3)
    for k in ("IPCA", "PRE"):
        cols["kind_" + k] = (x["kind"] == k).astype(float)
    for p in sorted(x["peer"].dropna().unique())[1:]:
        cols["peer_" + p] = (x["peer"] == p).astype(float)
    return pd.DataFrame(cols, index=x.index)


XT = design_deep(D)
FEATS = list(XT.columns)
Xnp = XT.to_numpy(np.float32)
log(f"tabular design: {len(FEATS)} features")

# ---- sequence tensor: last 26 weekly obs per decision row (strictly <= decision day) ----
LD["b"] = codes.get_indexer(LD["codigo"])
LD["pos"] = days.get_indexer(LD["day"])
LD = LD.sort_values(["b", "pos"])
LD["trade"] = (LD["date"] != LD.groupby("b")["date"].shift(1)).astype(float)
PRES = np.zeros((ND, NB), np.float32)
CDI = np.zeros((ND, NB), np.float32)
RB = np.zeros((ND, NB), np.float32)
RZ = np.zeros((ND, NB), np.float32)
TRC = np.zeros((ND, NB), np.float32)
pi, bi = LD["pos"].to_numpy(), LD["b"].to_numpy()
PRES[pi, bi] = 1
CDI[pi, bi] = LD["cdi_bps"].fillna(0).to_numpy()
RB[pi, bi] = LD["resid_bps"].fillna(0).clip(-500, 500).to_numpy()
RZ[pi, bi] = LD["resid_z"].fillna(0).clip(-4, 4).to_numpy()
TRC[pi, bi] = LD["trade"].to_numpy()
TRC = np.cumsum(TRC, axis=0)
# returns assigned to the END of each segment (so nothing after the decision leaks through a gap)
Gs = G.sort_values(["b", "pos"])
nxt = Gs.groupby("b")["pos"].shift(-1)
okr = nxt.notna().to_numpy()
LRE = np.zeros((ND, NB), np.float64)
np.add.at(LRE, (nxt[okr].astype(int).to_numpy(), Gs["b"].to_numpy()[okr]),
          np.log1p(Gs["r_patch"].to_numpy()[okr].clip(-0.95, 1.0)))
LRE = np.cumsum(LRE, axis=0)


def build_seq(D):
    p = D["dpos"].to_numpy()[:, None]
    b = D["b"].to_numpy()[:, None]
    qs = p - 5 * np.arange(SEQ_T - 1, -1, -1)[None, :]
    Q = np.clip(qs, 0, ND - 1)
    Q5 = np.clip(qs - 5, 0, ND - 1)
    inb = (qs >= 0).astype(np.float32)
    pres = PRES[Q, b] * inb
    cnow = CDI[p[:, 0], b[:, 0]][:, None]
    dcdi = np.clip((CDI[Q, b] - cnow) / 100, -5, 5) * pres
    rb = RB[Q, b] / 100 * pres
    rz = RZ[Q, b] * pres
    wr = np.clip((LRE[Q, b] - LRE[Q5, b]) * 100, -10, 10) * inb * (qs >= 5)
    tr = np.clip((TRC[Q, b] - TRC[Q5, b]) / 5, 0, 1) * inb * (qs >= 5)
    return np.stack([pres, dcdi, rb, rz, wr.astype(np.float32), tr], axis=-1).astype(np.float32)


Snp = build_seq(D)
del PRES, CDI, RB, RZ, TRC, LRE
log(f"sequence tensor {Snp.shape}")

# ---------------------------------------------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------------------------------------------
if HAVE_TORCH:
    class MLP(nn.Module):
        def __init__(self, f, h=(64, 32), p=0.25):
            super().__init__()
            L, i = [], f
            for k in h:
                L += [nn.Linear(i, k), nn.ReLU(), nn.Dropout(p)]
                i = k
            L += [nn.Linear(i, 1)]
            self.net = nn.Sequential(*L)

        def forward(self, x, s=None):
            return self.net(x).squeeze(-1)

    class SeqNet(nn.Module):
        def __init__(self, f, c, hs=16, p=0.25):
            super().__init__()
            self.gru = nn.GRU(c, hs, batch_first=True)
            self.st = nn.Sequential(nn.Linear(f, 32), nn.ReLU(), nn.Dropout(p))
            self.head = nn.Sequential(nn.Linear(hs + 32, 16), nn.ReLU(), nn.Dropout(p), nn.Linear(16, 1))

        def forward(self, x, s):
            _, h = self.gru(s)
            return self.head(torch.cat([h[-1], self.st(x)], dim=1)).squeeze(-1)


def per_date_corr(pred, y, gid):
    df = pd.DataFrame({"p": pred, "y": y, "g": gid})
    c = df.groupby("g").apply(lambda z: np.corrcoef(z["p"], z["y"])[0, 1] if len(z) > 5 and z["p"].std() > 0 else np.nan)
    return float(np.nanmean(c))


def train_one(kind, loss, Xtr, Str, ytr, gtr, Xva, Sva, yva, gva, seed, max_ep, patience=5):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = SeqNet(Xtr.shape[1], Str.shape[2]) if kind == "gru" else MLP(Xtr.shape[1])
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-3)
    Xt, yt = torch.from_numpy(Xtr), torch.from_numpy(ytr)
    St = torch.from_numpy(Str) if kind == "gru" else None
    Xv = torch.from_numpy(Xva)
    Sv = torch.from_numpy(Sva) if kind == "gru" else None
    n = len(Xtr)
    cap = 20000 if kind == "gru" else n  # rows per epoch (subsample for GRU cost)
    groups = [np.flatnonzero(gtr == g) for g in np.unique(gtr)]
    best, best_state, best_ep, bad = -9, None, 0, 0
    for ep in range(max_ep):
        net.train()
        if loss == "listnet":
            order = rng.permutation(len(groups))
            for j in range(0, len(order), 4):
                opt.zero_grad()
                tot = 0
                for gi in order[j:j + 4]:
                    idx = groups[gi]
                    if len(idx) > 400:
                        idx = rng.choice(idx, 400, replace=False)
                    out = net(Xt[idx], None)
                    tp = torch.softmax(yt[idx] * 8.0, 0)
                    tot = tot - (tp * torch.log_softmax(out, 0)).sum()
                tot.backward()
                opt.step()
        else:
            perm = rng.permutation(n)[:cap]
            for j in range(0, len(perm), 512):
                idx = perm[j:j + 512]
                opt.zero_grad()
                out = net(Xt[idx], St[idx] if St is not None else None)
                l = ((out - yt[idx]) ** 2).mean()
                l.backward()
                opt.step()
        net.eval()
        with torch.no_grad():
            pv = net(Xv, Sv).numpy()
        sc = per_date_corr(pv, yva, gva)
        if sc > best + 1e-4:
            best, best_ep, bad = sc, ep, 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    return net, best, best_ep


def walk_forward_deep(H, kind="mlp", loss="mse", nseed=5, retrain=RETRAIN, placebo=False, max_ep=30):
    """Returns ensemble prediction (per-date rank-average of seeds) and per-seed predictions + training diagnostics."""
    y = f"y{H}"
    tgt = (D.groupby("day")[y].rank(pct=True) - 0.5).to_numpy(np.float32)
    dayv = D["day"].to_numpy()
    dates = sorted(D["day"].unique())
    dpos = D.groupby("day")["dpos"].first()
    labend = D[f"lab_end{H}"].to_numpy()
    ynn = D[y].notna().to_numpy()
    preds = np.full((nseed, len(D)), np.nan, np.float32)
    nets, last_fit, diag = None, -99, []
    rng = np.random.default_rng(123)
    for i, d in enumerate(dates):
        p = dpos[d]
        tr = (labend <= p) & ynn
        trd = np.unique(dayv[tr])
        if len(trd) < MIN_TRAIN_DATES:
            continue
        if nets is None or i - last_fit >= retrain:
            nva = max(2, int(round(0.2 * len(trd))))
            va_d, tr_d = trd[-nva:], trd[:-(nva + 1)]  # 1-date purge between train and validation slices
            mtr, mva = tr & np.isin(dayv, tr_d), tr & np.isin(dayv, va_d)
            ytr = tgt[mtr].copy()
            gtr = pd.factorize(dayv[mtr])[0]
            if placebo:
                for g in np.unique(gtr):
                    ix = np.flatnonzero(gtr == g)
                    ytr[ix] = rng.permutation(ytr[ix])
            yva = tgt[mva].copy()
            gva = pd.factorize(dayv[mva])[0]
            if placebo:
                for g in np.unique(gva):
                    ix = np.flatnonzero(gva == g)
                    yva[ix] = rng.permutation(yva[ix])
            nets = []
            for s in range(nseed):
                net, sc, ep = train_one(kind, loss, Xnp[mtr], Snp[mtr] if kind == "gru" else None, ytr, gtr,
                                        Xnp[mva], Snp[mva] if kind == "gru" else None, yva, gva, seed=1000 * s + i,
                                        max_ep=max_ep)
                nets.append(net)
                diag.append({"date": str(pd.Timestamp(d).date()), "seed": s, "val_ic": sc, "best_epoch": ep,
                             "n_train": int(mtr.sum()), "n_train_dates": int(len(tr_d))})
            last_fit = i
        te = np.flatnonzero(dayv == d)
        with torch.no_grad():
            for s, net in enumerate(nets):
                preds[s, te] = net(torch.from_numpy(Xnp[te]), torch.from_numpy(Snp[te]) if kind == "gru" else None).numpy()
    P = pd.DataFrame(preds.T, index=D.index)
    R = P.groupby(D["day"]).rank(pct=True)
    ens = R.mean(axis=1).where(P.notna().all(axis=1))
    return ens, P, diag


def walk_forward_gbm(H, Xall, retrain=RETRAIN):
    y = f"y{H}"
    tgt = D.groupby("day")[y].rank(pct=True) - 0.5
    pred = pd.Series(np.nan, index=D.index)
    dates = sorted(D["day"].unique())
    dpos = D.groupby("day")["dpos"].first()
    m, last_fit = None, -99
    for i, d in enumerate(dates):
        p = dpos[d]
        tr = (D[f"lab_end{H}"] <= p) & D[y].notna()
        if D.loc[tr, "day"].nunique() < MIN_TRAIN_DATES:
            continue
        if m is None or i - last_fit >= retrain:
            m = HistGradientBoostingRegressor(max_iter=150, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=200,
                                              l2_regularization=1.0, random_state=0).fit(Xall[tr], tgt[tr])
            last_fit = i
        te = D["day"] == d
        pred[te] = m.predict(Xall[te])
    return pred


# ---------------------------------------------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------------------------------------------
X_lab = design_lab(D, BASE + FUNDF)
MODELS = {}
TRAIN_DIAG = {}
SEED_PREDS = {}
for H in HZ:
    D[f"gbm_lab_{H}"] = walk_forward_gbm(H, X_lab)
    D[f"gbm_same_{H}"] = walk_forward_gbm(H, XT)
    log(f"H={H} GBM baselines done")
    specs = [("mlp", "mlp", "mse", NSEED, RETRAIN, False, 30),
             ("listnet", "mlp", "listnet", NSEED, RETRAIN, False, 30),
             ("placebo", "mlp", "mse", NSEED_PLAC, RETRAIN, True, 30),
             ("gru", "gru", "mse", NSEED_SEQ, RETRAIN_SEQ, False, 15)]
    if not HAVE_TORCH:
        specs = []
    for nm, kind, loss, ns, rt, plc, me in specs:
        ts = time.time()
        ens, P, dg = walk_forward_deep(H, kind, loss, ns, rt, plc, me)
        D[f"{nm}_{H}"] = ens
        SEED_PREDS[(nm, H)] = P
        TRAIN_DIAG[(nm, H)] = dg
        log(f"H={H} {nm}: {len(dg)} fits in {time.time() - ts:.0f}s, mean val IC {np.mean([g['val_ic'] for g in dg]):+.3f},"
            f" mean best epoch {np.mean([g['best_epoch'] for g in dg]):.1f}")

MODEL_NAMES = {"mlp": "MLP (5 seeds)", "listnet": "MLP ListNet", "gru": "GRU seq+static", "placebo": "Placebo MLP",
               "gbm_lab": "GBM carry+RV+fund (sellab)", "gbm_same": "GBM same feats as MLP"}
DL = ["mlp", "listnet", "gru"]


def ic_series(col, H):
    y = f"y{H}"
    out = {}
    for d, x in D[D[y].notna() & D[f"dok{H}"] & D[col].notna()].groupby("day"):
        if len(x) >= 30:
            out[d] = sst.spearmanr(x[col], x[y])[0]
    return pd.Series(out, dtype=float).sort_index()


def restrict(W, dates):
    return {d: w for d, w in W.items() if d in dates}


def stats_on(s, uni, W, Wu, H, dates):
    s = s[s.index.isin(dates)]
    ex = (s - uni.reindex(s.index)).dropna()
    yrs = H / 252
    Wr, Wur = restrict(W, set(ex.index)), restrict(Wu, set(ex.index))
    c25, c50 = book_cost(Wr, H, 25), book_cost(Wr, H, 50)
    u25, u50 = book_cost(Wur, H, 25), book_cost(Wur, H, 50)
    sm = summarize(ex, H)
    return {"n_cohorts": int(len(ex)), "first": str(ex.index.min().date()) if len(ex) else None,
            "avg_bonds": float(np.mean([len(W[d]) for d in ex.index])) if len(ex) else np.nan,
            "excess_vs_U_ann_%_gross": float(ex.mean() / yrs * 100), "t_nw": sm["t_nw"], "t_nonoverlap": sm["t_nonoverlap"],
            "book_cost_ann_%_25": c25 * 100,
            "excess_vs_U_ann_%_net25": float((ex.mean() / yrs - (c25 - u25)) * 100),
            "excess_vs_U_ann_%_net50": float((ex.mean() / yrs - (c50 - u50)) * 100),
            "excess_2022_23_ann_%": float(ex[ex.index < SPLIT].mean() / yrs * 100) if (ex.index < SPLIT).any() else None,
            "excess_2024_26_ann_%": float(ex[ex.index >= SPLIT].mean() / yrs * 100),
            "hit_rate_vs_U": float((ex > 0).mean())}


results = {"design": {
    "data": "same cached panel as run_selection_lab (monthly decisions 2022-01.., patched hedged returns, entry = first "
            "trade within 20 bdays, strict fundamentals availability)",
    "training": f"expanding window, only labels with lab_end<=decision pos; retrain every {RETRAIN} months "
                f"(GRU {RETRAIN_SEQ}); min {MIN_TRAIN_DATES} label dates; last 20% of train dates = early-stopping "
                "validation (1-date purge); target = per-date rank; AdamW wd 1e-3, dropout 0.25",
    "features": FEATS, "seq_channels": ["present", "d_cdi_bps_vs_now/100", "resid_bps/100", "resid_z",
                                        "weekly_ret_%", "weekly_trades/5"], "seq_len_weeks": SEQ_T,
    "seeds": {"mlp": NSEED, "listnet": NSEED, "gru": NSEED_SEQ, "placebo": NSEED_PLAC}, "torch": HAVE_TORCH}}
ICR, PORT, PAIR, STAB = {}, {}, {}, {}
COH, WTS = {}, {}
for H in HZ:
    y = f"y{H}"
    uni, Wu = portfolio(D, lambda x: np.ones(len(x), bool), H, "U")
    variants = {"P4": lambda x: p4_mask(x).to_numpy(), "P4+quality": screen("f_quality", False)}
    for m in ["gbm_lab", "gbm_same"] + [k for k in DL + ["placebo"] if f"{k}_{H}" in D]:
        col = f"{m}_{H}"
        variants[f"{m} top20"] = (lambda c: (lambda x: x[c] if x[c].notna().any() else None))(col)
        variants[f"{m} P4∩top50"] = (lambda c: (lambda x: (p4_mask(x) & (x[c] >= x[c].median())).to_numpy()
                                                  if x[c].notna().any() else None))(col)
    for nm, f in variants.items():
        COH[(H, nm)], WTS[(H, nm)] = portfolio(D, f, H, nm)
    ref = "mlp top20" if (H, "mlp top20") in COH else "gbm_lab top20"
    common = set(COH[(H, ref)].index)
    PORT[H] = {"_common_dates": {"n": len(common), "first": str(min(common).date()), "last": str(max(common).date())}}
    for nm in variants:
        PORT[H][nm] = {"full_sample": stats_on(COH[(H, nm)], uni, WTS[(H, nm)], Wu, H, set(COH[(H, nm)].index)),
                       "common_dates": stats_on(COH[(H, nm)], uni, WTS[(H, nm)], Wu, H, common)}
        q = PORT[H][nm]["common_dates"]
        log(f"H={H} {nm:<22} n={q['n_cohorts']:3d} exU {q['excess_vs_U_ann_%_gross']:+6.2f} (t {q['t_nw']:+5.2f}) "
            f"net25 {q['excess_vs_U_ann_%_net25']:+6.2f} halves {q['excess_2022_23_ann_%']} / {q['excess_2024_26_ann_%']:+.2f}")
    # IC
    ICR[H] = {}
    for m in ["gbm_lab", "gbm_same"] + [k for k in DL + ["placebo"] if f"{k}_{H}" in D]:
        s = ic_series(f"{m}_{H}", H)
        ICR[H][m] = {**summarize(s, H), "ic_2022_23": float(s[s.index < SPLIT].mean()) if (s.index < SPLIT).any() else None,
                     "ic_2024_26": float(s[s.index >= SPLIT].mean())}
    D["_neg_cdipct"] = -D["cdi_pct"]
    s = ic_series("cdi_bps", H)
    ICR[H]["cdi_bps (reference, same dates)"] = summarize(s[s.index.isin(common)], H)
    log(f"H={H} IC " + ", ".join(f"{k} {v['mean']:+.3f}({v['t_nw']:+.1f})" for k, v in ICR[H].items()))
    # paired tests vs P4 and vs P4+quality
    for m in [k for k in DL + ["placebo", "gbm_lab", "gbm_same"] if (H, f"{k} top20") in COH]:
        for form in ("top20", "P4∩top50"):
            a = f"{m} {form}"
            for b in ("P4", "P4+quality"):
                sa, sb = COH[(H, a)], COH[(H, b)]
                dd = (sa - sb.reindex(sa.index)).dropna()
                if len(dd) < 4:
                    continue
                t = nw_t(dd, H // 21)
                dts = set(dd.index)
                cd25 = book_cost(restrict(WTS[(H, a)], dts), H, 25) - book_cost(restrict(WTS[(H, b)], dts), H, 25)
                cd50 = book_cost(restrict(WTS[(H, a)], dts), H, 50) - book_cost(restrict(WTS[(H, b)], dts), H, 50)
                PAIR[f"{a} vs {b} | H={H}"] = {
                    "model": m, "form": form, "base": b, "H": H, "n": int(len(dd)),
                    "diff_ann_%_gross": float(dd.mean() / (H / 252) * 100), "t_nw": t,
                    "p_two_sided": float(2 * (1 - sst.norm.cdf(abs(t)))) if t == t else np.nan,
                    "diff_ann_%_net25": float((dd.mean() / (H / 252) - cd25) * 100),
                    "diff_ann_%_net50": float((dd.mean() / (H / 252) - cd50) * 100),
                    "diff_2022_23": float(dd[dd.index < SPLIT].mean() / (H / 252) * 100) if (dd.index < SPLIT).any() else None,
                    "diff_2024_26": float(dd[dd.index >= SPLIT].mean() / (H / 252) * 100)}
    # seed stability
    for m in [k for k in DL + ["placebo"] if (k, H) in SEED_PREDS]:
        P = SEED_PREDS[(m, H)]
        ics, exs = [], []
        for s in P.columns:
            D["_tmp"] = P[s]
            ss = ic_series("_tmp", H)
            ics.append(float(ss.mean()))
            cs, _ = portfolio(D, lambda x: x["_tmp"] if x["_tmp"].notna().any() else None, H, "seed", keep_weights=False)
            ex = (cs - uni.reindex(cs.index)).dropna()
            exs.append(float(ex.mean() / (H / 252) * 100))
        rc = []
        for d, idx in D.groupby("day").groups.items():
            M = P.loc[idx].dropna()
            if len(M) > 30 and P.shape[1] > 1:
                c = M.rank().corr().to_numpy()
                rc.append(c[np.triu_indices_from(c, 1)].mean())
        dg = TRAIN_DIAG[(m, H)]
        STAB[f"{m} | H={H}"] = {"per_seed_ic": ics, "ic_seed_std": float(np.std(ics)), "per_seed_top20_exU_ann_%": exs,
                                "top20_exU_seed_std": float(np.std(exs)),
                                "mean_pairwise_rank_corr_between_seeds": float(np.mean(rc)) if rc else None,
                                "mean_val_ic": float(np.mean([g["val_ic"] for g in dg])),
                                "mean_best_epoch": float(np.mean([g["best_epoch"] for g in dg])),
                                "share_best_epoch_0": float(np.mean([g["best_epoch"] == 0 for g in dg])),
                                "n_fits": len(dg)}
        log(f"STAB {m} H={H}", json.dumps(STAB[f"{m} | H={H}"], default=float))

# Holm across DL variants (MLP/ListNet/GRU x top20/P4∩top50 x vs P4/P4+quality x horizons)
keys = [k for k, v in PAIR.items() if v["model"] in DL and v["p_two_sided"] == v["p_two_sided"]]
ps = np.array([PAIR[k]["p_two_sided"] for k in keys])
order = np.argsort(ps)
adj, run = np.empty(len(ps)), 0.0
for r_, i in enumerate(order):
    run = max(run, min(1.0, (len(ps) - r_) * ps[i]))
    adj[i] = run
for k, a in zip(keys, adj):
    PAIR[k]["p_holm_DL_family"] = float(a)
for k, v in PAIR.items():
    log(f"PAIR {k:<40} {v['diff_ann_%_gross']:+6.2f}%/a t {v['t_nw']:+5.2f} net25 {v['diff_ann_%_net25']:+6.2f} "
        f"holm {v.get('p_holm_DL_family', float('nan')):.3f}")
results.update({"ic": ICR, "portfolios": PORT, "paired": PAIR, "seed_stability": STAB,
                "holm_family_size": int(len(keys)),
                "training_log": {f"{m} | H={H}": v for (m, H), v in TRAIN_DIAG.items()}})

# ---------------------------------------------------------------------------------------------------------------
# equity curves (6m): daily tranche book, 25 bps, all variants on the MLP's cohort dates
# ---------------------------------------------------------------------------------------------------------------
H = 126
Rmat = np.zeros((ND, NB))
Rmat[G["pos"].to_numpy(), G["b"].to_numpy()] = G["r_patch"].to_numpy()
code_idx = {c: i for i, c in enumerate(codes)}
ent = D.set_index(["day", "codigo"])["entry_pos"]
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


ref = "mlp top20" if (H, "mlp top20") in COH else "gbm_lab top20"
cd = set(COH[(H, ref)].index)
ub = daily_book(restrict(WTS[(H, "U")] if (H, "U") in WTS else portfolio(D, lambda x: np.ones(len(x), bool), H, "U")[1], cd))
start, end = min(cd), max(cd) + pd.Timedelta(days=190)
fig, ax = plt.subplots(figsize=(12, 5.5), dpi=110)
curves = {}
plot_vars = [("P4", "P4", "--"), ("P4+quality", "P4 + excl. worst-quintile quality", "--"),
             ("gbm_lab top20", "GBM carry+RV+fund top20", ":"), ("mlp top20", "MLP top20", "-"),
             ("gru top20", "GRU seq top20", "-"), ("listnet top20", "ListNet MLP top20", "-"),
             ("mlp P4∩top50", "P4 ∩ MLP top50", "-.")]
for key, lab, ls in plot_vars:
    if (H, key) not in WTS:
        continue
    s = daily_book(restrict(WTS[(H, key)], cd))
    ex = (s - ub)[(days >= start) & (days <= end)]
    eq = (1 + ex).cumprod() - 1
    curves[lab] = float(eq.iloc[-1] * 100)
    ax.plot(eq.index, eq * 100, ls, lw=1.5, label=lab)
ax.axhline(0, color="k", lw=0.6)
ax.set_ylabel("cumulative % above universe")
ax.set_title(f"Cumulative excess vs universe, 6m horizon, overlapping monthly tranches, 25 bps book cost\n"
             f"all variants on the same {len(cd)} cohorts (first model date {min(cd).date()})", fontsize=10)
ax.legend(fontsize=8, frameon=False)
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(OUT / "deep_equity.png")
results["equity_6m_final_cum_excess_%"] = curves
log("equity", json.dumps(curves))


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if o != o else round(float(o), 5)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, pd.Timestamp):
        return str(o.date())
    return o


json.dump(clean(results), open(OUT / "deep_results.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
log("done")
