"""HY lab: can a high-yield (top-carry) debenture book beat P4, and do hedges of its tail risk pay for themselves?

Re-uses the selection-lab caches (READ ONLY): data/history/sellab_panel.pkl (monthly decision cross-sections, entry
positions) and sellab_returns.pkl (PATCHED daily rate-hedged excess returns incl. >20% moves and no-trade gaps).
Run research/run_selection_lab.py --rebuild first if those caches are missing.

Outputs research/out/hy_results.json, hy_equity.png (HY_README.md written by hand from the json). Log: data/hy.log.
Index cache: research/out/hy_indices.pkl (brapi ^BVSP / SMAL11, refreshed after 20h).

Run:  PYTHONPATH=. PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe research/run_hy_lab.py
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
from scipy.stats import norm, skew

from rfmonitor.config import DATA_DIR
from rfmonitor.history import bcb_series

warnings.filterwarnings("ignore")
OUT = _P(__file__).parent / "out"
HIST = DATA_DIR / "history"
LOGF = DATA_DIR / "hy.log"
t0 = time.time()
_logf = open(LOGF, "a", encoding="utf-8")


def log(*a):
    msg = f"[{time.strftime('%H:%M:%S')} +{time.time() - t0:6.0f}s] " + " ".join(str(x) for x in a)
    print(msg, flush=True)
    _logf.write(msg + "\n")
    _logf.flush()


HZ = (63, 126, 252)
ENTRY_MAX = 20
CAP = 0.10
SPLIT = pd.Timestamp("2024-01-01")
CRISIS = (pd.Timestamp("2022-12-01"), pd.Timestamp("2023-06-30"))  # Americanas (Jan-23), Light (May-23), etc.
FUND_STALE_DAYS = 460
REC = 0.40                    # recovery (fraction of par) for the stop-trading-below-0.90 sensitivity
BORROW = (0.02, 0.05)         # stock borrow cost per year (base / check)
STOCK_SIDE = 0.0010           # stock trading cost per side (10 bps)
FUT_SIDE = 0.0002             # index futures cost per side (2 bps)
ETF_BORROW = 0.02             # SMAL11 borrow per year
MIN_ADTV = 5e6                # R$/day: minimum stock liquidity to consider a short (borrow) implementable
OPT_ADTV = 200e6              # R$/day: proxy for names with a usable single-stock option market
TOP_RISK_4W = -0.15
EQ_ELAST = 1.0                # structural: d ln(spread) / d ln(equity) = -1  =>  h = dur * spread * 1
JTD_H = 0.5                   # jump-to-default sizing: bond -50% vs stock -80..-100% in default => ~0.5-0.6
PUT_OTM, PUT_T, PUT_SPREAD = 0.10, 63, 0.10   # 3m 10%-OTM put, 10% of premium paid as bid/ask on purchase

# ------------------------------------------------------------------------------------------------------------------
# data
# ------------------------------------------------------------------------------------------------------------------
D, info = pd.read_pickle(HIST / "sellab_panel.pkl")
G, codes = pd.read_pickle(HIST / "sellab_returns.pkl")
days = info["days"]
ND, NB = len(days), len(codes)
log(f"panel {D.shape}, returns {G.shape}, days {days[0].date()}..{days[-1].date()}")
for H in HZ:
    D[f"dok{H}"] = D["dpos"] + ENTRY_MAX + H <= ND

R0 = np.zeros((ND, NB))
R0[G["pos"].to_numpy(), G["b"].to_numpy()] = G["r_patch"].to_numpy()
PRES = np.zeros((ND, NB), bool)
PRES[G["pos"].to_numpy(), G["b"].to_numpy()] = True

# recovery sensitivity: bonds that stop trading (>30d before data end, before maturity) with last mark < 0.90
last = info["last"]
stopped = last[(last["last_day"] < days[-1] - pd.Timedelta(days=30))
               & (last["maturity"].isna() | (last["maturity"] > last["last_day"] + pd.Timedelta(days=60)))]
dist = stopped[stopped["last_ratio"] < 0.90]
lastpos = G.groupby("b")["pos"].max()
R40 = R0.copy()
DIST_B = set()
for c, r in dist.iterrows():
    b = codes.get_loc(c)
    p = int(lastpos[b])
    R40[p, b] = (1 + R40[p, b]) * (REC / r["last_ratio"]) - 1
    DIST_B.add(b)
log(f"distressed stop-trading bonds (last<0.90): {len(dist)}; jump to {REC:.0%} of par applied in the rec40 scenario")
RS = {"base": R0, "rec40": R40}
LC = {k: np.vstack([np.zeros((1, NB)), np.cumsum(np.log1p(np.clip(v, -0.99, None)), axis=0)]) for k, v in RS.items()}

# CDI on the grid (holidays: 0)
cdi = bcb_series(12, date(2020, 1, 1)) / 100
cidx = (1 + cdi).cumprod()
cg = cidx.reindex(cidx.index.union(days)).ffill().reindex(days)
CDI = (cg / cg.shift(1) - 1).fillna(0).to_numpy()
CDI_ANN = ((1 + cdi) ** 252 - 1).reindex(cidx.index.union(days)).ffill().reindex(days).to_numpy()

# indices (brapi), cached
ICACHE = OUT / "hy_indices.pkl"
if ICACHE.exists() and time.time() - ICACHE.stat().st_mtime < 20 * 3600:
    IDX = pd.read_pickle(ICACHE)
else:
    from rfmonitor.sources import brapi
    IDX = {}
    for t in ("^BVSP", "SMAL11"):
        h = brapi.history(t, "10y")
        s = pd.Series({pd.Timestamp(d): c for d, c, _ in h}).sort_index()
        IDX[t] = s[s.index < pd.Timestamp(date.today())]
    pd.to_pickle(IDX, ICACHE)


def on_grid(s: pd.Series, limit=5) -> np.ndarray:
    return s.reindex(s.index.union(days)).ffill(limit=limit).reindex(days).to_numpy()


IBOV = on_grid(IDX["^BVSP"])
SMAL = on_grid(IDX["SMAL11"])
IX = {"IBOV": np.nan_to_num(IBOV / np.roll(IBOV, 1) - 1) - CDI, "SMAL": np.nan_to_num(SMAL / np.roll(SMAL, 1) - 1) - CDI}
for k in IX:
    IX[k][0] = 0.0

# stocks
eq = pd.read_pickle(HIST / "equity_daily.pkl")
eq["date"] = pd.to_datetime(eq["date"])
eq = eq.sort_values(["ticker", "date"]).drop_duplicates(["ticker", "date"])
TICK = sorted(eq["ticker"].unique())
TI = {t: i for i, t in enumerate(TICK)}
NS = len(TICK)
SX = np.full((ND, NS), np.nan)     # daily stock excess return over CDI (NaN = no price)
SPX = np.full((ND, NS), np.nan)
ADTV = np.full((ND, NS), np.nan)
for t, g in eq.groupby("ticker"):
    s = g.set_index("date")["adj_close"].astype(float)
    s = s.where(s > 0)
    lr = np.log(s).diff()
    if g["source"].str.startswith("cotahist").all():   # unadjusted series: drop split jumps
        lr[lr.abs() > 0.5] = 0.0
    s = np.exp(lr.fillna(0).cumsum()) * s.dropna().iloc[0]
    px = on_grid(s)
    r = px / np.roll(px, 1) - 1
    r[0] = np.nan
    SX[:, TI[t]] = r - CDI
    SPX[:, TI[t]] = px
    tv = (g.set_index("date")["close"] * g.set_index("date")["volume"]).rolling(63, min_periods=20).median()
    ADTV[:, TI[t]] = on_grid(tv)
SX[:, :][np.isnan(SPX)] = np.nan
RET21 = SPX / np.vstack([np.full((21, NS), np.nan), SPX[:-21]]) - 1   # 21-grid-day stock return at close of t

mp = pd.read_csv(_P(__file__).parent / "data" / "equity_map.csv", dtype=str)
mp = mp[mp["ticker"].notna() & (mp["mapping_type"] != "none") & mp["ticker"].isin(TI)].copy()
mp["_c"] = mp["confidence"].map({"high": 0, "med": 1, "low": 2}).fillna(1)
CANDS = {k: list(zip(v["ticker"], v["mapping_type"])) for k, v in mp.sort_values(["cnpj8", "_c"]).groupby("cnpj8")}


def pick_ticker(cnpj8, p):
    """First mapped ticker (by confidence) with a price at the decision close and >=150 prices in the last year."""
    for t, mt in CANDS.get(cnpj8, []):
        j = TI[t]
        if not np.isnan(SPX[p, j]) and np.isfinite(SPX[max(0, p - 252):p, j]).sum() >= 150:
            return j, mt
    return -1, "none"


# fundamentals (composite quality exactly as in the selection lab, strict availability)
def fundamentals():
    f = pd.read_pickle(HIST / "fundamentals_pit.pkl").copy()
    f["avail"] = f["available_date_strict"]
    f = f.sort_values(["cnpj8", "avail", "period_end"])
    f["_mx"] = f.groupby("cnpj8")["period_end"].cummax().groupby(f["cnpj8"]).shift(1)
    f = f[f["_mx"].isna() | (f["period_end"] > f["_mx"])].drop_duplicates(["cnpj8", "avail"], keep="last")
    e = f["ebitda_ltm"]
    lev = np.where(e > 0, f["net_debt"] / e, 15.0)
    lev = np.where(f["net_debt"].isna() | e.isna(), np.nan, lev)
    fe = f["fin_exp_ltm"]
    cov = np.where(fe > 0, e / fe, np.where(fe.notna() & e.notna(), 30.0, np.nan))
    return pd.DataFrame({"cnpj8": f["cnpj8"], "avail": f["avail"], "period_end": f["period_end"],
                         "f_lev": np.clip(lev, -3, 15), "f_cov": np.clip(cov, -5, 30),
                         "f_cash_st": np.log1p(f["cash_to_st_debt"].clip(0, 50)),
                         "f_eq_ratio": f["equity_ratio"].clip(-1, 1),
                         "f_d_lev": f["d_net_debt_ebitda_4q"].clip(-10, 10)}).sort_values("avail")


QSIGN = {"f_lev": -1, "f_cov": 1, "f_cash_st": 1, "f_eq_ratio": 1, "f_d_lev": -1}
fu = fundamentals()
m = pd.merge_asof(D[["_i", "day", "cnpj8"]].sort_values("day"), fu, left_on="day", right_on="avail", by="cnpj8",
                  direction="backward")
m.loc[(m["day"] - m["period_end"]).dt.days > FUND_STALE_DAYS, list(QSIGN)] = np.nan
m = m.set_index("_i").reindex(D["_i"])
for c in QSIGN:
    D[c] = m[c].to_numpy()
D["covered"] = D["f_lev"].notna() & D["f_cov"].notna()
for c in QSIGN:
    D.loc[~D["covered"], c] = np.nan
D["f_quality"] = pd.DataFrame({c: D.groupby("day")[c].rank(pct=True) * s for c, s in QSIGN.items()}) \
    .mean(axis=1, skipna=True).where(D["covered"])
D["q_thr"] = D.groupby("day")["f_quality"].transform(lambda s: s.quantile(0.2))
D["worstQ"] = D["covered"] & (D["f_quality"] <= D["q_thr"])

# stock per (decision, bond)
tk = [pick_ticker(c, p) for c, p in zip(D["cnpj8"], D["dpos"])]
D["sj"] = [a for a, _ in tk]
D["map_type"] = [b for _, b in tk]
D["adtv"] = [ADTV[p, j] if j >= 0 else np.nan for p, j in zip(D["dpos"], D["sj"])]
D["hedgeable"] = (D["sj"] >= 0) & (D["adtv"] >= MIN_ADTV)
log(f"covered by fundamentals {D['covered'].mean():.3f}; listed stock {(D['sj'] >= 0).mean():.3f}; "
    f"hedgeable (ADTV>=R$5m) {D['hedgeable'].mean():.3f}")

nz = lambda s, v: s.fillna(v)
D["p4f"] = ~(nz(D["resid_z"], 0) <= -1.5) & (nz(D["press_neg_30d"], 0) < 1)
BUCKETS = {"D10": D["cdi_pct"] <= 0.10, "Q5": D["cdi_pct"] <= 0.20, "A300": D["cdi_bps"] >= 300}
D["top_risk"] = (D["cdi_pct"] <= 0.05) | D["worstQ"] | (nz(D["eq_ret_4w"], 0) <= TOP_RISK_4W)


# ------------------------------------------------------------------------------------------------------------------
# point-in-time hedge ratios (trailing 252 grid days, weekly Dimson regression lag 0+1)
# ------------------------------------------------------------------------------------------------------------------
def weekly(M, p, n=50):
    lo = p - 5 * n
    if lo < 0:
        n = p // 5
        lo = p - 5 * n
    return M[lo:p].reshape(n, 5, *M.shape[1:]).sum(axis=1) if n > 0 else None


def dimson(y, x, min_n=20):
    ok = np.isfinite(y) & np.isfinite(x)
    ok[0] = False
    ok[1:] &= np.isfinite(x[:-1])
    if ok.sum() < min_n:
        return np.nan
    X = np.column_stack([x[ok], np.roll(x, 1)[ok], np.ones(ok.sum())])
    beta, *_ = np.linalg.lstsq(X, y[ok], rcond=None)
    return float(beta[0] + beta[1])


anyHY = BUCKETS["Q5"] | BUCKETS["A300"]
D["beta_reg"] = np.nan
D["beta_raw"] = np.nan
for p, x in D[anyHY].groupby("dpos"):
    bw = weekly(R0, p)
    pw = weekly(PRES.astype(float), p)
    sw = weekly(np.where(np.isnan(SX), np.nan, SX), p)
    if bw is None:
        continue
    raw = {}
    for i, b, j in zip(x.index, x["b"], x["sj"]):
        if j < 0:
            continue
        y = np.where(pw[:, b] >= 3, bw[:, b], np.nan)
        raw[i] = dimson(y, sw[:, j])
    rv = pd.Series(raw, dtype=float)
    pool = float(np.nanmedian(rv)) if rv.notna().sum() >= 5 else 0.0
    D.loc[rv.index, "beta_raw"] = rv
    D.loc[x.index[x["sj"] >= 0], "beta_reg"] = np.clip(0.5 * rv.reindex(x.index[x["sj"] >= 0]).fillna(pool) + 0.5 * pool,
                                                      0, 1)
D["beta_struct"] = np.clip(EQ_ELAST * D["dur"].clip(0, 8) * (D["cdi_bps"].clip(lower=0) / 1e4), 0, 0.5)
log("hedge ratio (reg) on HY rows: " + D.loc[anyHY & (D['sj'] >= 0), "beta_reg"].describe().round(3).to_json())
log("hedge ratio (struct) on HY rows: " + D.loc[anyHY & (D['sj'] >= 0), "beta_struct"].describe().round(3).to_json())


def cap_weights(issuer):
    n = len(issuer)
    w = np.full(n, 1.0 / n)
    iss = pd.Series(issuer)
    for _ in range(20):
        tot = pd.Series(w).groupby(iss).transform("sum").to_numpy()
        over = tot > CAP + 1e-12
        if not over.any():
            break
        w[over] *= CAP / tot[over]
        free = ~over
        if not free.any():
            break
        w[free] *= (1 - w[over].sum()) / w[free].sum()
    return w / w.sum()


# ------------------------------------------------------------------------------------------------------------------
# book engine: overlapping monthly tranches held H days from entry; daily book = sum of live tranches / min(M, n)
# ------------------------------------------------------------------------------------------------------------------
DEC = sorted(D["dpos"].unique())
EQ_FLAV = ("reg", "struct", "top", "trig", "jtd")


def run_book(mask: pd.Series, H: int, hedges: bool):
    M = max(H // 21, 1)
    comp = {k: np.zeros(ND) for k in ("bond_base", "bond_rec40", "expo", "cost25", "cost50")}
    if hedges:
        for f in EQ_FLAV:
            for k in ("eq_", "eqnot_", "eqtc_"):
                comp[k + f] = np.zeros(ND)
        for k in ("IBOV", "SMAL"):
            comp["mac_" + k] = np.zeros(ND)
            comp["macnot_" + k] = np.zeros(ND)
            comp["mactc_" + k] = np.zeros(ND)
    ncoh = np.zeros(ND)
    coh, pos_rows, Wbook = [], [], []
    for p in DEC:
        x = D[(D["dpos"] == p) & mask]
        if len(x) < 5:
            Wbook.append((p, pd.Series(dtype=float)))
            continue
        w = cap_weights(x["cnpj8"].to_numpy())
        Wbook.append((p, pd.Series(w, index=x["b"].to_numpy()).groupby(level=0).sum()))
        ncoh[p + 1:] += 1
        rec = {"p": p, "day": days[p], "dok": bool(x[f"dok{H}"].iloc[0]), "n": len(x)}
        cs = {k: np.zeros(ND) for k in comp if k not in ("cost25", "cost50")}
        # macro beta of this cohort's basket (trailing weekly, Dimson)
        mb = {}
        ysum = {"base": 0.0, "rec40": 0.0}
        if hedges:
            bw, pw = weekly(R0, p), weekly(PRES.astype(float), p)
            basket = (np.where(pw[:, x["b"].to_numpy()] >= 3, bw[:, x["b"].to_numpy()], 0) * w).sum(axis=1)
            for k in ("IBOV", "SMAL"):
                mb[k] = float(np.clip(np.nan_to_num(dimson(basket, weekly(IX[k][:, None], p)[:, 0])), 0, 1))
            rec.update({f"beta_{k}": v for k, v in mb.items()})
        for wi, (i, r) in zip(w, x.iterrows()):
            k = int(r["entry_pos"])
            if k < 0:
                continue
            e = min(k + H, ND)
            b = int(r["b"])
            # buy-and-hold: the position's value drifts with its own (excess) return; hedges sized on current value
            gv = {sc: np.r_[1.0, np.cumprod(1 + RS[sc][k:e - 1, b])] for sc in ("base", "rec40")}
            wv = wi * gv["base"]
            cs["expo"][k:e] += wv
            yb = {}
            for s in ("base", "rec40"):
                cs["bond_" + s][k:e] += wi * gv[s] * RS[s][k:e, b]
                yb[s] = float(np.expm1(LC[s][e, b] - LC[s][k, b]))
                ysum[s] += wi * yb[s]
            row = {"i": i, "p": p, "b": b, "w": wi, "y_base": yb["base"], "y_rec40": yb["rec40"],
                   "full": e == k + H, "dist": b in DIST_B, "hedgeable": bool(r["hedgeable"]),
                   "listed": r["sj"] >= 0, "map_type": r["map_type"], "adtv": r["adtv"], "top_risk": bool(r["top_risk"])}
            if hedges:
                for kk in ("IBOV", "SMAL"):
                    cs["mac_" + kk][k:e] += -wv * mb[kk] * IX[kk][k:e]
                    cs["macnot_" + kk][k:e] += wv * mb[kk]
                j = int(r["sj"])
                if r["hedgeable"]:
                    sx = np.nan_to_num(SX[k:e, j])
                    live = np.isfinite(SX[k:e, j]).astype(float)
                    trig = np.maximum.accumulate(np.nan_to_num(np.r_[RET21[k - 1:e - 1, j]] <= TOP_RISK_4W).astype(float))
                    hr = {"reg": r["beta_reg"], "struct": r["beta_struct"],
                          "top": r["beta_reg"] if r["top_risk"] else 0.0, "trig": r["beta_reg"],
                          "jtd": JTD_H if r["top_risk"] else 0.0}
                    for f in EQ_FLAV:
                        h = float(np.nan_to_num(hr[f]))
                        act = live * (trig if f == "trig" else 1.0)
                        cs["eq_" + f][k:e] += -wv * h * sx * act
                        cs["eqnot_" + f][k:e] += wv * h * act
                        row["eq_" + f] = float(-(wv * h * sx * act).sum())
            pos_rows.append(row)
        for kk, v in cs.items():
            comp[kk] += v
        rec.update({kk: float(v.sum()) for kk, v in cs.items() if kk != "expo"})
        rec["y_base"] = ysum["base"]
        rec["y_rec40"] = ysum["rec40"]
        coh.append(rec)
    div = np.maximum(np.minimum(M, ncoh), 1)
    for kk in list(comp):
        if kk not in ("cost25", "cost50"):
            comp[kk] = comp[kk] / div
    # bond book turnover cost at each decision (average of the last M cohorts' weights)
    prev = None
    for i, (p, _) in enumerate(Wbook):
        live = [wv for _, wv in Wbook[max(0, i - M + 1): i + 1] if len(wv)]
        if not live:
            continue
        book = pd.concat(live, axis=1).fillna(0).sum(axis=1) / min(M, len(live)) if len(live) else None
        if prev is not None:
            to = book.sub(prev, fill_value=0).abs().sum()
            comp["cost25"][p] += to * 25 / 2 / 1e4
            comp["cost50"][p] += to * 50 / 2 / 1e4
        prev = book
    # hedge trading cost: round trip on every hedge notional that becomes active (conservative, no netting)
    if hedges:
        for f in EQ_FLAV:
            n = comp["eqnot_" + f]
            comp["eqtc_" + f] = np.maximum(np.diff(np.r_[0, n]), 0) * 2 * STOCK_SIDE
        for kk in ("IBOV", "SMAL"):
            n = comp["macnot_" + kk]
            comp["mactc_" + kk] = np.maximum(np.diff(np.r_[0, n]), 0) * 2 * FUT_SIDE
    return comp, pd.DataFrame(coh), pd.DataFrame(pos_rows)


# ------------------------------------------------------------------------------------------------------------------
# protective put proxy: rolling 3m 10%-OTM Ibovespa puts, Black-Scholes at a FIXED implied vol (approximation)
# ------------------------------------------------------------------------------------------------------------------
def bs_put(S, K, tau, r, sig):
    if tau <= 1e-9:
        return max(K - S, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sig ** 2) * tau) / (sig * np.sqrt(tau))
    return float(K * np.exp(-r * tau) * norm.cdf(-(d1 - sig * np.sqrt(tau))) - S * norm.cdf(-d1))


def put_overlay(sig, start):
    """Daily excess P&L per unit of Ibovespa notional (premium financed at CDI) and premium spent."""
    pnl = np.zeros(ND)
    prem = np.zeros(ND)
    p = start
    while p < ND - 1:
        S0 = IBOV[p]
        K = (1 - PUT_OTM) * S0
        r = np.log1p(CDI_ANN[p])
        exp = p + PUT_T
        V = bs_put(S0, K, PUT_T / 252, r, sig)
        prem[p] = V / S0
        pnl[p] -= PUT_SPREAD * V / S0
        for t in range(p + 1, min(exp, ND - 1) + 1):
            Vt = bs_put(IBOV[t], K, (exp - t) / 252, np.log1p(CDI_ANN[t]), sig)
            pnl[t] += (Vt - V) / S0 - V / S0 * CDI[t]
            V = Vt
        p = exp
    return pnl, prem


P0 = DEC[0]
PUTS = {s: put_overlay(s, P0) for s in (0.22, 0.27)}


# ------------------------------------------------------------------------------------------------------------------
# statistics
# ------------------------------------------------------------------------------------------------------------------
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


DS = pd.Series(days)
WIN = (days > days[P0])


def monthly(ex):
    s = pd.Series(ex[WIN], index=days[WIN])
    c = pd.Series(CDI[WIN], index=days[WIN])
    tot = (1 + c + s).groupby([s.index.year, s.index.month]).prod()
    cc = (1 + c).groupby([s.index.year, s.index.month]).prod()
    mth = tot / cc - 1
    mth.index = pd.to_datetime([f"{y}-{m:02d}-01" for y, m in mth.index])
    return mth


def mdd(m):
    eqc = (1 + m).cumprod()
    return float((eqc / eqc.cummax() - 1).min())


def stats(ex, uex, M, extra=None):
    m, mu = monthly(ex), monthly(uex)
    d = m - mu
    cr = pd.Series(ex, index=days)[(days >= CRISIS[0]) & (days <= CRISIS[1])]
    cr_eq = (1 + cr).cumprod()
    out = {
        "excess_vs_CDI_ann_%": m.mean() * 1200, "excess_vs_U_ann_%": d.mean() * 1200,
        "vol_ann_%": m.std() * np.sqrt(12) * 100, "sharpe": m.mean() * 12 / (m.std() * np.sqrt(12)) if m.std() > 0 else np.nan,
        "max_dd_%": mdd(m) * 100, "worst_month_%": m.min() * 100, "skew": float(skew(m)), "hit_rate": float((m > 0).mean()),
        "t_nw_vs_CDI": nw_t(m, M), "t_nw_vs_U": nw_t(d, M),
        "exCDI_2022_23_%": m[m.index < SPLIT].mean() * 1200, "exCDI_2024_26_%": m[m.index >= SPLIT].mean() * 1200,
        "exU_2022_23_%": d[d.index < SPLIT].mean() * 1200, "exU_2024_26_%": d[d.index >= SPLIT].mean() * 1200,
        "crisis_2022-12_2023-06_cum_excess_%": float(cr_eq.iloc[-1] - 1) * 100,
        "crisis_max_dd_%": float((cr_eq / cr_eq.cummax() - 1).min()) * 100,
        "n_months": int(len(m)),
    }
    if extra:
        out.update(extra)
    return {k: (round(float(v), 4) if isinstance(v, (float, np.floating)) else v) for k, v in out.items()}


def loss_share(pos, ycol="y_base", thr=-0.10):
    q = pos[pos["full"]] if len(pos) else pos
    if not len(q):
        return {}
    wl = q["w"] * q[ycol]
    neg = wl[wl < 0].sum()
    big = wl[q[ycol] < thr].sum()
    dd = wl[q["dist"] & (wl < 0)].sum()
    return {"share_losses_from_big_drops_<-10%": float(big / neg) if neg < 0 else 0.0,
            "share_losses_from_stopped_below_0.90": float(dd / neg) if neg < 0 else 0.0,
            "big_drop_positions_share_of_weight": float(q.loc[q[ycol] < thr, "w"].sum() / q["w"].sum())}


# ------------------------------------------------------------------------------------------------------------------
# run
# ------------------------------------------------------------------------------------------------------------------
BASES = {"U universe": (pd.Series(True, index=D.index), False),
         "P4": (D["cdi_pct"] <= 0.3) & D["p4f"],
         "P4 covered-only": (D["cdi_pct"] <= 0.3) & D["p4f"] & D["covered"],
         "P4 + quality": (D["cdi_pct"] <= 0.3) & D["p4f"] & ~D["worstQ"]}
BASES = {k: (v if isinstance(v, tuple) else (v, False)) for k, v in BASES.items()}
for bk, bm in BUCKETS.items():
    BASES[f"{bk} HY"] = (bm, True)
    BASES[f"{bk} HY+P4f"] = (bm & D["p4f"], False)
    BASES[f"{bk} HY+Q"] = (bm & ~D["worstQ"], False)
    BASES[f"{bk} HY+P4f+Q"] = (bm & D["p4f"] & ~D["worstQ"], True)
    BASES[f"{bk} HY+P4f+Q covered-only"] = (bm & D["p4f"] & ~D["worstQ"] & D["covered"], False)
    BASES[f"{bk} HY covered-only"] = (bm & D["covered"], False)

results = {"design": {
    "source": "selection-lab caches (patched daily rate-hedged excess returns); monthly decisions from "
              f"{days[DEC[0]].date()} to {days[DEC[-1]].date()}; tranches held H bdays from entry (first trade after "
              "decision, <=20 bdays); book = average of live tranches; issuer cap 10%; bond cost 25 bps (50 check) "
              "per round trip on book turnover",
    "buckets": {"D10": "top decile of eligible universe by cdi_bps (CDI+ equivalent), point-in-time",
                "Q5": "top quintile by cdi_bps", "A300": "cdi_bps >= 300"},
    "filters": {"P4f": "not rich vs peers (resid_z > -1.5) and no press_neg_30d",
                "Q": "drop worst quintile composite quality (uncovered kept)"},
    "equity_hedge": {"reg": "short issuer/parent stock, ratio = 0.5*own + 0.5*pooled HY median of trailing 252d weekly "
                            "Dimson beta (lag 0+1) of bond hedged returns on stock excess returns, clipped [0,1]",
                     "struct": f"ratio = {EQ_ELAST} * duration * spread (Merton-style, dln spread/dln E = -1), cap 0.5",
                     "top": "reg ratio, only positions flagged at decision: top 5% carry OR worst-quintile quality "
                            "OR stock <= -15% in 4w",
                     "trig": "reg ratio, switched on (for the rest of the tranche) once the stock's 21d return <= -15%",
                     "jtd": f"jump-to-default sizing: ratio {JTD_H} on top-risk positions only (bond LGD ~50% vs "
                            "stock -80..-100% in default)",
                     "eligibility": f"stock ADTV >= R$ {MIN_ADTV / 1e6:.0f}m/day (borrow proxy)",
                     "costs": f"borrow {BORROW[0]:.0%}/yr (check {BORROW[1]:.0%}), {STOCK_SIDE * 1e4:.0f} bps/side trading"},
    "macro_hedge": "short Ibovespa futures (P&L = -beta * (Ibov - CDI)), beta = trailing 252d weekly Dimson beta of the "
                   "cohort basket, clipped [0,1]; SMAL11 same with 2%/yr ETF borrow",
    "put_proxy": f"APPROXIMATION: roll 3m {PUT_OTM:.0%}-OTM Ibovespa puts priced Black-Scholes at a fixed IV (22%, 27% "
                 f"skew check), premium financed at CDI, {PUT_SPREAD:.0%} of premium paid as spread; notional = "
                 "book beta x exposure, or 25% of exposure",
    "recovery_sensitivity": f"bonds that stop trading >30d before data end and before maturity with last mark < 0.90 "
                            f"({len(dist)} bonds) jump to {REC:.0%} of par on their last grid day",
}}
TAB = {}
COH = {}
POS = {}
MAIN = {}
for H in HZ:
    M = max(H // 21, 1)
    comps = {}
    for nm, (mask, hed) in BASES.items():
        comps[nm], COH[(H, nm)], POS[(H, nm)] = run_book(mask, H, hed)
    log(f"H={H}: books built")
    U = comps["U universe"]
    u_ex = U["bond_base"] - U["cost25"]
    u_ex40 = U["bond_rec40"] - U["cost25"]
    T = {}

    def add(name, c, hed_pnl=None, hed_cost=None, extra=None, scen="base", cost="cost25"):
        ex = c["bond_" + scen] - c[cost]
        if hed_pnl is not None:
            ex = ex + hed_pnl - hed_cost
        ux = u_ex if scen == "base" else u_ex40
        e = dict(extra or {})
        if hed_pnl is not None:
            mw = WIN
            e["hedge_pnl_ann_%"] = float(hed_pnl[mw].sum() / mw.sum() * 252 * 100)
            e["hedge_cost_ann_%"] = float(hed_cost[mw].sum() / mw.sum() * 252 * 100)
        e["bond_cost_ann_%"] = float(c[cost][WIN].sum() / WIN.sum() * 252 * 100)
        e["avg_exposure"] = float(c["expo"][WIN].mean())
        T[name] = stats(ex, ux, M, e)
        return ex

    ser = {"CDI": np.zeros(ND)}
    for nm, c in comps.items():
        ls = loss_share(POS[(H, nm)])
        ser[nm] = add(nm, c, extra={**ls, "avg_bonds": float(COH[(H, nm)]["n"].mean())})
        add(nm + " [50bps]", c, cost="cost50")
        add(nm + " [rec40]", c, scen="rec40", extra=loss_share(POS[(H, nm)], "y_rec40"))
        if not BASES[nm][1]:
            continue
        pos = POS[(H, nm)]
        full = pos[pos["full"]]
        wt = full["w"].sum()
        cov = {"share_w_listed_stock": float(full.loc[full["listed"], "w"].sum() / wt),
               "share_w_listed_direct": float(full.loc[full["map_type"] == "direct", "w"].sum() / wt),
               "share_w_listed_parent": float(full.loc[full["map_type"] == "parent", "w"].sum() / wt),
               "share_w_hedgeable_ADTV>=5m": float(full.loc[full["hedgeable"], "w"].sum() / wt),
               "share_w_option_liquid_ADTV>=200m": float(full.loc[full["adtv"] >= OPT_ADTV, "w"].sum() / wt),
               "share_w_top_risk": float(full.loc[full["top_risk"], "w"].sum() / wt)}
        T[nm]["coverage"] = cov
        big = full[full["y_base"] < -0.10]
        for f in EQ_FLAV:
            for bi, br in enumerate(BORROW):
                if bi and f != "reg":
                    continue
                hp = c["eq_" + f]
                hc = c["eqnot_" + f] * br / 252 + c["eqtc_" + f]
                tag = f"{nm} + EQ-{f} ({br:.0%} borrow)"
                eff = {"eq_hedge_recoup_on_big_drops_%": float(big.get("eq_" + f, pd.Series(0.0)).fillna(0).mul(1).sum()
                                                              / -(big["w"] * big["y_base"]).sum() * 100) if len(big) else None,
                       "avg_hedge_notional_%_of_book": float(c["eqnot_" + f][WIN].mean() * 100)}
                ser[tag] = add(tag, c, hp, hc, eff)
                add(tag + " [rec40]", c, hp, hc, scen="rec40")
        for kk in ("IBOV", "SMAL"):
            hp = c["mac_" + kk]
            hc = c["mactc_" + kk] + (c["macnot_" + kk] * ETF_BORROW / 252 if kk == "SMAL" else 0)
            tag = f"{nm} + {kk} beta hedge"
            ser[tag] = add(tag, c, hp, hc, {"avg_beta_notional_%": float(c["macnot_" + kk][WIN].mean() * 100)})
            add(tag + " [rec40]", c, hp, hc, scen="rec40")
        for sig, (pp, prem) in PUTS.items():
            for sz, notional in (("beta", c["macnot_IBOV"]), ("25%", 0.25 * c["expo"])):
                if sig == 0.27 and sz == "beta":
                    continue
                hp = notional * pp
                tag = f"{nm} + PUT proxy ({sz} notional, IV {sig:.0%})"
                prem_ann = float((notional * prem)[WIN].sum() / WIN.sum() * 252 * 100)
                ser[tag] = add(tag, c, hp, np.zeros(ND), {"put_premium_paid_ann_%": prem_ann,
                                                           "note": "hedge_pnl is NET of premium, spread and financing"})
    # paired: variant minus P4 (monthly), NW t
    p4m = monthly(ser["P4"])
    for nm in list(T):
        if nm in ser:
            dm = monthly(ser[nm]) - p4m
            T[nm]["excess_vs_P4_ann_%"] = round(float(dm.mean() * 1200), 4)
            T[nm]["exP4_2022_23_%"] = round(float(dm[dm.index < SPLIT].mean() * 1200), 4)
            T[nm]["exP4_2024_26_%"] = round(float(dm[dm.index >= SPLIT].mean() * 1200), 4)
            T[nm]["t_nw_vs_P4"] = round(float(nw_t(dm, M)), 3) if nm != "P4" else None
    # cohort-level (sellab-comparable): mean H-horizon cohort excess vs CDI and vs U, annualised
    cl = {}
    ucoh = COH[(H, "U universe")].set_index("day")
    for nm in BASES:
        ch = COH[(H, nm)].set_index("day")
        ch = ch[ch["dok"]]
        y = ch["y_base"]
        dU = (y - ucoh["y_base"].reindex(y.index)).dropna()
        cl[nm] = {"n_cohorts": int(len(y)), "exCDI_ann_%": float(y.mean() / (H / 252) * 100),
                  "exU_ann_%": float(dU.mean() / (H / 252) * 100), "t_nw_vs_U": nw_t(dU, M),
                  "exCDI_ann_%_rec40": float(ch["y_rec40"].mean() / (H / 252) * 100)}
        if BASES[nm][1]:
            for f in EQ_FLAV:
                cl[nm][f"EQ-{f}_hedge_pnl_ann_%"] = float(ch["eq_" + f].mean() / (H / 252) * 100)
            cl[nm]["IBOV_hedge_pnl_ann_%"] = float(ch["mac_IBOV"].mean() / (H / 252) * 100)
            cl[nm]["avg_beta_IBOV"] = float(ch["beta_IBOV"].mean())
            cl[nm]["avg_beta_SMAL"] = float(ch["beta_SMAL"].mean())
    TAB[H] = {"book_monthly": T, "cohort_level": cl}
    MAIN[H] = ser
    for nm in ("P4", "P4 + quality", "U universe", "D10 HY", "D10 HY+P4f+Q", "D10 HY + EQ-reg (2% borrow)",
               "D10 HY + IBOV beta hedge", "Q5 HY", "Q5 HY+P4f+Q", "A300 HY"):
        q = T[nm]
        log(f"H={H} {nm:<36} exCDI {q['excess_vs_CDI_ann_%']:+6.2f} exU {q['excess_vs_U_ann_%']:+6.2f} "
            f"vsP4 {q['excess_vs_P4_ann_%']:+6.2f} vol {q['vol_ann_%']:5.2f} SR {q['sharpe']:5.2f} "
            f"MDD {q['max_dd_%']:6.2f} crisis {q['crisis_2022-12_2023-06_cum_excess_%']:+6.2f}")

results["results"] = TAB

# crisis anatomy: largest weighted losers inside the HY books during the crisis window (H=126)
anat = {}
for nm in ("D10 HY", "Q5 HY", "A300 HY", "P4"):
    pos = POS[(126, nm)].copy()
    pos["day"] = days[pos["p"].to_numpy()]
    q = pos[(pos["day"] >= CRISIS[0] - pd.Timedelta(days=180)) & (pos["day"] <= CRISIS[1])]
    q = q.assign(codigo=codes[q["b"].to_numpy()], cnpj8=D.loc[q["i"], "cnpj8"].to_numpy(),
                 contrib=q["w"] * q["y_base"])
    agg = dict(n_cohorts=("p", "size"), avg_w=("w", "mean"), worst_y=("y_base", "min"), sum_contrib=("contrib", "sum"))
    if "eq_reg" in q:
        agg.update(eq_reg_hedge_pnl=("eq_reg", "sum"), eq_struct_hedge_pnl=("eq_struct", "sum"))
    top = q.groupby(["codigo", "cnpj8"]).agg(**agg)
    anat[nm] = top.sort_values("sum_contrib").head(10).reset_index().round(4).to_dict("records")
am = D[D["cnpj8"] == "00776574"][["codigo", "day", "cdi_bps", "cdi_pct", "ratio", "eq_ret_4w", "worstQ", "p4f", "y126"]]
results["crisis_anatomy_H126"] = {"window": [str(CRISIS[0].date()), str(CRISIS[1].date())], "top_losers": anat,
                                  "americanas_rows": am.assign(day=am["day"].dt.date.astype(str)).round(4)
                                  .to_dict("records")}
results["puts"] = {f"IV_{s:.0%}": {"premium_per_roll_mean_%": float(pr[pr > 0].mean() * 100),
                                   "net_pnl_per_unit_notional_ann_%": float(pp[WIN].sum() / WIN.sum() * 252 * 100),
                                   "crisis_window_pnl_%": float(pp[(days >= CRISIS[0]) & (days <= CRISIS[1])].sum() * 100)}
                   for s, (pp, pr) in PUTS.items()}
ib = pd.Series(IBOV, index=days)
results["ibov_crisis_window_return_%"] = float(ib[(days >= CRISIS[0]) & (days <= CRISIS[1])].iloc[[0, -1]].pct_change()
                                               .iloc[-1] * 100)
results["distressed_stop_trading"] = {"n": int(len(dist)), "recovery_assumed": REC,
                                      "bonds": dist.reset_index().assign(last_day=lambda z: z["last_day"].dt.date.astype(str),
                                                                         maturity=lambda z: z["maturity"].astype(str))
                                      .round(4).to_dict("records")}

# ------------------------------------------------------------------------------------------------------------------
# plot (H = 126): total-return style CDI x (1 + excess)
# ------------------------------------------------------------------------------------------------------------------
H = 126
ser = MAIN[H]
mainb = "Q5"
lines = [("CDI", "CDI", "k", ":"), ("P4", "P4", "#1f77b4", "-"),
         (f"{mainb} HY", f"HY top quintile, unhedged", "#d62728", "-"),
         (f"{mainb} HY+P4f+Q", f"HY top quintile + P4 filters + quality", "#ff7f0e", "-"),
         (f"{mainb} HY + EQ-struct (2% borrow)", f"HY top quintile + equity short (structural ratio, 2% borrow)",
          "#2ca02c", "-"),
         (f"{mainb} HY + EQ-jtd (2% borrow)", f"HY top quintile + JTD-sized short on top-risk names (0.5)",
          "#8c564b", ":"),
         (f"{mainb} HY + IBOV beta hedge", f"HY top quintile + Ibov futures beta hedge", "#9467bd", "--"),
         ("D10 HY", "HY top decile, unhedged (missed Americanas)", "#e377c2", "-.")]
fig, axs = plt.subplots(1, 2, figsize=(15, 5.6), dpi=110)
w = WIN
cdi_tr = np.cumprod(1 + CDI[w])
for key, lab, col, ls in lines:
    tr = np.cumprod(1 + CDI[w] + ser[key][w])
    axs[0].plot(days[w], tr * 100, ls, color=col, lw=1.5, label=lab)
    axs[1].plot(days[w], (tr / cdi_tr - 1) * 100, ls, color=col, lw=1.5, label=lab)
for ax in axs:
    ax.axvspan(CRISIS[0], CRISIS[1], color="grey", alpha=0.12)
    ax.grid(alpha=0.25)
axs[0].set_title("Total return index (CDI x (1+excess)), base 100", fontsize=10)
axs[1].set_title("Cumulative excess over CDI (%)  - shaded: Dec-22..Jun-23 (Americanas / Light)", fontsize=10)
axs[0].legend(fontsize=8, frameon=False)
fig.suptitle("HY debenture books, 6m overlapping monthly tranches, rate-hedged, 25 bps/trade, issuer cap 10%", fontsize=11)
fig.tight_layout()
fig.savefig(OUT / "hy_equity.png")


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if o != o else round(float(o), 5)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, pd.Timestamp):
        return str(o.date())
    return o


json.dump(clean(results), open(OUT / "hy_results.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
log("done")
