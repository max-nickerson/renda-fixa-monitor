"""Alternative rate / inflation hedges for IPCA+ and Pre debentures, as additive adjustments to the harness return
matrix R (grid position x bond). DI+ floaters are untouched (no rate exposure).

The harness R already contains the bond-level hedge  + dur * d(bench_rate)  where bench_rate is the benchmark rate
interpolated at the bond's own duration on the trade date (NTN-B real curve for IPCA+, DI x Pre for Pre), i.e. an
'exact' key-rate hedge rebalanced at every trade. Variants (adjustment added to R):
  raw        : no rate hedge at all                                 -dur*d(bench)
  vertex     : duration-only hedge at ONE fixed vertex per curve    -dur*d(bench) + dur*d(curve@fixed tenor)
               (IPCA -> NTN-B 5y, Pre -> DI x Pre 3y; the classic 'hedge the book DV01 with one DI1/DAP contract')
  keyrate    : our own two-vertex (bracketing) interpolation on B3/Tesouro curves at the row's duration
  di1_ipca   : IPCA+ bonds hedged with NOMINAL DI1 at matching duration (DAP illiquid)  -dur*d(bench)+dur*d(PRE@dur)
  swap       : exact hedge + carry swap to CDI (IPCA x DI / Pre x DI swap): + cg - ig*(1+bench)^tau
               (removes the inflation/fixed carry vs CDI, leaving the pure credit spread return)
All curve inputs are the curve of the TRADE date of each mark (same timing as bench_rate). Adjustments are booked
at the harness's realisation position (gap moves at pos(next row)-1, AUDIT FIX v4)."""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from research.nightly import harness as H
from research.nightly.hedging_overlays import data as D

CACHE = D.CACHE / "rate_adj_v4.pkl"
VERTEX = {"IPCA": ("DIC", 1260), "PRE": ("PRE", 756)}


def _interp(cv: pd.DataFrame, pref: str, dates: pd.Series, years: np.ndarray) -> np.ndarray:
    ten = np.array(D.TENORS) / 252
    M = cv[[f"{pref}_{t}" for t in D.TENORS]]
    M = M.reindex(M.index.union(pd.DatetimeIndex(dates.unique()))).ffill(limit=5)
    vals = M.reindex(pd.DatetimeIndex(dates)).to_numpy()
    out = np.full(len(years), np.nan)
    ok = ~np.isnan(vals).any(axis=1) & np.isfinite(years)
    y = np.clip(years[ok], ten[0], ten[-1])
    idx = np.searchsorted(ten, y, side="right") - 1
    idx = np.clip(idx, 0, len(ten) - 2)
    w = (y - ten[idx]) / (ten[idx + 1] - ten[idx])
    v = vals[ok]
    ar = np.arange(len(v))
    out[ok] = v[ar, idx] * (1 - w) + v[ar, idx + 1] * w
    return out


def _fixed(cv, pref, tenor, dates):
    s = cv[f"{pref}_{tenor}"]
    s = s.reindex(s.index.union(pd.DatetimeIndex(dates.unique()))).ffill(limit=5)
    return s.reindex(pd.DatetimeIndex(dates)).to_numpy()


def build() -> dict:
    if CACHE.exists():
        return pd.read_pickle(CACHE)
    C = H._core()
    days, codes = C["days"], C["codes"]
    ND, NB = len(days), len(codes)
    g = pd.read_pickle(H.HIST / "lab_daily.pkl")[["codigo", "day", "date", "dur", "kind", "bench_rate", "contract"]]
    g = g.sort_values(["codigo", "day"]).reset_index(drop=True)
    grp = g.groupby("codigo")
    nday = grp["day"].shift(-1)
    ndate = grp["date"].shift(-1)
    has = nday.notna().to_numpy()
    rk = g["kind"].isin(["IPCA", "PRE"]).to_numpy() & has
    g = g[rk].copy()
    g["nday"], g["ndate"] = nday[rk], ndate[rk]
    g["nbench"] = grp["bench_rate"].shift(-1)[rk]
    # accruals (same construction as run_selection_lab.build_returns)
    from rfmonitor.history import bcb_series
    cdi = bcb_series(12, date(2020, 12, 1)) / 100
    Cix = (1 + cdi).cumprod()
    ipca_m = bcb_series(433, date(2019, 1, 1)) / 100
    bd = pd.bdate_range(Cix.index.min(), pd.Timestamp("2026-12-31"))
    per = []
    for d in bd:
        m = pd.Timestamp(d.year, d.month, 1)
        v = ipca_m.get(m, ipca_m.iloc[-1])
        n = len(pd.bdate_range(m, m + pd.offsets.MonthEnd(0)))
        per.append((1 + v) ** (1 / n))
    Iix = pd.Series(np.cumprod(per), index=bd)
    Cd = Cix.reindex(Cix.index.union(days)).ffill().reindex(days)
    Id = Iix.reindex(Iix.index.union(days)).ffill().reindex(days)
    cg = Cd.reindex(g["nday"]).to_numpy() / Cd.reindex(g["day"]).to_numpy()
    ig = Id.reindex(g["nday"]).to_numpy() / Id.reindex(g["day"]).to_numpy()
    nbd = np.busday_count(g["day"].values.astype("datetime64[D]"), g["nday"].values.astype("datetime64[D]"))
    tau = nbd / 252
    dur = g["dur"].to_numpy()
    ipca = g["kind"].eq("IPCA").to_numpy()
    db = ((g["nbench"] - g["bench_rate"]) / 100).fillna(0).to_numpy()
    hedge_now = dur * db
    pre, real = D.pre_curve(), D.real_curve()
    ycur = np.where(ipca, dur, dur)
    out = {}
    # raw
    out["raw"] = -hedge_now
    # key-rate on our curves (same duration held for the step)
    k0 = np.where(ipca, _interp(real, "DIC", g["date"], ycur), _interp(pre, "PRE", g["date"], ycur))
    k1 = np.where(ipca, _interp(real, "DIC", g["ndate"], ycur), _interp(pre, "PRE", g["ndate"], ycur))
    dk = np.nan_to_num((k1 - k0) / 100)
    out["keyrate"] = -hedge_now + dur * dk
    # single fixed vertex
    v0 = np.where(ipca, _fixed(real, "DIC", 1260, g["date"]), _fixed(pre, "PRE", 756, g["date"]))
    v1 = np.where(ipca, _fixed(real, "DIC", 1260, g["ndate"]), _fixed(pre, "PRE", 756, g["ndate"]))
    out["vertex"] = -hedge_now + dur * np.nan_to_num((v1 - v0) / 100)
    # IPCA with nominal DI1 at matching duration (Pre unchanged = exact)
    p0 = _interp(pre, "PRE", g["date"], dur)
    p1 = _interp(pre, "PRE", g["ndate"], dur)
    dp = np.nan_to_num((p1 - p0) / 100)
    out["di1_ipca"] = np.where(ipca, -hedge_now + dur * dp, 0.0)
    # carry swap to CDI (on top of the exact hedge)
    bench = g["bench_rate"].fillna(pd.Series(k0, index=g.index)).to_numpy() / 100
    sw = cg - np.where(ipca, ig, 1.0) * (1 + bench) ** tau
    out["swap"] = np.nan_to_num(sw)
    # diagnostics: bench vs our curve
    diag = {"n_rows": int(len(g)), "ipca_share_rows": float(ipca.mean()),
            "bench_minus_ourcurve_bps_median": float(np.nanmedian((g["bench_rate"].to_numpy() - k0) * 100)),
            "corr_dbench_dkeyrate": float(pd.Series(db).corr(pd.Series(dk))),
            "corr_dbench_dkeyrate_ipca": float(pd.Series(db[ipca]).corr(pd.Series(dk[ipca]))),
            "swap_adj_ann_mean_ipca_%": float(np.nansum(out["swap"][ipca]) / max(np.sum(tau[ipca]), 1) * 100),
            "swap_adj_ann_mean_pre_%": float(np.nansum(out["swap"][~ipca]) / max(np.sum(tau[~ipca]), 1) * 100)}
    # map to grid positions with the harness realisation rule
    pos = days.get_indexer(g["day"])
    npos = days.get_indexer(g["nday"])
    b = codes.get_indexer(g["codigo"])
    rpos = np.where(npos - pos > 1, npos - 1, pos)
    mats = {}
    for k, a in out.items():
        A = np.zeros((ND, NB), dtype=np.float32)
        np.add.at(A, (rpos, b), np.clip(np.nan_to_num(a), -0.5, 0.5).astype(np.float32))
        mats[k] = A
    res = {"mats": mats, "diag": diag}
    pd.to_pickle(res, CACHE)
    return res


class use_R:
    """Context manager: run harness backtests with R + adjustment (base scenario)."""

    def __init__(self, adj: np.ndarray | None):
        self.adj = adj

    def __enter__(self):
        R0 = H._Rmat("base")
        self.saved = R0
        H._MEM[("R64", "R")] = R0 + (self.adj.astype(np.float64) if self.adj is not None else 0.0)
        H._MEM.pop(("LC", "base"), None)
        return self

    def __exit__(self, *a):
        H._MEM[("R64", "R")] = self.saved
        H._MEM.pop(("LC", "base"), None)
        for k in [k for k in H._MEM if isinstance(k, tuple) and k and k[0] == "baseline"]:
            H._MEM.pop(k)


if __name__ == "__main__":
    r = build()
    print(r["diag"])
