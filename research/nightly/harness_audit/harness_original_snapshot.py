"""Common evaluation harness for the nightly research agents (debenture selection / portfolio research).

Everything is point-in-time, uses the PATCHED rate-hedged daily excess returns of the selection lab
(data/history/sellab_returns.pkl), next-fresh-trade execution, buy-and-hold drift inside a position, issuer cap,
book-level costs, and a sealed holdout (decisions >= HOLDOUT are invisible unless holdout=True).

Quick use (see research/nightly/HARNESS_README.md):

    from research.nightly import harness as H
    P = H.load_panel("M")                       # monthly decision cross-sections + features + fwd targets
    r = H.backtest(lambda x: x["cdi_bps"], freq="M", hold=126, top_frac=0.3)
    b = H.baseline("P4Q")
    print(H.stats(r["daily"], bench=b["daily"]))

Caches: data/history/nightly/harness/ (core.pkl, panel_M.pkl, panel_W.pkl, ibov.pkl). Delete them to rebuild.
Read-only inputs: data/history/{lab_daily,lab_weekly,sellab_panel,sellab_returns,fundamentals_pit,equity_daily}.pkl,
research/data/{equity_map,issuer_sectors}.csv, IDA-DI (rfmonitor.history.ida), CDI (BCB SGS 12).
"""
from __future__ import annotations

import sys as _sys
import time
import warnings
from datetime import date
from pathlib import Path as _P

_ROOT = _P(__file__).resolve().parents[2]
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
from scipy import stats as _sst

from rfmonitor.config import DATA_DIR

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

HIST = DATA_DIR / "history"
CACHE = HIST / "nightly" / "harness"
CACHE.mkdir(parents=True, exist_ok=True)
RDATA = _ROOT / "research" / "data"

HOLDOUT = pd.Timestamp("2026-01-01")
START = pd.Timestamp("2022-01-01")       # default first decision for backtests (as selection / HY labs)
PANEL_START = pd.Timestamp("2021-03-01")  # panel keeps earlier decisions for model training
SPLIT = pd.Timestamp("2024-01-01")
ENTRY_MAX = 20            # pending buys expire after 20 bdays without a fresh trade (-> cash)
SELL_MAX = 20             # pending sells: executed at the first fresh trade within 20 bdays, else at the last mark
ISSUER_CAP = 0.10
REC = 0.40
HZ = (21, 63, 126, 252)
FUND_STALE_DAYS = 460
MIN_NAMES = 5
_VERSION = 3              # bump to invalidate caches

_T0 = time.time()
_MEM: dict = {}


def _log(*a):
    print(f"[harness +{time.time() - _T0:6.1f}s]", *a, flush=True)


# =====================================================================================================================
# 1) core arrays: grid days, bonds, daily patched returns, trade dates, CDI
# =====================================================================================================================
def _core() -> dict:
    """Grid-level arrays shared by everything (cached in memory and on disk)."""
    if "core" in _MEM:
        return _MEM["core"]
    path = CACHE / "core.pkl"
    if path.exists():
        C = pd.read_pickle(path)
        if C.get("version") == _VERSION:
            _MEM["core"] = C
            return C
    _log("building core arrays (one-off, ~1 min)")
    D0, info = pd.read_pickle(HIST / "sellab_panel.pkl")
    G, codes = pd.read_pickle(HIST / "sellab_returns.pkl")
    days = info["days"]
    ND, NB = len(days), len(codes)
    R = np.zeros((ND, NB))
    R[G["pos"].to_numpy(), G["b"].to_numpy()] = G["r_patch"].to_numpy()
    g = pd.read_pickle(HIST / "lab_daily.pkl")[["codigo", "day", "date"]]
    pos = days.get_indexer(g["day"])
    b = codes.get_indexer(g["codigo"])
    assert (pos >= 0).all() and (b >= 0).all()
    TD = np.full((ND, NB), -1, dtype=np.int32)          # last trade date (days since epoch) of the row's mark
    TD[pos, b] = (g["date"].to_numpy().astype("datetime64[D]").astype(np.int64)).astype(np.int32)
    del g
    # distressed stop-trading bonds (>30d before data end, before maturity, last mark < 0.90): 40% recovery scenario
    last = info["last"]
    stopped = last[(last["last_day"] < days[-1] - pd.Timedelta(days=30))
                   & (last["maturity"].isna() | (last["maturity"] > last["last_day"] + pd.Timedelta(days=60)))]
    dist = stopped[stopped["last_ratio"] < 0.90]
    lastpos = G.groupby("b")["pos"].max()
    R40 = R.copy()
    for c, r in dist.iterrows():
        bb = codes.get_loc(c)
        p = int(lastpos[bb])
        R40[p, bb] = (1 + R40[p, bb]) * (REC / r["last_ratio"]) - 1
    # CDI on the grid
    from rfmonitor.history import bcb_series
    cdi = bcb_series(12, date(2020, 1, 1)) / 100
    cidx = (1 + cdi).cumprod()
    cg = cidx.reindex(cidx.index.union(days)).ffill().reindex(days)
    CDI = (cg / cg.shift(1) - 1).fillna(0).to_numpy()
    C = {"version": _VERSION, "days": days, "codes": codes, "R": R.astype(np.float32), "R40": R40.astype(np.float32),
         "TD": TD, "CDI": CDI, "cdi_daily_raw": cdi, "last": last,
         "dist_codes": list(dist.index), "survivorship": {"stopped_before_maturity": int(len(stopped)),
                                                          "stopped_below_0.90": int(len(dist))}}
    pd.to_pickle(C, path)
    _MEM["core"] = C
    return C


def days() -> pd.DatetimeIndex:
    return _core()["days"]


def _hpos() -> int:
    """First grid position on/after HOLDOUT."""
    return int(days().searchsorted(HOLDOUT))


def _Rmat(scenario: str = "base") -> np.ndarray:
    C = _core()
    key = "R" if scenario == "base" else "R40"
    if ("R64", key) not in _MEM:
        _MEM[("R64", key)] = C[key].astype(np.float64)
    return _MEM[("R64", key)]


def _LC(scenario: str = "base") -> np.ndarray:
    """Cumulative log returns: LC[p, b] = sum of log(1+R) of rows with pos < p (shape ND+1 x NB)."""
    k = ("LC", scenario)
    if k not in _MEM:
        R = _Rmat(scenario)
        _MEM[k] = np.vstack([np.zeros((1, R.shape[1])), np.cumsum(np.log1p(np.clip(R, -0.99, None)), axis=0)])
    return _MEM[k]


def exec_pos(p: int, max_wait: int = ENTRY_MAX) -> np.ndarray:
    """For a decision at grid position p (close of that day), the grid position at which each bond executes:
    first k in (p, p+max_wait] whose mark comes from a trade dated AFTER the decision day. -1 if none."""
    k = ("EX", p, max_wait)
    if k in _MEM:
        return _MEM[k]
    C = _core()
    TD, ND = C["TD"], len(C["days"])
    dnum = np.int32(C["days"][p].to_datetime64().astype("datetime64[D]").astype(np.int64))
    hi = min(p + max_wait + 1, ND)
    if hi <= p + 1:
        out = np.full(TD.shape[1], -1)
    else:
        M = TD[p + 1:hi] > dnum
        any_ = M.any(axis=0)
        out = np.where(any_, p + 1 + M.argmax(axis=0), -1)
    _MEM[k] = out
    return out


def cdi_daily() -> pd.Series:
    """Daily CDI return on the grid (0 on grid days that are holidays)."""
    C = _core()
    return pd.Series(C["CDI"], index=C["days"], name="CDI")


# =====================================================================================================================
# 2) point-in-time panel
# =====================================================================================================================
QSIGN = {"f_lev": -1, "f_cov": 1, "f_cash_st": 1, "f_eq_ratio": 1, "f_d_lev": -1}
FCOLS = ["f_lev", "f_cov", "f_cash_st", "f_eq_ratio", "f_gde", "f_margin", "f_rev_g", "f_d_lev", "f_d_cov", "f_size"]
WEEKLY_COLS = ["trades_30d", "own_z", "d_spread_4w", "d_spread_1w", "d_ratio_4w", "carry_per_dur", "n_fact_30d",
               "n_fact_90d", "n_distress_30d", "n_distress_90d", "n_deb_mtg_30d", "n_deb_mtg_90d", "n_rating_30d",
               "n_rating_90d", "n_oficio_30d", "n_oficio_90d", "days_since_distress", "news_any_30d", "distress_90d",
               "mkt_distress_z", "maturity"]


def _fundamentals(strict: bool = True) -> pd.DataFrame:
    """Same construction as research/run_selection_lab.py (strict availability dates by default)."""
    f = pd.read_pickle(HIST / "fundamentals_pit.pkl").copy()
    f["avail"] = f["available_date_strict" if strict else "available_date"]
    f = f.sort_values(["cnpj8", "avail", "period_end"])
    f["_mx"] = f.groupby("cnpj8")["period_end"].cummax().groupby(f["cnpj8"]).shift(1)
    f = f[f["_mx"].isna() | (f["period_end"] > f["_mx"])].drop_duplicates(["cnpj8", "avail"], keep="last")
    e = f["ebitda_ltm"]
    lev = np.where(e > 0, f["net_debt"] / e, 15.0)
    lev = np.where(f["net_debt"].isna() | e.isna(), np.nan, lev)
    fe = f["fin_exp_ltm"]
    cov = np.where(fe > 0, e / fe, np.where(fe.notna() & e.notna(), 30.0, np.nan))
    return pd.DataFrame({
        "cnpj8": f["cnpj8"].to_numpy(), "avail": f["avail"].to_numpy(), "period_end": f["period_end"].to_numpy(),
        "f_lev": np.clip(lev, -3, 15), "f_cov": np.clip(cov, -5, 30),
        "f_cash_st": np.log1p(f["cash_to_st_debt"].clip(0, 50)).to_numpy(),
        "f_eq_ratio": f["equity_ratio"].clip(-1, 1).to_numpy(),
        "f_gde": np.where(f["equity"] <= 0, 10.0, f["gross_debt_equity"].clip(0, 10)),
        "f_margin": f["ebitda_margin"].clip(-1, 1).to_numpy(), "f_rev_g": f["revenue_growth_yoy"].clip(-0.9, 3).to_numpy(),
        "f_d_lev": f["d_net_debt_ebitda_4q"].clip(-10, 10).to_numpy(),
        "f_d_cov": f["d_interest_coverage_4q"].clip(-10, 10).to_numpy(),
        "f_size": np.log(f["total_assets"].where(f["total_assets"] > 0)).to_numpy(),
        "f_is_parent": f["is_parent"].astype(float).to_numpy(), "f_source": f["source"].to_numpy(),
    }).assign(cnpj8=lambda z: z["cnpj8"].astype("str")).sort_values("avail")


def _equity_grid():
    """Stock prices / ADTV on the grid (ND x NS), same cleaning as run_hy_lab.py."""
    if "eqgrid" in _MEM:
        return _MEM["eqgrid"]
    C = _core()
    dd = C["days"]

    def on_grid(s, limit=5):
        return s.reindex(s.index.union(dd)).ffill(limit=limit).reindex(dd).to_numpy()

    eq = pd.read_pickle(HIST / "equity_daily.pkl")
    eq["date"] = pd.to_datetime(eq["date"])
    eq = eq.sort_values(["ticker", "date"]).drop_duplicates(["ticker", "date"])
    tick = sorted(eq["ticker"].unique())
    TI = {t: i for i, t in enumerate(tick)}
    SPX = np.full((len(dd), len(tick)), np.nan)
    ADTV = np.full((len(dd), len(tick)), np.nan)
    for t, g in eq.groupby("ticker"):
        s = g.set_index("date")["adj_close"].astype(float)
        s = s.where(s > 0)
        lr = np.log(s).diff()
        if g["source"].str.startswith("cotahist").all():
            lr[lr.abs() > 0.5] = 0.0
        s = np.exp(lr.fillna(0).cumsum()) * s.dropna().iloc[0]
        SPX[:, TI[t]] = on_grid(s)
        tv = (g.set_index("date")["close"] * g.set_index("date")["volume"]).rolling(63, min_periods=20).median()
        ADTV[:, TI[t]] = on_grid(tv)
    mp = pd.read_csv(RDATA / "equity_map.csv", dtype=str)
    mp = mp[mp["ticker"].notna() & (mp["mapping_type"] != "none") & mp["ticker"].isin(TI)].copy()
    mp["_c"] = mp["confidence"].map({"high": 0, "med": 1, "low": 2}).fillna(1)
    cands = {k: list(zip(v["ticker"], v["mapping_type"], v["confidence"]))
             for k, v in mp.sort_values(["cnpj8", "_c"]).groupby("cnpj8")}
    _MEM["eqgrid"] = (tick, TI, SPX, ADTV, cands)
    return _MEM["eqgrid"]


def index_levels() -> pd.DataFrame:
    """Daily levels of IDA-DI, IDA-Geral, IDA-IPCA and Ibovespa (cached). Ibovespa from brapi (via the HY-lab cache
    research/out/hy_indices.pkl when present, read-only)."""
    if "idx" in _MEM:
        return _MEM["idx"]
    path = CACHE / "indices.pkl"
    if path.exists() and time.time() - path.stat().st_mtime < 24 * 3600:
        _MEM["idx"] = pd.read_pickle(path)
        return _MEM["idx"]
    from rfmonitor.history import ida
    out = {}
    for nm in ("IDADI", "IDAGERAL", "IDAIPCA"):
        try:
            out[nm] = ida(nm)["index"]
        except Exception as e:  # pragma: no cover
            _log(f"IDA {nm} unavailable: {e}")
    ib = None
    hy = _ROOT / "research" / "out" / "hy_indices.pkl"
    if hy.exists():
        ib = pd.read_pickle(hy).get("^BVSP")
    if ib is None or ib.index.max() < pd.Timestamp.today() - pd.Timedelta(days=10):
        try:
            from rfmonitor.sources import brapi
            h = brapi.history("^BVSP", "10y")
            ib = pd.Series({pd.Timestamp(d): c for d, c, _ in h}).sort_index()
        except Exception as e:  # pragma: no cover
            _log(f"Ibovespa unavailable: {e}")
    if ib is not None:
        out["IBOV"] = ib
    df = pd.DataFrame(out).sort_index()
    df.to_pickle(path)
    _MEM["idx"] = df
    return df


def ida_regime(threshold: float = 0.0, window: int = 21) -> pd.Series:
    """1.0 when IDA-DI's trailing `window`-day excess return over CDI > threshold (known at the close), else 0.0,
    on the grid. This is P4's regime overlay (lab_daily mkt_mom_21 definition)."""
    ix = index_levels()["IDADI"]
    cdi = _core()["cdi_daily_raw"]
    mom = (ix.pct_change() - cdi.reindex(ix.index)).rolling(window).sum()
    s = mom.reindex(mom.index.union(days())).ffill().reindex(days())
    return (s > threshold).astype(float).rename("ida_regime")


def _decision_positions(freq: str) -> list[int]:
    dd = days()
    s = pd.Series(np.arange(len(dd)), index=dd)
    s = s[dd >= PANEL_START]
    if freq == "M":
        g = s.groupby([s.index.year, s.index.month]).min()
    elif freq == "W":
        iso = s.index.isocalendar()
        g = s.groupby([iso["year"].to_numpy(), iso["week"].to_numpy()]).min()
    else:
        raise ValueError("freq must be 'M' or 'W'")
    return sorted(int(v) for v in g.to_numpy())


def _build_panel(freq: str) -> pd.DataFrame:
    C = _core()
    dd, codes, ND = C["days"], C["codes"], len(C["days"])
    decs = _decision_positions(freq)
    _log(f"building panel {freq}: {len(decs)} decision dates")
    g = pd.read_pickle(HIST / "lab_daily.pkl")
    g["pos"] = dd.get_indexer(g["day"])
    g = g[g["pos"].isin(decs)].copy()
    g = g.drop(columns=["cnpj"])
    for c in ("codigo", "cnpj8", "kind", "peer"):
        g[c] = g[c].astype("str")
    g["b"] = codes.get_indexer(g["codigo"])
    g["dpos"] = g["pos"]
    g = g.drop(columns=["pos"]).reset_index(drop=True)
    g["univ"] = g["eligible"] & g["cdi_bps"].notna()

    # --- execution and forward targets (patched rate-hedged excess over CDI, from the entry trade)
    ent = np.full(len(g), -1)
    for p, idx in g.groupby("dpos").groups.items():
        ep = exec_pos(int(p))
        ent[g.index.get_indexer(idx)] = ep[g.loc[idx, "b"].to_numpy()]
    g["entry_pos"] = ent
    g["executed"] = ent >= 0
    bb = g["b"].to_numpy()
    for sc, suf in (("base", ""), ("rec40", "_rec40")):
        L = _LC(sc)
        for H in HZ:
            ok = (ent >= 0) & (ent + H <= ND)
            v = np.full(len(g), np.nan)
            v[ok] = np.expm1(L[ent[ok] + H, bb[ok]] - L[ent[ok], bb[ok]])
            g[f"fwd_{H}{suf}"] = v
    for H in HZ:
        g[f"lab_end_{H}"] = np.where(ent >= 0, ent + H, g["dpos"] + ENTRY_MAX + H)  # grid pos where label is known
        g[f"dok_{H}"] = g["dpos"] + ENTRY_MAX + H <= ND                       # every bond's label complete
    g["dist_stop_LOOKAHEAD"] = g["codigo"].isin(set(C["dist_codes"]))

    # --- weekly-lab extras (CVM filings, trade counts, spread changes): row of the last Monday <= decision day
    try:
        w = pd.read_pickle(HIST / "lab_weekly.pkl")[["codigo", "week"] + WEEKLY_COLS]
        w = w.sort_values("week")
        w["codigo"] = w["codigo"].astype("str")
        m = pd.merge_asof(g[["codigo", "day"]].reset_index().sort_values("day"), w, left_on="day", right_on="week",
                          by="codigo", direction="backward", tolerance=pd.Timedelta(days=8))
        m = m.set_index("index").reindex(g.index)
        for c in WEEKLY_COLS:
            g["wk_" + c if c == "maturity" else c] = m[c].to_numpy()
        g = g.rename(columns={"wk_maturity": "maturity"})
        del w
    except Exception as e:  # pragma: no cover
        _log(f"weekly extras failed: {e}")
    g["years_to_mat"] = (pd.to_datetime(g["maturity"]) - g["day"]).dt.days / 365.25

    # --- fundamentals (strict availability), composite quality among the universe rows
    fu = _fundamentals(strict=True)
    m = pd.merge_asof(g[["day", "cnpj8"]].reset_index().sort_values("day"), fu, left_on="day", right_on="avail",
                      by="cnpj8", direction="backward")
    m = m.set_index("index").reindex(g.index)
    stale = (m["day"] - m["period_end"]).dt.days > FUND_STALE_DAYS
    for c in FCOLS + ["f_is_parent"]:
        g[c] = m[c].where(~stale).to_numpy()
    g["f_age_days"] = (g["day"] - m["period_end"]).dt.days.where(~stale).to_numpy()
    g["f_source"] = m["f_source"].where(~stale).to_numpy()
    g["covered"] = g["f_lev"].notna() & g["f_cov"].notna()
    for c in FCOLS:
        g.loc[~g["covered"], c] = np.nan
    U = g[g["univ"]]
    rk = pd.DataFrame({c: U.groupby("day")[c].rank(pct=True) * s for c, s in QSIGN.items()})
    g["f_quality"] = rk.mean(axis=1, skipna=True).reindex(g.index).where(g["covered"] & g["univ"])
    thr = g[g["univ"]].groupby("day")["f_quality"].quantile(0.2)
    g["worstQ"] = g["covered"] & g["univ"] & (g["f_quality"] <= g["day"].map(thr))

    # --- issuer / parent equity features (ticker chosen point-in-time as in the HY lab)
    tick, TI, SPX, ADTV, cands = _equity_grid()
    pairs = g[["cnpj8", "dpos"]].drop_duplicates()
    rows = []
    lr = np.diff(np.log(SPX), axis=0, prepend=np.nan)
    for c8, p in zip(pairs["cnpj8"], pairs["dpos"]):
        j, mt, cf = -1, "none", None
        for t, mtt, cff in cands.get(c8, []):
            jj = TI[t]
            if not np.isnan(SPX[p, jj]) and np.isfinite(SPX[max(0, p - 252):p, jj]).sum() >= 150:
                j, mt, cf = jj, mtt, cff
                break
        if j < 0:
            rows.append((c8, p, None, "none", None) + (np.nan,) * 8)
            continue
        px = SPX[:, j]

        def ret(n):
            return px[p] / px[p - n] - 1 if p - n >= 0 and px[p - n] > 0 else np.nan
        win = lr[max(1, p - 62):p + 1, j]
        vol = np.nanstd(win) * np.sqrt(252) if np.isfinite(win).sum() >= 40 else np.nan
        hi = np.nanmax(px[max(0, p - 251):p + 1])
        rows.append((c8, p, tick[j], mt, cf, ret(5), ret(21), ret(63), ret(126), ret(252), vol, px[p] / hi - 1,
                     ADTV[p, j]))
    E = pd.DataFrame(rows, columns=["cnpj8", "dpos", "eq_ticker", "eq_map_type", "eq_confidence", "eq_r5", "eq_r21",
                                    "eq_r63", "eq_r126", "eq_r252", "eq_vol63", "eq_dd252", "eq_adtv"])
    E["cnpj8"] = E["cnpj8"].astype("str")
    g = g.merge(E, on=["cnpj8", "dpos"], how="left")
    g["listed"] = g["eq_ticker"].notna()

    # --- market features (known at the close of the decision day)
    ix = index_levels()
    cdi = pd.Series(C["CDI"], index=dd)
    cdi_c = (1 + cdi).cumprod()
    mk = pd.DataFrame(index=dd)
    for nm in ("IDADI", "IBOV"):
        if nm in ix:
            lv = ix[nm].reindex(ix[nm].index.union(dd)).ffill(limit=5).reindex(dd)
            for n in (21, 63):
                mk[f"{nm.lower()}_x{n}"] = lv / lv.shift(n) - cdi_c / cdi_c.shift(n)
    mk["ida_regime"] = ida_regime().to_numpy()
    mk["cdi_ann"] = (1 + cdi.where(cdi > 0).ffill()) ** 252 - 1
    disp = g[g["univ"]].groupby("day")["cdi_bps"].agg(univ_cdi_med="median",
                                                      univ_cdi_iqr=lambda s: s.quantile(.75) - s.quantile(.25))
    g = g.join(mk, on="day").join(disp, on="day")

    # --- sector
    try:
        sec = pd.read_csv(RDATA / "issuer_sectors.csv", dtype=str)[["cnpj8", "sector"]].drop_duplicates("cnpj8")
        sec = sec.astype("str")
        g = g.merge(sec, on="cnpj8", how="left")
    except Exception:  # pragma: no cover
        g["sector"] = np.nan
    g["sector"] = g["sector"].fillna("unknown")

    # --- convenience flags (baseline rules)
    nz = lambda s, v: s.fillna(v)
    g["p4f"] = ~(nz(g["resid_z"], 0) <= -1.5) & (nz(g["press_neg_30d"], 0) < 1)
    g["p4"] = g["univ"] & (nz(g["cdi_pct"], 1) <= 0.3) & g["p4f"]
    g["p4q"] = g["p4"] & ~g["worstQ"]
    g["rich"] = nz(g["resid_z"], 0) <= -1.5
    g["decision_end_ok"] = True
    g = g.sort_values(["day", "codigo"]).reset_index(drop=True)
    return g


def load_panel(freq: str = "M", holdout: bool = False, universe_only: bool = False) -> pd.DataFrame:
    """Point-in-time decision panel. One row per (decision day, bond on the daily grid) at the CLOSE of the first
    grid day of each month ('M', 58+ dates) or week ('W'); decisions from 2021-03 (backtests start 2022-01 by default).

    Columns (main): codigo, day, cnpj8, kind, cdi_bps (CDI+ equiv. spread), cdi_pct (rank, 0=highest carry),
    dur, resid_bps / resid_z (vs peer curve), incent, ratio, age, fresh, eligible, univ (= eligible & cdi_bps),
    press_neg_7d/30d, distress_2d, fact_2d, rat_days_since_down, eq_ret_1w/4w, mkt_mom_21 (lab_daily);
    weekly-lab extras (trades_30d, own_z, d_spread_1w/4w, d_ratio_4w, carry_per_dur, n_fact_*, n_distress_*,
    n_deb_mtg_*, n_rating_*, n_oficio_*, days_since_distress, news_any_30d, distress_90d, mkt_distress_z, maturity,
    years_to_mat); fundamentals f_* (strict dates), covered, f_quality, worstQ; equity eq_ticker, eq_map_type,
    eq_r5/21/63/126/252, eq_vol63, eq_dd252, eq_adtv, listed; market idadi_x21/63, ibov_x21/63, ida_regime, cdi_ann,
    univ_cdi_med/iqr; sector; flags p4f, p4, p4q, rich.
    Targets: fwd_{21,63,126,252} (+ _rec40): compounded patched rate-hedged excess over CDI from the entry trade
    (entry_pos = first trade after the decision within 20 bdays; NaN if never executed -> treat as cash = 0);
    lab_end_H = grid position at which the label is known (use `lab_end_H <= dpos` for walk-forward training);
    dok_H = every bond's label is complete for that date.
    holdout=False: decisions >= HOLDOUT are dropped AND labels whose window reaches HOLDOUT are set to NaN.
    dist_stop_LOOKAHEAD is a look-ahead flag for diagnostics only — never use it as a feature."""
    key = ("panel", freq)
    if key not in _MEM:
        path = CACHE / f"panel_{freq}.pkl"
        P = None
        if path.exists():
            P = pd.read_pickle(path)
            if P.attrs.get("version") != _VERSION:
                P = None
        if P is None:
            P = _build_panel(freq)
            P.attrs["version"] = _VERSION
            P.to_pickle(path)
        _MEM[key] = P
    P = _MEM[key]
    if not holdout:
        hp = _hpos()
        P = P[P["day"] < HOLDOUT].copy()
        for H in HZ:
            bad = P[f"lab_end_{H}"] >= hp
            for c in (f"fwd_{H}", f"fwd_{H}_rec40"):
                P.loc[bad, c] = np.nan
            P[f"dok_{H}"] = P[f"dok_{H}"] & (P["dpos"] + ENTRY_MAX + H < hp)
    else:
        P = P.copy()
    if universe_only:
        P = P[P["univ"]]
    return P


# =====================================================================================================================
# 3) portfolio construction
# =====================================================================================================================
def cap_weights(issuer, cap: float = ISSUER_CAP, w0=None) -> np.ndarray:
    """Equal weight (or w0) with an issuer cap, redistributing the excess to uncapped issuers."""
    n = len(issuer)
    if n == 0:
        return np.array([])
    w = np.full(n, 1.0 / n) if w0 is None else np.asarray(w0, float) / np.sum(w0)
    if cap is None or cap >= 1:
        return w
    iss = pd.Series(np.asarray(issuer))
    for _ in range(30):
        tot = pd.Series(w).groupby(iss).transform("sum").to_numpy()
        over = tot > cap + 1e-12
        if not over.any():
            break
        w[over] *= cap / tot[over]
        free = ~over
        if not free.any() or w[free].sum() <= 0:
            break
        w[free] *= (1 - w[over].sum()) / w[free].sum()
    return w / w.sum()


def _targets(signal, panel: pd.DataFrame, top_frac: float, issuer_cap: float, as_weights: bool,
             min_names: int, universe: str) -> dict:
    """-> {dpos: pd.Series(weight, index=b)} for every decision date in `panel`."""
    X = panel if universe == "all" else panel[panel["univ"]]
    if isinstance(signal, pd.DataFrame):
        col = next(c for c in ("weight", "score", "select") if c in signal.columns)
        s = signal[["day", "codigo", col]].drop_duplicates(["day", "codigo"])
        X = X.merge(s, on=["day", "codigo"], how="left")
        as_weights = as_weights or col == "weight"
        fn = lambda x: x[col].to_numpy()
    elif isinstance(signal, str):
        fn = lambda x: x[signal].to_numpy()
    else:
        fn = signal
    out = {}
    for p, x in X.groupby("dpos", sort=True):
        v = fn(x)
        if v is None:
            continue
        v = np.asarray(v)
        if v.dtype == bool:
            sel = v
            w0 = None
        elif as_weights:
            v = np.nan_to_num(v.astype(float), nan=0.0)
            sel = v > 0
            w0 = v[sel]
        else:
            v = v.astype(float)
            ok = np.isfinite(v)
            if ok.sum() == 0:
                continue
            thr = np.quantile(v[ok], 1 - top_frac)
            sel = ok & (v >= thr)
            w0 = None
        if sel.sum() < min_names:
            out[int(p)] = pd.Series(dtype=float)
            continue
        xs = x[sel]
        w = cap_weights(xs["cnpj8"].to_numpy(), issuer_cap, w0)
        out[int(p)] = pd.Series(w, index=xs["b"].to_numpy()).groupby(level=0).sum()
    return out


def _filter_rebalance(T: dict, rebalance: str | None) -> dict:
    if rebalance in (None, "W"):
        return T
    dd = days()
    if rebalance == "M":
        keep, seen = {}, set()
        for p in sorted(T):
            k = (dd[p].year, dd[p].month)
            if k not in seen:
                seen.add(k)
                keep[p] = T[p]
        return keep
    if rebalance == "Q":
        keep, seen = {}, set()
        for p in sorted(T):
            k = (dd[p].year, (dd[p].month - 1) // 3)
            if k not in seen:
                seen.add(k)
                keep[p] = T[p]
        return keep
    raise ValueError("rebalance must be None/'W'/'M'/'Q'")


# =====================================================================================================================
# 4) book engines
# =====================================================================================================================
def _tranche_book(T: dict, hold: int, cost_bps: float, scenario: str, exit_flags: dict | None, t_end: int):
    """Overlapping tranches (one per decision), each held `hold` bdays from its entry trade, buy-and-hold drift;
    daily book = sum of live tranche P&L / min(M, #tranches started) with M = hold / rebalance spacing
    (identical to run_hy_lab.run_book). Cost = book-level turnover of averaged tranche weights x cost/2 at each
    decision, + early exits (exit_flags) x cost/2."""
    R = _Rmat(scenario)
    ND = len(days())
    decs = sorted(T)
    step = int(np.median(np.diff(decs))) if len(decs) > 1 else 21
    M = max(int(round(hold / step)), 1)
    pnl = np.zeros(ND)
    expo = np.zeros(ND)
    cost = np.zeros(ND)
    ncoh = np.zeros(ND)
    hold_rows = []
    for p in decs:
        w = T[p]
        ncoh[p + 1:] += 1
        if not len(w):
            continue
        ep = exec_pos(p)
        ex_after = None
        if exit_flags is not None:
            ex_after = [(q, exit_flags[q]) for q in sorted(exit_flags) if q > p]
        for b, wi in w.items():
            k = int(ep[b])
            hold_rows.append((p, b, wi, k))
            if k < 0 or k >= t_end:
                continue
            e = min(k + hold, ND, t_end)
            early = False
            if ex_after:
                for q, fl in ex_after:
                    if q >= e:
                        break
                    if q > k and b in fl:
                        xq = exec_pos(q, SELL_MAX)[b]
                        xq = xq if xq >= 0 else min(q + SELL_MAX, ND - 1)
                        if xq < e:
                            e, early = xq, True
                        break
            if e <= k:
                continue
            r = R[k:e, b]
            gv = np.r_[1.0, np.cumprod(1 + r[:-1])]
            pnl[k:e] += wi * gv * r
            expo[k:e] += wi * gv
            if early:
                cost[e] += wi * gv[-1] * (1 + r[-1]) * cost_bps / 2 / 1e4 / M
    div = np.maximum(np.minimum(M, ncoh), 1)
    pnl /= div
    expo /= div
    # book-level turnover (averaged live tranche weights)
    prev, turn = None, np.zeros(ND)
    for i, p in enumerate(decs):
        live = [T[x] for x in decs[max(0, i - M + 1): i + 1]]
        live = [s for s in live if len(s)]
        book = (pd.concat(live, axis=1).fillna(0).sum(axis=1) / M) if live else pd.Series(dtype=float)
        if prev is not None:
            to = book.sub(prev, fill_value=0).abs().sum()
            turn[p] = to
            cost[p] += to * cost_bps / 2 / 1e4
        prev = book
    return pnl, cost, expo, turn, hold_rows


def _rebal_book(T: dict, cost_bps: float, scenario: str, t_end: int, exit_flags: dict | None, present: dict,
                issuer: np.ndarray, issuer_cap: float, band: float = 0.25):
    """Rebalancing engine (hold=None). Orders execute at the bond's next fresh trade: buys within 20 bdays (else
    expire -> cash) and never beyond available cash (gross exposure <= 1); sells within 20 bdays, else at the last
    mark on day +20. Positions drift (buy-and-hold) between trades; NAV compounds.
    - exit_flags None ('replace'): at each target date the book becomes T[p]; kept names are re-sized only if they
      deviate more than `band` (relative) from target.
    - exit_flags given ('sticky', like the original daily P4): held names are kept until flagged by the exit signal
      (checked on every panel date) or until they leave the grid; at each target date new names from T[p] are
      added and the whole book is re-targeted to equal weight with the issuer cap (tolerance band as above)."""
    R = _Rmat(scenario)
    ND = len(days())
    tdates = set(p for p in T if p < t_end)
    xdates = set(p for p in (exit_flags or {}) if p < t_end)
    events = tdates | xdates
    v = np.zeros(R.shape[1])
    nav = 1.0
    pnl, cost, expo, turn = np.zeros(ND), np.zeros(ND), np.zeros(ND), np.zeros(ND)
    sched = {}
    pend = {}           # b -> scheduled exec pos (so a re-decision can cancel it)
    hold_rows = []
    t0 = min(tdates) if tdates else t_end

    def order(b, tf, k):
        if b in pend:
            sched.get(pend[b], {}).pop(b, None)
        sched.setdefault(k, {})[b] = tf
        pend[b] = k

    for t in range(t0, t_end):
        c = 0.0
        if t in sched:
            orders = sched.pop(t)
            for b, tf in sorted(orders.items(), key=lambda z: z[1]):      # sells first -> frees cash
                pend.pop(b, None)
                tgt = tf * nav
                if tgt > v[b]:
                    tgt = v[b] + min(tgt - v[b], max(nav - v.sum(), 0.0))
                tr = abs(tgt - v[b])
                c += tr * cost_bps / 2 / 1e4
                turn[t] += tr / nav
                v[b] = tgt
        r = R[t]
        gain = float(v @ r)
        pnl[t] = gain / nav
        cost[t] = c / nav
        expo[t] = v.sum() / nav
        nav += gain - c
        v *= (1 + r)
        if t not in events:
            continue
        held = set(np.nonzero(v > 1e-12)[0].tolist()) | {b for b, k in pend.items() if sched.get(k, {}).get(b, 0) > 0}
        eps = exec_pos(t, SELL_MAX)
        ep = exec_pos(t)
        sell = set()
        if exit_flags is not None:
            fl = exit_flags.get(t, set())
            pr = present.get(t)
            sell = {b for b in held if b in fl or (pr is not None and b not in pr)}
        if t in tdates:
            w = T[t]
            if exit_flags is None:
                target = w
                sell |= {b for b in held if b not in w.index}
            else:
                keep = sorted((held - sell) | set(int(z) for z in w.index))
                target = pd.Series(cap_weights(issuer[keep], issuer_cap), index=keep) if keep else pd.Series(dtype=float)
            for b, wi in target.items():
                b = int(b)
                hold_rows.append((t, b, float(wi), int(ep[b])))
                cur = v[b] / nav
                if cur > 0 and abs(cur / wi - 1) <= band:
                    if b in pend:                      # cancel a pending order, keep as is
                        sched.get(pend.pop(b), {}).pop(b, None)
                    continue
                if cur > wi:
                    k = int(eps[b]) if eps[b] >= 0 else min(t + SELL_MAX, ND - 1)
                    order(b, float(wi), k)
                elif ep[b] >= 0:
                    order(b, float(wi), int(ep[b]))
        for b in sell:
            if v[b] <= 1e-12:
                if b in pend:
                    sched.get(pend.pop(b), {}).pop(b, None)
                continue
            k = int(eps[b]) if eps[b] >= 0 else min(t + SELL_MAX, ND - 1)
            order(b, 0.0, k)
    return pnl, cost, expo, turn, hold_rows


def _issuer_of_b() -> np.ndarray:
    if "iss" not in _MEM:
        P = load_panel("W", holdout=True)
        m = P.drop_duplicates("b", keep="last").set_index("b")["cnpj8"]
        arr = np.array(["?"] * len(_core()["codes"]), dtype=object)
        arr[m.index.to_numpy()] = m.to_numpy()
        _MEM["iss"] = arr
    return _MEM["iss"]


def backtest(signal, freq: str = "M", rebalance: str | None = None, hold: int | None = 126, top_frac: float = 0.2,
             as_weights: bool = False, issuer_cap: float = ISSUER_CAP, cost_bps: float = 25.0, overlay=None,
             exit_signal=None, scenario: str = "base", holdout: bool = False, start=START, panel=None,
             min_names: int = MIN_NAMES, universe: str = "univ", name: str | None = None) -> dict:
    """Backtest a selection rule on the point-in-time panel.

    signal: callable(x) on the decision cross-section x (universe rows of one date) returning a bool mask,
            a float score (top `top_frac` taken) or weights (as_weights=True); OR a column name of the panel;
            OR a DataFrame with columns day, codigo and one of weight / score / select.
    freq: panel used for decisions ('M' or 'W'); rebalance: None (every panel date) / 'M' / 'Q' subsample.
    hold: int -> overlapping tranches held `hold` bdays (the selection/HY-lab design);
          None -> replace-the-book at each rebalance (live style).
    overlay: None | 'ida' (P4 regime: IDA-DI 21d excess momentum > 0 else CDI) | pd.Series on grid days (0..1,
             value at close t applies to the t->t+1 return). Switching costs |Δoverlay| x exposure x cost/2.
    exit_signal: None | panel bool column name | callable(x)->bool, evaluated on ALL grid rows of every panel date.
             Tranche mode: a held bond flagged on a later panel date is sold at its next fresh trade (slot -> cash).
             hold=None: 'sticky' book - names are held until flagged (or off the grid), new names added at each
             rebalance, book re-targeted to capped equal weight with a 25% tolerance band (original daily-P4 style).
    scenario: 'base' | 'rec40' (bonds that stop trading below 0.90 jump to 40% of par at their last mark).
    Returns dict: daily (net excess over CDI, pd.Series on grid days), gross, cost, exposure, turnover_ann (one-way,
    fraction of book per year), cost_ann_%, holdings (DataFrame day, codigo, weight, entry_day), n_avg, params.
    """
    t0 = time.time()
    C = _core()
    dd, codes, ND = C["days"], C["codes"], len(C["days"])
    P = panel if panel is not None else load_panel(freq, holdout=holdout)
    t_end = ND if holdout else _hpos()
    P = P[(P["day"] >= pd.Timestamp(start)) & (P["dpos"] < t_end)]
    T = _targets(signal, P, top_frac, issuer_cap, as_weights, min_names, universe)
    T = _filter_rebalance(T, rebalance)
    exit_flags = None
    if exit_signal is not None:
        exit_flags = {}
        for p, x in P.groupby("dpos"):   # all grid rows (a held bond may have become ineligible)
            f = x[exit_signal].to_numpy() if isinstance(exit_signal, str) else np.asarray(exit_signal(x))
            exit_flags[int(p)] = set(x["b"].to_numpy()[np.asarray(f, dtype=bool)].tolist())
    if hold is not None:
        pnl, cost, expo, turn, rows = _tranche_book(T, hold, cost_bps, scenario, exit_flags, t_end)
    else:
        present = {int(p): set(x.to_numpy().tolist()) for p, x in P.groupby("dpos")["b"]}
        iss = _issuer_of_b()
        pnl, cost, expo, turn, rows = _rebal_book(T, cost_bps, scenario, t_end, exit_flags, present, iss, issuer_cap)
    gross = pnl.copy()
    if overlay is not None:
        ov = ida_regime() if (isinstance(overlay, str) and overlay == "ida") else \
            pd.Series(overlay).reindex(dd).ffill().fillna(1.0)
        ov = ov.to_numpy()
        gross = gross * ov
        sw = np.abs(np.diff(np.r_[ov[0], ov]))
        cost = cost * ov + sw * expo * cost_bps / 2 / 1e4
        expo = expo * ov
    first = min(T) + 1 if T else 0
    idx = dd[first:t_end]
    net = pd.Series((gross - cost)[first:t_end], index=idx, name=name or "strategy")
    H = pd.DataFrame(rows, columns=["dpos", "b", "weight", "entry_pos"])
    if len(H):
        H["day"] = dd[H["dpos"].to_numpy()]
        H["codigo"] = codes[H["b"].to_numpy()]
        H["entry_day"] = [dd[k] if k >= 0 else pd.NaT for k in H["entry_pos"]]
        H = H[["day", "codigo", "weight", "entry_day"]]
    yrs = len(idx) / 252
    res = {"daily": net, "gross": pd.Series(gross[first:t_end], index=idx), "cost": pd.Series(cost[first:t_end], index=idx),
           "exposure": pd.Series(expo[first:t_end], index=idx), "holdings": H,
           "turnover_ann": float(turn[first:t_end].sum() / yrs) if yrs else np.nan,
           "cost_ann_%": float(cost[first:t_end].sum() / yrs * 100) if yrs else np.nan,
           "n_avg": float(np.mean([len(v) for v in T.values()])) if T else 0.0,
           "n_decisions": len(T), "runtime_s": round(time.time() - t0, 2),
           "params": dict(freq=freq, rebalance=rebalance, hold=hold, top_frac=top_frac, issuer_cap=issuer_cap,
                          cost_bps=cost_bps, overlay=None if overlay is None else str(overlay)[:20],
                          scenario=scenario, holdout=holdout, start=str(pd.Timestamp(start).date()))}
    return res


# ---------------------------------------------------------------------------------------------------------------------
BASELINES = {
    "U": lambda x: x["univ"].to_numpy(),
    "P4": lambda x: x["p4"].to_numpy(),
    "P4Q": lambda x: x["p4q"].to_numpy(),
    "P4_live": lambda x: x["p4"].to_numpy(),
}


def baseline(name: str = "P4Q", **kw) -> dict:
    """Baselines on the same engine. name: 'U' (eligible universe), 'P4' (top 30% CDI+ carry, not rich, no negative
    press 30d), 'P4Q' (P4 minus worst-quintile composite quality; uncovered kept), 'P4_live' (weekly decisions, sticky book:
    buy P4 names, sell only when rich (resid_z <= -1.5) or off the grid, IDA-DI regime overlay -> closest to the
    original lab_daily 'P4 semanal'),
    'CDI' (zeros). Defaults: monthly decisions, 126-bday tranches, 25 bps — the HY-lab 'book' design.
    kwargs are passed to backtest (e.g. cost_bps=50, scenario='rec40', holdout=True, hold=63)."""
    if name == "CDI":
        dd = days()
        t_end = len(dd) if kw.get("holdout") else _hpos()
        i0 = dd.searchsorted(pd.Timestamp(kw.get("start", START)))
        return {"daily": pd.Series(0.0, index=dd[i0 + 1:t_end], name="CDI")}
    if name == "P4_live":
        kw = {"freq": "W", "hold": None, "overlay": "ida", "exit_signal": "rich", **kw}
    key = ("baseline", name, tuple(sorted((k, str(v)) for k, v in kw.items())))
    if key not in _MEM:
        _MEM[key] = backtest(BASELINES[name], name=name, **kw)
    return _MEM[key]


def placebo(signal, n: int = 20, seed: int = 0, **kw) -> dict:
    """Null distribution: the same number of names (and the same weights) per date, drawn at random from the
    universe of that date (issuer cap re-applied). Returns {'ann_excess_%': array, 'mean', 'p95'} over n runs."""
    kw = dict(kw)
    freq = kw.get("freq", "M")
    P = kw.pop("panel", None)
    if P is None:
        P = load_panel(freq, holdout=kw.get("holdout", False))
    base = backtest(signal, panel=P, **kw)
    H = base["holdings"]
    U = P[P["univ"]]
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        rows = []
        for d, h in H.groupby("day"):
            cand = U.loc[U["day"] == d, "codigo"].to_numpy()
            pick = rng.choice(cand, size=min(len(h), len(cand)), replace=False)
            rows.append(pd.DataFrame({"day": d, "codigo": pick, "select": True}))
        sig = pd.concat(rows)
        r = backtest(sig, panel=P, **{k: v for k, v in kw.items() if k not in ("top_frac", "as_weights")})
        out.append(stats(r["daily"])["ann_excess_%"])
    out = np.array(out)
    return {"ann_excess_%": out, "mean": float(out.mean()), "p95": float(np.percentile(out, 95)),
            "actual": stats(base["daily"])["ann_excess_%"]}


# =====================================================================================================================
# 5) statistics
# =====================================================================================================================
def nw_t(x, lags: int) -> float:
    x = np.asarray(pd.Series(x).dropna(), float)
    n = len(x)
    if n < 4:
        return np.nan
    e = x - x.mean()
    v = e @ e / n
    for l in range(1, min(lags, n - 1) + 1):
        v += 2 * (1 - l / (lags + 1)) * (e[l:] @ e[:-l]) / n
    return float(x.mean() / np.sqrt(v / n)) if v > 0 else np.nan


def monthly(excess: pd.Series) -> pd.Series:
    """Monthly excess over CDI from a daily excess series: prod(1+cdi+ex)/prod(1+cdi) - 1 per calendar month."""
    s = excess.dropna()
    c = cdi_daily().reindex(s.index).fillna(0)
    tot = (1 + c + s).groupby([s.index.year, s.index.month]).prod()
    cc = (1 + c).groupby([s.index.year, s.index.month]).prod()
    m = tot / cc - 1
    m.index = pd.to_datetime([f"{y}-{mm:02d}-01" for y, mm in m.index])
    return m


def _mdd(m: pd.Series) -> float:
    eq = (1 + m).cumprod()
    return float((eq / eq.cummax() - 1).min())


def stats(series: pd.Series, bench: pd.Series | None = None, lag: int = 6, holdout: bool = False) -> dict:
    """Stats of a DAILY excess-over-CDI series, computed on calendar months (smooth marks: daily stats would be
    even more inflated). lag = Newey-West lags in months (6 ~ 126-bday overlapping tranches).
    holdout=False truncates at HOLDOUT; holdout='only' keeps only >= HOLDOUT (report once, at the end).
    Returns ann excess (mean monthly x12, %), vol, sharpe, max_dd, worst month, hit, skew, t_nw, halves
    (2022-23 / 2024-25), and if bench given: paired diff (ann %), its NW t, two-sided p, halves of the diff."""
    s = series.dropna()
    if holdout is False:
        s = s[s.index < HOLDOUT]
    elif holdout == "only":
        s = s[s.index >= HOLDOUT]
    m = monthly(s)
    out = {"ann_excess_%": m.mean() * 1200, "vol_%": m.std() * np.sqrt(12) * 100,
           "sharpe": m.mean() * 12 / (m.std() * np.sqrt(12)) if m.std() > 0 else np.nan,
           "max_dd_%": _mdd(m) * 100, "worst_month_%": m.min() * 100, "hit": float((m > 0).mean()),
           "skew": float(_sst.skew(m)) if len(m) > 2 else np.nan, "t_nw": nw_t(m, lag),
           "h1_2022_23_%": m[m.index < SPLIT].mean() * 1200,
           "h2_2024_25_%": m[(m.index >= SPLIT) & (m.index < HOLDOUT)].mean() * 1200,
           "cum_%": float((1 + m).prod() - 1) * 100, "n_months": int(len(m)),
           "period": f"{m.index.min():%Y-%m}..{m.index.max():%Y-%m}" if len(m) else ""}
    if bench is not None:
        mb = monthly(bench.reindex(s.index).fillna(0))
        d = (m - mb).dropna()
        t = nw_t(d, lag)
        out.update({"diff_ann_%": d.mean() * 1200, "diff_t_nw": t,
                    "diff_p": float(2 * (1 - _sst.norm.cdf(abs(t)))) if t == t else np.nan,
                    "diff_h1_%": d[d.index < SPLIT].mean() * 1200,
                    "diff_h2_%": d[(d.index >= SPLIT) & (d.index < HOLDOUT)].mean() * 1200,
                    "diff_hit": float((d > 0).mean()), "diff_te_%": d.std() * np.sqrt(12) * 100})
    return {k: (round(float(v), 4) if isinstance(v, (float, np.floating)) else v) for k, v in out.items()}


def holm(pvals) -> np.ndarray:
    """Holm step-down adjusted p-values (same order as input; NaN kept)."""
    p = np.asarray(pvals, float)
    ok = np.isfinite(p)
    q = p[ok]
    order = np.argsort(q)
    m = len(q)
    adj = np.empty(m)
    run = 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (m - r) * q[i]))
        adj[i] = run
    out = np.full(len(p), np.nan)
    out[ok] = adj
    return out


def halves(series: pd.Series) -> dict:
    m = monthly(series[series.index < HOLDOUT])
    return {"2022_23": m[m.index < SPLIT].mean() * 1200, "2024_25": m[m.index >= SPLIT].mean() * 1200}


def compare(results: dict, bench: str | pd.Series = "P4Q", universe: str | pd.Series = "U", lag: int = 6,
            cost_bps: float = 25.0, **bkw) -> pd.DataFrame:
    """Table: for each {name: backtest-result or daily series}: stats vs CDI, excess vs universe, paired vs bench
    (NW t, p, Holm across the rows given), halves. bench/universe: baseline name or daily series."""
    b = baseline(bench, cost_bps=cost_bps, **bkw)["daily"] if isinstance(bench, str) else bench
    u = baseline(universe, cost_bps=cost_bps, **bkw)["daily"] if isinstance(universe, str) else universe
    rows = {}
    for nm, r in results.items():
        s = r["daily"] if isinstance(r, dict) else r
        a = stats(s, bench=b, lag=lag)
        au = stats(s, bench=u, lag=lag)
        rows[nm] = {"exCDI_%": a["ann_excess_%"], "exU_%": au["diff_ann_%"], "t_vsU": au["diff_t_nw"],
                    "vs_bench_%": a["diff_ann_%"], "t_vs_bench": a["diff_t_nw"], "p_vs_bench": a["diff_p"],
                    "vol_%": a["vol_%"], "sharpe": a["sharpe"], "maxDD_%": a["max_dd_%"],
                    "h1_vs_bench": a["diff_h1_%"], "h2_vs_bench": a["diff_h2_%"],
                    "turnover": r.get("turnover_ann") if isinstance(r, dict) else np.nan,
                    "n_avg": r.get("n_avg") if isinstance(r, dict) else np.nan}
    df = pd.DataFrame(rows).T
    df["p_holm"] = holm(df["p_vs_bench"].astype(float).to_numpy())
    return df


# =====================================================================================================================
# 6) cross-sectional helpers (IC, cohort returns)
# =====================================================================================================================
def ic(panel: pd.DataFrame, feature: str, target: str = "fwd_126", min_n: int = 30, universe: bool = True) -> dict:
    """Per-date Spearman IC of `feature` vs `target` (NaN target of an unexecuted bond -> dropped). NW lag =
    horizon in months. Returns mean, t_nw, n_dates, and the per-date series."""
    X = panel[panel["univ"]] if universe else panel
    H = int(target.split("_")[1])
    X = X[X[f"dok_{H}"] & X[target].notna() & X[feature].notna()]
    per = {}
    for d, x in X.groupby("day"):
        if len(x) >= min_n and x[feature].nunique() > 2:
            per[d] = _sst.spearmanr(x[feature], x[target])[0]
    s = pd.Series(per).sort_index()
    step = max(H // 21, 1) if (s.index.to_series().diff().dt.days.median() or 30) > 20 else max(H // 5, 1)
    return {"mean": float(s.mean()), "t_nw": nw_t(s, step), "n_dates": int(len(s)), "series": s}


def cohort_excess(signal, H: int = 126, freq: str = "M", top_frac: float = 0.2, as_weights: bool = False,
                  issuer_cap: float = ISSUER_CAP, holdout: bool = False, start=START, panel=None) -> pd.Series:
    """Selection-lab style: per decision date, the tranche's H-bday forward return (unexecuted weight = 0) minus the
    equal-weight (capped) universe tranche. Fast (no daily book). Annualise with mean / (H/252)."""
    P = panel if panel is not None else load_panel(freq, holdout=holdout)
    P = P[(P["day"] >= pd.Timestamp(start)) & P[f"dok_{H}"]]
    T = _targets(signal, P, top_frac, issuer_cap, as_weights, MIN_NAMES, "univ")
    Tu = _targets(BASELINES["U"], P, top_frac, issuer_cap, False, MIN_NAMES, "univ")
    C = _core()
    y = P.set_index(["dpos", "b"])[f"fwd_{H}"]
    y = y[~y.index.duplicated()]
    out = {}
    for p, w in T.items():
        if not len(w) or p not in Tu:
            continue
        yy = y.reindex(pd.MultiIndex.from_product([[p], w.index])).fillna(0).to_numpy()
        yu = y.reindex(pd.MultiIndex.from_product([[p], Tu[p].index])).fillna(0).to_numpy()
        out[C["days"][p]] = float(w.to_numpy() @ yy - Tu[p].to_numpy() @ yu)
    return pd.Series(out).sort_index()


# =====================================================================================================================
# 7) curves and plots
# =====================================================================================================================
def total_return_curve(excess: pd.Series, base: float = 100.0) -> pd.Series:
    """Total-return index CDI x (1 + excess), base 100 on the day before the series starts."""
    s = excess.fillna(0)
    c = cdi_daily().reindex(s.index).fillna(0)
    return base * (1 + c + s).cumprod()


def index_excess(name: str = "IDADI") -> pd.Series:
    """Daily excess over CDI of an index (IDADI / IDAGERAL / IDAIPCA / IBOV) on the grid."""
    lv = index_levels()[name]
    dd = days()
    lv = lv.reindex(lv.index.union(dd)).ffill(limit=5).reindex(dd)
    return (lv.pct_change() - cdi_daily()).fillna(0).rename(name)


def plot_curves(strategies: dict, path, title: str = "", refs=("CDI", "U", "P4", "P4Q", "IDADI", "IBOV"),
                holdout: bool = False, start=START, cost_bps: float = 25.0):
    """Two panels: total-return index (CDI x (1+excess), base 100) and cumulative excess vs the universe.
    strategies: {label: backtest result or daily excess series}. refs drawn dashed/grey. Saves PNG to `path`."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ser = {k: (v["daily"] if isinstance(v, dict) else v) for k, v in strategies.items()}
    t0 = min(s.index.min() for s in ser.values())
    t1 = max(s.index.max() for s in ser.values())
    ref = {}
    for r in refs:
        if r == "CDI":
            ref["CDI"] = None
        elif r in ("U", "P4", "P4Q", "P4_live"):
            ref[{"U": "Universe", "P4Q": "P4+Q"}.get(r, r)] = baseline(r, holdout=holdout, start=start,
                                                                       cost_bps=cost_bps)["daily"]
        elif r in ("IDADI", "IBOV", "IDAGERAL", "IDAIPCA"):
            try:
                ref[{"IDADI": "IDA-DI", "IBOV": "Ibovespa"}.get(r, r)] = index_excess(r)
            except Exception:
                pass
    uni = ref.get("Universe", baseline("U", holdout=holdout, start=start, cost_bps=cost_bps)["daily"])
    fig, axs = plt.subplots(1, 2, figsize=(15, 5.8), dpi=110)
    idx = days()[(days() >= t0) & (days() <= t1)]
    styles = {"CDI": ("k", ":"), "Universe": ("#7f7f7f", "--"), "P4": ("#1f77b4", "--"), "P4+Q": ("#17becf", "--"),
              "IDA-DI": ("#bcbd22", "-."), "Ibovespa": ("#c7c7c7", "-."), "P4_live": ("#9edae5", "--")}
    for nm, s in ref.items():
        s = pd.Series(0.0, index=idx) if s is None else s.reindex(idx).fillna(0)
        col, ls = styles.get(nm, ("grey", "--"))
        tr = total_return_curve(s)
        axs[0].plot(idx, tr, ls, color=col, lw=1.1, label=nm)
        if nm not in ("Universe", "CDI", "Ibovespa"):
            axs[1].plot(idx, ((1 + s - uni.reindex(idx).fillna(0)).cumprod() - 1) * 100, ls, color=col, lw=1.1,
                        label=nm)
    for nm, s in ser.items():
        s = s.reindex(idx).fillna(0)
        axs[0].plot(idx, total_return_curve(s), lw=1.8, label=nm)
        axs[1].plot(idx, ((1 + s - uni.reindex(idx).fillna(0)).cumprod() - 1) * 100, lw=1.8, label=nm)
    for ax in axs:
        ax.grid(alpha=0.25)
        if holdout:
            ax.axvline(HOLDOUT, color="red", lw=0.8, ls=":")
    axs[0].set_title("Total return index: CDI x (1 + excess), base 100", fontsize=10)
    axs[1].set_title("Cumulative excess vs universe (%)", fontsize=10)
    axs[0].set_yscale("log")
    axs[0].legend(fontsize=7, frameon=False)
    axs[1].axhline(0, color="k", lw=0.6)
    fig.suptitle(title or "Strategies vs references (rate-hedged debenture books, net of costs)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


# =====================================================================================================================
# 8) self-test / validation:  python research/nightly/harness.py
# =====================================================================================================================
if __name__ == "__main__":
    import json
    out_dir = _P(__file__).parent / "harness"
    out_dir.mkdir(exist_ok=True)
    tm = {}
    t = time.time(); _core(); tm["core"] = time.time() - t
    for f in ("M", "W"):
        t = time.time(); P = load_panel(f); tm[f"load_panel_{f}"] = time.time() - t
        _log(f"panel {f}: {P.shape}, {P['day'].nunique()} dates, {P.memory_usage(deep=True).sum() / 1e6:.0f} MB")
    res = {}
    for nm in ("U", "P4", "P4Q"):
        t = time.time(); r = baseline(nm); tm[f"baseline_{nm}"] = time.time() - t
        res[nm] = r
    t = time.time(); res["P4_live"] = baseline("P4_live"); tm["baseline_P4_live"] = time.time() - t
    tab = compare(res, bench="P4Q")
    print(tab.round(3).to_string())
    print(json.dumps(tm, indent=1))
