"""Factor zoo: ~80 point-in-time bond / issuer features on the harness decision grid.

All features are known at the CLOSE of the decision day d (the harness executes at the first trade AFTER d):
  * spread / residual / price paths: lab_daily grid values at positions <= d (a grid value is the last SND mark
    <= 14 days old, so it is PIT);
  * returns: harness patched rate-hedged returns R[t] (t -> t+1); at close d only R[.. d-1] is known;
  * liquidity: SND daily trade files (trade date <= d; SND publishes end of day);
  * registry (SND caracteristicas, snapshot of 2026-09): ONLY issuance-time fields (size at issue, guarantee, offering
    regime, coupon frequency, amortisation type, call clause, issue date, original maturity). Quantities currently
    outstanding / redeemed are NOT used (look-ahead);
  * issuer / sector aggregates: cross-sections of the same day;
  * equity: harness price grid for the PIT-chosen ticker (closes <= d); fundamentals: harness f_* (strict CVM dates).

Public entry points:
  build_features(force=False) -> DataFrame [day, codigo, <features>] for every M and W decision date (cached pkl)
  attach(panel)               -> panel with the features merged (day, codigo)
  signals(panel)              -> DataFrame [day, codigo, value] = the factor-zoo composite (see run.py / README)
"""
from __future__ import annotations

import glob
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from research.nightly import harness as H

CACHE = ROOT / "data" / "history" / "nightly" / "factor_zoo"
CACHE.mkdir(parents=True, exist_ok=True)
FEAT_PATH = CACHE / f"features_v{H._VERSION}.pkl"
_T0 = time.time()


def log(*a):
    print(f"[fz +{time.time() - _T0:6.1f}s]", *a, flush=True)


# ---------------------------------------------------------------------------------------------------------------------
def _grid_mats():
    """Pivot lab_daily onto the harness (day x bond) grid for the columns we need."""
    C = H._core()
    dd, codes = C["days"], C["codes"]
    ND, NB = len(dd), len(codes)
    cols = ["codigo", "day", "cdi_bps", "resid_bps", "resid_z", "ratio", "fresh", "cnpj8", "peer", "dur"]
    g = pd.read_pickle(H.HIST / "lab_daily.pkl")[cols]
    pos = dd.get_indexer(g["day"])
    b = codes.get_indexer(g["codigo"])
    M = {}
    for c in ("cdi_bps", "resid_bps", "resid_z", "ratio", "dur"):
        a = np.full((ND, NB), np.nan, dtype=np.float64)
        a[pos, b] = g[c].to_numpy(dtype=float)
        M[c] = a
    fr = np.zeros((ND, NB), dtype=bool)
    fr[pos, b] = g["fresh"].to_numpy(dtype=bool)
    M["fresh"] = fr
    iss = pd.Series(g["cnpj8"].astype(str).to_numpy(), index=b).groupby(level=0).last()
    issuer = np.array(["?"] * NB, dtype=object)
    issuer[iss.index.to_numpy()] = iss.to_numpy()
    M["issuer"] = issuer
    del g
    return M


def _lag(A, h):
    out = np.full_like(A, np.nan)
    out[h:] = A[:-h]
    return out


def _roll(A, w, fn, minp):
    return getattr(pd.DataFrame(A).rolling(w, min_periods=minp), fn)().to_numpy()


def _snd_trades():
    """(ND x NB) matrices of trade count and R$ volume from SND daily trade files (trade date -> grid day on/after)."""
    C = H._core()
    dd, codes = C["days"], C["codes"]
    ND, NB = len(dd), len(codes)
    NT = np.zeros((ND, NB))
    VO = np.zeros((ND, NB))
    cidx = pd.Index(codes)
    for f in sorted(glob.glob(str(H.HIST / "snd_trades_*.csv.gz"))):
        t = pd.read_csv(f, usecols=["date", "codigo", "qty", "trades", "pu_avg"])
        t["date"] = pd.to_datetime(t["date"])
        p = dd.searchsorted(t["date"])          # grid day on/after the trade date
        b = cidx.get_indexer(t["codigo"])
        ok = (b >= 0) & (p < ND)
        np.add.at(NT, (p[ok], b[ok]), t["trades"].to_numpy(float)[ok])
        np.add.at(VO, (p[ok], b[ok]), (t["qty"] * t["pu_avg"]).to_numpy(float)[ok] / 1e6)   # R$ mn
    return NT, VO


def _registry(codes):
    """Issuance-time SND registry fields per bond (static; no outstanding-quantity fields)."""
    from rfmonitor.sources import snd
    t = snd.table()
    t = t.drop_duplicates("Codigo do Ativo").set_index("Codigo do Ativo")
    t = t.reindex(list(codes))
    num = lambda s: pd.to_numeric(s.astype(str).str.replace(",", ".", regex=False), errors="coerce")
    dt = lambda s: pd.to_datetime(s, format="%d/%m/%Y", errors="coerce")
    R = pd.DataFrame(index=t.index)
    size = num(t["Quantidade Emitida"]) * num(t["Valor Nominal na Emissao"])
    R["log_issue_size"] = np.log(size.where(size > 0))
    ga = t["Garantia/Especie"].astype(str)
    R["guar_real"] = (ga == "Real").astype(float)
    R["guar_sub"] = (ga == "Subordinada").astype(float)
    R["guar_flut"] = (ga == "Flutuante").astype(float)
    R["guar_quiro"] = (ga.str.startswith("Quirograf")).astype(float)
    R.loc[t["Garantia/Especie"].isna(), ["guar_real", "guar_sub", "guar_flut", "guar_quiro"]] = np.nan
    reg = t["Registro CVM da Emissao"].astype(str).str.upper()
    R["is_476"] = reg.str.contains("476").astype(float)
    R["call_clause"] = (t["Resgate Antecipado"].astype(str) == "S").astype(float)
    tam = t["Tipo de Amortizacao"].astype(str)
    R["amortizing"] = (~tam.isin(["-", "<NA>", "nan"])).astype(float)
    cada = num(t["Juros Criterio Novo - Cada"])
    un = t["Juros Criterio Novo - Unidade"].astype(str)
    R["coupon_freq_m"] = np.where(un == "DIA", cada / 21, cada)
    R.loc[R["coupon_freq_m"] <= 0, "coupon_freq_m"] = np.nan
    R["emission_no"] = num(t["Emissao"])
    R["issue_date"] = dt(t["Data de Emissao"])
    R["mat_orig"] = dt(t["Data de Vencimento"])
    R["amort_start"] = dt(t["Amortizacao - Carencia"])
    R["issuer_cnpj8"] = t["CNPJ"].astype(str).str.replace(r"\D", "", regex=True).str[:8]
    return R


def build_features(force: bool = False) -> pd.DataFrame:
    if FEAT_PATH.exists() and not force:
        return pd.read_pickle(FEAT_PATH)
    C = H._core()
    dd, codes = C["days"], C["codes"]
    ND, NB = len(dd), len(codes)
    decs = sorted(set(H._decision_positions("M")) | set(H._decision_positions("W")))
    log(f"{len(decs)} decision positions; loading grid")
    M = _grid_mats()
    CS, RB, RZ, RAT = M["cdi_bps"], M["resid_bps"], M["resid_z"], M["ratio"]
    F = {}
    # --- spread momentum / reversal, spread vol, drawdown from peak spread
    for h in (5, 21, 63, 126, 252):
        F[f"ds_{h}"] = CS - _lag(CS, h)
    for h in (21, 63):
        F[f"dres_{h}"] = RB - _lag(RB, h)
        F[f"drz_{h}"] = RZ - _lag(RZ, h)
    CSf = pd.DataFrame(CS).ffill(limit=5).to_numpy()
    d5 = CSf - _lag(CSf, 5)
    F["svol_63"] = _roll(d5, 63, "std", 30)
    F["svol_126"] = _roll(d5, 126, "std", 60)
    mx = _roll(CS, 252, "max", 60)
    mn = _roll(CS, 252, "min", 60)
    F["s_from_peak_252"] = CS - mx
    F["s_from_trough_252"] = CS - mn
    mu, sd = _roll(CS, 252, "mean", 60), _roll(CS, 252, "std", 60)
    F["s_ownz_252"] = (CS - mu) / np.where(sd > 1, sd, np.nan)
    # --- residual (distance to peer curve) at several smoothings
    for w in (5, 21, 63):
        F[f"rz_ma{w}"] = _roll(RZ, w, "mean", max(3, w // 3))
    F["rb_ma21"] = _roll(RB, 21, "mean", 7)
    F["rz_minus_ma63"] = RZ - F["rz_ma63"]
    # --- price
    F["ratio_ch_21"] = RAT - _lag(RAT, 21)
    F["ratio_ch_63"] = RAT - _lag(RAT, 63)
    F["ratio_from_hi_252"] = RAT / _roll(RAT, 252, "max", 60) - 1
    F["par_dist"] = np.abs(RAT - 1)
    # --- realised (hedged excess) return momentum / vol / drawdown: R[t] is t->t+1, known at close t+1
    R = H._Rmat("base")
    Rk = _lag(R, 1)   # at row d: R[d-1] (move to d's mark)
    Rk[0] = 0
    LR = np.log1p(np.clip(Rk, -0.99, None))
    cum = np.cumsum(np.nan_to_num(LR), axis=0)
    first = np.where(np.isfinite(CS).any(axis=0), np.isfinite(CS).argmax(axis=0), ND)
    rowi = np.arange(ND)[:, None]
    for h in (21, 63, 126, 252):
        F[f"rmom_{h}"] = np.where(rowi - h >= first[None, :], np.expm1(cum - _lag(cum, h)), np.nan)
    F["rmom_252_21"] = np.where(rowi - 252 >= first[None, :], np.expm1(_lag(cum, 21) - _lag(cum, 252)), np.nan)
    F["rvol_126"] = _roll(Rk, 126, "std", 60) * np.sqrt(252)
    F["rdd_252"] = cum - _roll(cum, 252, "max", 60)
    F["rmin_126"] = _roll(Rk, 126, "min", 60)
    del R, Rk, LR, cum, d5, CSf, mx, mn, mu, sd
    F = {k: v.astype(np.float32) for k, v in F.items()}
    # --- liquidity (SND trades, date <= d)
    log("liquidity")
    NT, VO = _snd_trades()
    for w in (21, 63):
        F[f"ntr_{w}"] = _roll(NT, w, "sum", 1)
    F["tdays_63"] = _roll((NT > 0).astype(float), 63, "sum", 1)
    v63 = _roll(VO, 63, "sum", 1)
    v21 = _roll(VO, 21, "sum", 1)
    v126 = _roll(VO, 126, "sum", 1)
    F["log_vol_63"] = np.log1p(v63)
    F["ticket_63"] = np.log1p(v63 / np.where(F["ntr_63"] > 0, F["ntr_63"], np.nan))
    F["vol_trend"] = np.log1p(v21 * 6) - np.log1p(v126)
    # Amihud: |move realised on a trade day| / R$ volume that day, mean over 126d
    Rr = _lag(H._Rmat("base"), 1)
    amh = np.where(VO > 0, np.abs(np.nan_to_num(Rr)) * 1e4 / np.maximum(VO, 0.01), np.nan)
    F["amihud_126"] = np.log1p(_roll(amh, 126, "mean", 3))
    del Rr, amh
    TD = C["TD"]
    dnum = dd.to_numpy().astype("datetime64[D]").astype(np.int64)[:, None]
    F["days_since_trade"] = np.where(TD >= 0, dnum - TD, np.nan).astype(float)
    F["days_since_trade"] = np.where(np.isnan(CS), np.nan, F["days_since_trade"])
    del NT, VO, v21, v63, v126
    F = {k: v.astype(np.float32) for k, v in F.items()}
    # --- extract at decision positions (rows present on the grid)
    log("extract")
    rows = []
    issuer = M["issuer"]
    for p in decs:
        ok = np.isfinite(CS[p])
        bb = np.where(ok)[0]
        d = {"day": dd[p], "codigo": codes[bb], "_iss": issuer[bb], "_cs": CS[p, bb], "_rb": RB[p, bb]}
        for k, A in F.items():
            d[k] = A[p, bb]
        rows.append(pd.DataFrame(d))
    X = pd.concat(rows, ignore_index=True)
    del F, M
    # --- issuer aggregates (same-day cross-section of the issuer's bonds on the grid)
    gi = X.groupby(["day", "_iss"])
    X["iss_resid_bps"] = gi["_rb"].transform("mean")
    X["bond_minus_iss_resid"] = X["_rb"] - X["iss_resid_bps"]
    X["iss_n_bonds"] = gi["_rb"].transform("size").astype(float)
    X["iss_resid_disp"] = gi["_rb"].transform("std")
    X["iss_ds_63"] = gi["ds_63"].transform("mean")
    X["iss_rmom_126"] = gi["rmom_126"].transform("mean")
    X["cs_minus_iss_cs"] = X["_cs"] - gi["_cs"].transform("mean")
    # --- registry (issuance-time fields)
    log("registry")
    RG = _registry(codes)
    X = X.join(RG, on="codigo")
    X["bond_age_y"] = (X["day"] - X["issue_date"]).dt.days / 365.25
    X["ttm_orig_y"] = (X["mat_orig"] - X["day"]).dt.days / 365.25
    X["life_frac"] = X["bond_age_y"] / (X["bond_age_y"] + X["ttm_orig_y"])
    X["y_to_amort"] = ((X["amort_start"] - X["day"]).dt.days / 365.25).clip(lower=0)
    X.loc[X["amortizing"] == 0, "y_to_amort"] = X.loc[X["amortizing"] == 0, "ttm_orig_y"]
    # issuer's number of issues already dated before d (serial issuer), from registry issue dates
    reg = RG.dropna(subset=["issue_date"])[["issuer_cnpj8", "issue_date"]].copy()
    reg = reg.drop_duplicates()
    cnt = []
    for dday, x in X.groupby("day"):
        r = reg[reg["issue_date"] <= dday].groupby("issuer_cnpj8").size()
        cnt.append(x["_iss"].map(r).rename("iss_n_issues"))
    X["iss_n_issues"] = pd.concat(cnt).reindex(X.index).fillna(0).astype(float)
    X = X.drop(columns=["issue_date", "mat_orig", "amort_start", "issuer_cnpj8", "_iss", "_cs", "_rb"])
    # --- equity volume trend / idiosyncratic return (PIT ticker chosen by the harness at the panel level)
    X["codigo"] = X["codigo"].astype(str)
    X = X.astype({c: "float32" for c in X.columns if X[c].dtype == np.float64})
    pd.to_pickle(X, FEAT_PATH)
    log(f"features: {X.shape}")
    return X


def _equity_extra(P: pd.DataFrame) -> pd.DataFrame:
    """Equity volume trend and idiosyncratic momentum for the harness-chosen PIT ticker."""
    tick, TI, SPX, ADTV, cands = H._equity_grid()
    eq = pd.read_pickle(H.HIST / "equity_daily.pkl")
    eq["date"] = pd.to_datetime(eq["date"])
    dd = H.days()
    VT = np.full((len(dd), len(tick)), np.nan)
    for t, g in eq.sort_values("date").drop_duplicates(["ticker", "date"]).groupby("ticker"):
        tv = (g.set_index("date")["close"] * g.set_index("date")["volume"])
        a21 = tv.rolling(21, min_periods=10).mean()
        a126 = tv.rolling(126, min_periods=60).mean()
        s = np.log((a21 + 1) / (a126 + 1))
        VT[:, TI[t]] = s.reindex(s.index.union(dd)).ffill(limit=5).reindex(dd).to_numpy()
    j = P["eq_ticker"].map(TI)
    ok = j.notna()
    out = pd.Series(np.nan, index=P.index)
    out[ok] = VT[P.loc[ok, "dpos"].to_numpy(int), j[ok].to_numpy(int)]
    return out


BASE_FEATS = ["cdi_bps", "resid_z", "resid_bps", "dur", "ratio", "incent", "contract", "age", "carry_per_dur", "own_z",
              "trades_30d", "d_spread_1w", "d_spread_4w", "d_ratio_4w", "years_to_mat",
              "rat_days_since_down", "n_rating_90d", "n_fact_90d", "n_distress_90d", "news_any_30d", "press_neg_30d",
              "days_since_distress", "n_deb_mtg_90d",
              "eq_r21", "eq_r63", "eq_r126", "eq_r252", "eq_vol63", "eq_dd252",
              "f_lev", "f_cov", "f_cash_st", "f_eq_ratio", "f_gde", "f_margin", "f_rev_g", "f_d_lev", "f_d_cov",
              "f_size", "f_quality", "f_age_days"]


def attach(P: pd.DataFrame) -> pd.DataFrame:
    """Merge the zoo features into a harness panel and add a few derived panel-level features."""
    X = build_features()
    P = P.merge(X, on=["day", "codigo"], how="left")
    P["is_ipca"] = (P["kind"] == "IPCA").astype(float)
    P["log_eq_adtv"] = np.log1p(P["eq_adtv"])
    P["eq_idio63"] = P["eq_r63"] - P["ibov_x63"]
    P["eq_mom_12_1"] = (1 + P["eq_r252"]) / (1 + P["eq_r21"]) - 1
    P["eq_vol_trend"] = _equity_extra(P)
    P["rat_down_365"] = (P["rat_days_since_down"] < 365).astype(float)
    # same-day peer / sector relative spread moves
    P["ds_63_vs_peer"] = P["ds_63"] - P.groupby(["day", "peer"])["ds_63"].transform("median")
    P["sec_ds_63"] = P.groupby(["day", "sector"])["ds_63"].transform("median")
    P["sec_resid"] = P.groupby(["day", "sector"])["resid_bps"].transform("median")
    P["cs_x_dur"] = P["cdi_bps"] * P["dur"]
    return P


ZOO = None  # filled by run.py: list of all candidate feature names


def signals(panel: pd.DataFrame | None = None, spec: dict | None = None) -> pd.DataFrame:
    """The frozen factor-zoo composite as (day, codigo, value): mean of per-date normal scores (ranked within the
    eligible universe) of the survivors with their signs (research/nightly/factor_zoo/results.json 'composite_spec').
    Higher = better expected forward rate-hedged excess return. panel: a harness panel (default monthly, pre-2026)."""
    import json
    from scipy.stats import norm
    if spec is None:
        spec = json.loads((Path(__file__).parent / "results.json").read_text())["composite_spec"]
    P = attach(panel if panel is not None else H.load_panel("M"))
    U = P[P["univ"]].copy()
    parts = []
    for f, s in spec.items():
        r = U.groupby("day")[f].rank(pct=True)
        n = U.groupby("day")[f].transform("count")
        z = pd.Series(norm.ppf((r * n - 0.5) / n), index=U.index) * s
        parts.append(z)
    v = pd.concat(parts, axis=1).mean(axis=1, skipna=True)
    return pd.DataFrame({"day": U["day"], "codigo": U["codigo"], "value": v}).dropna()
