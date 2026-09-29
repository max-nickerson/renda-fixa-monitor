"""Universe expansion & new sleeves: backtests on the common harness (v4).

Sleeves (all as daily excess over CDI on the harness grid, index convention of H.index_excess):
  * NTN-B / pre / LFT via ANBIMA indices (IMA-B5, IMA-B, IDKA-IPCA-2A, IRF-M, IMA-S), minus a 0.20%/yr ETF fee.
  * Listed CRI funds (FII de papel, B3 prices + distributions from brapi, NAV from CVM) with P4-like selection.
  * FIDC senior quotas (CVM informe mensal, self-reported monthly returns) with P4-like selection (paper sleeve).
  * CRI/CRA securitisation reports: arrears flag of certificates whose debtor/cedent is a debenture issuer, used
    as an extra exclusion on P4+Q.
  * Multi-sleeve allocations (P4+Q debentures + NTN-B + CRI funds + FIDC + CDI).
Sealed holdout: every choice uses decisions < 2026-01-01; holdout reported once at the end (section HOLDOUT).

Run:  PYTHONPATH=. PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
      .venv/Scripts/python.exe research/nightly/universe_expansion/run.py
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from research.nightly import harness as H  # noqa: E402

D = Path("data/history/nightly/universe_expansion")
OUTD = Path("research/nightly/universe_expansion")
RNG = np.random.default_rng(0)
DD = H.days()
CDI = H.cdi_daily()
ETF_FEE = 0.0020 / 252


def pos_of(day):
    return DD.get_loc(day)


# =====================================================================================================================
# generic monthly book for a sleeve with its own daily return matrix
# =====================================================================================================================
def sleeve_book(X: pd.DataFrame, targets: dict, cost_bps: float = 25.0, lag: int = 2, end=H.HOLDOUT) -> pd.Series:
    """X: grid x asset daily excess over CDI (NaN = no print -> 0). targets: {decision_day: pd.Series weights}.
    Weights decided at close d take effect on the return indexed d+lag (buy at close d+1). Buy-and-hold drift within
    the month; cost = |dw| x cost/2 at each rebalance. Returns daily excess (net)."""
    Xf = X.fillna(0.0)
    out = pd.Series(0.0, index=DD)
    ds = sorted(targets)
    w_prev = pd.Series(dtype=float)
    for k, d in enumerate(ds):
        p0 = pos_of(d) + lag
        p1 = pos_of(ds[k + 1]) + lag if k + 1 < len(ds) else len(DD)
        if p0 >= len(DD):
            break
        w = targets[d]
        w = w[w > 0]
        allk = w.index.union(w_prev.index)
        dw = (w.reindex(allk).fillna(0) - w_prev.reindex(allk).fillna(0)).abs().sum()
        cost = dw * cost_bps / 2 / 1e4
        if len(w) == 0:
            out.iloc[p0] -= cost
            w_prev = w
            continue
        seg = Xf.iloc[p0:p1][w.index] + CDI.iloc[p0:p1].to_numpy()[:, None]  # total return
        val = (1 + seg).cumprod()
        v = (val * w.to_numpy()).sum(axis=1)
        vprev = np.r_[w.sum(), v.to_numpy()[:-1]]
        port_tr = v.to_numpy() / vprev - 1
        ex = port_tr - CDI.iloc[p0:p1].to_numpy()
        # cash part (1 - sum w) earns CDI -> 0 excess; scale
        ex = ex * w.sum()
        out.iloc[p0:p1] = ex
        out.iloc[p0] -= cost
        wend = (val.iloc[-1] * w)
        w_prev = wend / wend.sum() * w.sum()
    out = out[out.index >= H.START]
    if end is not None:
        out = out[out.index < end]
    return out


def ew(names):
    names = list(names)
    return pd.Series(1.0 / len(names), index=names) if names else pd.Series(dtype=float)


# =====================================================================================================================
# 1) index sleeves
# =====================================================================================================================
def index_sleeves(end=H.HOLDOUT):
    IS = pd.read_pickle(D / "index_sleeves.pkl")
    out = {}
    for nm, lab in [("IMAB5", "NTNB_IMAB5"), ("IMAB", "NTNB_IMAB"), ("IDKAIPCA2A", "NTNB_IDKA2"), ("IRFM", "PRE_IRFM"),
                    ("IMAS", "LFT_IMAS"), ("IDAIPCA", "DEB_IDAIPCA")]:
        s = IS[nm] - (ETF_FEE if nm not in ("IDAIPCA",) else 0.0)
        s = s[(s.index >= H.START)]
        out[lab] = s[s.index < end] if end is not None else s
    return out


# =====================================================================================================================
# 2) listed CRI funds (FII de papel)
# =====================================================================================================================
def fii_targets(rule: str, holdout=False, min_adtv=5e5, seed=None, n_like=None):
    P = pd.read_pickle(D / "fii_panel.pkl")
    P = P[(P["adtv"] >= min_adtv) & P["dy12"].notna() & (P["dy12"] > 0.02) & (P["crish"].fillna(1) >= 0.5)]
    if not holdout:
        P = P[P["day"] < H.HOLDOUT]
    P = P[P["day"] >= "2021-12-01"]
    T = {}
    for d, x in P.groupby("day"):
        x = x.set_index("ticker")
        if len(x) < 8:
            continue
        top = x["dy12"] >= x["dy12"].quantile(0.7)
        rich = x["pnav"] > x["pnav"].quantile(0.8)            # expensive vs NAV (NaN -> not rich)
        worst = x["nav_chg12"] <= x["nav_chg12"].quantile(0.2)  # NAV erosion = realised credit losses (NaN kept)
        if rule == "U":
            sel = x.index
        elif rule == "carry":
            sel = x.index[top]
        elif rule == "P4":
            sel = x.index[top & ~rich]
        elif rule == "P4Q":
            sel = x.index[top & ~rich & ~worst]
        elif rule == "Qonly":
            sel = x.index[~worst]
        elif rule == "cheap":
            sel = x.index[x["pnav"] <= x["pnav"].quantile(0.3)]
        elif rule == "placebo":
            k = n_like.get(d, 0)
            sel = list(seed.choice(x.index.to_numpy(), size=min(k, len(x)), replace=False)) if k else []
        else:
            raise ValueError(rule)
        T[d] = ew(sel)
    return T


def fii_matrix():
    F = pickle.load(open(D / "fii_daily.pkl", "rb"))
    return F["ex"]


# =====================================================================================================================
# 3) FIDC senior quotas (paper sleeve, monthly self-reported returns)
# =====================================================================================================================
def fidc_series(rule: str, holdout=False, seed=None, n_like=None, missing="cdi"):
    F = pd.read_pickle(D / "fidc_panel.pkl")
    F = F[(F["ret"] > -1) & (F["ret"] < 0.2) & (F["ret"] != 0)]  # 0 = not reported (55k rows), not a 0% month
    F = F[F["FUNDO_EXCLUSIVO"] != "S"]
    cdi_m = H.monthly(pd.Series(0.0, index=DD))  # zeros -> used only for index
    cdi_tot = (1 + CDI).groupby([DD.year, DD.month]).prod() - 1
    cdi_tot.index = pd.to_datetime([f"{y}-{m:02d}-01" for y, m in cdi_tot.index])
    F["ex"] = (1 + F["ret"]) / (1 + F["ref"].map(cdi_tot)) - 1
    piv = F.pivot_table(index="ref", columns="cls", values="ex", aggfunc="first")
    pl = F.pivot_table(index="ref", columns="cls", values="pl", aggfunc="first")
    inad = F.pivot_table(index="ref", columns="cls", values="inad_ratio", aggfunc="first")
    months = pd.date_range("2022-01-01", "2026-08-01" if holdout else "2025-12-01", freq="MS")
    mret = {}
    n_sel = {}
    for m in months:
        # decision at start of month m: reports with ref + month-end + 45d <= m  -> ref <= m - 2 months
        last_ref = m - pd.DateOffset(months=2)
        hist = piv.loc[(piv.index <= last_ref) & (piv.index > last_ref - pd.DateOffset(months=3))]
        if len(hist) < 3 or m not in piv.index:
            continue
        carry = hist.mean()
        ok = hist.notna().all() & (pl.reindex([last_ref]).iloc[0] > 50e6).reindex(carry.index).fillna(False) & (hist != 0).all()
        c = carry[ok]
        # zero/NaN monthly returns in the target month: treat as CDI (0 excess) -- optimistic for defaults
        tgt = piv.loc[m].reindex(c.index)
        # missing report in the target month: 'cdi' = earns CDI (optimistic); 'zero' = earns 0% (excess = -CDI)
        tgt = tgt.fillna(0.0 if missing == "cdi" else -float(cdi_tot.get(m, 0.0)) / (1 + float(cdi_tot.get(m, 0.0))))
        q = inad.reindex([last_ref]).iloc[0].reindex(c.index)
        worst = q >= q.quantile(0.8)
        top = c >= c.quantile(0.7)
        if rule == "U":
            sel = c.index
        elif rule == "P4":
            sel = c.index[top]
        elif rule == "P4Q":
            sel = c.index[top & ~worst.fillna(False)]
        elif rule == "Qonly":
            sel = c.index[~worst.fillna(False)]
        elif rule == "placebo":
            k = n_like.get(m, 0)
            sel = seed.choice(c.index.to_numpy(), size=min(k, len(c)), replace=False) if k else []
        n_sel[m] = len(sel)
        mret[m] = float(tgt.reindex(sel).clip(-1, 0.5).mean()) if len(sel) else 0.0
    ms = pd.Series(mret)
    # spread each month's excess evenly over its grid days (daily series for the harness)
    out = pd.Series(0.0, index=DD)
    for m, r in ms.items():
        idx = (DD.year == m.year) & (DD.month == m.month)
        n = idx.sum()
        if n:
            out[idx] = (1 + r) ** (1 / n) - 1
    out = out[(out.index >= H.START)]
    if not holdout:
        out = out[out.index < H.HOLDOUT]
    return out, n_sel, ms


# =====================================================================================================================
# 4) CRI/CRA arrears link -> extra exclusion on P4+Q
# =====================================================================================================================
def crisec_flags():
    C = pd.read_pickle(D / "crisec_panel.pkl")
    Pp = pd.read_pickle(D / "crisec_parties.pkl")
    arr = C[C["arrears"]][["Codigo_Identificacao_Certificado", "ref", "avail"]].drop_duplicates()
    L = Pp.merge(arr, on=["Codigo_Identificacao_Certificado", "ref"])
    L = L[L["cnpj8"] != "00000000"].dropna(subset=["avail"])
    return L[["cnpj8", "avail"]].sort_values("avail")


def signal_crisec(L):
    def f(x):
        d = x["day"].iloc[0]
        bad = set(L.loc[(L["avail"] <= d) & (L["avail"] > d - pd.Timedelta(days=365)), "cnpj8"])
        return (x["p4q"].to_numpy() & ~x["cnpj8"].isin(bad).to_numpy())
    return f


def crisec_signal_table(L):
    """Reusable PIT signal: (cnpj8, date=avail, value=1) = issuer is debtor/cedent of a CRI/CRA reported in arrears."""
    t = L.rename(columns={"avail": "date"}).assign(value=1.0)
    t.to_pickle(D / "signal_crisec_arrears.pkl")
    return t


# =====================================================================================================================
# 5) allocation helpers
# =====================================================================================================================
def mix(parts: dict, weights, switch_cost_bps=10.0):
    """parts {name: daily excess}; weights: dict of constants or {name: daily weight series (already lagged)}."""
    idx = None
    for s in parts.values():
        idx = s.index if idx is None else idx.intersection(s.index)
    out = pd.Series(0.0, index=idx)
    for k, s in parts.items():
        w = weights[k]
        w = w.reindex(idx).ffill().fillna(0) if isinstance(w, pd.Series) else pd.Series(w, index=idx)
        out += w * s.reindex(idx).fillna(0)
        dw = w.diff().abs().fillna(0)
        out -= dw * switch_cost_bps / 2 / 1e4
    return out


def month_start_weights(fn, idx):
    """fn(decision_day) -> weight, evaluated at each month's first grid day, applied from 2 days later."""
    ds = [DD[p] for p in H._decision_positions("M")]
    w = {}
    for d in ds:
        p = pos_of(d) + 2
        if p < len(DD):
            w[DD[p]] = fn(d)
    return pd.Series(w).reindex(idx, method="ffill")


def unsmooth_sharpe(daily):
    """Geltner/Getmansky AR(1) de-smoothing of monthly excess: r* = (r - rho r_{-1})/(1-rho)."""
    m = H.monthly(daily[daily.index < H.HOLDOUT])
    rho = m.autocorr(1)
    rho = min(max(rho, 0.0), 0.9)
    u = ((m - rho * m.shift(1)) / (1 - rho)).dropna()
    return {"rho": round(float(rho), 3), "vol_unsm_%": round(float(u.std() * np.sqrt(12) * 100), 3),
            "sharpe_unsm": round(float(u.mean() * 12 / (u.std() * np.sqrt(12))), 3)}


def corr_monthly(a, b):
    ma, mb = H.monthly(a[a.index < H.HOLDOUT]), H.monthly(b[b.index < H.HOLDOUT])
    j = pd.concat([ma, mb], axis=1).dropna()
    return round(float(j.corr().iloc[0, 1]), 3)


# =====================================================================================================================
def main():
    res = {"notes": "pre-2026 unless stated; monthly stats (harness.stats); paired vs P4+Q (126-bday tranches)"}
    base = {k: H.baseline(k) for k in ("U", "P4", "P4Q")}
    b50 = {k: H.baseline(k, cost_bps=50) for k in ("U", "P4Q")}
    P4Q = base["P4Q"]["daily"]
    U = base["U"]["daily"]

    # ---- sleeves
    S = index_sleeves()
    X = fii_matrix()
    fiiT = {r: fii_targets(r) for r in ("U", "carry", "P4", "P4Q", "Qonly", "cheap")}
    for r, T in fiiT.items():
        S[f"FII_{r}"] = sleeve_book(X, T, 25)
    S["FII_P4Q_50bps"] = sleeve_book(X, fiiT["P4Q"], 50)
    fidc = {}
    for r in ("U", "P4", "P4Q", "Qonly"):
        s, n, ms = fidc_series(r)
        S[f"FIDC_{r}"] = s
        fidc[r] = n
    S["FIDC_P4Q_missing0"] = fidc_series("P4Q", missing="zero")[0]
    S["FIDC_U_missing0"] = fidc_series("U", missing="zero")[0]
    # placebos
    nl = {d: int((w > 0).sum()) for d, w in fiiT["P4Q"].items()}
    pl = [sleeve_book(X, fii_targets("placebo", seed=np.random.default_rng(i), n_like=nl), 25) for i in range(30)]
    plv = [H.stats(p)["ann_excess_%"] for p in pl]
    pl_fidc = [fidc_series("placebo", seed=np.random.default_rng(i), n_like=fidc["P4Q"])[0] for i in range(30)]
    plf = [H.stats(p)["ann_excess_%"] for p in pl_fidc]

    # ---- CRI/CRA arrears link on debentures
    L = crisec_flags()
    crisec_signal_table(L)
    r_cs = H.backtest(signal_crisec(L), name="P4Q_minus_CRIarrears")
    r_cs50 = H.backtest(signal_crisec(L), cost_bps=50)
    Pm = H.load_panel("M")
    Pm = Pm[Pm["univ"]]
    n_flag = []
    for d, x in Pm.groupby("day"):
        bad = set(L.loc[(L["avail"] <= d) & (L["avail"] > d - pd.Timedelta(days=365)), "cnpj8"])
        n_flag.append({"day": d, "univ_flag": int(x["cnpj8"].isin(bad).sum()),
                       "p4q_flag": int((x["p4q"] & x["cnpj8"].isin(bad)).sum())})
    n_flag = pd.DataFrame(n_flag)
    res["crisec_link"] = {"issuers_flagged_ever": int(L["cnpj8"].nunique()),
                          "avg_univ_bonds_flagged": round(float(n_flag["univ_flag"].mean()), 2),
                          "avg_p4q_bonds_flagged": round(float(n_flag["p4q_flag"].mean()), 2),
                          "stats_vs_P4Q": H.stats(r_cs["daily"], bench=P4Q),
                          "stats_vs_P4Q_50bps": H.stats(r_cs50["daily"], bench=H.baseline("P4Q", cost_bps=50)["daily"])}

    # ---- allocations
    ntnb = S["NTNB_IMAB5"]
    fii = S["FII_P4Q"]
    fd = S["FIDC_P4Q"]
    parts = {"deb": P4Q, "ntnb": ntnb, "fii": fii, "fidc": fd}
    idx = P4Q.index
    cumx = lambda s, n: (1 + s).rolling(n).apply(np.prod, raw=True) - 1
    ntnb_mom = cumx(ntnb, 126)
    fii_mom = cumx(fii, 126)
    vols = {k: v.rolling(126, min_periods=60).std() for k, v in parts.items()}

    def invvol(d, k, keys=("deb", "ntnb", "fii")):
        iv = {j: 1 / max(vols[j].get(d, np.nan), 1e-6) if np.isfinite(vols[j].get(d, np.nan)) else 0 for j in keys}
        tot = sum(iv.values())
        return iv[k] / tot if tot > 0 else (1.0 if k == "deb" else 0.0)

    A = {}
    A["MIX_80deb_20ntnb"] = mix(parts, {"deb": .8, "ntnb": .2, "fii": 0, "fidc": 0})
    A["MIX_80deb_20fii"] = mix(parts, {"deb": .8, "ntnb": 0, "fii": .2, "fidc": 0})
    A["MIX_80deb_20fidc"] = mix(parts, {"deb": .8, "ntnb": 0, "fii": 0, "fidc": .2})
    A["MIX_70deb_15ntnb_15fii"] = mix(parts, {"deb": .7, "ntnb": .15, "fii": .15, "fidc": 0})
    A["MIX_60deb_15ntnb_10fii_15fidc"] = mix(parts, {"deb": .6, "ntnb": .15, "fii": .10, "fidc": .15})
    A["MIX_80deb_20ntnb_trend"] = mix(parts, {
        "deb": .8, "ntnb": month_start_weights(lambda d: 0.2 if ntnb_mom.get(d, -1) > 0 else 0.0, idx),
        "fii": 0, "fidc": 0})
    A["MIX_80deb_20best_trend"] = mix(parts, {
        "deb": .8,
        "ntnb": month_start_weights(lambda d: 0.2 if (ntnb_mom.get(d, -1) > max(fii_mom.get(d, -1), 0)) else 0.0, idx),
        "fii": month_start_weights(lambda d: 0.2 if (fii_mom.get(d, -1) >= max(ntnb_mom.get(d, -1), 0)) and fii_mom.get(d, -1) > 0 else 0.0, idx),
        "fidc": 0})
    A["MIX_invvol_deb_ntnb_fii"] = mix(parts, {k: month_start_weights(lambda d, k=k: invvol(d, k), idx)
                                               for k in ("deb", "ntnb", "fii")} | {"fidc": 0})

    # ---- one table, every variant tried (Holm)
    allv = {**{k: v for k, v in S.items()}, **A, "DEB_P4Q_minus_CRIarrears": r_cs["daily"],
            "P4": base["P4"]["daily"]}
    tab = H.compare(allv, bench="P4Q")
    tab = tab.drop(columns=["turnover", "n_avg"])
    extra = {k: unsmooth_sharpe(v) for k, v in {**allv, "P4Q": P4Q, "U": U}.items()}
    tab["sharpe_unsm"] = [extra[k]["sharpe_unsm"] for k in tab.index]
    tab["rho_ar1"] = [extra[k]["rho"] for k in tab.index]
    tab["corr_P4Q"] = [corr_monthly(allv[k], P4Q) for k in tab.index]
    st50 = {}
    for k in ("FII_P4Q", "FII_U"):
        pass
    res["n_variants_tried"] = int(len(tab))
    res["table"] = json.loads(tab.round(4).to_json(orient="index"))
    res["P4Q_ref"] = {**H.stats(P4Q), **extra["P4Q"]}
    res["U_ref"] = {**H.stats(U), **extra["U"]}
    res["P4Q_50bps"] = H.stats(b50["P4Q"]["daily"])
    res["FII_P4Q_vs_FII_U"] = H.stats(S["FII_P4Q"], bench=S["FII_U"])
    res["FII_P4_vs_FII_U"] = H.stats(S["FII_P4"], bench=S["FII_U"])
    res["FII_carry_vs_FII_U"] = H.stats(S["FII_carry"], bench=S["FII_U"])
    res["FII_P4Q_50_vs_FII_U"] = H.stats(S["FII_P4Q_50bps"], bench=S["FII_U"])
    res["FII_placebo"] = {"real": H.stats(S["FII_P4Q"])["ann_excess_%"], "placebo_mean": float(np.mean(plv)),
                          "placebo_p95": float(np.percentile(plv, 95)),
                          "p_value": float(np.mean(np.array(plv) >= H.stats(S["FII_P4Q"])["ann_excess_%"]))}
    res["FIDC_P4Q_vs_FIDC_U"] = H.stats(S["FIDC_P4Q"], bench=S["FIDC_U"])
    res["FIDC_P4_vs_FIDC_U"] = H.stats(S["FIDC_P4"], bench=S["FIDC_U"])
    res["FIDC_placebo"] = {"real": H.stats(S["FIDC_P4Q"])["ann_excess_%"], "placebo_mean": float(np.mean(plf)),
                           "placebo_p95": float(np.percentile(plf, 95)),
                           "p_value": float(np.mean(np.array(plf) >= H.stats(S["FIDC_P4Q"])["ann_excess_%"]))}
    res["fii_avg_names"] = {r: float(np.mean([len(w) for w in T.values()])) for r, T in fiiT.items()}
    res["fidc_avg_names"] = {r: float(np.mean(list(n.values()))) for r, n in fidc.items()}
    for k in ("MIX_80deb_20ntnb", "MIX_80deb_20fii", "MIX_70deb_15ntnb_15fii", "MIX_80deb_20fidc"):
        res[f"{k}_50bps_vs_P4Q50"] = H.stats(
            mix({"deb": b50["P4Q"]["daily"], "ntnb": ntnb, "fii": S["FII_P4Q_50bps"], "fidc": fd},
                {"deb": .8 if "80" in k else .7, "ntnb": .2 if k.endswith("ntnb") else (.15 if "15ntnb" in k else 0),
                 "fii": .2 if k.endswith("20fii") else (.15 if "15fii" in k else 0),
                 "fidc": .2 if "fidc" in k else 0}),
            bench=b50["P4Q"]["daily"])
    # correlation matrix of monthly excess
    cm = pd.DataFrame({k: H.monthly(v[v.index < H.HOLDOUT]) for k, v in
                       {"P4Q": P4Q, "U": U, "NTNB_IMAB5": ntnb, "PRE_IRFM": S["PRE_IRFM"], "FII_U": S["FII_U"],
                        "FII_P4Q": fii, "FIDC_U": S["FIDC_U"], "FIDC_P4Q": fd, "IDAIPCA": S["DEB_IDAIPCA"]}.items()})
    res["corr_monthly"] = json.loads(cm.corr().round(3).to_json())

    print(tab.round(3).to_string())
    pickle.dump({"S": S, "A": A, "cs": r_cs["daily"]}, open(D / "series_pre2026.pkl", "wb"))
    json.dump(res, open(OUTD / "results_pre2026.json", "w"), indent=1, default=str)
    return res, tab, S, A


if __name__ == "__main__":
    main()
