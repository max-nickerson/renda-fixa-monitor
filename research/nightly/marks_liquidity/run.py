"""marks_liquidity / step 2: better marks -> signals, de-smoothed risk, costs by liquidity, capacity, liquidity-aware P4+Q.

Run (after build_marks.py):
  PYTHONPATH=. PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/Scripts/python.exe research/nightly/marks_liquidity/run.py
  add --holdout to also report the sealed 2026 holdout (once, at the end, for the frozen best variant).
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
from research.nightly.marks_liquidity import signals as SG  # noqa: E402

HERE = Path(__file__).resolve().parent
CACHE = ROOT / "data" / "history" / "nightly" / "marks_liquidity"
T0 = time.time()
RES: dict = {}
AUMS = (100e6, 500e6, 2e9)
PART = 0.20          # max share of the bond's SND volume in the 20-bday entry window
Y_IMPACT = 1.0       # square-root impact coefficient: cost = Y * sigma_daily * sqrt(Q / ADV)


def log(*a):
    print(f"[run +{time.time() - T0:6.1f}s]", *a, flush=True)


def rnd(o, k=4):
    if isinstance(o, dict):
        return {str(a): rnd(b, k) for a, b in o.items()}
    if isinstance(o, (list, tuple)):
        return [rnd(v, k) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else round(float(o), k)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


# =====================================================================================================================
# 0) cost table by liquidity quintile (training prints < 2024 only) + print-size premium
# =====================================================================================================================
def build_cost_table(PW: pd.DataFrame) -> dict:
    dd = H.days()
    prm = json.load(open(CACHE / "params.json"))
    t = pd.read_pickle(CACHE / "trades.pkl")
    F = pd.read_pickle(CACHE / "peer_factor.pkl")
    M = pd.read_pickle(CACHE / "marks.pkl")[["codigo", "day", "adv63", "kf_prior", "kf_z"]]
    t = t.merge(M.rename(columns={"day": "date"}), on=["codigo", "date"], how="left")
    t["pos"] = dd.get_indexer(t["date"])
    t = t[t["pos"] >= 0].sort_values(["codigo", "pos"])
    fl = F.reindex(columns=sorted(t["peer"].unique())).fillna(0.0)
    t["F"] = fl.to_numpy()[t["pos"].to_numpy(), fl.columns.get_indexer(t["peer"])]
    g = t.groupby("codigo")
    t["gp"] = t["pos"] - g["pos"].shift(1)
    t["gn"] = g["pos"].shift(-1) - t["pos"]
    t["dpa"] = ((t["s"] - g["s"].shift(1)) - (t["F"] - g["F"].shift(1))) * t["dur"]
    t["dpa_n"] = g["dpa"].shift(-1)
    t["grp"] = np.where(t["kind"] == "DI_SPREAD", "DI", "IPCA_PRE")
    # ADV quintile thresholds of the universe on the latest weekly decision date <= print date
    U = PW[PW["univ"] & PW["adv63"].notna()]
    thr = U.groupby("day")["adv63"].quantile([0.2, 0.4, 0.6, 0.8]).unstack()
    thr.columns = ["q20", "q40", "q60", "q80"]
    thr = thr.reset_index()
    t = pd.merge_asof(t.sort_values("date"), thr.sort_values("day"), left_on="date", right_on="day",
                      direction="backward")
    t["liq_q"] = 1 + (t["adv63"] > t["q20"]).astype(int) + (t["adv63"] > t["q40"]) + (t["adv63"] > t["q60"]) + \
        (t["adv63"] > t["q80"])
    t.loc[t["q20"].isna() | t["adv63"].isna(), "liq_q"] = np.nan
    tr = t[(t["date"] < pd.Timestamp("2024-01-01")) & (t["gp"] == 1) & (t["gn"] == 1) & t["dpa"].abs().lt(1000)
           & t["dpa_n"].abs().lt(1000) & t["liq_q"].notna()]
    table, detail = {}, {}
    for gname, z in tr.groupby("grp"):
        table[gname] = {}
        for q, zz in z.groupby("liq_q"):
            pr = (zz["dpa"] * zz["dpa_n"]).to_numpy()
            hs = float(np.sqrt(max(-pr.mean(), 0))) if len(pr) > 100 else np.nan
            table[gname][str(int(q))] = hs
            detail[f"{gname}_q{int(q)}"] = {"roll_half_spread_price_bps": hs, "n_triples": int(len(pr)),
                                            "median_dur": float(zz["dur"].median())}
        # fill missing quintiles (few triples in illiquid buckets, which rarely trade on consecutive days) with the
        # least-liquid estimate available x 1.25 (conservative)
        vals = [table[gname].get(str(q)) for q in range(1, 6)]
        known = [v for v in vals if v is not None and np.isfinite(v)]
        for q in range(1, 6):
            v = table[gname].get(str(q))
            if v is None or not np.isfinite(v):
                table[gname][str(q)] = float(max(known) * 1.25)
        # monotone (less liquid never cheaper): cumulative max from liquid to illiquid
        arr = [table[gname][str(q)] for q in range(5, 0, -1)]
        arr = np.maximum.accumulate(arr)
        for i, q in enumerate(range(5, 0, -1)):
            table[gname][str(q)] = float(arr[i])
    json.dump(table, open(HERE / "cost_table.json", "w"), indent=1)
    # print-size premium: signed deviation of the print from the Kalman one-step prediction (price bps; > 0 = the
    # print is RICHER than fair, i.e. the buyer paid up). All prints dated < 2024 with a prediction.
    z = t[(t["date"] < pd.Timestamp("2024-01-01")) & t["kf_prior"].notna()].copy()
    z["dev_price_bps"] = -(z["s"] - z["kf_prior"]) * z["dur"]
    z["tb"] = pd.cut(z["ticket"], [0, 5e4, 2e5, 1e6, 5e6, np.inf], labels=["<50k", "50-200k", "200k-1m", "1-5m", ">5m"],
                     right=False)
    prem = {}
    for gname, zz in z.groupby("grp"):
        lo, hi = zz["dev_price_bps"].quantile([0.01, 0.99])
        zz = zz.assign(dv=zz["dev_price_bps"].clip(lo, hi))
        prem[gname] = {str(k): {"mean_dev_price_bps": float(v["dv"].mean()), "median": float(v["dv"].median()),
                                "abs_median": float(v["dv"].abs().median()), "n": int(len(v))}
                       for k, v in zz.groupby("tb", observed=True)}
    # institutional tier: Roll on prints with ticket >= R$200k only (consecutive such prints <= 5 bdays apart), vs the
    # same estimator on all prints -> scale factor applied to the bucket table ('inst' cost model)
    tt = t[t["date"] < pd.Timestamp("2024-01-01")]
    inst = {}
    for lab, sub in (("inst", tt[tt["ticket"] >= 2e5]), ("all", tt)):
        sub = sub.sort_values(["codigo", "pos"])
        g2 = sub.groupby("codigo")
        gp = sub["pos"] - g2["pos"].shift(1)
        dp = ((sub["s"] - g2["s"].shift(1)) - (sub["F"] - g2["F"].shift(1))) * sub["dur"]
        dpn = dp.groupby(sub["codigo"]).shift(-1)
        gn = gp.groupby(sub["codigo"]).shift(-1)
        ok = (gp <= 5) & (gn <= 5) & dp.abs().lt(1000) & dpn.abs().lt(1000)
        for gname in ("DI", "IPCA_PRE"):
            m = ok & (sub["grp"] == gname)
            pr = (dp[m] * dpn[m]).to_numpy()
            inst[f"{gname}_{lab}"] = float(np.sqrt(max(-pr.mean(), 0)))
            inst[f"{gname}_{lab}_n"] = int(m.sum())
    inst_scale = {g: inst[f"{g}_inst"] / inst[f"{g}_all"] for g in ("DI", "IPCA_PRE")}
    out = {"inst_roll": inst, "inst_scale": inst_scale, "half_spread_price_bps_by_liq_quintile": table, "detail": detail, "print_size_premium": prem,
           "params": {g: {k: prm["groups"][g][k] for k in ("R_price_bps2_typical", "q_spread_bps2_per_bday",
                                                          "issuer_share", "roll_half_spread_price_bps_by_ticket",
                                                          "roll_half_spread_price_bps_all",
                                                          "noise_mult_by_ntrades(1|2-4|5+)")}
                      for g in prm["groups"]}}
    return out


# =====================================================================================================================
# 1) custom tranche engine (mirrors harness._tranche_book, no early exits) with per-bond costs and partial fills
# =====================================================================================================================
def _vol_matrix() -> tuple[np.ndarray, np.ndarray]:
    """V[pos, b] = SND R$ volume of bond b on grid day pos (harness bond order); cumulative with a zero row."""
    dd, codes = H.days(), H._core()["codes"]
    t = pd.read_pickle(CACHE / "trades.pkl")[["codigo", "date", "vol_brl"]]
    p = dd.get_indexer(t["date"])
    b = codes.get_indexer(t["codigo"])
    ok = (p >= 0) & (b >= 0)
    V = np.zeros((len(dd), len(codes)), np.float64)
    V[p[ok], b[ok]] = t["vol_brl"].to_numpy()[ok]
    return V, np.vstack([np.zeros((1, V.shape[1])), np.cumsum(V, 0)])


def tranche_book(T: dict, hold: int = 126, cost_side_bps=12.5, aum: float | None = None, part: float = PART,
                 sigma_b: dict | None = None, adv_b: dict | None = None, hs_b: dict | None = None,
                 cumV: np.ndarray | None = None) -> dict:
    """T: {dpos: pd.Series(weight, index=b)} (sum 1). If aum is given: each new tranche (aum / M) buys w*aum/M of each
    bond, filled up to part x the bond's actual SND volume over its entry window (p+1 .. p+20); the unfilled part
    stays in cash. Costs per side: flat `cost_side_bps`, or (hs_b given) the bond's half-spread + square-root impact
    Y * sigma_b * sqrt(trade / ADV63) on the book-level trade |d book| x aum. hs_b/sigma_b/adv_b: {dpos: {b: value}}."""
    R = H._Rmat("base")
    ND = len(H.days())
    t_end = H._hpos()
    NR = H._next_row()
    decs = sorted(T)
    step = int(np.median(np.diff(decs))) if len(decs) > 1 else 21
    M = max(int(round(hold / step)), 1)
    pnl, expo, ncoh = np.zeros(ND), np.zeros(ND), np.zeros(ND)
    Tf = {}
    fills = []
    for p in decs:
        w = T[p]
        ncoh[p + 1:] += 1
        if not len(w):
            Tf[p] = w
            continue
        ep = H.exec_pos(p)
        if aum is not None:
            lo, hi = p + 1, min(p + 21, ND)
            vwin = cumV[hi, w.index.to_numpy()] - cumV[lo, w.index.to_numpy()]
            req = w.to_numpy() * aum / M
            f = np.minimum(1.0, part * vwin / np.maximum(req, 1.0))
            w = pd.Series(w.to_numpy() * f, index=w.index)
            fills.append((p, float(w.sum())))
        Tf[p] = w
        for b, wi in w.items():
            k = int(ep[b])
            if k < 0 or k >= t_end or wi <= 0:
                continue
            e = k + hold
            if e < ND:
                x = int(NR[e, b])
                e = x if x < ND else e
            e = min(e, ND, t_end)
            if e <= k:
                continue
            r = R[k:e, b]
            gv = np.r_[1.0, np.cumprod(1 + r[:-1])]
            pnl[k:e] += wi * gv * r
            expo[k:e] += wi * gv
    div = np.maximum(np.minimum(M, ncoh), 1)
    pnl /= div
    expo /= div
    cost = np.zeros(ND)
    turn = np.zeros(ND)
    prev = pd.Series(dtype=float)
    cbd = []
    for i, p in enumerate(decs):
        live = [Tf[x] for x in decs[max(0, i - M + 1): i + 1]]
        live = [s for s in live if len(s)]
        book = (pd.concat(live, axis=1).fillna(0).sum(axis=1) / min(M, i + 1)) if live else pd.Series(dtype=float)
        d = book.sub(prev, fill_value=0).abs()
        d = d[d > 0]
        pc = min(p + 1, ND - 1)
        turn[pc] += d.sum()
        if hs_b is None:
            c = d.sum() * cost_side_bps / 1e4
        else:
            hs = np.array([hs_b[p].get(b, np.nan) for b in d.index])
            hs = np.where(np.isfinite(hs), hs, np.nanmax(list(hs_b[p].values()) or [50.0]))
            if aum is not None:
                sg = np.array([sigma_b[p].get(b, np.nan) for b in d.index])
                sg = np.where(np.isfinite(sg), sg, np.nanmedian(list(sigma_b[p].values()) or [10.0]))
                av = np.array([adv_b[p].get(b, np.nan) for b in d.index])
                av = np.where(np.isfinite(av) & (av > 0), av, 1e4)
                imp = Y_IMPACT * sg * np.sqrt(d.to_numpy() * aum / av)
            else:
                imp = 0.0
            c = float((d.to_numpy() * (hs + imp)).sum() / 1e4)
            cbd.append((p, float((d.to_numpy() * hs).sum() / 1e4), float((d.to_numpy() * imp).sum() / 1e4)
                        if aum is not None else 0.0, float(d.sum())))
        cost[pc] += c
        prev = book
    first = min(decs) + 1
    idx = H.days()[first:t_end]
    yrs = len(idx) / 252
    net = pd.Series((pnl - cost)[first:t_end], index=idx)
    out = {"daily": net, "gross": pd.Series(pnl[first:t_end], index=idx),
           "exposure": pd.Series(expo[first:t_end], index=idx),
           "turnover_ann": float(turn[first:t_end].sum() / yrs), "cost_ann_%": float(cost[first:t_end].sum() / yrs * 100),
           "n_avg": float(np.mean([len(v) for v in T.values()]))}
    if fills:
        out["avg_fill"] = float(np.mean([f for _, f in fills]))
    if cbd:
        cb = np.array([c[1:] for c in cbd])
        out["spread_cost_ann_%"] = float(cb[:, 0].sum() / yrs * 100)
        out["impact_cost_ann_%"] = float(cb[:, 1].sum() / yrs * 100)
    return out


# =====================================================================================================================
# 2) GLM (Getmansky-Lo-Makarov) de-smoothing
# =====================================================================================================================
def glm(series: pd.Series, k: int = 2, freq: str = "M") -> dict:
    import statsmodels.api as sm
    if freq == "M":
        x = H.monthly(series) * 100
        ann = 12
    else:
        c = H.cdi_daily().reindex(series.index).fillna(0)
        x = (((1 + c + series).groupby(series.index.to_period("W")).prod()
              / (1 + c).groupby(series.index.to_period("W")).prod()) - 1) * 100
        ann = 52
    x = x.dropna()
    out = {"n": int(len(x)), "mean_ann_%": float(x.mean() * ann), "vol_obs_ann_%": float(x.std() * np.sqrt(ann))}
    out["sharpe_obs"] = out["mean_ann_%"] / out["vol_obs_ann_%"]
    ac = [float(x.autocorr(j)) for j in range(1, k + 1)]
    out["autocorr"] = ac
    try:
        m = sm.tsa.arima.ARIMA(x.to_numpy(), order=(0, 0, k), trend="c").fit()
        ma = np.array([m.params[1 + j] for j in range(k)])
        th = np.r_[1.0, ma] / (1 + ma.sum())
        xi = float((th ** 2).sum())
        out["theta"] = th.tolist()
        out["xi_sum_theta2"] = xi
        out["vol_true_ann_%"] = out["vol_obs_ann_%"] / np.sqrt(xi)
        out["sharpe_true_glm"] = out["sharpe_obs"] * np.sqrt(xi)
    except Exception as e:  # pragma: no cover
        out["glm_error"] = str(e)[:100]
    # Lo (2002) autocorrelation-adjusted annualisation (uses the first k autocorrelations)
    q = ann
    rho = [x.autocorr(j) for j in range(1, min(k, q - 1) + 1)]
    denom = q + 2 * sum((q - j) * r for j, r in enumerate(rho, start=1))
    out["sharpe_lo_adj"] = float((x.mean() / x.std()) * q / np.sqrt(max(denom, 1e-9)))
    return out


# =====================================================================================================================
def main(holdout: bool = False):
    log("panels")
    PW0 = H.load_panel("W")
    PW = PW0[["codigo", "day", "univ"]].merge(pd.read_pickle(CACHE / "marks.pkl")[["codigo", "day", "adv63"]],
                                              on=["codigo", "day"], how="left")
    del PW0
    log("cost table")
    CT = build_cost_table(PW)
    RES["costs"] = CT
    PM = SG.signals(H.load_panel("M"))           # reload now that cost_table.json exists -> cost_rt_bps
    U = PM[PM["univ"]]
    RES["coverage"] = {"univ_rows": int(len(U)), "kf_s": float(U["kf_s"].notna().mean()),
                       "vw_s": float(U["vw_s"].notna().mean()), "adv63": float(U["adv63"].notna().mean()),
                       "roll_hs_bond": float(U["roll_hs"].notna().mean())}
    # how different are the marks?
    comp = {}
    for kind, z in U.groupby(U["kind"].astype(str)):
        comp[kind] = {"n": int(len(z)), "corr_cdi_kf": float(z[["cdi_bps", "kf_s"]].corr().iloc[0, 1]),
                      "median_abs_gap_bps": float(z["kf_gap"].abs().median()),
                      "p90_abs_gap_bps": float(z["kf_gap"].abs().quantile(0.9)),
                      "median_kf_sd": float(z["kf_sd"].median()),
                      "resid_corr_raw_vs_kf": float(z[["resid_bps", "kf_resid_bps"]].corr().iloc[0, 1]),
                      "rich_raw_share": float((z["resid_z"] <= -1.5).mean()),
                      "rich_kf_share": float((z["kf_resid_z"] <= -1.5).mean()),
                      "rich_both_share": float(((z["resid_z"] <= -1.5) & (z["kf_resid_z"] <= -1.5)).mean())}
    RES["mark_comparison"] = comp
    log("mark comparison", json.dumps(rnd(comp, 3)))

    # ---------------------------------------------------------------- IC: does RV still work on better marks?
    PM["noise_part"] = PM["resid_bps"] - PM["kf_resid_bps"]
    PM["neg_kf_sd"] = -PM["kf_sd"]
    ics = {}
    for f in ("cdi_bps", "kf_s", "vw_s2", "resid_bps", "kf_resid_bps", "vw_resid_bps", "kf_gap", "noise_part",
              "adv63", "tdays63", "roll_hs", "kf_sd"):
        ics[f] = {}
        for tg in ("fwd_21", "fwd_63", "fwd_126"):
            r = H.ic(PM, f, tg)
            ics[f][tg] = {"ic": r["mean"], "t": r["t_nw"], "n": r["n_dates"]}
    # RV by liquidity: resid IC within the least vs most liquid 40%
    for lab, m in (("illiquid_q1_2", PM["liq_q"] <= 2), ("liquid_q4_5", PM["liq_q"] >= 4)):
        for f in ("resid_bps", "kf_resid_bps"):
            r = H.ic(PM[m | ~PM["univ"]], f, "fwd_126", min_n=20)
            ics[f"{f}|{lab}"] = {"fwd_126": {"ic": r["mean"], "t": r["t_nw"], "n": r["n_dates"]}}
    for lab, kd in (("DI", PM["kind"].astype(str) == "DI_SPREAD"), ("IPCA", PM["kind"].astype(str) == "IPCA")):
        for f in ("resid_bps", "kf_resid_bps", "kf_gap"):
            r = H.ic(PM[kd | ~PM["univ"]], f, "fwd_126", min_n=20)
            ics[f"{f}|{lab}"] = {"fwd_126": {"ic": r["mean"], "t": r["t_nw"], "n": r["n_dates"]}}
    RES["ic"] = ics
    log("IC", json.dumps(rnd({k: v.get("fwd_126", v) for k, v in ics.items()}, 3)))

    # ---------------------------------------------------------------- selection variants (harness engine, 25 bps)
    def p4like(carry="cdi_bps", rz="resid_z", extra=None, frac=0.3):
        def f(x):
            c = x[carry]
            pct = c.rank(pct=True, ascending=False)
            m = (pct <= frac) & ~(x[rz].fillna(0) <= -1.5) & (x["press_neg_30d"].fillna(0) < 1) & ~x["worstQ"]
            m = m & c.notna()
            if extra is not None:
                m = m & extra(x)
            return m.to_numpy()
        return f

    V = {
        "P4Q_kf": p4like("kf_s", "kf_resid_z"),
        "P4Q_kf_rich_only": p4like("cdi_bps", "kf_resid_z"),
        "P4Q_vw": p4like("vw_s2", "vw_resid_z"),
        "P4Q_gapfilter": p4like(extra=lambda x: ~(x["kf_gap_z"] < -2)),
        "P4Q_liq_q2plus": p4like(extra=lambda x: x["liq_q"].fillna(1) >= 2),
        "P4Q_netcost": p4like("carry_net"),
    }
    PM["carry_net"] = PM["cdi_bps"] - PM["cost_rt_bps"].fillna(PM["cost_rt_bps"].max())
    res = {}
    chk = H.backtest(lambda x: x["p4q"].to_numpy(), panel=PM, name="P4Q_check")
    b = H.baseline("P4Q")
    RES["p4q_rebuild_max_abs_diff"] = float((chk["daily"] - b["daily"]).abs().max())
    for nm, f in V.items():
        res[nm] = H.backtest(f, panel=PM, name=nm)
        log(nm, round(H.stats(res[nm]["daily"], bench=b["daily"])["diff_ann_%"], 3))

    # ---------------------------------------------------------------- per-bond costs, capacity (custom engine)
    log("custom engine")
    cumV = _vol_matrix()[1]
    codes = H._core()["codes"]
    PM["b"] = codes.get_indexer(PM["codigo"])
    prm = json.load(open(CACHE / "params.json"))["groups"]
    PM["grp"] = np.where(PM["kind"].astype(str) == "DI_SPREAD", "DI", "IPCA_PRE")
    PM["q_s"] = PM["grp"].map({g: prm[g]["q_spread_bps2_per_bday"] for g in prm})
    PM["sigma_px"] = np.sqrt(PM["q_s"] * (PM["kf_s"].clip(50, 1000).fillna(100) / 100)) * PM["dur"]
    PM["hs_all"] = PM["cost_rt_bps"] / 2
    PM["hs"] = PM["hs_all"] * PM["grp"].map(CT["inst_scale"])          # main cost model: institutional tier
    U = PM[PM["univ"]]
    hs_b = {int(p): dict(zip(x["b"], x["hs"])) for p, x in PM.groupby("dpos")}
    hs_b_all = {int(p): dict(zip(x["b"], x["hs_all"])) for p, x in PM.groupby("dpos")}
    sg_b = {int(p): dict(zip(x["b"], x["sigma_px"])) for p, x in PM.groupby("dpos")}
    adv_b = {int(p): dict(zip(x["b"], x["adv63"].fillna(0))) for p, x in PM.groupby("dpos")}
    Pd = PM[(PM["day"] >= H.START)]

    def targets(sig, P=Pd):
        return H._targets(sig, P, 0.2, H.ISSUER_CAP, False, H.MIN_NAMES, "univ")

    T_p4q = targets(lambda x: x["p4q"].to_numpy())
    T_u = targets(lambda x: x["univ"].to_numpy())
    T_p4 = targets(lambda x: x["p4"].to_numpy())
    mine = tranche_book(T_p4q, cost_side_bps=12.5)
    RES["engine_check_max_abs_diff_vs_harness"] = float((mine["daily"] - b["daily"]).abs().max())
    log("engine check", RES["engine_check_max_abs_diff_vs_harness"])
    cap = {}
    books = {}
    for nm, T in (("U", T_u), ("P4", T_p4), ("P4Q", T_p4q)):
        r = tranche_book(T, hs_b=hs_b)
        books[f"{nm}_bucketcost"] = r
        cap[f"{nm}|no_aum|bucket_cost"] = {**H.stats(r["daily"]), "cost_ann_%": r["cost_ann_%"],
                                          "turnover": r["turnover_ann"]}
        r2 = tranche_book(T, hs_b=hs_b_all)
        books[f"{nm}_bucketcost_all"] = r2
        cap[f"{nm}|no_aum|bucket_cost_allprints(upper)"] = {**H.stats(r2["daily"]), "cost_ann_%": r2["cost_ann_%"]}
    for aum in AUMS:
        for nm, T in (("U", T_u), ("P4Q", T_p4q)):
            r = tranche_book(T, aum=aum, hs_b=hs_b, sigma_b=sg_b, adv_b=adv_b, cumV=cumV)
            books[f"{nm}_aum{int(aum / 1e6)}m"] = r
            s = H.stats(r["daily"])
            cap[f"{nm}|aum={int(aum / 1e6)}m"] = {**s, "avg_fill": r.get("avg_fill"), "cost_ann_%": r["cost_ann_%"],
                                                 "spread_cost_ann_%": r.get("spread_cost_ann_%"),
                                                 "impact_cost_ann_%": r.get("impact_cost_ann_%"),
                                                 "avg_exposure": float(r["exposure"].mean())}
            log(nm, aum, round(s["ann_excess_%"], 3), "fill", round(r.get("avg_fill", 1), 3))

    # ---------------------------------------------------------------- liquidity-aware P4+Q (sized by PIT ADV)
    def la_weights(aum, part=PART, ext_frac=0.5):
        M = 6

        def f(x):
            core = x["p4q"].to_numpy()
            pct = x["cdi_bps"].rank(pct=True, ascending=False).to_numpy()
            ext = (~core) & (pct <= ext_frac) & ~(x["resid_z"].fillna(0) <= -1.5).to_numpy() & \
                (x["press_neg_30d"].fillna(0) < 1).to_numpy() & ~x["worstQ"].to_numpy()
            capw = part * x["adv63"].fillna(0).to_numpy() * 20 / (aum / M)     # max tranche weight fillable (PIT)
            iss = x["cnpj8"].to_numpy()
            w = np.zeros(len(x))
            left = 1.0
            for tier in (core, ext):
                idx = np.nonzero(tier)[0]
                for _ in range(40):
                    room = np.minimum(capw[idx] - w[idx], 1.0)
                    # issuer cap
                    it = pd.Series(w).groupby(iss).transform("sum").to_numpy()
                    room = np.minimum(room, H.ISSUER_CAP - it[idx])
                    act = idx[room > 1e-6]
                    if left <= 1e-6 or not len(act):
                        break
                    add = min(left / len(act), 1.0)
                    r_ = np.minimum(capw[act] - w[act], H.ISSUER_CAP - it[act])
                    inc = np.minimum(add, r_)
                    inc = np.maximum(inc, 0)
                    w[act] += inc
                    left -= inc.sum()
                if left <= 1e-6:
                    break
            return w
        return f

    def la_targets(aum):
        f = la_weights(aum)
        out = {}
        for p, x in Pd[Pd["univ"]].groupby("dpos"):
            w = f(x)
            s = w.sum()
            if (w > 0).sum() < H.MIN_NAMES:
                out[int(p)] = pd.Series(dtype=float)
                continue
            out[int(p)] = pd.Series(w[w > 0], index=x["b"].to_numpy()[w > 0]).groupby(level=0).sum()
        return out
    for aum in AUMS:
        T = la_targets(aum)
        r = tranche_book(T, aum=aum, hs_b=hs_b, sigma_b=sg_b, adv_b=adv_b, cumV=cumV)
        books[f"P4Q_LA_aum{int(aum / 1e6)}m"] = r
        s = H.stats(r["daily"])
        cap[f"P4Q_LA|aum={int(aum / 1e6)}m"] = {**s, "avg_fill": r.get("avg_fill"), "cost_ann_%": r["cost_ann_%"],
                                               "spread_cost_ann_%": r.get("spread_cost_ann_%"),
                                               "impact_cost_ann_%": r.get("impact_cost_ann_%"),
                                               "avg_exposure": float(r["exposure"].mean()),
                                               "avg_invested_target": float(np.mean([v.sum() for v in T.values()
                                                                                     if len(v)])),
                                               "n_avg": r["n_avg"]}
        log("LA", aum, round(s["ann_excess_%"], 3), "fill", round(r.get("avg_fill", 1), 3))
    RES["capacity"] = cap
    # paired vs P4Q at the same AUM and cost model
    pair = {}
    for aum in AUMS:
        a = f"{int(aum / 1e6)}m"
        st = H.stats(books[f"P4Q_LA_aum{a}"]["daily"], bench=books[f"P4Q_aum{a}"]["daily"])
        su = H.stats(books[f"P4Q_aum{a}"]["daily"], bench=books[f"U_aum{a}"]["daily"])
        pair[a] = {"LA_minus_P4Q_%": st["diff_ann_%"], "t": st["diff_t_nw"], "p": st["diff_p"],
                   "P4Q_minus_U_%": su["diff_ann_%"], "t_P4Q_vs_U": su["diff_t_nw"]}
    RES["capacity_paired"] = pair

    # ---------------------------------------------------------------- the official comparison table (harness stats)
    allv = dict(res)
    tab = H.compare({**allv, "P4": H.baseline("P4"), "U": H.baseline("U")}, bench="P4Q")
    # Holm over the variants I tried only (exclude references)
    tried = list(allv)
    tab.loc[tried, "p_holm"] = H.holm(tab.loc[tried, "p_vs_bench"].astype(float).to_numpy())
    tab.loc[["P4", "U"], "p_holm"] = np.nan
    RES["variants_table_25bps"] = tab.round(4).to_dict(orient="index")
    RES["n_variants_tried"] = len(tried) + 1      # + the liquidity-aware book (tested on its own cost model)
    pv = list(tab.loc[tried, "p_vs_bench"].astype(float)) + [pair["500m"]["p"]]
    RES["holm_all_variants"] = dict(zip(tried + ["P4Q_LA_500m_vs_P4Q_500m"], H.holm(np.array(pv)).tolist()))
    log("\n" + tab.round(3).to_string())
    # 50 bps
    t50 = {}
    b50 = H.baseline("P4Q", cost_bps=50)
    for nm, f in V.items():
        r = H.backtest(f, panel=PM, cost_bps=50)
        s = H.stats(r["daily"], bench=b50["daily"])
        t50[nm] = {"exCDI_%": s["ann_excess_%"], "vs_P4Q_%": s["diff_ann_%"], "t": s["diff_t_nw"]}
    RES["variants_50bps"] = t50
    # like-for-like on the bucket cost model (no AUM): variants vs P4Q
    tb = {}
    for nm, f in V.items():
        r = tranche_book(targets(f), hs_b=hs_b)
        s = H.stats(r["daily"], bench=books["P4Q_bucketcost"]["daily"])
        tb[nm] = {"exCDI_%": s["ann_excess_%"], "vs_P4Q_bucketcost_%": s["diff_ann_%"], "t": s["diff_t_nw"],
                  "cost_ann_%": r["cost_ann_%"]}
        books[f"{nm}_bucketcost"] = r
    RES["variants_bucket_cost"] = tb

    # ---------------------------------------------------------------- de-smoothed risk (GLM)
    log("GLM")
    ser = {"U": H.baseline("U")["daily"], "P4": H.baseline("P4")["daily"], "P4Q": b["daily"],
           "IDA-DI": H.index_excess("IDADI").reindex(b["daily"].index).fillna(0),
           "IBOV": H.index_excess("IBOV").reindex(b["daily"].index).fillna(0),
           "P4Q_minus_U": b["daily"] - H.baseline("U")["daily"]}
    RES["glm"] = {nm: {"monthly_k2": glm(s, 2, "M"), "weekly_k4": glm(s, 4, "W")} for nm, s in ser.items()}
    log("GLM", json.dumps(rnd({k: {"obs": v["monthly_k2"]["sharpe_obs"], "glm": v["monthly_k2"].get("sharpe_true_glm"),
                                    "wk": v["weekly_k4"].get("sharpe_true_glm")} for k, v in RES["glm"].items()}, 2)))

    # ---------------------------------------------------------------- liquidity profile of the books
    prof = {}
    hold_share = {}
    for nm, sel in (("U", U["univ"]), ("P4", U["p4"]), ("P4Q", U["p4q"])):
        z = U[sel]
        prof[nm] = {"median_adv63_R$m": float(z["adv63"].median() / 1e6), "median_tdays63": float(z["tdays63"].median()),
                    "share_liq_q1": float((z["liq_q"] == 1).mean()), "share_liq_q5": float((z["liq_q"] == 5).mean()),
                    "median_cost_rt_bps": float(z["cost_rt_bps"].median()), "mean_cost_rt_bps": float(z["cost_rt_bps"].mean())}
    RES["book_liquidity_profile"] = prof
    # per-quintile carry and forward returns (universe)
    qt = {}
    for q, z in U.groupby("liq_q"):
        qt[str(int(q))] = {"n": int(len(z)), "median_adv_R$m": float(z["adv63"].median() / 1e6),
                           "median_tdays63": float(z["tdays63"].median()),
                           "median_cdi_bps": float(z["cdi_bps"].median()),
                           "fwd126_mean_%": float(z["fwd_126"].mean() * 100),
                           "median_kf_sd": float(z["kf_sd"].median()),
                           "share_executed": float(z["executed"].mean()),
                           "est_round_trip_DI_bps": CT["half_spread_price_bps_by_liq_quintile"]["DI"][str(int(q))] * 2,
                           "est_round_trip_IPCA_PRE_bps": CT["half_spread_price_bps_by_liq_quintile"]["IPCA_PRE"][str(int(q))] * 2}
    RES["liquidity_quintiles"] = qt
    # total SND volume (market depth) and P4Q name volume
    V_, cV = _vol_matrix()
    dd = H.days()
    tv = pd.Series(V_.sum(1), index=dd)
    RES["snd_total_volume_R$bn_per_bday_median_by_year"] = {str(y): float(s[s > 0].median() / 1e9)
                                                            for y, s in tv.groupby(dd.year)}

    # ---------------------------------------------------------------- placebo for the best selection variant
    best = max(res, key=lambda k: RES["variants_table_25bps"][k]["vs_bench_%"])
    RES["best_variant"] = best
    log("placebo", best)
    pl = H.placebo(V[best], n=20, panel=PM)
    RES["placebo_best"] = {"actual": pl["actual"], "mean": pl["mean"], "p95": pl["p95"]}

    # ---------------------------------------------------------------- charts
    log("charts")
    plot(books, res, best, b)
    H.plot_curves({best: res[best], "P4Q @R$500m (costs+impact)": books["P4Q_aum500m"],
                   "P4Q-LA @R$500m": books["P4Q_LA_aum500m"]}, HERE / "equity_harness_style.png",
                  title="marks_liquidity: best variant and capacity-constrained P4+Q (pre-2026)")

    # ---------------------------------------------------------------- sealed holdout (ONCE; frozen choices)
    if holdout:
        log("HOLDOUT")
        PH = SG.signals(H.load_panel("M", holdout=True))
        PH["carry_net"] = PH["cdi_bps"] - PH["cost_rt_bps"].fillna(PH["cost_rt_bps"].max())
        rh = H.backtest(V[best], panel=PH, holdout=True)
        bh = H.baseline("P4Q", holdout=True)
        uh = H.baseline("U", holdout=True)
        RES["holdout_2026"] = {"best": best, "stats": H.stats(rh["daily"], bench=bh["daily"], holdout="only"),
                               "P4Q": H.stats(bh["daily"], bench=uh["daily"], holdout="only")}
    json.dump(rnd(RES), open(HERE / "results.json", "w"), indent=1, default=str)
    log("done")


def plot(books, res, best, b):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    idx = b["daily"].index
    u = H.baseline("U")["daily"].reindex(idx).fillna(0)
    lines = {"CDI": pd.Series(0.0, index=idx), "Universe": u, "P4": H.baseline("P4")["daily"],
             "P4+Q": b["daily"], "IDA-DI": H.index_excess("IDADI"), "Ibovespa": H.index_excess("IBOV"),
             f"{best}": res[best]["daily"],
             "P4+Q, liquidity-bucket costs (no AUM)": books["P4Q_bucketcost"]["daily"],
             "P4+Q @R$500m": books["P4Q_aum500m"]["daily"], "P4+Q @R$2bn": books["P4Q_aum2000m"]["daily"],
             "P4+Q-LA @R$500m": books["P4Q_LA_aum500m"]["daily"], "P4+Q-LA @R$2bn": books["P4Q_LA_aum2000m"]["daily"]}
    sty = {"CDI": ("k", ":"), "Universe": ("#7f7f7f", "--"), "P4": ("#1f77b4", "--"), "P4+Q": ("#17becf", "-"),
           "IDA-DI": ("#bcbd22", "-."), "Ibovespa": ("#c7c7c7", "-.")}
    fig, ax = plt.subplots(figsize=(11, 6), dpi=110)
    for nm, s in lines.items():
        s = s.reindex(idx).fillna(0)
        c, ls = sty.get(nm, (None, "-"))
        ax.plot(idx, H.total_return_curve(s), ls=ls, color=c, lw=1.8 if nm not in sty else 1.1, label=nm)
    ax.set_yscale("log")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, frameon=False, ncol=2)
    ax.set_title("Total return, CDI x (1 + excess), base 100 - pre-2026, net of costs (harness 25 bps unless noted)")
    fig.tight_layout()
    fig.savefig(HERE / "equity_total_return.png")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(11, 6), dpi=110)
    for nm, s in lines.items():
        if nm in ("Universe", "CDI", "Ibovespa"):
            continue
        s = s.reindex(idx).fillna(0)
        c, ls = sty.get(nm, (None, "-"))
        ax.plot(idx, ((1 + s - u).cumprod() - 1) * 100, ls=ls, color=c, lw=1.8 if nm not in sty else 1.1, label=nm)
    ax.axhline(0, color="k", lw=0.6)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, frameon=False, ncol=2)
    ax.set_title("Cumulative excess vs the universe (%) - pre-2026")
    fig.tight_layout()
    fig.savefig(HERE / "cum_excess.png")
    plt.close(fig)
    # capacity chart
    cap = RES["capacity"]
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5), dpi=110)
    xs = [100, 500, 2000]
    for nm, c in (("U", "#7f7f7f"), ("P4Q", "#17becf"), ("P4Q_LA", "#d62728")):
        axs[0].plot(xs, [cap[f"{nm}|aum={a}m"]["ann_excess_%"] for a in xs], "o-", color=c, label=nm)
        axs[1].plot(xs, [cap[f"{nm}|aum={a}m"]["avg_exposure"] for a in xs], "o-", color=c, label=nm)
    axs[0].axhline(cap["P4Q|no_aum|bucket_cost"]["ann_excess_%"], color="#17becf", ls=":", label="P4Q, bucket cost, no AUM")
    for ax in axs:
        ax.set_xscale("log")
        ax.set_xlabel("AUM (R$ m)")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, frameon=False)
    axs[0].set_title("Excess over CDI %/yr, net of spread + sqrt impact")
    axs[1].set_title(f"Average invested share (fills <= {int(PART * 100)}% of SND volume)")
    fig.tight_layout()
    fig.savefig(HERE / "capacity.png")
    plt.close(fig)


if __name__ == "__main__":
    main(holdout="--holdout" in sys.argv)
