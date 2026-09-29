"""w2_expected_loss_carry / model.py

Issuer-month distress hazard panel + walk-forward PD models.

Rows: (cnpj8, d) for d = first business day of each month 2019-04 .. 2026-09
   * every CVM filer with a fundamentals row <= 460 days stale at d (broad training panel, 900+ firms), and
   * every debenture issuer with a bond on the harness grid at d (2021-03+), covered or not.
Label y12: the issuer has ANY distress event (events.pkl: bias_audit distress, severe rating action, bond loss
   episode, CVM RJ/RE/default filing) dated in (d, d + 365 days].  label_end = d + 365 days.
Features (all known at the close of d):
   fundamentals (fund_all.pkl CVM for all filers + fundamentals_pit.pkl incl. brapi; as-of available_date_strict)
   equity (harness eq_dd252 / eq_vol63 / eq_r126, structural_credit sc_dd) - debenture issuers only, 2021+
   bond market (harness panel + factor_zoo features: spread level, resid vs peer, ds_5/21/63, dres_21, marks)
   ratings (days since downgrade; # downgrades 365d from rating_events), past events, sector.
Models (walk-forward, refit monthly, train rows with label_end <= d  -> 12-month embargo):
   'logit'  L2 logistic hazard (C = 0.05 on standardised features + missing flags + sector dummies)
   'lgbm'   LightGBM with monotone constraints (n_jobs 2, 7 leaves, 150 trees, lr 0.05, min_child 80)
Output: data/history/nightly/w2_expected_loss_carry/pd_oos.pkl  (cnpj8, day, pd_logit, pd_lgbm, y12, ...)
"""
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from research.nightly import harness as H

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "data" / "history" / "nightly" / "w2_expected_loss_carry"
RD = ROOT / "research" / "nightly" / "w2_expected_loss_carry"
T0 = time.time()
HOLD = pd.Timestamp(H.HOLDOUT)


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


# ------------------------------------------------------------------------------------------------ fundamentals
def fundamentals() -> pd.DataFrame:
    A = pd.read_pickle(OUT / "fund_all.pkl")
    B = pd.read_pickle(H.HIST / "fundamentals_pit.pkl")
    # the universe file wins for the issuers it covers (it adds brapi gap-fill / parent rows)
    A = A[~A["cnpj8"].isin(set(B["cnpj8"]))]
    F = pd.concat([A, B], ignore_index=True)
    F["avail"] = pd.to_datetime(F["available_date_strict"]).fillna(pd.to_datetime(F["available_date"]))
    F = F.sort_values(["cnpj8", "avail", "period_end"])
    F = F.drop_duplicates(["cnpj8", "avail"], keep="last")
    prev = F.groupby("cnpj8")["period_end"].transform(lambda s: s.cummax().shift())
    F = F[prev.isna() | (F["period_end"] > prev)].copy()
    ta = F["total_assets"].where(F["total_assets"] > 0)
    eb = F["ebitda_ltm"]
    X = pd.DataFrame({"cnpj8": F["cnpj8"], "avail": F["avail"], "period_end": F["period_end"]})
    X["f_nde"] = np.where(eb <= 0, 20.0, F["net_debt_ebitda"].clip(-5, 20))
    X["f_lcov"] = np.where(eb <= 0, np.log(0.05), np.log(F["interest_coverage"].clip(0.05, 100)))
    X["f_lcash_st"] = np.log(F["cash_to_st_debt"].clip(0.02, 50))
    X["f_eqr"] = F["equity_ratio"].clip(-1, 1)
    X["f_gde"] = np.where(F["equity"] <= 0, 20.0, F["gross_debt_equity"].clip(0, 20))
    X["f_margin"] = F["ebitda_margin"].clip(-1, 1)
    X["f_dnde"] = F["d_net_debt_ebitda_4q"].clip(-10, 10)
    X["f_dcov"] = F["d_interest_coverage_4q"].clip(-10, 10)
    X["f_rev_g"] = F["revenue_growth_yoy"].clip(-1, 2)
    X["f_size"] = np.log(ta.clip(1, None))
    X["f_roa"] = (F["net_income_ltm"] / ta).clip(-1, 1)
    X["f_ebit_ta"] = (F["ebit_ltm"] / ta).clip(-1, 1)
    X["f_cash_ta"] = (F["cash"] / ta).clip(0, 1)
    X["f_std_share"] = (F["st_debt"] / F["gross_debt"].where(F["gross_debt"] > 0)).clip(0, 1)
    X["f_neg_eq"] = (F["equity"] <= 0).astype(float)
    X["f_neg_ebitda"] = (eb <= 0).astype(float)
    # margin trend: margin minus the margin of the period one year earlier (same issuer)
    lag = X[["cnpj8", "period_end", "f_margin"]].copy()
    lag["period_end"] = lag["period_end"] + pd.DateOffset(years=1) + pd.offsets.MonthEnd(0)
    X = X.merge(lag.drop_duplicates(["cnpj8", "period_end"]).rename(columns={"f_margin": "f_margin_4q"}),
                on=["cnpj8", "period_end"], how="left")
    X["f_dmargin"] = (X["f_margin"] - X["f_margin_4q"]).clip(-1, 1)
    X = X.drop(columns=["f_margin_4q"])
    num = X.select_dtypes("number").columns
    X[num] = X[num].replace([np.inf, -np.inf], np.nan)
    return X


FUND = ["f_nde", "f_lcov", "f_lcash_st", "f_eqr", "f_gde", "f_margin", "f_dnde", "f_dcov", "f_rev_g", "f_size",
        "f_roa", "f_ebit_ta", "f_cash_ta", "f_std_share", "f_neg_eq", "f_neg_ebitda", "f_dmargin"]
EQ = ["eq_dd252", "eq_vol63", "eq_r126", "sc_dd"]
BOND = ["b_lcdi_med", "b_lcdi_max", "b_resid_med", "b_ratio_min", "b_ds5_max", "b_ds21_max", "b_ds63_max",
        "b_dres21_max", "b_rdd252_min", "b_rfromhi_min", "b_press30", "b_ndist90", "b_nbonds"]
RAT = ["r_days_down", "r_ndown365", "ev_past365", "ev_days_since"]
MONO = {  # +1: higher -> riskier
    "f_nde": 1, "f_lcov": -1, "f_lcash_st": -1, "f_eqr": -1, "f_gde": 1, "f_margin": -1, "f_dnde": 1, "f_dcov": -1,
    "f_rev_g": 0, "f_size": -1, "f_roa": -1, "f_ebit_ta": -1, "f_cash_ta": -1, "f_std_share": 1, "f_neg_eq": 1,
    "f_neg_ebitda": 1, "f_dmargin": -1,
    "eq_dd252": -1, "eq_vol63": 1, "eq_r126": -1, "sc_dd": -1,
    "b_lcdi_med": 1, "b_lcdi_max": 1, "b_resid_med": 1, "b_ratio_min": -1, "b_ds5_max": 1, "b_ds21_max": 1,
    "b_ds63_max": 1, "b_dres21_max": 1, "b_rdd252_min": -1, "b_rfromhi_min": -1, "b_press30": 1, "b_ndist90": 1,
    "b_nbonds": 0,
    "r_days_down": -1, "r_ndown365": 1, "ev_past365": 1, "ev_days_since": -1,
    "has_fund": 0, "has_eq": 0, "has_bond": 0,
}
FEATS = FUND + EQ + BOND + RAT + ["has_fund", "has_eq", "has_bond"]


# ------------------------------------------------------------------------------------------------ panel
def month_days() -> pd.DatetimeIndex:
    P = H.load_panel("M", holdout=True)
    grid = pd.DatetimeIndex(sorted(P["day"].unique()))
    early = pd.bdate_range("2019-04-01", grid[0] - pd.Timedelta(days=1), freq="BMS")
    return early.append(grid)


def bond_block() -> pd.DataFrame:
    from research.nightly.structural_credit.signals import signals as sc_signals
    P = H.load_panel("M", holdout=True)
    Z = pd.read_pickle(H.HIST / "nightly" / "factor_zoo" / "features_v4.pkl")[
        ["day", "codigo", "ds_5", "ds_21", "ds_63", "dres_21", "rdd_252", "ratio_from_hi_252"]]
    P = P.merge(Z, on=["day", "codigo"], how="left")
    P = sc_signals(P)
    P["lcdi"] = np.log1p(P["cdi_bps"].clip(0, 5000))
    g = P.groupby(["cnpj8", "day"])
    B = pd.DataFrame({
        "b_lcdi_med": g["lcdi"].median(), "b_lcdi_max": g["lcdi"].max(), "b_resid_med": g["resid_bps"].median().clip(-500, 1500),
        "b_ratio_min": g["ratio"].min().clip(0.3, 1.2), "b_ds5_max": g["ds_5"].max().clip(-300, 1000),
        "b_ds21_max": g["ds_21"].max().clip(-300, 1500), "b_ds63_max": g["ds_63"].max().clip(-500, 2000),
        "b_dres21_max": g["dres_21"].max().clip(-300, 1500), "b_rdd252_min": g["rdd_252"].min().clip(-0.8, 0),
        "b_rfromhi_min": g["ratio_from_hi_252"].min().clip(-0.8, 0.1), "b_press30": g["press_neg_30d"].max().clip(0, 10),
        "b_ndist90": g["n_distress_90d"].max().clip(0, 10), "b_nbonds": np.log1p(g["codigo"].count()),
        "eq_dd252": g["eq_dd252"].first(), "eq_vol63": g["eq_vol63"].first(), "eq_r126": g["eq_r126"].first(),
        "sc_dd": g["sc_dd"].first().clip(-5, 20), "rat_days_since_down": g["rat_days_since_down"].min(),
        "sector": g["sector"].first(), "listed": g["listed"].max(), "in_univ": g["univ"].max(),
    }).reset_index()
    return B


def build_panel() -> pd.DataFrame:
    days = month_days()
    F = fundamentals()
    B = bond_block()
    E = pd.read_pickle(OUT / "events.pkl")
    Rt = pd.read_pickle(H.HIST / "rating_events.pkl")
    Rt = Rt[Rt["direction"] < 0][["cnpj8", "date"]].copy()
    Rt["date"] = pd.to_datetime(Rt["date"])
    # skeleton: fundamentals-covered issuer-days + bond issuer-days
    sk = []
    Fs = F.sort_values("avail")
    for d in days:
        last = Fs[Fs["avail"] <= d].drop_duplicates("cnpj8", keep="last")
        last = last[(d - last["period_end"]).dt.days <= 460]
        sk.append(pd.DataFrame({"cnpj8": last["cnpj8"].to_numpy(), "day": d}))
    S = pd.concat(sk + [B[["cnpj8", "day"]]], ignore_index=True).drop_duplicates()
    S = S.sort_values("day").reset_index(drop=True)
    # fundamentals as-of
    M = pd.merge_asof(S, Fs, left_on="day", right_on="avail", by="cnpj8", direction="backward")
    stale = (M["day"] - M["period_end"]).dt.days > 460
    M.loc[stale, FUND] = np.nan
    M["has_fund"] = M["f_nde"].notna().astype(float)
    M = M.drop(columns=["avail", "period_end"])
    M = M.merge(B, on=["cnpj8", "day"], how="left")
    M["has_bond"] = M["b_lcdi_med"].notna().astype(float)
    M["has_eq"] = M["eq_vol63"].notna().astype(float)
    M["r_days_down"] = np.log1p(M["rat_days_since_down"].fillna(3650).clip(0, 3650))
    # rating downgrades (any) in the past 365 days, events in the past / future
    ev = E.groupby("cnpj8")["date"].apply(lambda s: np.sort(s.to_numpy()))
    rt = Rt.groupby("cnpj8")["date"].apply(lambda s: np.sort(s.to_numpy()))
    dn = M["day"].to_numpy()
    y = np.zeros(len(M))
    past = np.zeros(len(M))
    since = np.full(len(M), 3650.0)
    nd = np.zeros(len(M))
    first_ev = np.full(len(M), np.datetime64("NaT"), dtype="datetime64[ns]")
    for i, (c, d) in enumerate(zip(M["cnpj8"].to_numpy(), dn)):
        a = ev.get(c)
        if a is not None:
            fut = a[(a > d) & (a <= d + np.timedelta64(365, "D"))]
            y[i] = float(len(fut) > 0)
            if len(fut):
                first_ev[i] = fut[0]
            pa = a[a <= d]
            if len(pa):
                gap = (d - pa[-1]) / np.timedelta64(1, "D")
                since[i] = min(gap, 3650)
                past[i] = float(gap <= 365)
        r = rt.get(c)
        if r is not None:
            nd[i] = float(((r <= d) & (r > d - np.timedelta64(365, "D"))).sum())
    M["y12"] = y
    M["first_event"] = first_ev
    M["ev_past365"] = past
    M["ev_days_since"] = np.log1p(since)
    M["r_ndown365"] = np.clip(nd, 0, 5)
    M["label_end"] = M["day"] + pd.Timedelta(days=365)
    M["sector"] = M["sector"].fillna("unknown")
    return M


# ------------------------------------------------------------------------------------------------ models
def _design(tr: pd.DataFrame, te: pd.DataFrame, sectors):
    Xtr = tr[FEATS].astype(float)
    Xte = te[FEATS].astype(float)
    mu = Xtr.mean()
    sd = Xtr.std().replace(0, 1).fillna(1)
    Ztr = ((Xtr - mu) / sd).clip(-5, 5)
    Zte = ((Xte - mu) / sd).clip(-5, 5)
    miss_tr = Xtr.isna().astype(float).add_prefix("m_")
    miss_te = Xte.isna().astype(float).add_prefix("m_")
    keep = [c for c in miss_tr.columns if miss_tr[c].std() > 0]
    Ztr = pd.concat([Ztr.fillna(0), miss_tr[keep]], axis=1)
    Zte = pd.concat([Zte.fillna(0), miss_te[keep]], axis=1)
    for s in sectors:
        Ztr["s_" + s] = (tr["sector"] == s).astype(float).to_numpy()
        Zte["s_" + s] = (te["sector"] == s).astype(float).to_numpy()
    return Ztr.to_numpy(), Zte.to_numpy(), list(Ztr.columns)


def walk_forward(M: pd.DataFrame, pred_days, C: float = 0.05, n_trees: int = 150) -> pd.DataFrame:
    from sklearn.linear_model import LogisticRegression
    import lightgbm as lgb
    out = []
    sectors = [s for s, n in M["sector"].value_counts().items() if n >= 300 and s != "unknown"]
    mono = [MONO[f] for f in FEATS]
    coefs = {}
    imps = {}
    for d in pred_days:
        tr = M[(M["label_end"] <= d)]
        te = M[M["day"] == d]
        if len(te) == 0 or tr["y12"].sum() < 5:
            continue
        Ztr, Zte, cols = _design(tr, te, sectors)
        lr = LogisticRegression(C=C, max_iter=2000)
        lr.fit(Ztr, tr["y12"].to_numpy())
        p1 = lr.predict_proba(Zte)[:, 1]
        coefs[str(d.date())] = dict(zip(cols, np.round(lr.coef_[0], 4)))
        gb = lgb.LGBMClassifier(n_estimators=n_trees, learning_rate=0.05, num_leaves=7, min_child_samples=80,
                                subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0,
                                monotone_constraints=mono, n_jobs=2, verbose=-1, random_state=0)
        gb.fit(tr[FEATS].astype(float), tr["y12"].to_numpy())
        p2 = gb.predict_proba(te[FEATS].astype(float))[:, 1]
        imps[str(d.date())] = dict(zip(FEATS, np.round(gb.booster_.feature_importance("gain"), 1).tolist()))
        o = te[["cnpj8", "day", "y12", "first_event", "has_fund", "has_eq", "has_bond", "in_univ", "listed",
                "ev_past365", "sector"]].copy()
        o["pd_logit"] = p1
        o["pd_lgbm"] = p2
        o["n_train"] = len(tr)
        o["n_train_pos"] = int(tr["y12"].sum())
        o["n_train_pos_issuers"] = int(tr.loc[tr["y12"] > 0, "cnpj8"].nunique())
        out.append(o)
        if d.month == 1:
            log("pred", d.date(), "train", len(tr), "pos", int(tr["y12"].sum()), "test", len(te))
    return pd.concat(out, ignore_index=True), coefs, imps


def evaluate(O: pd.DataFrame) -> dict:
    from sklearn.metrics import roc_auc_score, brier_score_loss
    res = {}
    O = O.copy()
    # labels usable pre-holdout: the outcome window must end before 2026 (else censor to events < 2026)
    O["y_pre"] = np.where(O["first_event"].notna() & (O["first_event"] < HOLD), 1.0, 0.0)
    O = O[O["day"] < HOLD]
    O["yr"] = O["day"].dt.year
    subsets = {
        "debenture_issuers": O["has_bond"] > 0,
        "debenture_listed": (O["has_bond"] > 0) & (O["has_eq"] > 0),
        "debenture_unlisted": (O["has_bond"] > 0) & (O["has_eq"] == 0),
        "debenture_fund_covered": (O["has_bond"] > 0) & (O["has_fund"] > 0),
        "debenture_no_fund": (O["has_bond"] > 0) & (O["has_fund"] == 0),
        "debenture_clean(no event past 365d)": (O["has_bond"] > 0) & (O["ev_past365"] == 0),
        "broad_cvm_no_bond": O["has_bond"] == 0,
    }
    for nm, m in subsets.items():
        r = {}
        for yr, g in O[m].groupby("yr"):
            yy = g["y_pre"].to_numpy()
            row = {"n": int(len(g)), "pos_rows": int(yy.sum()), "pos_issuers": int(g.loc[yy > 0, "cnpj8"].nunique()),
                   "censored_2025": bool(yr >= 2025)}
            for k in ("pd_logit", "pd_lgbm"):
                p = g[k].to_numpy()
                row[f"auc_{k}"] = float(roc_auc_score(yy, p)) if 0 < yy.sum() < len(yy) else None
                row[f"brier_{k}"] = float(brier_score_loss(yy, p))
                row[f"mean_{k}"] = float(p.mean())
            row["base_rate"] = float(yy.mean())
            r[int(yr)] = row
        g = O[m]
        yy = g["y_pre"].to_numpy()
        allr = {"n": int(len(g)), "pos_rows": int(yy.sum())}
        for k in ("pd_logit", "pd_lgbm"):
            allr[f"auc_{k}"] = float(roc_auc_score(yy, g[k])) if 0 < yy.sum() < len(yy) else None
            allr[f"brier_{k}"] = float(brier_score_loss(yy, g[k]))
        # naive benchmarks: spread level alone, past event alone
        if nm.startswith("debenture") and "b_lcdi_med" in g:
            pass
        r["all"] = allr
        res[nm] = r
    return res


def main():
    path = OUT / "issuer_month.pkl"
    if path.exists():
        M = pd.read_pickle(path)
    else:
        M = build_panel()
        M.to_pickle(path)
    log("panel", M.shape, "issuers", M["cnpj8"].nunique(), "pos rows", int(M["y12"].sum()))
    days = sorted(M["day"].unique())
    pred_days = [pd.Timestamp(d) for d in days if pd.Timestamp(d) >= pd.Timestamp("2020-07-01")]
    O, coefs, imps = walk_forward(M, pred_days)
    O.to_pickle(OUT / "pd_oos.pkl")
    ev = evaluate(O.merge(M[["cnpj8", "day", "b_lcdi_med"]], on=["cnpj8", "day"], how="left"))
    # simple benchmark AUCs on debenture issuers: spread level and past event flag
    from sklearn.metrics import roc_auc_score
    X = O.merge(M[["cnpj8", "day", "b_lcdi_med", "f_nde"]], on=["cnpj8", "day"], how="left")
    X = X[(X["day"] < HOLD) & (X["has_bond"] > 0)]
    yy = (X["first_event"].notna() & (X["first_event"] < HOLD)).astype(float)
    ev["benchmarks_debenture_issuers"] = {
        "auc_spread_level": float(roc_auc_score(yy, X["b_lcdi_med"].fillna(X["b_lcdi_med"].median()))),
        "auc_past_event": float(roc_auc_score(yy, X["ev_past365"])),
        "auc_nde_covered": float(roc_auc_score(yy[X["f_nde"].notna()], X.loc[X["f_nde"].notna(), "f_nde"])),
    }
    # last pre-holdout logit coefficients + LGBM gain importances (for the README)
    last = [k for k in coefs if k < "2026"][-1]
    json.dump({"eval": ev, "logit_coefs_last_pre2026": coefs[last], "last_fit": last,
               "lgbm_gain_last_pre2026": imps[last]},
              open(RD / "model_eval.json", "w"), indent=1, default=str)
    log("done")


if __name__ == "__main__":
    main()
