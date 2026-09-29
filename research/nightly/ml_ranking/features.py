"""Point-in-time feature matrix for the ml_ranking agent.

Everything here is known at the close of the decision day `day` (grid position `dpos`):
  * panel columns (harness v4 load_panel: lab_daily / weekly extras / fundamentals / equity / market);
  * NEW bond-path features from the harness return matrix R (R[t] = move t->t+1, realised at close t+1, so the
    moves known at close p are R[:p], i.e. the cumulative log return LC[p]);
  * NEW cross-sectional context: carry vs same-date sector / kind median, per-date ranks, issuer aggregates
    (bond count, mean residual, mean past return, worst past return among the issuer's bonds).
No label or look-ahead column is used (dist_stop_LOOKAHEAD, fwd_*, entry_pos, executed, lab_end_* excluded).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from research.nightly import harness as H

BASE_NUM = [
    "cdi_bps", "cdi_pct", "dur", "ratio", "incent", "age", "bench_rate", "resid_bps", "resid_z",
    "press_neg_7d", "press_neg_30d", "distress_2d", "fact_2d", "rat_days_since_down", "eq_ret_1w", "eq_ret_4w",
    "trades_30d", "own_z", "d_spread_1w", "d_spread_4w", "d_ratio_4w", "carry_per_dur",
    "n_fact_30d", "n_fact_90d", "n_distress_30d", "n_distress_90d", "n_deb_mtg_30d", "n_deb_mtg_90d",
    "n_rating_30d", "n_rating_90d", "n_oficio_30d", "n_oficio_90d", "days_since_distress", "news_any_30d",
    "distress_90d", "years_to_mat",
    "f_lev", "f_cov", "f_cash_st", "f_eq_ratio", "f_gde", "f_margin", "f_rev_g", "f_d_lev", "f_d_cov", "f_size",
    "f_is_parent", "f_age_days", "covered", "f_quality", "worstQ",
    "eq_r5", "eq_r21", "eq_r63", "eq_r126", "eq_r252", "eq_vol63", "eq_dd252", "eq_adtv_log", "listed",
    # market state (constant within a date: only useful through interactions in trees)
    "mkt_mom_21", "mkt_distress_z", "idadi_x21", "idadi_x63", "ibov_x21", "ibov_x63", "ida_regime", "cdi_ann",
    "univ_cdi_med", "univ_cdi_iqr",
    "kind_c", "sector_c",
]
NEW = ["br_21", "br_63", "br_126", "bvol_63", "bmdd_126", "bstale_63", "bmin_126",
       "cdi_vs_sector", "cdi_vs_kind", "cdi_x_dur", "resid_rank", "cdi_rank_kind",
       "iss_n", "iss_resid_mean", "iss_br63_mean", "iss_br126_min", "iss_cdi_mean", "iss_worst_bmdd",
       "sec_br63_mean", "sec_n"]
FEATURES = BASE_NUM + NEW
MONO_POS = {"cdi_bps", "resid_z", "resid_bps", "cdi_vs_sector", "cdi_vs_kind"}
MONO_NEG = {"cdi_pct"}


def _path_features(P: pd.DataFrame) -> pd.DataFrame:
    LC = H._LC("base")                      # LC[p, b] = sum log(1+R[:p, b])
    R = H._Rmat("base")
    p = P["dpos"].to_numpy()
    b = P["b"].to_numpy()
    out = {}
    for k in (21, 63, 126):
        out[f"br_{k}"] = np.expm1(LC[p, b] - LC[np.maximum(p - k, 0), b])
    vol = np.full(len(P), np.nan)
    stale = np.full(len(P), np.nan)
    mdd = np.full(len(P), np.nan)
    mn = np.full(len(P), np.nan)
    for i, (pp, bb) in enumerate(zip(p, b)):
        w = R[max(pp - 63, 0):pp, bb]
        if len(w) > 5:
            vol[i] = w.std()
            stale[i] = (w == 0).mean()
        w2 = R[max(pp - 126, 0):pp, bb]
        if len(w2) > 5:
            c = np.cumsum(np.log1p(np.clip(w2, -0.99, None)))
            mdd[i] = np.expm1((c - np.maximum.accumulate(np.r_[0.0, c])[1:]).min())
            mn[i] = w2.min()
    out.update({"bvol_63": vol, "bstale_63": stale, "bmdd_126": mdd, "bmin_126": mn})
    return pd.DataFrame(out, index=P.index)


def build(freq: str = "M", holdout: bool = False) -> pd.DataFrame:
    P = H.load_panel(freq, holdout=holdout).copy()
    P["eq_adtv_log"] = np.log1p(P["eq_adtv"])
    P["kind_c"] = P["kind"].map({"DI_SPREAD": 0, "IPCA": 1, "PRE": 2}).astype(float)
    sectors = sorted(P["sector"].astype(str).unique())
    P["sector_c"] = P["sector"].astype(str).map({s: i for i, s in enumerate(sectors)}).astype(float)
    P = pd.concat([P, _path_features(P)], axis=1)
    # cross-sectional context, computed on the universe rows of each date (other rows get NaN-safe values too)
    g = P.groupby("day")
    P["cdi_vs_sector"] = P["cdi_bps"] - P.groupby(["day", "sector"])["cdi_bps"].transform("median")
    P["cdi_vs_kind"] = P["cdi_bps"] - P.groupby(["day", "kind"])["cdi_bps"].transform("median")
    P["cdi_x_dur"] = P["cdi_bps"] * P["dur"]
    P["resid_rank"] = g["resid_bps"].rank(pct=True)
    P["cdi_rank_kind"] = P.groupby(["day", "kind"])["cdi_bps"].rank(pct=True)
    gi = P.groupby(["day", "cnpj8"])
    P["iss_n"] = gi["codigo"].transform("count")
    P["iss_resid_mean"] = gi["resid_z"].transform("mean")
    P["iss_br63_mean"] = gi["br_63"].transform("mean")
    P["iss_br126_min"] = gi["br_126"].transform("min")
    P["iss_cdi_mean"] = gi["cdi_bps"].transform("mean")
    P["iss_worst_bmdd"] = gi["bmdd_126"].transform("min")
    gs = P.groupby(["day", "sector"])
    P["sec_br63_mean"] = gs["br_63"].transform("mean")
    P["sec_n"] = gs["codigo"].transform("count")
    for c in FEATURES:
        P[c] = pd.to_numeric(P[c], errors="coerce").astype(float)
    return P


if __name__ == "__main__":
    import time
    t = time.time()
    X = build("M")
    print(X.shape, round(time.time() - t, 1))
    U = X[X["univ"]]
    print(U[NEW].describe().T)
