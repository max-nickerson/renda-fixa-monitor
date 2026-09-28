"""Walk-forward ML scores for the weekly lab (with / without news features)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BASE = ["cdi_bps", "resid_bps", "resid_z", "own_z", "d_spread_1w", "d_spread_4w", "d_ratio_4w", "carry_per_dur",
        "dur", "T", "incent", "kind_ipca", "kind_pre", "log_trades_30d", "age_days", "ratio", "mkt_mom_21"]
NEWS = ["n_fact_30d", "n_fact_90d", "n_distress_30d", "n_distress_90d", "n_deb_mtg_30d", "n_deb_mtg_90d",
        "n_rating_90d", "n_oficio_30d", "n_oficio_90d", "log_days_since_distress", "mkt_distress_z", "news_any_30d"]
H = 4          # forecast horizon (weeks)
RETRAIN = 13   # weeks
PURGE = H + 1


def prepare(g: pd.DataFrame, ret_col: str = "ret") -> pd.DataFrame:
    """ret_col='ret_hedged' → labels and backtests on rate-hedged (pure credit) returns."""
    g = g.copy()
    if ret_col != "ret":
        g["ret_unhedged"] = g["ret"]
        g["ret"] = g[ret_col]
    g["kind_ipca"] = g["kind"].eq("IPCA").astype(int)
    g["kind_pre"] = g["kind"].eq("PRE").astype(int)
    g["log_trades_30d"] = np.log1p(g["trades_30d"].fillna(0))
    g["log_days_since_distress"] = np.log1p(g["days_since_distress"].clip(upper=3650))
    # forward H-week excess return, starting one week after the decision (execution lag)
    r = g.groupby("codigo")["ret"]
    fwd = sum(r.shift(-k) for k in range(1, H + 1))
    g["y_fwd"] = fwd
    g["y_blowup"] = (fwd < -0.03).astype(float).where(fwd.notna())
    return g


def walk_forward_scores(g: pd.DataFrame, features: list[str], start: str = "2022-01-01",
                        kind: str = "rank") -> pd.Series:
    """kind='rank' → Ridge on forward return; kind='blowup' → logistic P(4w return < −3%)."""
    weeks = sorted(g["week"].unique())
    test_weeks = [w for w in weeks if w >= pd.Timestamp(start)]
    out = pd.Series(np.nan, index=g.index)
    el = g[g["eligible"]]
    for i in range(0, len(test_weeks), RETRAIN):
        block = test_weeks[i:i + RETRAIN]
        cutoff = block[0] - pd.Timedelta(weeks=PURGE)
        tr = el[(el["week"] <= cutoff) & el["y_fwd"].notna()]
        med = tr[features].median()
        X = tr[features].fillna(med)
        if kind == "rank":
            y = tr["y_fwd"].clip(tr["y_fwd"].quantile(0.01), tr["y_fwd"].quantile(0.99))
            m = make_pipeline(StandardScaler(), Ridge(alpha=10.0)).fit(X, y)
            pred = lambda Z: m.predict(Z)
        else:
            y = tr["y_blowup"].astype(int)
            if y.nunique() < 2:
                continue
            m = make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=1000, class_weight="balanced")).fit(X, y)
            pred = lambda Z: m.predict_proba(Z)[:, 1]
        rows = g[g["week"].isin(block)]
        out.loc[rows.index] = pred(rows[features].fillna(med))
    return out


def weekly_pct(g: pd.DataFrame, col: str, ascending: bool = False) -> pd.Series:
    """Percentile within each week among eligible bonds (0 = best)."""
    s = pd.Series(np.nan, index=g.index)
    el = g["eligible"] & g[col].notna()
    s[el] = g[el].groupby("week")[col].rank(pct=True, ascending=ascending)
    return s
