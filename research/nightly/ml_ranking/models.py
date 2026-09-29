"""Walk-forward model zoo for the ml_ranking agent.

Protocol (per model):
  * decision dates = monthly panel dates; predictions for every date >= PRED_START (2022-01);
  * refit every REFIT_EVERY decision dates (semi-annual); a model used at date p was fit only on universe rows
    whose 126-bday label is realised by the refit date (lab_end_126 <= refit dpos  -> embargo = horizon);
  * hyper-parameters: nested time-series CV with optuna, tuned ONCE at the first 2024 refit (data realised by
    2024-01) and reused for 2024-25 (2022-23 use defaults; CatBoost defaults only - CPU budget): inner validation = last 6 labelled decision dates, inner train =
    rows whose label is realised before the first validation date (embargo again). 2022 uses defaults.
    Objective: mean per-date Spearman IC (AUC for the loss classifier). Holdout (>= 2026) never touched.
"""
from __future__ import annotations

import time
import warnings

import numpy as np
import optuna
import pandas as pd
from scipy.stats import spearmanr, rankdata
from sklearn.metrics import roc_auc_score

from research.nightly.ml_ranking.features import FEATURES, MONO_NEG, MONO_POS

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

H_LAB = 126
LAB = f"fwd_{H_LAB}"
LOSS_THR = -0.03
PRED_START = pd.Timestamp("2022-01-01")
REFIT_EVERY = 6
NTHREAD = 1
TUNE_YEAR = 2024   # one nested-CV tuning (data to 2023-12), reused for 2024-25; 2022-23 use defaults


# ------------------------------------------------------------------------------------------------ targets
def make_target(df: pd.DataFrame, target: str) -> np.ndarray:
    y = df[LAB].to_numpy(float)
    if target == "raw":        # winsorised, demeaned per date (cross-sectional model)
        lo, hi = np.nanpercentile(y, [1, 99])
        y = np.clip(y, lo, hi)
        return y - df.assign(_y=y).groupby("dpos")["_y"].transform("mean").to_numpy()
    if target == "rank":       # per-date percentile rank
        return df.assign(_y=y).groupby("dpos")["_y"].rank(pct=True).to_numpy()
    if target == "grade":      # 0..4 quintile relevance for learning-to-rank
        r = df.assign(_y=y).groupby("dpos")["_y"].rank(pct=True).to_numpy()
        return np.minimum((r * 5).astype(int), 4)
    if target == "peer":       # excess over the same-date peer-group mean (winsorised)
        lo, hi = np.nanpercentile(y, [1, 99])
        y = np.clip(y, lo, hi)
        return y - df.assign(_y=y).groupby(["dpos", "peer"])["_y"].transform("mean").to_numpy()
    if target == "loss":
        return (y < LOSS_THR).astype(int)
    raise ValueError(target)


def per_date_ic(dpos, pred, y) -> float:
    d = pd.DataFrame({"d": dpos, "p": pred, "y": y})
    v = [spearmanr(x["p"], x["y"])[0] for _, x in d.groupby("d") if len(x) > 30 and x["p"].nunique() > 2]
    return float(np.nanmean(v)) if v else -1.0


def rank_gauss(df: pd.DataFrame, cols) -> np.ndarray:
    """Per-date rank -> (-0.5, 0.5), NaN -> 0 (the date median)."""
    out = np.zeros((len(df), len(cols)))
    g = df.groupby("dpos")
    for j, c in enumerate(cols):
        r = g[c].rank(pct=True).to_numpy() - 0.5
        out[:, j] = np.nan_to_num(r, nan=0.0)
    return out


# ------------------------------------------------------------------------------------------------ learners
class Learner:
    target = "raw"
    trials = 6

    def __init__(self, params=None):
        self.params = params or self.default()

    def default(self):
        return {}

    def space(self, trial):
        return {}

    def fit(self, tr: pd.DataFrame):
        raise NotImplementedError

    def predict(self, te: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    def score(self, va: pd.DataFrame, pred) -> float:
        if self.target == "loss":
            y = make_target(va, "loss")
            return roc_auc_score(y, pred) if 0 < y.sum() < len(y) else 0.5
        return per_date_ic(va["dpos"].to_numpy(), pred, va[LAB].to_numpy())


def _sorted(df):
    return df.sort_values(["dpos", "b"], kind="mergesort")


class LGBReg(Learner):
    target = "raw"
    objective = "regression"
    mono = False

    def default(self):
        return dict(num_leaves=15, min_data_in_leaf=200, learning_rate=0.03, n_estimators=300,
                    feature_fraction=0.7, bagging_fraction=0.8, lambda_l2=5.0)

    def space(self, t):
        return dict(num_leaves=t.suggest_int("num_leaves", 4, 63, log=True),
                    min_data_in_leaf=t.suggest_int("min_data_in_leaf", 50, 1000, log=True),
                    learning_rate=t.suggest_float("learning_rate", 0.01, 0.1, log=True),
                    n_estimators=t.suggest_int("n_estimators", 100, 600, step=50),
                    feature_fraction=t.suggest_float("feature_fraction", 0.4, 0.9),
                    bagging_fraction=t.suggest_float("bagging_fraction", 0.5, 0.9),
                    lambda_l2=t.suggest_float("lambda_l2", 0.1, 50, log=True))

    def _mk(self):
        import lightgbm as lgb
        kw = dict(objective=self.objective, n_jobs=NTHREAD, verbose=-1, bagging_freq=1, random_state=7, max_bin=63, force_col_wise=True,
                  **self.params)
        if self.mono:
            kw["monotone_constraints"] = [1 if f in MONO_POS else -1 if f in MONO_NEG else 0 for f in FEATURES]
            kw["monotone_constraints_method"] = "advanced"
        if self.objective == "lambdarank":
            return lgb.LGBMRanker(**kw, lambdarank_truncation_level=60, label_gain=[0, 1, 3, 7, 15])
        if self.objective == "binary":
            return lgb.LGBMClassifier(**kw)
        return lgb.LGBMRegressor(**kw)

    def fit(self, tr):
        tr = _sorted(tr)
        y = make_target(tr, self.target)
        self.m = self._mk()
        if self.objective == "lambdarank":
            grp = tr.groupby("dpos", sort=True).size().to_numpy()
            self.m.fit(tr[FEATURES], y, group=grp)
        else:
            self.m.fit(tr[FEATURES], y)
        return self

    def predict(self, te):
        if self.objective == "binary":
            return self.m.predict_proba(te[FEATURES])[:, 1]
        return self.m.predict(te[FEATURES])


class LGBRank(LGBReg):
    target = "grade"
    objective = "lambdarank"


class LGBMono(LGBReg):
    mono = True


class LGBPeer(LGBReg):
    target = "peer"


class LGBLoss(LGBReg):
    target = "loss"
    objective = "binary"


class XGBPair(Learner):
    target = "grade"
    trials = 6

    def default(self):
        return dict(max_depth=4, learning_rate=0.05, n_estimators=300, subsample=0.8, colsample_bytree=0.7,
                    min_child_weight=20, reg_lambda=5.0)

    def space(self, t):
        return dict(max_depth=t.suggest_int("max_depth", 2, 7),
                    learning_rate=t.suggest_float("learning_rate", 0.01, 0.15, log=True),
                    n_estimators=t.suggest_int("n_estimators", 100, 500, step=50),
                    subsample=t.suggest_float("subsample", 0.5, 0.9),
                    colsample_bytree=t.suggest_float("colsample_bytree", 0.4, 0.9),
                    min_child_weight=t.suggest_float("min_child_weight", 1, 100, log=True),
                    reg_lambda=t.suggest_float("reg_lambda", 0.1, 50, log=True))

    def fit(self, tr):
        import xgboost as xgb
        tr = _sorted(tr)
        y = make_target(tr, self.target)
        self.m = xgb.XGBRanker(objective="rank:pairwise", tree_method="hist", max_bin=63, n_jobs=NTHREAD, random_state=7,
                               **self.params)
        self.m.fit(tr[FEATURES], y, qid=tr["dpos"].to_numpy())
        return self

    def predict(self, te):
        return self.m.predict(te[FEATURES])


class CatYeti(Learner):
    target = "rank"
    trials = 0          # too slow on this machine (~1 min per fit): defaults only

    def default(self):
        return dict(depth=5, learning_rate=0.1, iterations=150, l2_leaf_reg=5.0)

    def space(self, t):
        return dict(depth=t.suggest_int("depth", 3, 7),
                    learning_rate=t.suggest_float("learning_rate", 0.03, 0.15, log=True),
                    iterations=t.suggest_int("iterations", 150, 350, step=50),
                    l2_leaf_reg=t.suggest_float("l2_leaf_reg", 1, 30, log=True))

    def fit(self, tr):
        from catboost import CatBoost, Pool
        tr = _sorted(tr)
        y = make_target(tr, self.target)
        pool = Pool(tr[FEATURES].to_numpy(np.float32), label=y, group_id=tr["dpos"].to_numpy())
        self.m = CatBoost(dict(loss_function="YetiRank", thread_count=NTHREAD, random_seed=7, verbose=False,
                               **self.params))
        self.m.fit(pool)
        return self

    def predict(self, te):
        return self.m.predict(te[FEATURES].to_numpy(np.float32))


# linear block: per-date rank features + pairwise interactions of the strongest drivers
LIN_CORE = ["cdi_bps", "resid_z", "f_quality", "dur", "eq_r63", "br_63", "bvol_63", "trades_30d", "press_neg_30d",
            "iss_resid_mean", "cdi_vs_sector", "years_to_mat"]


class ENetInt(Learner):
    target = "raw"
    trials = 6

    def default(self):
        return dict(alpha=1e-4, l1_ratio=0.2)

    def space(self, t):
        return dict(alpha=t.suggest_float("alpha", 1e-6, 1e-2, log=True),
                    l1_ratio=t.suggest_float("l1_ratio", 0.0, 1.0))

    def _X(self, df):
        Z = rank_gauss(df, FEATURES)
        idx = [FEATURES.index(c) for c in LIN_CORE]
        inter = [Z[:, i] * Z[:, j] for a, i in enumerate(idx) for j in idx[a + 1:]]
        miss = np.c_[df["f_quality"].isna(), df["eq_r63"].isna()].astype(float)
        return np.c_[Z, np.array(inter).T, miss]

    def fit(self, tr):
        from sklearn.linear_model import ElasticNet
        y = make_target(tr, self.target)
        self.m = ElasticNet(max_iter=3000, **self.params).fit(self._X(tr), y)
        return self

    def predict(self, te):
        return self.m.predict(self._X(te))


class HierShrink(ENetInt):
    """Ridge on ranked features + empirical-Bayes issuer and sector random effects on its training residuals:
    effect_g = sum(resid_g) / (n_g + k). Uses only realised labels (issuer 'alpha persistence')."""
    trials = 6

    def default(self):
        return dict(alpha=1e-4, l1_ratio=0.0, k_iss=20.0, k_sec=200.0)

    def space(self, t):
        return dict(alpha=t.suggest_float("alpha", 1e-6, 1e-2, log=True), l1_ratio=0.0,
                    k_iss=t.suggest_float("k_iss", 1, 500, log=True),
                    k_sec=t.suggest_float("k_sec", 10, 5000, log=True))

    def fit(self, tr):
        from sklearn.linear_model import ElasticNet
        pr = dict(self.params)
        k_iss, k_sec = pr.pop("k_iss"), pr.pop("k_sec")
        y = make_target(tr, self.target)
        X = self._X(tr)
        self.m = ElasticNet(max_iter=3000, **pr).fit(X, y)
        res = y - self.m.predict(X)
        d = pd.DataFrame({"iss": tr["cnpj8"].to_numpy(), "sec": tr["sector"].astype(str).to_numpy(), "r": res})
        gs = d.groupby("sec")["r"].agg(["sum", "count"])
        self.sec = gs["sum"] / (gs["count"] + k_sec)
        d["r2"] = d["r"] - d["sec"].map(self.sec).fillna(0)
        gi = d.groupby("iss")["r2"].agg(["sum", "count"])
        self.iss = gi["sum"] / (gi["count"] + k_iss)
        return self

    def predict(self, te):
        return (self.m.predict(self._X(te)) + te["sector"].astype(str).map(self.sec).fillna(0).to_numpy()
                + te["cnpj8"].map(self.iss).fillna(0).to_numpy())


MODELS = {
    "lgb_reg": LGBReg, "lgb_rank": LGBRank, "xgb_pair": XGBPair, "cat_yeti": CatYeti, "lgb_mono": LGBMono,
    "enet_int": ENetInt, "hier": HierShrink, "lgb_peer": LGBPeer, "lgb_loss": LGBLoss,
}


# ------------------------------------------------------------------------------------------------ walk-forward
def labelled(P: pd.DataFrame, upto: int) -> pd.DataFrame:
    return P[P["univ"] & (P[f"lab_end_{H_LAB}"] <= upto) & P[LAB].notna()]


def tune(cls, P: pd.DataFrame, p0: int, seed: int = 0) -> dict:
    if cls.trials == 0:
        return cls().default()
    L = labelled(P, p0)
    dts = np.sort(L["dpos"].unique())
    if len(dts) < 12:
        return cls().default()
    val_d = dts[-6:]
    va = L[L["dpos"].isin(val_d)]
    itr = L[L[f"lab_end_{H_LAB}"] <= val_d[0]]
    if itr["dpos"].nunique() < 4:
        return cls().default()

    def obj(t):
        m = cls(cls().space(t) or None)
        try:
            m.fit(itr)
            return m.score(va, m.predict(va))
        except Exception as e:  # noqa
            print("   trial failed", e)
            return -1.0

    st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    st.enqueue_trial({k: v for k, v in cls().default().items()})
    st.optimize(obj, n_trials=cls.trials)
    best = cls().default()
    best.update(st.best_params)
    return best


def walk_forward(name: str, P: pd.DataFrame, pred_dates=None, params_by_year=None, log=print):
    """Returns (predictions DataFrame [day, dpos, codigo, cnpj8, pred], params_by_year, last fitted model)."""
    cls = MODELS[name]
    U = P[P["univ"]]
    dates = np.sort(U.loc[U["day"] >= PRED_START, "dpos"].unique()) if pred_dates is None else np.sort(pred_dates)
    dday = U.drop_duplicates("dpos").set_index("dpos")["day"]
    params_by_year = dict(params_by_year or {})
    out, model, last_fit, t0 = [], None, None, time.time()
    for i, p in enumerate(dates):
        yr = dday[p].year
        if model is None or (i % REFIT_EVERY == 0):
            if yr not in params_by_year:
                if yr == TUNE_YEAR:
                    params_by_year[yr] = tune(cls, P, int(p), seed=yr)
                elif yr > TUNE_YEAR and TUNE_YEAR in params_by_year:
                    params_by_year[yr] = dict(params_by_year[TUNE_YEAR])
                else:
                    params_by_year[yr] = cls().default()
                log(f"  [{name}] {yr} params {params_by_year[yr]}")
            tr = labelled(P, int(p))
            model = cls(dict(params_by_year[yr])).fit(tr)
            last_fit = int(p)
        te = U[U["dpos"] == p]
        pr = model.predict(te)
        out.append(pd.DataFrame({"day": te["day"].to_numpy(), "dpos": p, "codigo": te["codigo"].to_numpy(),
                                 "cnpj8": te["cnpj8"].to_numpy(), "pred": pr, "fit_pos": last_fit}))
    log(f"  [{name}] done {len(dates)} dates in {time.time() - t0:.0f}s")
    return pd.concat(out, ignore_index=True), params_by_year, model
