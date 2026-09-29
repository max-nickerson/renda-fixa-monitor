"""ml_ranking: state-of-the-art tabular models as rankers / overlays on P4+Q (nightly agent).

Stages (each cached; rerun is idempotent):
  python research/nightly/ml_ranking/run.py preds [model ...]   walk-forward predictions  -> data/history/nightly/ml_ranking/
  python research/nightly/ml_ranking/run.py eval                 IC, backtests, overlays, compare, charts, SHAP -> results.json
  python research/nightly/ml_ranking/run.py holdout              ONE-TIME sealed-holdout report of the frozen best variant
  python research/nightly/ml_ranking/run.py all                  preds + eval
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from research.nightly import harness as H
from research.nightly.ml_ranking import features as F
from research.nightly.ml_ranking import models as M

OUT = Path("research/nightly/ml_ranking")
CACHE = Path("data/history/nightly/ml_ranking")
CACHE.mkdir(parents=True, exist_ok=True)
LOG = OUT / "run.log"
BASE_MODELS = ["lgb_reg", "lgb_rank", "xgb_pair", "cat_yeti", "lgb_mono", "enet_int", "hier", "lgb_peer", "lgb_loss"]


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(time.strftime("%H:%M:%S ") + s + "\n")


def feats(holdout=False) -> pd.DataFrame:
    path = CACHE / f"features_M{'_hold' if holdout else ''}.pkl"
    if path.exists():
        return pd.read_pickle(path)
    X = F.build("M", holdout=holdout)
    X.to_pickle(path)
    return X


# ---------------------------------------------------------------------------------------------------------- preds
def stage_preds(names):
    P = feats()
    for nm in names:
        path = CACHE / f"pred_{nm}.pkl"
        if path.exists():
            log(f"{nm}: cached")
            continue
        log(f"{nm}: walk-forward")
        pr, params, _ = M.walk_forward(nm, P, log=log)
        pd.to_pickle({"pred": pr, "params": params}, path)


def load_preds(names=BASE_MODELS) -> dict:
    return {nm: pd.read_pickle(CACHE / f"pred_{nm}.pkl")["pred"] for nm in names if (CACHE / f"pred_{nm}.pkl").exists()}


def add_pct(df: pd.DataFrame, sign=1.0) -> pd.DataFrame:
    df = df.copy()
    df["pct"] = (sign * df["pred"]).groupby(df["dpos"]).rank(pct=True)
    return df


def ensembles(preds: dict, P: pd.DataFrame) -> dict:
    """ens = equal rank average of the return models; stack = non-negative weights re-estimated each date on
    PAST out-of-sample predictions whose labels are realised (lab_end_126 <= date)."""
    ret_models = [m for m in preds if m != "lgb_loss"]
    R = None
    for m in ret_models:
        d = add_pct(preds[m])[["dpos", "codigo", "day", "cnpj8", "pct"]].rename(columns={"pct": m})
        R = d if R is None else R.merge(d[["dpos", "codigo", m]], on=["dpos", "codigo"], how="inner")
    R["pred"] = R[ret_models].mean(axis=1)
    ens = R[["day", "dpos", "codigo", "cnpj8", "pred"]].copy()
    # stacking
    lab = P[["dpos", "codigo", M.LAB, f"lab_end_{M.H_LAB}"]]
    RL = R.merge(lab, on=["dpos", "codigo"], how="left")
    RL["yr"] = RL.groupby("dpos")[M.LAB].rank(pct=True) - 0.5
    from scipy.optimize import nnls
    out = []
    for p in sorted(RL["dpos"].unique()):
        hist = RL[(RL[f"lab_end_{M.H_LAB}"] <= p) & RL["yr"].notna()]
        te = RL[RL["dpos"] == p]
        if hist["dpos"].nunique() >= 4:
            w, _ = nnls(hist[ret_models].to_numpy() - 0.5, hist["yr"].to_numpy())
            w = w / w.sum() if w.sum() > 0 else np.ones(len(ret_models)) / len(ret_models)
        else:
            w = np.ones(len(ret_models)) / len(ret_models)
        out.append(pd.DataFrame({"day": te["day"], "dpos": p, "codigo": te["codigo"], "cnpj8": te["cnpj8"],
                                 "pred": te[ret_models].to_numpy() @ w}))
    stack = pd.concat(out, ignore_index=True)
    return {"ens": ens, "stack": stack}


# ---------------------------------------------------------------------------------------------------------- eval
def ic_table(allp: dict, P: pd.DataFrame) -> dict:
    lab = P[P["univ"]][["dpos", "codigo", M.LAB, f"dok_{M.H_LAB}", "cdi_bps", "p4q"]]
    out = {}
    for nm, d in list(allp.items()) + [("cdi_bps (ref)", None)]:
        if d is None:
            X = lab.assign(pred=lab["cdi_bps"])
            X = X[X["dpos"].isin(allp["lgb_reg"]["dpos"].unique())]
        else:
            X = d.merge(lab, on=["dpos", "codigo"], how="left")
        sgn = -1 if nm == "lgb_loss" else 1
        X = X[X[f"dok_{M.H_LAB}"] & X[M.LAB].notna()]
        s = X.groupby("dpos").apply(lambda x: x["pred"].corr(x[M.LAB] * sgn, method="spearman"))
        # IC inside P4+Q (what an overlay can use)
        q = X[X["p4q"]].groupby("dpos").apply(
            lambda x: x["pred"].corr(x[M.LAB] * sgn, method="spearman") if len(x) > 15 else np.nan).dropna()
        out[nm] = {"ic": round(float(s.mean()), 4), "ic_t_nw": round(H.nw_t(s, 6), 2), "n_dates": int(len(s)),
                   "ic_within_p4q": round(float(q.mean()), 4), "ic_within_p4q_t": round(H.nw_t(q, 6), 2)}
    return out


def overlay_signals(nm: str, d: pd.DataFrame, P: pd.DataFrame) -> dict:
    """Signals (DataFrames day,codigo,select|weight|score) for the overlay family of one model."""
    sign = -1.0 if nm == "lgb_loss" else 1.0          # loss model: high P(loss) = bad
    d = add_pct(d, sign)
    U = P[P["univ"]][["day", "dpos", "codigo", "p4q"]].merge(d[["dpos", "codigo", "pct"]], on=["dpos", "codigo"],
                                                            how="inner")
    sig = {}
    sig["veto10"] = U.assign(select=U["p4q"] & (U["pct"] > 0.10))[["day", "codigo", "select"]]
    sig["veto20"] = U.assign(select=U["p4q"] & (U["pct"] > 0.20))[["day", "codigo", "select"]]
    sig["tilt"] = U.assign(weight=np.where(U["p4q"], 0.5 + U["pct"], 0.0))[["day", "codigo", "weight"]]
    if nm != "lgb_loss":
        sig["top20"] = U.assign(score=U["pct"])[["day", "codigo", "score"]]
    return sig


def run_backtests(sigs: dict, **kw) -> dict:
    res = {}
    for k, s in sigs.items():
        res[k] = H.backtest(s, name=k, **kw)
    return res


def fmt_table(df: pd.DataFrame) -> str:
    cols = ["exCDI_%", "exU_%", "t_vsU", "vs_bench_%", "t_vs_bench", "p_holm", "h1_vs_bench", "h2_vs_bench",
            "sharpe", "maxDD_%", "turnover", "n_avg"]
    d = df[cols].astype(float).round(2)
    lines = ["| variant | " + " | ".join(cols) + " |", "|" + "---|" * (len(cols) + 1)]
    for i, r in d.iterrows():
        lines.append(f"| {i} | " + " | ".join(f"{v:g}" for v in r.to_numpy()) + " |")
    return "\n".join(lines)


def shap_best(model_name: str, P: pd.DataFrame, params: dict):
    """SHAP (TreeSHAP via LightGBM pred_contrib / xgboost / catboost) of the last pre-2026 refit of `model_name`,
    evaluated on the 2025 out-of-sample decision rows."""
    cls = M.MODELS[model_name]
    U = P[P["univ"]]
    p_fit = int(U.loc[U["day"] >= "2025-10-01", "dpos"].min())
    tr = M.labelled(P, p_fit)
    m = cls(dict(params[2025])).fit(tr)
    te = U[U["day"] >= "2025-01-01"]
    if model_name.startswith("lgb"):
        contrib = m.m.predict(te[F.FEATURES], pred_contrib=True)[:, :-1]
    elif model_name == "xgb_pair":
        import xgboost as xgb
        contrib = m.m.get_booster().predict(xgb.DMatrix(te[F.FEATURES]), pred_contribs=True)[:, :-1]
    elif model_name == "cat_yeti":
        from catboost import Pool
        contrib = m.m.get_feature_importance(Pool(te[F.FEATURES].to_numpy(np.float32),
                                                  group_id=te["dpos"].to_numpy()), type="ShapValues")[:, :-1]
    else:
        return None
    imp = pd.Series(np.abs(contrib).mean(0), index=F.FEATURES).sort_values(ascending=False)
    # direction: correlation of the feature (per-date rank) with its SHAP value
    dirn = {}
    for f in imp.index[:20]:
        x = te[f].to_numpy()
        ok = np.isfinite(x)
        dirn[f] = float(np.corrcoef(x[ok], contrib[ok, F.FEATURES.index(f)])[0, 1]) if ok.sum() > 50 else np.nan
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    top = imp.iloc[:20][::-1]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(top.index, top.to_numpy(), color=["#2a7ab9" if dirn.get(f, 0) >= 0 else "#c0504d" for f in top.index])
    ax.set_title(f"SHAP mean |contribution| — {model_name} (fit 2025-10, OOS rows 2025)\nblue: higher value -> higher score; red: lower")
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "shap_best.png", dpi=110)
    plt.close(fig)
    return {"model": model_name, "mean_abs_shap": imp.round(6).iloc[:25].to_dict(),
            "direction_corr": {k: round(v, 3) for k, v in dirn.items()}}


def stage_eval():
    t0 = time.time()
    P = feats()
    preds = load_preds()
    params = {nm: pd.read_pickle(CACHE / f"pred_{nm}.pkl")["params"] for nm in preds}
    allp = dict(preds)
    allp.update(ensembles(preds, P))
    # non-ML controls through the SAME overlay family (does ML add anything beyond one raw column?)
    U0 = P[P["univ"] & P["dpos"].isin(allp["lgb_reg"]["dpos"].unique())]
    for c, col in (("ctrl_carry", "cdi_bps"), ("ctrl_resid", "resid_z")):
        allp[c] = U0[["day", "dpos", "codigo", "cnpj8"]].assign(pred=U0[col].fillna(U0[col].median()).to_numpy())
    for k, v in allp.items():
        v.to_pickle(CACHE / f"oos_{k}.pkl") if k in ("ens", "stack") else None
    res = {"n_models": len(allp), "models": list(allp), "params": {k: {str(y): p for y, p in v.items()}
                                                                    for k, v in params.items()}}
    log("IC table")
    res["ic"] = ic_table(allp, P)
    for k, v in res["ic"].items():
        log(f"  {k:14s} {v}")
    # backtests: every overlay of every model = the Holm family
    B, sigs_all = {}, {}
    for nm, d in allp.items():
        for ov, s in overlay_signals(nm, d, P).items():
            sigs_all[f"{nm}:{ov}"] = s
    log(f"{len(sigs_all)} variants")
    for k, s in sigs_all.items():
        B[k] = H.backtest(s, name=k)
    refs = {"P4": H.baseline("P4"), "U": H.baseline("U"), "P4Q": H.baseline("P4Q")}
    tab = H.compare(B, bench="P4Q")
    tab = tab.sort_values("vs_bench_%", ascending=False)
    reft = H.compare(refs, bench="P4Q")
    log(fmt_table(tab.head(15)))
    res["n_variants"] = len(B)
    res["table_25bps"] = tab.round(4).to_dict(orient="index")
    res["refs_25bps"] = reft.round(4).to_dict(orient="index")
    best = tab.index[0]
    res["best_variant_pre2026"] = best
    log("best", best)
    # robustness of the best + top-5 variants
    top5 = list(tab.index[:5])
    rob = {}
    for k in top5:
        s = sigs_all[k]
        r50 = H.backtest(s, cost_bps=50)
        b50 = H.baseline("P4Q", cost_bps=50)
        r40 = H.backtest(s, scenario="rec40")
        b40 = H.baseline("P4Q", scenario="rec40")
        rW = H.backtest(s, hold=63)
        bW = H.baseline("P4Q", hold=63)
        coh = H.cohort_excess(s, H=126)
        cohq = H.cohort_excess("p4q", H=126)
        dco = (coh - cohq).dropna()
        rob[k] = {"net50_diff_vs_p4q": H.stats(r50["daily"], bench=b50["daily"])["diff_ann_%"],
                  "net50_t": H.stats(r50["daily"], bench=b50["daily"])["diff_t_nw"],
                  "rec40_diff_vs_p4q": H.stats(r40["daily"], bench=b40["daily"])["diff_ann_%"],
                  "hold63_diff_vs_p4q": H.stats(rW["daily"], bench=bW["daily"])["diff_ann_%"],
                  "cohort126_diff_vs_p4q_ann": float(dco.mean() / 0.5 * 100), "cohort_t": H.nw_t(dco, 6),
                  "stats": H.stats(B[k]["daily"], bench=refs["P4Q"]["daily"])}
        log(k, {a: (round(b, 3) if isinstance(b, float) else "") for a, b in rob[k].items() if a != "stats"})
    res["robust_top5"] = rob
    # placebo for the best: random vetoes of the same size inside P4+Q
    res["placebo_best"] = placebo_veto(sigs_all[best], P)
    log("placebo", res["placebo_best"])
    # SHAP of the best tree model (by IC among tree models)
    trees = ["lgb_reg", "lgb_rank", "xgb_pair", "cat_yeti", "lgb_mono", "lgb_peer", "lgb_loss"]
    best_model = best.split(":")[0]
    shap_model = best_model if best_model in trees else max([t for t in trees if t != "lgb_loss"],
                                                            key=lambda t: res["ic"][t]["ic"])
    try:
        res["shap"] = shap_best(shap_model, P, params[shap_model])
    except Exception as e:  # noqa
        log("shap failed", e)
    # curves
    showk = [best] + [k for k in tab.index[1:] if k.split(":")[0] != best.split(":")[0]][:1] + \
            [k for k in ["stack:top20", "ens:veto10"] if k in B and k != best]
    H.plot_curves({k: B[k] for k in showk}, OUT / "equity_total_return.png",
                  title="ml_ranking: model overlays on P4+Q (pre-2026, 25 bps, 126d tranches)")
    cum_excess_plot({k: B[k] for k in showk}, refs)
    # save the best signal pieces for the combiner
    export_signals(allp)
    res["runtime_eval_s"] = round(time.time() - t0, 1)
    json.dump(res, open(OUT / "results.json", "w"), indent=1, default=float)
    log("eval done", res["runtime_eval_s"])


def placebo_veto(sig: pd.DataFrame, P: pd.DataFrame, n: int = 30) -> dict:
    """Null for an overlay: remove the same NUMBER of names from P4+Q at random each date (or random tilt)."""
    base = H.backtest(sig)
    b = H.baseline("P4Q")
    act = H.stats(base["daily"], bench=b["daily"])["diff_ann_%"]
    U = P[P["univ"] & (P["day"] >= H.START)][["day", "codigo", "p4q"]]
    rng = np.random.default_rng(0)
    col = "select" if "select" in sig.columns else "weight"
    out = []
    for i in range(n):
        if col == "select":
            S = U.merge(sig, on=["day", "codigo"], how="left")
            S["select"] = S["select"].fillna(False).astype(bool)
            rows = []
            for d, x in S.groupby("day"):
                k = int(x["select"].sum())
                q = x[x["p4q"]]
                keep = set(rng.choice(q["codigo"].to_numpy(), size=min(k, len(q)), replace=False)) if len(q) else set()
                rows.append(pd.DataFrame({"day": d, "codigo": x["codigo"], "select": x["codigo"].isin(keep)}))
            s = pd.concat(rows)
        else:
            S = U.merge(sig, on=["day", "codigo"], how="left").fillna({"weight": 0})
            S["weight"] = np.where(S["p4q"], 0.5 + rng.random(len(S)), 0.0)
            s = S[["day", "codigo", "weight"]]
        r = H.backtest(s)
        out.append(H.stats(r["daily"], bench=b["daily"])["diff_ann_%"])
    out = np.array(out)
    return {"actual_diff_vs_p4q": act, "placebo_mean": float(out.mean()), "placebo_p95": float(np.percentile(out, 95)),
            "p_value": float((out >= act).mean()), "n": n}


def cum_excess_plot(B: dict, refs: dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    u = H.monthly(refs["U"]["daily"])
    q = H.monthly(refs["P4Q"]["daily"])
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    for k, r in list(B.items()) + [("P4", refs["P4"]), ("P4+Q", refs["P4Q"])]:
        m = H.monthly(r["daily"])
        ax[0].plot((1 + m - u.reindex(m.index).fillna(0)).cumprod() * 100 - 100, label=k,
                   lw=2 if k in B else 1.2, ls="-" if k in B else "--")
        if k != "P4+Q":
            ax[1].plot((1 + m - q.reindex(m.index).fillna(0)).cumprod() * 100 - 100, label=k,
                       lw=2 if k in B else 1.2, ls="-" if k in B else "--")
    ax[0].set_title("cumulative excess vs universe (%), monthly, 25 bps")
    ax[1].set_title("cumulative paired difference vs P4+Q (%)")
    for a in ax:
        a.axhline(0, color="k", lw=0.5)
        a.axvline(pd.Timestamp("2024-01-01"), color="grey", lw=0.5, ls=":")
        a.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "cum_excess.png", dpi=110)
    plt.close(fig)


def export_signals(allp: dict):
    """Reusable PIT signal: per decision date, the per-date percentile of each model's score (codigo, cnpj8, date)."""
    rows = []
    for k in ("stack", "ens", "lgb_peer", "lgb_loss", "lgb_rank", "hier"):
        if k in allp:
            d = add_pct(allp[k], -1.0 if k == "lgb_loss" else 1.0)
            rows.append(d.assign(signal=f"ml_{k}_pct")[["codigo", "cnpj8", "day", "signal", "pct"]]
                        .rename(columns={"day": "date", "pct": "value"}))
    pd.concat(rows, ignore_index=True).to_pickle(CACHE / "signals_ml_ranking.pkl")


def signals(panel: pd.DataFrame | None = None, name: str = "ml_stack_pct") -> pd.DataFrame:
    """Combiner entry point. Returns (codigo, cnpj8, date, value) for pre-2026 monthly decision dates (walk-forward,
    known at the close of `date`; higher = better). If `panel` is given, merged onto its (codigo, day)."""
    S = pd.read_pickle(CACHE / "signals_ml_ranking.pkl")
    S = S[S["signal"] == name].drop(columns="signal")
    if panel is None:
        return S
    return panel.merge(S.rename(columns={"date": "day", "value": name})[["codigo", "day", name]],
                       on=["codigo", "day"], how="left")


# ---------------------------------------------------------------------------------------------------------- holdout
def stage_holdout(variant: str):
    """ONE-TIME: frozen variant (model, overlay, hyper-params of 2025) predicted walk-forward on 2026 dates."""
    nm, ov = variant.split(":")
    PH = feats(holdout=True)
    U = PH[PH["univ"]]
    dates26 = np.sort(U.loc[U["day"] >= H.HOLDOUT, "dpos"].unique())
    base = {}
    names = [nm] if nm not in ("ens", "stack") else [m for m in BASE_MODELS if m != "lgb_loss"]
    for m in names:
        old = pd.read_pickle(CACHE / f"pred_{m}.pkl")
        prm = dict(old["params"])
        prm[2026] = prm[2025]                        # frozen: no tuning on holdout data
        new, _, _ = M.walk_forward(m, PH, pred_dates=dates26, params_by_year=prm, log=log)
        base[m] = pd.concat([old["pred"], new], ignore_index=True)
    if nm in ("ens", "stack"):
        d = ensembles(base, PH)[nm]
    else:
        d = base[nm]
    sig = overlay_signals(nm, d, PH)[ov]
    r = H.backtest(sig, holdout=True)
    b = H.baseline("P4Q", holdout=True)
    u = H.baseline("U", holdout=True)
    out = {"variant": variant, "holdout_only": H.stats(r["daily"], bench=b["daily"], holdout="only"),
           "holdout_vs_U": H.stats(r["daily"], bench=u["daily"], holdout="only"),
           "p4q_holdout_only": H.stats(b["daily"], bench=u["daily"], holdout="only"),
           "full_incl_2026": H.stats(r["daily"], bench=b["daily"], holdout=True)}
    log("HOLDOUT", json.dumps(out, default=float))
    res = json.load(open(OUT / "results.json"))
    res["holdout"] = out
    json.dump(res, open(OUT / "results.json", "w"), indent=1, default=float)


if __name__ == "__main__":


    st = sys.argv[1] if len(sys.argv) > 1 else "all"
    if st in ("preds", "all"):
        stage_preds(sys.argv[2:] if (st == "preds" and len(sys.argv) > 2) else BASE_MODELS)
    if st in ("eval", "all"):
        stage_eval()
    if st == "holdout":
        stage_holdout(sys.argv[2])
