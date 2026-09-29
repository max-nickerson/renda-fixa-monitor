"""alt_signals: alternative & supply signals as overlays on P4+Q (pre-2026 only; holdout reported once at the end).

Rerun:
  PYTHONPATH=. PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
    .venv/Scripts/python.exe research/nightly/alt_signals/run.py [--stage ic|books|holdout|all]
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from research.nightly import harness as H
from research.nightly.alt_signals import signals as S

OUT = Path(__file__).resolve().parent
CACHE = S.CACHE
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


def panel(holdout: bool = False) -> pd.DataFrame:
    f = CACHE / f"panel_alt_M{'_hold' if holdout else ''}.pkl"
    if f.exists():
        return pd.read_pickle(f)
    Q = S.signals(H.load_panel("M", holdout=holdout))
    Q.to_pickle(f)
    return Q


# ------------------------------------------------------------------ stage 1: information coefficients
IC_FEATS = [("sup_iss_90d", "all"), ("sup_iss_365d", "all"), ("sup_sec_mom", "all"),
            ("rat_neg_180d", "all"), ("rat_out_neg_180d", "all"), ("rat_pos_180d", "all"),
            ("nc_margin", "exposed"), ("nc_expo", "exposed"), ("tar_adj_last", "distribution")]


def stage_ic(Q: pd.DataFrame) -> dict:
    res = {}
    for f, scope in IC_FEATS:
        X = Q
        if scope == "exposed":
            X = Q[Q[f].notna()]
        if scope == "distribution":
            X = Q[Q["sector"] == "utilities_distribution"]
        for tgt in ("fwd_63", "fwd_126"):
            for sub in ("univ", "p4q"):
                Z = X if sub == "univ" else X[X["p4q"]]
                try:
                    r = H.ic(Z, f, tgt, min_n=15)
                    res[f"{f}|{tgt}|{sub}"] = {"mean": round(r["mean"], 4), "t_nw": round(r["t_nw"], 2),
                                              "n_dates": r["n_dates"]}
                except Exception as e:  # noqa: BLE001
                    res[f"{f}|{tgt}|{sub}"] = {"err": str(e)[:80]}
            log("IC", f, tgt, res[f"{f}|{tgt}|univ"], res[f"{f}|{tgt}|p4q"])
    # event-conditional forward excess (fwd_126 minus date-universe mean), flagged vs not, within P4+Q
    cond = {}
    U = Q[Q["univ"] & Q["dok_126"] & Q["fwd_126"].notna()].copy()
    U["xs"] = U["fwd_126"] - U.groupby("day")["fwd_126"].transform("mean")
    flags = {"sup_iss_90d>0": U["sup_iss_90d"] > 0, "rat_neg_180d>0": U["rat_neg_180d"] > 0,
             "rat_out_neg>0": U["rat_out_neg_180d"] > 0, "rat_pos_180d>0": U["rat_pos_180d"] > 0,
             "nc_margin<-1": U["nc_margin"] < -1, "nc_margin>1": U["nc_margin"] > 1,
             "nc_expo<-1": U["nc_expo"] < -1, "sup_sec_mom>0.4": U["sup_sec_mom"] > 0.4,
             "tar_adj_last<0": U["tar_adj_last"] < 0}
    for nm, fl in flags.items():
        for sub in ("univ", "p4q"):
            m = fl if sub == "univ" else fl & U["p4q"]
            base = U["p4q"] if sub == "p4q" else pd.Series(True, index=U.index)
            a = U.loc[m].groupby("day")["xs"].mean()
            b = U.loc[base & ~fl].groupby("day")["xs"].mean()
            d = (a - b).dropna()
            cond[f"{nm}|{sub}"] = {"n_rows": int(m.sum()), "n_dates": int(len(d)),
                                   "diff_ann_%": round(float(d.mean()) * 100 * 2, 3) if len(d) else None,
                                   "t_nw": round(H.nw_t(d, 6), 2) if len(d) > 8 else None}
        log("COND", nm, cond[f"{nm}|univ"], cond[f"{nm}|p4q"])
    return {"ic": res, "conditional_fwd126_vs_rest": cond}


# ------------------------------------------------------------------ stage 2: book overlays on P4+Q
def variants() -> dict:
    p = lambda x: x["p4q"].to_numpy()
    nz = lambda x, c: x[c].fillna(0).to_numpy()
    V = {
        "P4Q_ex_nc_margin_bad": lambda x: p(x) & ~(x["nc_margin"].to_numpy() < -1),
        "P4Q_ex_nc_expo_bad": lambda x: p(x) & ~(x["nc_expo"].to_numpy() < -1),
        "P4Q_ex_issuer_new_supply": lambda x: p(x) & ~(nz(x, "sup_iss_90d") > 0),
        "P4Q_only_issuer_new_supply": lambda x: p(x) & (nz(x, "sup_iss_365d") > 0),
        "P4Q_ex_rating_neg": lambda x: p(x) & ~(nz(x, "rat_neg_180d") > 0),
        "P4Q_ex_outlook_neg": lambda x: p(x) & ~(nz(x, "rat_out_neg_180d") > 0),
        "P4Q_ex_hot_sector_supply": lambda x: p(x) & ~(x["sup_sec_mom"].to_numpy() > 0.4),
        "P4Q_ex_tariff_cut": lambda x: p(x) & ~(x["tar_adj_last"].to_numpy() < 0),
        "P4Q_ex_all_alt": lambda x: p(x) & ~((x["nc_margin"].to_numpy() < -1) | (nz(x, "rat_neg_180d") > 0)
                                             | (x["tar_adj_last"].to_numpy() < 0)),
        # added AFTER seeing the IC stage (counted in Holm like the rest)
        "P4Q_ex_supply_or_nc_expo": lambda x: p(x) & ~((nz(x, "sup_iss_90d") > 0) | (x["nc_expo"].to_numpy() < -1)),
        # positive tilt: extend P4+Q with top-40% carry names whose sector nowcast is strong (> +1)
        "P4Q_plus_nc_strong": lambda x: p(x) | ((x["cdi_pct"].to_numpy() <= 0.4) & (x["nc_margin"].to_numpy() > 1)
                                                & (x["resid_z"].to_numpy() > -1.5) & ~x["worstQ"].fillna(False).to_numpy()),
    }
    return V


def stage_books(Q: pd.DataFrame) -> dict:
    V = variants()
    R = {}
    for k, f in V.items():
        R[k] = H.backtest(f, freq="M", hold=126, panel=Q, name=k)
        log("book", k, round(H.stats(R[k]["daily"])["ann_excess_%"], 3), "n", round(R[k]["n_avg"], 1))
    base = {"P4Q": H.baseline("P4Q"), "P4": H.baseline("P4"), "U": H.baseline("U")}
    tab = H.compare({**R, **base}, bench="P4Q")
    log("\n" + tab.round(3).to_string())
    # 50 bps
    R50 = {k: H.backtest(V[k], freq="M", hold=126, panel=Q, cost_bps=50) for k in V}
    tab50 = H.compare({**R50, "P4Q": H.baseline("P4Q", cost_bps=50)}, bench="P4Q", cost_bps=50)
    # rec40
    R40 = {k: H.backtest(V[k], freq="M", hold=126, panel=Q, scenario="rec40") for k in V}
    tab40 = H.compare({**R40, "P4Q": H.baseline("P4Q", scenario="rec40")}, bench="P4Q", scenario="rec40")
    # pick best by t vs P4Q at 25bps (excluding baselines)
    vt = tab.loc[list(V), "t_vs_bench"].astype(float)
    best = vt.idxmax()
    log("best", best)
    # placebo: drop the same number of P4+Q names at random each date (same exclusion count)
    pl = placebo_exclusion(Q, V[best], n=20)
    H.plot_curves({best: R[best], "P4Q_ex_all_alt": R["P4Q_ex_all_alt"]}, OUT / "equity_total_return.png",
                  title=f"alt_signals: {best} vs baselines (pre-2026, 25 bps)")
    cum_excess_plot(R, best, OUT / "cum_excess.png")
    return {"table_25bps": tab.round(4).to_dict(orient="index"), "table_50bps": tab50.round(4).to_dict(orient="index"),
            "table_rec40": tab40.round(4).to_dict(orient="index"), "best": best, "placebo": pl,
            "n_variants": len(V)}


def placebo_exclusion(Q: pd.DataFrame, f, n: int = 20) -> dict:
    """Null: at each date, drop from P4+Q as many random P4+Q names as the rule drops."""
    b = H.stats(H.baseline("P4Q")["daily"])["ann_excess_%"]
    real = H.stats(H.backtest(f, freq="M", hold=126, panel=Q)["daily"])["ann_excess_%"] - b
    vals = []
    for s in range(n):
        rng = np.random.default_rng(1000 + s)

        def g(x, rng=rng):
            p = x["p4q"].to_numpy()
            m = f(x)
            k = int(p.sum() - (m & p).sum())
            out = p.copy()
            if k > 0:
                idx = np.where(p)[0]
                out[rng.choice(idx, size=min(k, len(idx)), replace=False)] = False
            return out | (m & ~p)
        vals.append(H.stats(H.backtest(g, freq="M", hold=126, panel=Q)["daily"])["ann_excess_%"] - b)
    vals = np.array(vals)
    return {"real_vs_P4Q_%": round(real, 3), "placebo_mean_%": round(float(vals.mean()), 3),
            "placebo_p95_%": round(float(np.quantile(vals, 0.95)), 3), "p_placebo": float((vals >= real).mean()),
            "n": n}


def cum_excess_plot(R: dict, best: str, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    b = H.monthly(H.baseline("P4Q")["daily"])
    u = H.monthly(H.baseline("U")["daily"])
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    for k, r in R.items():
        m = H.monthly(r["daily"])
        lw, al = (2.4, 1) if k == best else (1, 0.55)
        ax[0].plot((m - u).cumsum() * 100, lw=lw, alpha=al, label=k)
        ax[1].plot((m - b).cumsum() * 100, lw=lw, alpha=al, label=k)
    ax[0].plot((b - u).cumsum() * 100, color="k", lw=2, ls="--", label="P4Q")
    ax[0].set_title("cumulative excess vs universe (%, monthly sum)")
    ax[1].set_title("cumulative difference vs P4+Q (%)")
    for a in ax:
        a.axhline(0, color="grey", lw=0.6)
        a.grid(alpha=0.3)
    ax[1].legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ------------------------------------------------------------------ stage 3: sealed holdout (once)
def stage_holdout(best: str) -> dict:
    Qh = panel(holdout=True)
    f = variants()[best]
    rh = H.backtest(f, freq="M", hold=126, panel=Qh, holdout=True)
    bh = H.baseline("P4Q", holdout=True)
    uh = H.baseline("U", holdout=True)
    s = H.stats(rh["daily"], bench=bh["daily"], holdout="only")
    su = H.stats(rh["daily"], bench=uh["daily"], holdout="only")
    return {"variant": best, "exCDI_%": s["ann_excess_%"], "vs_P4Q_%": s["diff_ann_%"], "t_vs_P4Q": s["diff_t_nw"],
            "vs_U_%": su["diff_ann_%"], "note": "months >= 2026-01 only; a few months, not significant by construction"}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all")
    a = ap.parse_args()
    res_f = OUT / "results.json"
    res = json.loads(res_f.read_text()) if res_f.exists() else {}
    Q = panel()
    S.export_long(Q, CACHE / "alt_signals_M.pkl")
    if a.stage in ("ic", "all"):
        res["ic_stage"] = stage_ic(Q)
        res_f.write_text(json.dumps(res, indent=1, default=str))
    if a.stage in ("books", "all"):
        res["books"] = stage_books(Q)
        res_f.write_text(json.dumps(res, indent=1, default=str))
    if a.stage in ("holdout", "all"):
        res["holdout"] = stage_holdout(res["books"]["best"])
        res_f.write_text(json.dumps(res, indent=1, default=str))
    log("done")
