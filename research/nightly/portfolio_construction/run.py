"""Portfolio construction study on the P4+Q alpha source (harness v4, sealed holdout).

  PYTHONPATH=. PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
      .venv/Scripts/python.exe research/nightly/portfolio_construction/run.py            # pre-2026 study
  ... run.py --holdout     # ONCE, after the recommendation is frozen (RECOMMENDED below)
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

from research.nightly import harness as H
from research.nightly.portfolio_construction import build_weights as BW

OUTD = Path(__file__).resolve().parent
RECOMMENDED = "mv_emp_g20_fund500m"      # frozen before the holdout run (see README)


def wsig(W, v):
    return W.loc[W["variant"] == v, ["day", "codigo", "weight"]]


def buffer_signal(P):
    """Hysteresis (turnover penalty by rule): hold a name until it leaves a wider band (carry top 45%, not rich,
    no negative press, not worst-quality) instead of the P4+Q entry band (top 30%)."""
    X = P[P["univ"] & (P["day"] >= H.START)]
    held, rows = set(), []
    for d, x in X.groupby("day", sort=True):
        keep = (x["codigo"].isin(held) & (x["cdi_pct"] <= 0.45) & ~x["rich"].astype(bool)
                & (x["press_neg_30d"].fillna(0) == 0) & ~x["worstQ"].fillna(False).astype(bool))
        sel = x["p4q"].astype(bool) | keep
        held = set(x.loc[sel, "codigo"])
        rows.append(pd.DataFrame({"day": d, "codigo": x["codigo"], "select": sel.to_numpy()}))
    return pd.concat(rows)


def engine_variants(holdout=False):
    PM = H.load_panel("M", holdout=holdout)
    out = {}
    kw = dict(holdout=holdout)
    out["live_W_replace"] = H.backtest("p4q", freq="W", hold=None, **kw)
    out["live_M_replace"] = H.backtest("p4q", freq="M", hold=None, **kw)
    out["live_Q_replace"] = H.backtest("p4q", freq="M", rebalance="Q", hold=None, **kw)
    orig = H._rebal_book
    try:
        H._rebal_book = lambda *a, **k: orig(*a, **{**k, "band": 0.5})
        out["live_M_band50"] = H.backtest("p4q", freq="M", hold=None, **kw)
    finally:
        H._rebal_book = orig
    out["live_M_buffer45"] = H.backtest(buffer_signal(PM), freq="M", hold=None, **kw)
    out["live_M_sticky_rich"] = H.backtest("p4q", freq="M", hold=None, exit_signal="rich", **kw)
    out["tranche_h63"] = H.backtest("p4q", hold=63, **kw)
    out["tranche_h252"] = H.backtest("p4q", hold=252, **kw)
    out["tranche_W126"] = H.backtest("p4q", freq="W", hold=126, **kw)
    out["tranche_Q126"] = H.backtest("p4q", freq="M", rebalance="Q", hold=126, **kw)
    out["tranche_buffer45"] = H.backtest(buffer_signal(PM), freq="M", hold=126, **kw)
    return out


def construction_variants(W, holdout=False, **kw):
    out = {}
    for v in sorted(W["variant"].unique()):
        out[v] = H.backtest(wsig(W, v), freq="M", hold=126, issuer_cap=1.0, holdout=holdout, name=v, **kw)
    return out


def rnd(d, k=3):
    if isinstance(d, dict):
        return {a: rnd(b, k) for a, b in d.items()}
    if isinstance(d, (float, np.floating)):
        return None if not np.isfinite(d) else round(float(d), k)
    if isinstance(d, (np.integer,)):
        return int(d)
    return d


def topk_carry(holdout=False, k=25, fund=False):
    """Optimizer-free explanation of the MV books: EW over the k highest-carry P4+Q names, issuer cap 5%
    (fund=True: + sector 25%, duration band, liquidity at R$500m, min lot, via the same projection)."""
    from research.nightly.portfolio_construction import construct as CO
    P = H.load_panel("M", holdout=holdout)
    L = BW.build_liq()[["codigo", "day", "vol91_brl"]]
    X = P[P["univ"] & (P["day"] >= H.START)].merge(L, on=["codigo", "day"], how="left")
    X["sector"] = X["sector"].astype(str)
    X["cnpj8"] = X["cnpj8"].astype(str)
    rows = []
    for d, x in X.groupby("day", sort=True):
        q = x[x["p4q"]].sort_values("cdi_bps", ascending=False).reset_index(drop=True)
        s = q.head(k)
        if len(s) < 5:
            continue
        if not fund:
            w = CO.project(s, np.ones(len(s)), {"issuer_cap": 0.05})
        else:
            # liquidity-feasible: extend down the carry ranking (k, 1.4k, ...) before relaxing the liquidity cap
            ud = float(x["dur"].mean())
            w, m, kk = None, 1.0, k
            while w is None and m < 50:
                s = q.head(kk).reset_index(drop=True)
                c = {"issuer_cap": 0.05, "sector_cap": 0.25, "dur_band": (ud - 1, ud + 1),
                     "wmax": np.minimum(BW.liq_wmax(s, BW.AUM) * m, 1.0), "min_lot": BW.MIN_LOT}
                w = CO.project(s, np.ones(len(s)), c)
                if w is None:
                    if kk < len(q):
                        kk = int(kk * 1.4) + 1
                    else:
                        m *= 1.5
        if w is None:
            continue
        rows.append(pd.DataFrame({"day": d, "codigo": s["codigo"], "weight": w}))
    return pd.concat(rows)


def main():
    t0 = time.time()
    W, M = BW.build(holdout=False, tag="pre")
    W = pd.concat([W, topk_carry().assign(variant="top25_carry_cap5"),
                   topk_carry(fund=True).assign(variant="top25_carry_fund500m")], ignore_index=True)
    res = construction_variants(W)
    res.update(engine_variants())
    tab = H.compare(res, bench="P4Q")
    n_var = int(tab["p_vs_bench"].notna().sum())
    # same-engine diffs for live books
    b_live = res["live_M_replace"]["daily"]
    tab["vs_live_M_replace_%"] = [H.stats(r["daily"], bench=b_live)["diff_ann_%"] for r in res.values()]
    tab["cost_ann_%"] = [r["cost_ann_%"] for r in res.values()]
    # rec40 + 50 bps for every construction variant (cheap)
    r40 = {v: H.backtest(wsig(W, v), freq="M", hold=126, issuer_cap=1.0, scenario="rec40") for v in W["variant"].unique()}
    b40 = H.baseline("P4Q", scenario="rec40")["daily"]
    r50 = {v: H.backtest(wsig(W, v), freq="M", hold=126, issuer_cap=1.0, cost_bps=50) for v in W["variant"].unique()}
    b50 = H.baseline("P4Q", cost_bps=50)["daily"]
    for v in r40:
        tab.loc[v, "rec40_vs_P4Q_%"] = H.stats(r40[v]["daily"], bench=b40)["diff_ann_%"]
        s50 = H.stats(r50[v]["daily"], bench=b50)
        tab.loc[v, "net50_vs_P4Q_%"] = s50["diff_ann_%"]
        tab.loc[v, "net50_exCDI_%"] = s50["ann_excess_%"]
    for nm in ("live_M_replace", "live_W_replace", "live_Q_replace", "live_M_buffer45", "tranche_buffer45"):
        r = H.backtest(buffer_signal(H.load_panel("M")) if "buffer" in nm else "p4q",
                       freq="W" if "_W_" in nm else "M", rebalance="Q" if "_Q_" in nm else None,
                       hold=126 if nm.startswith("tranche") else None, cost_bps=50)
        tab.loc[nm, "net50_vs_P4Q_%"] = H.stats(r["daily"], bench=b50)["diff_ann_%"]
        tab.loc[nm, "net50_exCDI_%"] = H.stats(r["daily"])["ann_excess_%"]
    # ex-ante construction metrics (averages over decision dates)
    mm = M[~M["variant"].isin(["_params"])].groupby("variant").mean(numeric_only=True)
    for c in ["n", "eff_n_issuer", "max_issuer", "max_sector", "dur", "spread_bps", "cvar99_6m_%", "exp_loss_6m_%",
              "liq_excess_500m", "liq_excess_2bn", "days_to_trade_500m_med", "liq_relax"]:
        if c in mm:
            tab[c] = mm[c].reindex(tab.index)
    params = M[M["variant"] == "_params"].mean(numeric_only=True).to_dict()
    # placebo for the recommended construction (same weights, random names)
    pl = H.placebo(wsig(W, RECOMMENDED).assign(), n=20, freq="M", hold=126, issuer_cap=1.0)
    stats_rec = H.stats(res[RECOMMENDED]["daily"], bench=H.baseline("P4Q")["daily"])
    stats_p4q = H.stats(H.baseline("P4Q")["daily"])
    tab = tab.sort_values("exCDI_%", ascending=False)
    pd.set_option("display.width", 250)
    print(tab.round(3).to_string())
    tab.to_csv(OUTD / "results_table.csv")
    # ---------------- charts
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    show = {f"Recommended: {RECOMMENDED}": res[RECOMMENDED],
            "MV emp cov gamma20 (unconstrained liquidity)": res["mv_emp_g20"],
            "Top-25 carry in P4+Q, issuer 5%": res["top25_carry_cap5"],
            "EW fund-constrained R$500m": res["EW_fund500m"],
            "Live monthly replace (P4+Q)": res["live_M_replace"]}
    H.plot_curves(show, OUTD / "equity_total_return.png",
                  title="Portfolio construction on P4+Q (pre-2026, net 25 bps; tranche 126d unless 'live')")
    fig, axs = plt.subplots(1, 2, figsize=(15, 5.5), dpi=110)
    u = H.baseline("U")["daily"]
    pq = H.baseline("P4Q")["daily"]
    for nm, r in {**show, "P4+Q": {"daily": pq}, "P4": H.baseline("P4")}.items():
        s = r["daily"].reindex(u.index).fillna(0)
        ls = "--" if nm in ("P4+Q", "P4") else "-"
        axs[0].plot(s.index, ((1 + s - u).cumprod() - 1) * 100, ls, label=nm)
        axs[1].plot(s.index, ((1 + s - pq.reindex(s.index).fillna(0)).cumprod() - 1) * 100, ls, label=nm)
    axs[0].set_title("Cumulative excess vs universe (%)")
    axs[1].set_title("Cumulative excess vs P4+Q (%)")
    for a in axs:
        a.grid(alpha=0.3); a.axhline(0, color="k", lw=0.6); a.legend(fontsize=7, frameon=False)
    fig.tight_layout(); fig.savefig(OUTD / "cum_excess.png"); plt.close(fig)
    # frontier
    fig, axs = plt.subplots(1, 3, figsize=(17, 5.5), dpi=110)
    cons_rows = tab[tab["cvar99_6m_%"].notna()]
    for ax, xc, xl in ((axs[0], "cvar99_6m_%", "ex-ante 6m CVaR99 with default scenarios (%)"),
                       (axs[1], "maxDD_%", "realised max drawdown of monthly excess (%)"),
                       (axs[2], "turnover", "one-way turnover (x book / yr)")):
        d = tab if xc != "cvar99_6m_%" else cons_rows
        ax.scatter(d[xc], d["exCDI_%"], s=25, c=["C3" if i == RECOMMENDED else ("C0" if i in cons_rows.index else "C2")
                                               for i in d.index])
        for i, r in d.iterrows():
            ax.annotate(i, (r[xc], r["exCDI_%"]), fontsize=6)
        ax.axhline(H.stats(pq)["ann_excess_%"], color="grey", ls="--", lw=0.8)
        ax.set_xlabel(xl); ax.set_ylabel("excess over CDI %/yr (net 25 bps)"); ax.grid(alpha=0.3)
    fig.suptitle("Construction trade-offs (blue: tranche constructions, green: engine/rebalancing, red: recommended; dashed = P4+Q)")
    fig.tight_layout(); fig.savefig(OUTD / "frontier.png"); plt.close(fig)
    # reusable output: recommended weights
    wsig(W, RECOMMENDED).to_pickle(BW.OUT / "recommended_weights_pre2026.pkl")
    out = {"slug": "portfolio_construction", "harness_version": H._VERSION, "n_variants_holm": n_var,
           "recommended": RECOMMENDED, "table": rnd(tab.to_dict(orient="index")),
           "recommended_stats_vs_P4Q": rnd(stats_rec), "P4Q_stats": rnd(stats_p4q),
           "placebo_recommended": rnd({"actual": pl["actual"], "mean": pl["mean"], "p95": pl["p95"]}),
           "risk_model_params_avg": rnd(params), "runtime_s": round(time.time() - t0)}
    prev = {}
    if (OUTD / "results.json").exists():
        prev = json.loads((OUTD / "results.json").read_text())
    if "holdout" in prev:
        out["holdout"] = prev["holdout"]
    (OUTD / "results.json").write_text(json.dumps(out, indent=1, default=str))
    print("placebo", pl["mean"], pl["p95"], pl["actual"], "runtime", time.time() - t0)


def holdout_run():
    """ONCE, after RECOMMENDED is frozen. 2026 months only."""
    W, M = BW.build(holdout=True, tag="all")
    W = pd.concat([W, topk_carry(holdout=True).assign(variant="top25_carry_cap5"),
                   topk_carry(holdout=True, fund=True).assign(variant="top25_carry_fund500m")], ignore_index=True)
    bq = H.baseline("P4Q", holdout=True)["daily"]
    bu = H.baseline("U", holdout=True)["daily"]
    out = {}
    for v in (RECOMMENDED, "mv_emp_g20", "top25_carry_cap5", "top25_carry_fund500m", "EW_fund500m", "cvar_l5"):
        r = H.backtest(wsig(W, v), freq="M", hold=126, issuer_cap=1.0, holdout=True)
        s = H.stats(r["daily"], bench=bq, holdout="only")
        su = H.stats(r["daily"], bench=bu, holdout="only")
        out[v] = {"exCDI_%": s["ann_excess_%"], "vs_P4Q_%": s["diff_ann_%"], "t": s["diff_t_nw"],
                  "vs_U_%": su["diff_ann_%"], "n_months": s["n_months"], "period": s["period"]}
    out["P4Q"] = {"exCDI_%": H.stats(bq, holdout="only")["ann_excess_%"]}
    out["U"] = {"exCDI_%": H.stats(bu, holdout="only")["ann_excess_%"]}
    r = H.backtest(wsig(W, RECOMMENDED), freq="M", hold=126, issuer_cap=1.0, holdout=True)
    H.plot_curves({f"Recommended ({RECOMMENDED})": r}, OUTD / "equity_total_return_incl_holdout.png",
                  holdout=True, title="Incl. sealed 2026 holdout (red line)")
    print(json.dumps(rnd(out), indent=1))
    res = json.loads((OUTD / "results.json").read_text())
    res["holdout"] = rnd(out)
    (OUTD / "results.json").write_text(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    if "--holdout" in sys.argv:
        holdout_run()
    else:
        main()
