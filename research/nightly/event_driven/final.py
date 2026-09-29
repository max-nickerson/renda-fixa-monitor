"""Full pipeline: events -> event studies -> overlays/sleeves -> robustness -> placebo -> charts -> sealed holdout (once).

  PYTHONPATH=. PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
      .venv/Scripts/python.exe research/nightly/event_driven/final.py
"""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from research.nightly import harness as H
from research.nightly.event_driven import study, run
from research.nightly.event_driven.events import build

OUTD = run.OUTD
log = run.log
BEST = "P4Q_exNegEvents"


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else round(float(o), 4)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, pd.Timestamp):
        return str(o.date())
    return o


def main():
    out = {"slug": "event_driven", "harness_version": 4}
    ref, E = build()
    out["event_counts"] = E[E["date"] < H.HOLDOUT].groupby("etype").agg(n=("date", "size"),
                                                                         issuers=("cnpj8", "nunique")).to_dict("index")
    # ---------------- event studies (weekly panel, pre-2026)
    PWar, tab, season, _ = study.run(E=E, ref=ref)
    log("event studies done")
    extra = []
    # redemption announcements split by price vs par ("buy bonds likely to be called above market price")
    rows = study.issuer_event_rows(E, PWar, "resgate")
    rows = rows[rows["univ"]]
    extra.append(study.summarize(rows[rows["ratio"] < 0.995], "resgate [univ, ratio<0.995]"))
    extra.append(study.summarize(rows[rows["ratio"] > 1.005], "resgate [univ, ratio>1.005]"))
    # negative convexity: callable bonds trading above par vs non-callable above par (all universe rows, monthly dates)
    cal = PWar["codigo"].map(ref.set_index("codigo")["callable"])
    U = PWar[PWar["univ"]].assign(callable=cal[PWar["univ"]].to_numpy())
    Um = U[U["day"].dt.day <= 7]    # first week of each month to limit overlap
    for lab, m in [("callable & ratio>1.01", (Um["callable"] == True) & (Um["ratio"] > 1.01)),
                   ("non-callable & ratio>1.01", (Um["callable"] == False) & (Um["ratio"] > 1.01)),
                   ("callable & ratio<0.99", (Um["callable"] == True) & (Um["ratio"] < 0.99)),
                   ("non-callable & ratio<0.99", (Um["callable"] == False) & (Um["ratio"] < 0.99))]:
        g = Um[m]
        extra.append(study.summarize(g.assign(key=g["codigo"]), f"convexity {lab}"))
    ext = pd.DataFrame(extra)
    evt = pd.concat([tab, ext], ignore_index=True)
    out["event_study"] = evt.to_dict("records")
    out["seasoning_curve"] = season.to_dict("records")
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print(evt[["event", "n_events", "n_dates", "ar_c_21_%", "ar_c_63_%", "t_c_63", "ar_c_126_%", "t_c_126", "pre63_ar_%"]])
    # event-study chart (carry/kind-adjusted AR path)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    show = ["ma [univ]", "rating_down [univ]", "agd [univ]", "resgate [univ]", "rating_up [univ]",
            "equity_raise [univ]", "supply_new_issue [univ]", "new_issue_first_week [univ]", "new_issue [incent]"]
    for nm in show:
        r = evt.set_index("event").loc[nm]
        x = [-63, 0, 21, 63, 126]
        pre = r.get("pre63_ar_%", np.nan)
        pre = 0.0 if nm.startswith("new_issue") else pre
        y = [-pre if np.isfinite(pre) else 0, 0, r["ar_c_21_%"], r["ar_c_63_%"], r["ar_c_126_%"]]
        ax.plot(x, y, marker="o", label=f"{nm} (n={int(r['n_events'])})")
    ax.axhline(0, color="grey", lw=0.8)
    ax.axvline(0, color="grey", lw=0.8, ls=":")
    ax.set_xlabel("bdays relative to first decision after the event (pre-window = realised, post = fwd from entry)")
    ax.set_ylabel("cumulative abnormal hedged return, % (carry/dur/kind-adjusted)")
    ax.set_title("Event studies on debentures, 2021-2025 (pre-holdout)")
    ax.legend(fontsize=7.5, loc="lower left")
    fig.tight_layout()
    fig.savefig(OUTD / "event_studies.png", dpi=120)
    plt.close(fig)
    del PWar

    # ---------------- overlays / sleeves (all variants tried -> Holm)
    PM, PW, res, tab2 = run.main()
    out["variants"] = tab2.reset_index().rename(columns={"index": "variant"}).to_dict("records")
    out["n_variants_tried"] = len(res)
    t50 = H.compare({k: H.backtest(run.RULES[k], panel=PM, cost_bps=50, name=k) for k in run.RULES},
                    bench="P4Q", cost_bps=50)
    out["variants_50bps"] = t50[["exCDI_%", "exU_%", "vs_bench_%", "t_vs_bench"]].reset_index().to_dict("records")
    print(t50[["exCDI_%", "exU_%", "vs_bench_%", "t_vs_bench"]])
    # ---------------- robustness of the best variant (same engine/kwargs as the baseline in each row)
    rob = {}
    f = run.RULES[BEST]
    for lab, kw in [("50bps", dict(cost_bps=50)), ("rec40", dict(scenario="rec40")), ("hold63", dict(hold=63)),
                    ("hold252", dict(hold=252)), ("cap5", dict(issuer_cap=0.05))]:
        r = H.backtest(f, panel=PM, name=BEST, **kw)
        b = H.baseline("P4Q", **kw)
        s = H.stats(r["daily"], bench=b["daily"])
        rob[lab] = {"exCDI_%": s["ann_excess_%"], "vs_P4Q_%": s["diff_ann_%"], "t": s["diff_t_nw"],
                    "h1": s["diff_h1_%"], "h2": s["diff_h2_%"]}
    rW = H.backtest(f, freq="W", panel=PW, name=BEST)
    bW = H.baseline("P4Q", freq="W")
    s = H.stats(rW["daily"], bench=bW["daily"])
    rob["weekly_tranches"] = {"exCDI_%": s["ann_excess_%"], "vs_P4Q_%": s["diff_ann_%"], "t": s["diff_t_nw"],
                              "h1": s["diff_h1_%"], "h2": s["diff_h2_%"]}
    out["robustness_best"] = rob
    print(pd.DataFrame(rob).T)
    # ---------------- placebo: drop the SAME number of P4Q issuers at random each date
    def make_placebo(seed):
        def g(x):
            keep = x["p4q"].to_numpy()
            drop_real = keep & run.neg(x)
            iss = x["cnpj8"].to_numpy()
            n_iss = len(set(iss[drop_real]))
            cand = sorted(set(iss[keep]))
            rng = np.random.default_rng(seed * 100003 + int(x["dpos"].iloc[0]))
            dropped = set(rng.choice(cand, size=min(n_iss, len(cand)), replace=False)) if n_iss else set()
            return keep & ~np.isin(iss, list(dropped))
        return g
    best_r = res[BEST]
    b = H.baseline("P4Q")
    real = H.stats(best_r["daily"], bench=b["daily"])["diff_ann_%"]
    pl = []
    for sd in range(40):
        r = H.backtest(make_placebo(sd), panel=PM, name="placebo")
        pl.append(H.stats(r["daily"], bench=b["daily"])["diff_ann_%"])
    pl = np.array(pl)
    out["placebo_random_issuer_drop"] = {"real_vs_P4Q_%": real, "placebo_mean_%": float(pl.mean()),
                                         "placebo_p95_%": float(np.percentile(pl, 95)),
                                         "p_placebo": float((pl >= real).mean()), "n": len(pl)}
    log("placebo", out["placebo_random_issuer_drop"])
    # ---------------- charts (pre-2026)
    curves = {BEST: res[BEST], "P4Q_combo": res["P4Q_combo"], "Sleeve_NewIssue": res["Sleeve_NewIssue"]}
    ser = {k: v["daily"] for k, v in curves.items()}
    refs = {"P4+Q": H.baseline("P4Q")["daily"], "P4": H.baseline("P4")["daily"], "Universe": H.baseline("U")["daily"]}
    idx = refs["P4+Q"].index
    ida = H.index_excess("IDADI").reindex(idx).fillna(0)
    ibov = H.index_excess("IBOV").reindex(idx).fillna(0)
    fig, ax = plt.subplots(figsize=(11, 6))
    for k, v in ser.items():
        ax.plot(H.total_return_curve(v.reindex(idx).fillna(0)), lw=2, label=k)
    for k, v in refs.items():
        ax.plot(H.total_return_curve(v), lw=1.2, ls="--", label=k)
    ax.plot(H.total_return_curve(ida), lw=1, ls=":", label="IDA-DI")
    ax.plot(H.total_return_curve(ibov), lw=1, ls=":", color="grey", label="Ibovespa")
    ax.plot(H.total_return_curve(pd.Series(0.0, index=idx)), lw=1, color="black", label="CDI")
    ax.set_yscale("log")
    ax.set_title("Total return (CDI x (1+excess)), base 100, 2022-2025, 25 bps, 126d tranches")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTD / "equity_total_return.png", dpi=120)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    u = refs["Universe"]
    for k, v in {**ser, "P4+Q": refs["P4+Q"], "P4": refs["P4"]}.items():
        d = (v.reindex(idx).fillna(0) - u.reindex(idx).fillna(0))
        ax.plot(d.cumsum() * 100, lw=2 if k in ser else 1.2, ls="-" if k in ser else "--", label=k)
    ax.plot((ida - u).cumsum() * 100, lw=1, ls=":", label="IDA-DI")
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_ylabel("cumulative excess vs universe, %")
    ax.set_title("Cumulative excess vs the eligible universe (pre-2026)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTD / "cum_excess.png", dpi=120)
    plt.close(fig)
    # ---------------- sealed holdout: ONCE, frozen choices (BEST, P4Q_combo, Sleeve_NewIssue)
    PMh = run.load("M", holdout=True)
    ho = {}
    bh = H.baseline("P4Q", holdout=True)
    uh = H.baseline("U", holdout=True)
    for k in [BEST, "P4Q_combo", "Sleeve_NewIssue"]:
        r = H.backtest(run.RULES[k], panel=PMh, holdout=True, name=k)
        s = H.stats(r["daily"], bench=bh["daily"], holdout="only")
        su = H.stats(r["daily"], bench=uh["daily"], holdout="only")
        ho[k] = {"exCDI_%": s["ann_excess_%"], "vs_P4Q_%": s["diff_ann_%"], "t": s["diff_t_nw"],
                 "vs_U_%": su["diff_ann_%"], "cum_%": s["cum_%"], "months": int(len(H.monthly(r["daily"])[H.monthly(r["daily"]).index >= "2026-01"]))}
    ho["P4Q"] = {"exCDI_%": H.stats(bh["daily"], holdout="only")["ann_excess_%"]}
    out["holdout_2026"] = ho
    log("holdout", ho)
    (OUTD / "results.json").write_text(json.dumps(jsonable(out), indent=1, default=str), encoding="utf-8")
    log("done")


if __name__ == "__main__":
    main()
