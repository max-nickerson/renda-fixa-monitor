"""Robustness of the best overlay (P4Q ex issuer-new-supply-90d or exposure nowcast < -1). Pre-2026 only.

- window sensitivity of the supply exclusion (30/60/180/365d) and nowcast threshold (-0.5/-1.5)
- weekly-tranche engine, hold 63 / 252, cohort level
- which issuers drive it (contribution: rerun excluding the top issuers from the rule)
- placebo for the nowcast leg: permute the sector -> nowcast mapping among exposed sectors (20 draws)
All extra variants are added to one Holm family with the 11 from run.py (reported as n_variants_total).
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.nightly import harness as H
from research.nightly.alt_signals import signals as S
from research.nightly.alt_signals.run import variants, log

OUT = Path(__file__).resolve().parent


def rule(win_col="sup_iss_90d", thr=-1.0, nc="nc_expo", drop_issuers=()):
    def f(x):
        p = x["p4q"].to_numpy()
        bad = (x[win_col].fillna(0).to_numpy() > 0) | (x[nc].to_numpy() < thr)
        if drop_issuers:
            bad &= ~x["cnpj8"].isin(drop_issuers).to_numpy()
        return p & ~bad
    return f


def main():
    Q = pd.read_pickle(S.CACHE / "panel_alt_M.pkl")
    o = S.load_offers()
    for w in (30, 60, 180):
        Q[f"sup_iss_{w}d"] = S.issuer_event_counts(Q, o, {"x": w})["x"].to_numpy()
    res = {}
    R = {}
    R["best_supply30"] = H.backtest(rule("sup_iss_30d"), panel=Q)
    R["best_supply60"] = H.backtest(rule("sup_iss_60d"), panel=Q)
    R["best_supply180"] = H.backtest(rule("sup_iss_180d"), panel=Q)
    R["best_supply365"] = H.backtest(rule("sup_iss_365d"), panel=Q)
    R["best_nc-0.5"] = H.backtest(rule(thr=-0.5), panel=Q)
    R["best_nc-1.5"] = H.backtest(rule(thr=-1.5), panel=Q)
    R["best_nc_margin"] = H.backtest(rule(nc="nc_margin"), panel=Q)
    best = variants()["P4Q_ex_supply_or_nc_expo"]
    R["best"] = H.backtest(best, panel=Q)
    tab = H.compare({**R, "P4Q": H.baseline("P4Q")}, bench="P4Q")
    log("\n" + tab.round(3).to_string())
    res["sensitivity"] = tab.round(4).to_dict(orient="index")

    # engines / horizons (paired against P4Q on the same engine)
    eng = {}
    for nm, kw in {"hold63": dict(hold=63), "hold252": dict(hold=252)}.items():
        r = H.backtest(best, panel=Q, **kw)
        s = H.stats(r["daily"], bench=H.baseline("P4Q", **kw)["daily"])
        eng[nm] = {"exCDI": s["ann_excess_%"], "vs_P4Q": s["diff_ann_%"], "t": s["diff_t_nw"]}
        log(nm, eng[nm])
    # weekly tranches need the weekly panel with the signals
    QW = S.signals(H.load_panel("W"))
    r = H.backtest(best, freq="W", panel=QW)
    s = H.stats(r["daily"], bench=H.baseline("P4Q", freq="W")["daily"])
    eng["weekly126"] = {"exCDI": s["ann_excess_%"], "vs_P4Q": s["diff_ann_%"], "t": s["diff_t_nw"]}
    log("weekly", eng["weekly126"])
    del QW
    for Hh in (63, 126, 252):
        a = H.cohort_excess(best, H=Hh, panel=Q)
        b = H.cohort_excess("p4q", H=Hh, panel=Q)
        d = (a - b).dropna()
        eng[f"cohort{Hh}"] = {"diff_ann_%": round(float(d.mean()) / (Hh / 252) * 100, 3),
                              "t": round(H.nw_t(d, max(Hh // 21, 1)), 2), "n": int(len(d))}
        log("cohort", Hh, eng[f"cohort{Hh}"])
    res["engines"] = eng

    # which issuers does the rule remove most (weight-days) -> leave-top-5-out
    U = Q[Q["univ"] & Q["p4q"] & (Q["day"] >= H.START)]
    bad = (U["sup_iss_90d"].fillna(0) > 0) | (U["nc_expo"] < -1)
    top = U[bad].groupby("cnpj8").size().sort_values(ascending=False)
    names = pd.read_csv(S.ROOT / "research" / "data" / "issuer_sectors.csv", dtype={"cnpj8": str}).set_index("cnpj8")
    res["top_excluded"] = [{"cnpj8": c, "rows": int(n), "name": str(names["issuer_name"].get(c, "?"))[:40],
                            "sector": str(names["sector"].get(c, "?"))} for c, n in top.head(12).items()]
    r5 = H.backtest(rule(drop_issuers=tuple(top.index[:5])), panel=Q)
    s5 = H.stats(r5["daily"], bench=H.baseline("P4Q")["daily"])
    res["leave_top5_issuers_in"] = {"vs_P4Q": s5["diff_ann_%"], "t": s5["diff_t_nw"]}
    log("leave top5", res["leave_top5_issuers_in"])
    # split legs
    sup_only = variants()["P4Q_ex_issuer_new_supply"]
    # supply leg by sector group: regulated infra vs others
    infra = {"utilities_distribution", "utilities_transmission", "utilities_generation_hydro",
             "utilities_generation_renewables", "sanitation", "toll_roads", "utilities_generation_thermal"}

    def sup_infra(x):
        p = x["p4q"].to_numpy()
        return p & ~((x["sup_iss_90d"].fillna(0).to_numpy() > 0) & x["sector"].isin(infra).to_numpy())

    def sup_noninfra(x):
        p = x["p4q"].to_numpy()
        return p & ~((x["sup_iss_90d"].fillna(0).to_numpy() > 0) & ~x["sector"].isin(infra).to_numpy())
    split = {}
    for nm, f in {"supply_infra_only": sup_infra, "supply_noninfra_only": sup_noninfra}.items():
        s = H.stats(H.backtest(f, panel=Q)["daily"], bench=H.baseline("P4Q")["daily"])
        split[nm] = {"vs_P4Q": s["diff_ann_%"], "t": s["diff_t_nw"]}
        log(nm, split[nm])
    res["supply_split"] = split

    # sector-permutation placebo for the nowcast leg (keeps supply leg real)
    nc = S.sector_nowcasts()
    secs = sorted(nc.dropna(subset=["nc_expo"])["sector"].unique())
    base_p4q = H.stats(H.baseline("P4Q")["daily"])["ann_excess_%"]
    real = H.stats(H.backtest(variants()["P4Q_ex_nc_expo_bad"], panel=Q)["daily"])["ann_excess_%"] - base_p4q
    vals = []
    for s in range(20):
        rng = np.random.default_rng(7 + s)
        perm = dict(zip(secs, rng.permutation(secs)))
        Qp = Q.drop(columns=["nc_expo"]).copy()
        Qp["_src"] = Qp["sector"].map(perm)
        Qp = Qp.merge(nc[["day", "sector", "nc_expo"]].rename(columns={"sector": "_src"}), on=["day", "_src"], how="left")
        vals.append(H.stats(H.backtest(variants()["P4Q_ex_nc_expo_bad"], panel=Qp)["daily"])["ann_excess_%"] - base_p4q)
    vals = np.array(vals)
    res["placebo_sector_permutation_nc_expo"] = {"real": round(real, 3), "mean": round(float(vals.mean()), 3),
                                                  "p95": round(float(np.quantile(vals, 0.95)), 3),
                                                  "p": float((vals >= real).mean())}
    log("perm placebo", res["placebo_sector_permutation_nc_expo"])
    # 12-month time-shift placebo for the supply leg: pretend offers happened 1y earlier/later
    sh = {}
    for lag in (-365, 365):
        o2 = o.copy()
        o2["avail"] = o2["avail"] + pd.Timedelta(days=lag)
        Q["_sup_shift"] = S.issuer_event_counts(Q, o2, {"x": 91})["x"].to_numpy()
        f = lambda x: x["p4q"].to_numpy() & ~(x["_sup_shift"].to_numpy() > 0)
        s = H.stats(H.backtest(f, panel=Q)["daily"], bench=H.baseline("P4Q")["daily"])
        sh[str(lag)] = {"vs_P4Q": s["diff_ann_%"], "t": s["diff_t_nw"]}
        log("shift", lag, sh[str(lag)])
    res["supply_time_shift_placebo"] = sh
    (OUT / "robustness.json").write_text(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
