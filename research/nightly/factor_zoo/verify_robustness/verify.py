"""Adversarial robustness verification of factor_zoo's zoo_screen20 (pre-2026; holdout re-read once only to reproduce)."""
import json
import time

import numpy as np
import pandas as pd

from research.nightly import harness as H
from research.nightly.factor_zoo import features as FZ
from research.nightly.factor_zoo.run import composite, screen_fn

OUT = "research/nightly/factor_zoo/verify_robustness/"
T0 = time.time()


def log(*a):
    print(f"[+{time.time()-T0:6.1f}s]", *a, flush=True)


spec = json.load(open("research/nightly/factor_zoo/results.json"))["composite_spec"]
P = FZ.attach(H.load_panel("M"))
P = composite(P, spec, "zoo")
B = H.baseline("P4Q")["daily"]
res = {}


def md(r, b=B):
    r = r[r.index < H.HOLDOUT]
    return (H.monthly(r) - H.monthly(b.reindex(r.index).fillna(0))).dropna()


def summ(d):
    d = d.dropna()
    return {"ann": round(d.mean() * 1200, 3), "t": round(H.nw_t(d, 6), 2), "n": int(len(d))}


def run(fn, panel=None, bench=None, **kw):
    panel = P if panel is None else panel
    r = H.backtest(fn, freq="M", hold=126, panel=panel, **kw)
    b = bench if bench is not None else B
    d = md(r["daily"], b)
    s = summ(d)
    s.update({"h1": round(d[d.index < H.SPLIT].mean() * 1200, 3), "h2": round(d[d.index >= H.SPLIT].mean() * 1200, 3),
              "ex2023": summ(d[d.index.year != 2023]), "n_avg": round(r["n_avg"], 1)})
    return s, d, r


base, dbase, rbase = run(screen_fn("zoo", 0.2))
res["reproduce"] = base
log("reproduce", base)

res["periods"] = {str(y): summ(dbase[dbase.index.year == y]) for y in (2022, 2023, 2024, 2025)}
res["periods"]["ex2023"] = summ(dbase[dbase.index.year != 2023])
res["periods"]["ex_2023H1"] = summ(dbase[~((dbase.index.year == 2023) & (dbase.index.month <= 6))])
top = dbase.sort_values(ascending=False)
res["periods"]["top3_months"] = {str(k.date()): round(v * 1200, 3) for k, v in top.head(3).items()}
res["periods"]["ex_top3_months"] = summ(dbase.drop(top.index[:3]))
log("periods", res["periods"])

# clean OOS: decisions after the selection window (>= 2023-07), P&L months >= 2024 (no overlap with selection labels)
r_oos = H.backtest(screen_fn("zoo", 0.2), freq="M", hold=126, panel=P, start="2023-07-01")
b_oos = H.backtest("p4q", freq="M", hold=126, panel=P, start="2023-07-01")
d = md(r_oos["daily"], b_oos["daily"])
res["clean_oos_2024_25"] = summ(d[d.index >= "2024-01-01"])
log("clean oos", res["clean_oos_2024_25"])

# costs
r50 = H.backtest(screen_fn("zoo", 0.2), freq="M", hold=126, panel=P, cost_bps=50)
res["cost50"] = summ(md(r50["daily"], H.baseline("P4Q", cost_bps=50)["daily"]))
r100 = H.backtest(screen_fn("zoo", 0.2), freq="M", hold=126, panel=P, cost_bps=100)
res["cost100"] = summ(md(r100["daily"], H.baseline("P4Q", cost_bps=100)["daily"]))
res["turnover"] = {"screen": round(rbase["turnover_ann"], 2), "p4q": round(H.baseline("P4Q")["turnover_ann"], 2)}
log("costs", res["cost50"], res["cost100"], res["turnover"])

res["q_grid"] = {}
for q in (0.1, 0.15, 0.25, 0.3, 0.4):
    res["q_grid"][str(q)] = run(screen_fn("zoo", q))[0]
log("q grid", {k: (v["ann"], v["t"]) for k, v in res["q_grid"].items()})

res["loo"] = {}
for f in spec:
    sp = {k: v for k, v in spec.items() if k != f}
    P = composite(P, sp, "_c")
    res["loo"]["minus_" + f] = run(screen_fn("_c", 0.2))[0]
subsets = {"stable_eq_ds5_dres": {k: spec[k] for k in ("eq_r21", "eq_r126", "eq_vol63", "ds_5", "dres_21")},
           "drop_unstable3": {k: v for k, v in spec.items() if k not in ("days_since_distress", "bond_age_y", "age")},
           "eq_only": {k: spec[k] for k in ("eq_r21", "eq_r126", "eq_vol63")},
           "eq_r126_only": {"eq_r126": 1}, "eq_vol63_only": {"eq_vol63": -1},
           "distress_age_only": {"days_since_distress": -1, "age": 1}}
for nm, sp in subsets.items():
    P = composite(P, sp, "_c")
    res["loo"]["SUBSET_" + nm] = run(screen_fn("_c", 0.2))[0]
log("loo", {k: (v["ann"], v["t"], v["h1"], v["h2"]) for k, v in res["loo"].items()})

# who is screened out
x = P[P["p4q"]].copy()
x["thr"] = x.groupby("day")["zoo"].transform(lambda s: s.quantile(0.2))
x["drop"] = x["zoo"] <= x["thr"]
x["listed_"] = x["eq_r126"].notna().astype(float)
x["dss_hit"] = (x["days_since_distress"] < 9999).astype(float)
chars = ["listed_", "dss_hit", "cdi_bps", "age", "days_since_trade", "trades_30d", "log_vol_63", "ntr_63",
         "amihud_126", "bond_age_y", "dur", "resid_z", "fwd_126"]
res["dropped_vs_kept"] = x.groupby("drop")[[c for c in chars if c in x]].mean().round(4).to_dict(orient="index")
log("dropped vs kept", res["dropped_vs_kept"])

# exclude top issuers by 'avoided loss' (screened-out fwd_126 minus date P4Q mean) and by kept winners
x["rel"] = x["fwd_126"] - x.groupby("day")["fwd_126"].transform("mean")
contrib = x[x["drop"]].groupby("cnpj8")["rel"].sum().sort_values()
kept_contrib = x[~x["drop"]].groupby("cnpj8")["rel"].sum().sort_values(ascending=False)
res["top_avoided_issuers"] = {str(k): round(float(v), 3) for k, v in contrib.head(10).items()}
for k_ in (5, 10):
    for nm, iss in (("avoided", list(contrib.index[:k_])), ("kept_winners", list(kept_contrib.index[:k_]))):
        P2 = P.copy()
        m = P2["cnpj8"].isin(iss)
        for c in ("univ", "p4q", "p4"):
            P2.loc[m, c] = False
        b2 = H.backtest("p4q", freq="M", hold=126, panel=P2)["daily"]
        res[f"excl_top{k_}_{nm}"] = run(screen_fn("zoo", 0.2), panel=P2, bench=b2)[0]
        log(f"excl top{k_} {nm}", res[f"excl_top{k_}_{nm}"])

rng = np.random.default_rng(11)


def rand_screen(kind):
    def f(xx):
        b = xx["p4q"].to_numpy()
        if kind == "bond":
            v = pd.Series(rng.random(len(xx)), index=xx.index).where(xx["p4q"])
            return b & ~((v <= v.quantile(0.2)).to_numpy())
        if kind == "issuer":
            iss = xx.loc[xx["p4q"], "cnpj8"].unique()
            k = int(round(0.2 * len(iss)))
            bad = set(rng.choice(iss, size=k, replace=False)) if k else set()
            return b & ~xx["cnpj8"].isin(bad).to_numpy()
        # matched: drop the same number of listed / unlisted names as the real screen does, at random
        thr = xx["zoo"].where(xx["p4q"]).quantile(0.2)
        real = ((xx["zoo"] <= thr) & xx["p4q"]).to_numpy()
        lst = xx["eq_r126"].notna().to_numpy()
        out = b.copy()
        for g in (True, False):
            cand = np.where(b & (lst == g))[0]
            k = int((real & (lst == g)).sum())
            if k and len(cand):
                out[rng.choice(cand, size=min(k, len(cand)), replace=False)] = False
        return out
    return f


res["placebo"] = {}
for kind in ("bond", "issuer", "matched"):
    vals = np.array([summ(md(H.backtest(rand_screen(kind), freq="M", hold=126, panel=P)["daily"]))["ann"]
                     for _ in range(40)])
    res["placebo"][kind] = {"mean": round(vals.mean(), 3), "sd": round(vals.std(), 3),
                            "p95": round(np.percentile(vals, 95), 3), "max": round(vals.max(), 3),
                            "p_emp": float((vals >= base["ann"]).mean())}
    log("placebo", kind, res["placebo"][kind])

P = P.sort_values(["codigo", "day"])
P["zoo_lag12"] = P.groupby("codigo")["zoo"].shift(12)
P = P.sort_index()
res["zoo_lag12_screen"] = run(screen_fn("zoo_lag12", 0.2))[0]
P["zoo_neg"] = -P["zoo"]
res["inverted_screen"] = run(screen_fn("zoo_neg", 0.2))[0]
log("lag12", res["zoo_lag12_screen"], "inverted", res["inverted_screen"])

P3 = P.copy()
med = P3[P3["univ"]].groupby("day")["trades_30d"].median()
liq = P3["trades_30d"] >= P3["day"].map(med)
for c in ("univ", "p4q"):
    P3[c] = P3[c] & liq.fillna(False)
b3 = H.backtest("p4q", freq="M", hold=126, panel=P3)["daily"]
res["liquid_half_only"] = run(screen_fn("zoo", 0.2), panel=P3, bench=b3)[0]
log("liquid", res["liquid_half_only"])
r = H.backtest(screen_fn("zoo", 0.2), freq="M", hold=252, panel=P)
res["hold252"] = summ(md(r["daily"], H.baseline("P4Q", hold=252)["daily"]))
r = H.backtest(screen_fn("zoo", 0.2), freq="M", hold=126, panel=P, issuer_cap=0.05)
res["cap5"] = summ(md(r["daily"], H.baseline("P4Q", issuer_cap=0.05)["daily"]))
log("hold252", res["hold252"], "cap5", res["cap5"])

PA = composite(FZ.attach(H.load_panel("M", holdout=True)), spec, "zoo")
rh = H.backtest(screen_fn("zoo", 0.2), freq="M", hold=126, panel=PA, holdout=True)
bh = H.baseline("P4Q", holdout=True)
res["holdout_reproduced"] = H.stats(rh["daily"], bench=bh["daily"], holdout="only")
log("holdout", {k: res["holdout_reproduced"][k] for k in ("diff_ann_%", "diff_t_nw", "diff_hit")})

json.dump(res, open(OUT + "verify_results.json", "w"), indent=1, default=str)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(9, 4.5), dpi=110)
ax.plot(dbase.index, dbase.cumsum() * 100, label="zoo_screen20 - P4+Q, cumulative monthly diff (%)")
ax.axvspan(pd.Timestamp("2021-12-15"), pd.Timestamp("2023-12-31"), color="grey", alpha=.15,
           label="overlaps selection-period labels")
ax.axhline(0, color="k", lw=.6)
ax.legend(fontsize=8)
ax.grid(alpha=.3)
fig.tight_layout()
fig.savefig(OUT + "cum_diff_vs_p4q.png")
log("done")
