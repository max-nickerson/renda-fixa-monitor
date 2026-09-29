"""Adversarial ROBUSTNESS check of fund_flows X4 (P4+Q ex bottom-quintile cshare) and side claims (X5 orphans,
X2 1m cohort). Read-only w.r.t. fund_flows files; pre-2026 only (holdout untouched)."""
import json, time
import numpy as np, pandas as pd
from research.nightly import harness as H
from research.nightly.fund_flows import features as F

OUT = "research/nightly/fund_flows/verify_robustness"
t0 = time.time()
P0 = H.load_panel("M")
X = F.signals(P0)
P = pd.concat([P0.reset_index(drop=True), X.drop(columns=["day", "codigo"]).reset_index(drop=True)], axis=1)
B = H.baseline("P4Q"); B50 = H.baseline("P4Q", cost_bps=50)
res = {}


def exq(f, q=0.2, low=True):
    def g(x):
        r = x[f].rank(pct=True)
        fl = ((r <= q) if low else (r > 1 - q)).fillna(False).to_numpy()
        return x["p4q"].to_numpy() & ~fl
    return g


def st(r, b=B):
    s = H.stats(r["daily"], bench=b["daily"])
    return {k: s[k] for k in ["ann_excess_%", "diff_ann_%", "diff_t_nw", "diff_h1_%", "diff_h2_%", "diff_hit"]}


def monthly_diff(r, b=B):
    s = r["daily"][r["daily"].index < H.HOLDOUT]
    return (H.monthly(s) - H.monthly(b["daily"].reindex(s.index).fillna(0))).dropna()


X4 = exq("cshare")
r4 = H.backtest(X4, panel=P)
res["X4_repro"] = st(r4)
res["X4_50bps"] = st(H.backtest(X4, panel=P, cost_bps=50), B50)
d = monthly_diff(r4)
res["X4_monthly_diff"] = {str(k.date()): round(float(v) * 100, 4) for k, v in d.items()}
ex23 = d[d.index.year != 2023]
res["X4_ex2023"] = {"diff_ann_%": float(ex23.mean() * 1200), "t": H.nw_t(ex23.to_numpy(), 6)}
by_year = d.groupby(d.index.year).mean() * 1200
res["X4_by_year"] = {int(k): round(float(v), 3) for k, v in by_year.items()}
# drop top-3 best months
dd_ = d.sort_values()
res["X4_ex_top3_months"] = float(dd_.iloc[:-3].mean() * 1200)
res["X4_ex_top1_month"] = float(dd_.iloc[:-1].mean() * 1200)
print("repro", res["X4_repro"], res["X4_ex2023"], res["X4_by_year"], round(time.time() - t0), flush=True)

# quantile perturbations
res["q_perturb"] = {}
for q in (0.1, 0.15, 0.25, 0.3, 0.4):
    res["q_perturb"][q] = st(H.backtest(exq("cshare", q), panel=P))
    print("q", q, res["q_perturb"][q], flush=True)
# holding period / rebalance perturbations (each vs P4Q on same engine)
res["hold_perturb"] = {}
for hold, reb in [(63, None), (252, None), (126, "Q")]:
    b = H.backtest("p4q", panel=P, hold=hold, rebalance=reb)
    res["hold_perturb"][f"h{hold}_{reb}"] = st(H.backtest(X4, panel=P, hold=hold, rebalance=reb), b)
    print(hold, reb, res["hold_perturb"][f"h{hold}_{reb}"], flush=True)

# credit-fund threshold perturbation: recompute cshare with deb_share >= thr
deb, _ = F._load()
cls = F.fund_class()
days = sorted(pd.to_datetime(P["day"].unique()))
vint = F._vintage_for(days)
byref = {r: g[["fund", "codigo", "val"]] for r, g in deb.groupby("ref")}
res["thr_perturb"] = {}
for thr in (0.10, 0.50, 0.75):
    cr = cls[cls["deb_share"] >= thr]
    crs = {r: set(g["fund"]) for r, g in cr.groupby("ref")}
    rows = []
    for day in days:
        v = vint.loc[day]
        if pd.isna(v):
            continue
        h = byref[v]
        c = h["fund"].isin(crs.get(v, set())).astype(float) * h["val"]
        s = (c.groupby(h["codigo"]).sum() / h.groupby("codigo")["val"].sum()).rename("cs2")
        rows.append(s.reset_index().assign(day=day))
    CS = pd.concat(rows)
    PP = P.merge(CS, on=["day", "codigo"], how="left")
    res["thr_perturb"][thr] = st(H.backtest(exq("cs2"), panel=PP))
    print("thr", thr, res["thr_perturb"][thr], flush=True)
del byref

# placebo: 100 random drops of the same count from P4Q
rng = np.random.default_rng(123)
pl = []
for i in range(100):
    seed = int(rng.integers(1e9))
    def rnd(x, seed=seed):
        base = x["p4q"].to_numpy(); keep = X4(x); nd = int(base.sum() - (base & keep).sum())
        rr = np.random.default_rng(seed + int(x["dpos"].iloc[0]))
        idx = np.flatnonzero(base); m = base.copy()
        if nd > 0 and len(idx) > nd:
            m[rr.choice(idx, nd, replace=False)] = False
        return m
    pl.append(H.stats(H.backtest(rnd, panel=P)["daily"], bench=B["daily"])["diff_ann_%"])
pl = np.array(pl)
res["placebo100"] = {"mean": float(pl.mean()), "sd": float(pl.std()), "p95": float(np.percentile(pl, 95)),
                     "share_ge_actual": float((pl >= res["X4_repro"]["diff_ann_%"]).mean())}
print("placebo", res["placebo100"], flush=True)
# placebo 2: random feature (shuffle cshare within date among covered)
pl2 = []
for i in range(30):
    PP = P.copy()
    rr = np.random.default_rng(1000 + i)
    PP["csh"] = PP.groupby("day")["cshare"].transform(lambda s: pd.Series(rr.permutation(s.to_numpy()), index=s.index))
    pl2.append(H.stats(H.backtest(exq("csh"), panel=PP)["daily"], bench=B["daily"])["diff_ann_%"])
pl2 = np.array(pl2)
res["placebo_shuffle30"] = {"mean": float(pl2.mean()), "p95": float(np.percentile(pl2, 95)),
                            "share_ge_actual": float((pl2 >= res["X4_repro"]["diff_ann_%"]).mean())}
print("placebo shuffle", res["placebo_shuffle30"], flush=True)

# excluded names: issuer contributions. Excluded = p4q & low cshare; measure fwd_126 relative to P4Q date-mean
rows = []
for dpos, x in P[P["univ"]].groupby("dpos"):
    base = x["p4q"].to_numpy(); keep = X4(x); exc = base & ~keep
    if exc.sum() == 0 or "fwd_126" not in x:
        continue
    mu = x.loc[base, "fwd_126"].mean()
    e = x.loc[exc, ["cnpj8", "fwd_126"]].copy(); e["rel"] = e["fwd_126"] - mu
    rows.append(e)
E = pd.concat(rows).dropna()
contrib = E.groupby("cnpj8")["rel"].agg(["sum", "count"]).sort_values("sum")
res["excluded_issuers_worst10"] = contrib.head(10).round(4).reset_index().to_dict(orient="records")
res["excluded_rel_mean_%"] = float(E["rel"].mean() * 100)
top5 = contrib.head(5).index.tolist()
PP = P[~P["cnpj8"].isin(top5)]
b5 = H.backtest("p4q", panel=PP)
res["X4_ex_top5_helpful_issuers"] = st(H.backtest(X4, panel=PP), b5)
print("ex top5", res["X4_ex_top5_helpful_issuers"], flush=True)
top10 = contrib.head(10).index.tolist()
PP = P[~P["cnpj8"].isin(top10)]
res["X4_ex_top10_helpful_issuers"] = st(H.backtest(X4, panel=PP), H.backtest("p4q", panel=PP))

# carry control: excluded names carry vs kept
x = P[P["univ"] & P["p4q"]]
res["excluded_share_of_p4q"] = float(1 - np.mean(np.concatenate([X4(g)[g["p4q"].to_numpy()] for _, g in P[P["univ"]].groupby("dpos")])))

# side claims: X5 orphans, X2 cohort 1m
X5 = lambda x: x["p4q"].to_numpy() & (x["nh"] > 0).to_numpy()
r5 = H.backtest(X5, panel=P)
res["X5_repro"] = st(r5)
d5 = monthly_diff(r5)
res["X5_ex2023"] = {"diff": float(d5[d5.index.year != 2023].mean() * 1200), "t": H.nw_t(d5[d5.index.year != 2023].to_numpy(), 6)}
res["X5_50bps"] = st(H.backtest(X5, panel=P, cost_bps=50), B50)
# orphans' contribution concentration
rows = []
for dpos, x in P[P["univ"]].groupby("dpos"):
    base = x["p4q"].to_numpy(); o = base & (x["nh"] == 0).to_numpy()
    if o.sum() == 0:
        continue
    mu = x.loc[base, "fwd_126"].mean()
    e = x.loc[o, ["cnpj8", "codigo", "fwd_126"]].copy(); e["rel"] = e["fwd_126"] - mu
    rows.append(e)
EO = pd.concat(rows).dropna()
co = EO.groupby("cnpj8")["rel"].sum().sort_values(ascending=False)
res["orphan_top5_issuer_share_of_rel_sum"] = float(co.head(5).sum() / co.sum()) if co.sum() != 0 else None
res["orphan_n_issuers"] = int(co.size)
PP = P[~P["cnpj8"].isin(co.head(5).index)]
res["X5_ex_top5_orphan_issuers"] = st(H.backtest(X5, panel=PP), H.backtest("p4q", panel=PP))
print("X5", res["X5_repro"], res["X5_ex_top5_orphan_issuers"], flush=True)
X2 = exq("press21")
a = H.cohort_excess(X2, H=21, panel=P); b = H.cohort_excess("p4q", H=21, panel=P)
dc = (a - b).dropna()
res["X2_cohort21"] = {"ann": float(dc.mean() / (21 / 252) * 100), "t": H.nw_t(dc.to_numpy(), 1),
                      "ex2023_ann": float(dc[dc.index.year != 2023].mean() / (21 / 252) * 100) if hasattr(dc.index, "year") else None}
print("X2", res["X2_cohort21"], flush=True)
res["runtime_s"] = round(time.time() - t0)
json.dump(res, open(f"{OUT}/results.json", "w"), indent=1, default=float)
print("done", res["runtime_s"])
