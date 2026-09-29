"""Adversarial robustness check of structural_credit A3 (P4+Q minus worst CreditGrades quintile). Pre-2026 only."""
import json, time
import numpy as np, pandas as pd
from research.nightly import harness as H
from research.nightly.structural_credit.run import worst_q, in_book_worst

OUT = "research/nightly/structural_credit/verify_robust/"
T0 = time.time()


def log(*a):
    print(f"[{time.time()-T0:5.0f}s]", *a, flush=True)


PM = pd.read_pickle("data/history/nightly/structural_credit/panel_M_sc.pkl")
PM = PM[PM["day"] < H.HOLDOUT]
R = {}
B = H.baseline("P4Q")["daily"]


def st(r, bench=B):
    s = H.stats(r["daily"] if isinstance(r, dict) else r, bench=bench)
    return {k: s[k] for k in ["ann_excess_%", "diff_ann_%", "diff_t_nw", "diff_p", "diff_h1_%", "diff_h2_%",
                              "max_dd_%", "vol_%"]}


def A3q(q=0.2, col="sc_cg", low_bad=False):
    return lambda x: x["p4q"].to_numpy() & ~worst_q(x, col, q=q, low_bad=low_bad)


def save():
    json.dump(R, open(OUT + "verify_results.json", "w"), indent=1, default=float)


# ---------- A. reproduce + perturbations
V = {f"A3_q{q}": A3q(q) for q in (0.1, 0.15, 0.2, 0.25, 0.3, 0.4)}
V["A1_dd_q0.2"] = A3q(0.2, "sc_dd", True)
V["A3_inbook_q0.2"] = lambda x: x["p4q"].to_numpy() & ~in_book_worst(x.assign(neg=-x["sc_cg"]), "neg")
V["C2_eqdd252_q0.2"] = A3q(0.2, "eq_dd252", True)
V["C3_eqr63_q0.2"] = A3q(0.2, "eq_r63", True)
# confound: pure unlisted tilt
V["X1_P4Q_drop_all_withCG"] = lambda x: x["p4q"].to_numpy() & x["sc_cg"].isna().to_numpy()
V["X2_P4Q_unlisted_only"] = lambda x: x["p4q"].to_numpy() & ~x["listed"].fillna(False).astype(bool).to_numpy()
books = {}
for k, f in V.items():
    books[k] = H.backtest(f, panel=PM, name=k)
    log(k, st(books[k]))
R["perturb_25bps"] = {k: st(v) for k, v in books.items()}
tab = H.compare(books, bench="P4Q")
p = list(tab["p_vs_bench"].astype(float))
aut = json.load(open("research/nightly/structural_credit/results.json"))["books_25bps"]
p_all = p + [aut[k]["p_vs_bench"] for k in aut if k not in ("P4", "U")]
R["holm_family_size"] = len(p_all)
R["A3_holm_in_extended_family"] = float(H.holm(np.array(p_all))[list(books).index("A3_q0.2")])
R["min_raw_p_perturb"] = float(min(p))
save()

# ---------- B. costs, horizons
a3 = A3q(0.2)
for c in (50, 100):
    R[f"cost{c}"] = st(H.backtest(a3, panel=PM, cost_bps=c), bench=H.baseline("P4Q", cost_bps=c)["daily"])
for h in (63, 252):
    R[f"hold{h}"] = st(H.backtest(a3, panel=PM, hold=h), bench=H.baseline("P4Q", hold=h)["daily"])
R["rec40"] = st(H.backtest(a3, panel=PM, scenario="rec40"), bench=H.baseline("P4Q", scenario="rec40")["daily"])
R["start_2022_07"] = st(H.backtest(a3, panel=PM, start="2022-07-01"),
                        bench=H.baseline("P4Q", start="2022-07-01")["daily"])
log("costs/horizons", {k: R[k]["diff_ann_%"] for k in ("cost50", "cost100", "hold63", "hold252", "rec40", "start_2022_07")})
save()

# ---------- C. monthly diff: exclude years, concentration
ra3 = books["A3_q0.2"]["daily"]
d = (H.monthly(ra3) - H.monthly(B.reindex(ra3.index).fillna(0)))
d = d[d.index < H.HOLDOUT]
R["monthly_diff_ann"] = float(d.mean() * 1200)
for y in (2022, 2023, 2024, 2025):
    dd_ = d[d.index.year != y]
    R[f"excl_{y}"] = {"diff_ann_%": float(dd_.mean() * 1200), "t": H.nw_t(dd_, 6)}
    R[f"only_{y}"] = float(d[d.index.year == y].mean() * 1200)
ds = d.sort_values(ascending=False)
R["top3_months"] = {str(k.date()): float(v * 100) for k, v in ds.head(3).items()}
R["diff_excl_top3_months_ann"] = float(ds.iloc[3:].mean() * 1200)
R["diff_excl_top3_t"] = H.nw_t(d.drop(ds.index[:3]).sort_index(), 6)
R["share_months_pos"] = float((d > 0).mean())
log("years", {k: R[k] for k in R if k.startswith(("excl_", "only_", "top3", "diff_excl"))})
save()

# ---------- D. top issuer contributors to the avoid gain
X = PM[PM["univ"] & PM["dok_126"] & (PM["day"] >= H.START)]
contrib = {}
for day, x in X.groupby("day"):
    m = a3(x)
    p4q = x["p4q"].to_numpy()
    kept = x.loc[p4q & m, "fwd_126"].fillna(0).mean()
    dr = x[p4q & ~m]
    for c, v in (dr["fwd_126"].fillna(0) - kept).groupby(dr["cnpj8"]).sum().items():
        contrib[c] = contrib.get(c, 0) + v
cs = pd.Series(contrib).sort_values()
tick = PM.dropna(subset=["eq_ticker"]).drop_duplicates("cnpj8").set_index("cnpj8")["eq_ticker"]
R["top_contrib_issuers"] = {f"{c}/{tick.get(c)}": float(v) for c, v in cs.head(10).items()}
R["n_dropped_issuers"] = int(len(cs))


def excl(drop):
    P2 = PM.copy()
    msk = P2["cnpj8"].isin(drop)
    for col in ("univ", "p4", "p4q"):
        P2.loc[msk, col] = False
    return st(H.backtest(a3, panel=P2), bench=H.backtest("p4q", panel=P2)["daily"])


for k in (1, 3, 5, 10):
    R[f"excl_top{k}_issuers"] = excl(set(cs.head(k).index))
    log(f"excl top{k}", R[f"excl_top{k}_issuers"])
R["excl_top5_hurting_issuers"] = excl(set(cs.tail(5).index))
save()

# ---------- E. placebos: drop the same number of P4Q names at random
rng = np.random.default_rng(123)
Pp = PM[(PM["day"] >= H.START) & PM["univ"]]
groups = [(dday, x, a3(x), x["p4q"].to_numpy()) for dday, x in Pp.groupby("day")]
act = R["perturb_25bps"]["A3_q0.2"]["diff_ann_%"]


def placebo(pool_fn, n=100):
    out = []
    for it in range(n):
        rows = []
        for dday, x, m, p4q in groups:
            nd = int((p4q & ~m).sum())
            sel = p4q.copy()
            cand = np.where(p4q & pool_fn(x))[0]
            if nd and len(cand):
                sel[rng.choice(cand, size=min(nd, len(cand)), replace=False)] = False
            rows.append(pd.DataFrame({"day": dday, "codigo": x["codigo"].to_numpy()[sel], "select": True}))
        out.append(H.stats(H.backtest(pd.concat(rows), panel=PM)["daily"], bench=B)["diff_ann_%"])
    out = np.array(out)
    return {"mean": float(out.mean()), "sd": float(out.std()), "p95": float(np.percentile(out, 95)),
            "max": float(out.max()), "p_ge_actual": float((out >= act).mean()), "actual": act, "n": n}


R["placebo_listed_withCG"] = placebo(lambda x: x["sc_cg"].notna().to_numpy(), 100)
log("placebo1", R["placebo_listed_withCG"])
save()


def hi_carry(x):
    c = x["cdi_bps"].where(x["p4q"] & x["sc_cg"].notna())
    return (c >= c.median()).to_numpy()


R["placebo_listed_highcarry"] = placebo(hi_carry, 50)
log("placebo2", R["placebo_listed_highcarry"])
sh = []
for dday, x, m, p4q in groups:
    dr = x[p4q & ~m]
    kp = x[p4q & m & x["sc_cg"].notna().to_numpy()]
    if len(dr):
        sh.append((dr["cdi_bps"].mean(), kp["cdi_bps"].mean(), len(dr), p4q.sum()))
sh = np.array(sh, float)
R["dropped_carry_bps_mean"] = float(np.nanmean(sh[:, 0]))
R["kept_listed_carry_bps_mean"] = float(np.nanmean(sh[:, 1]))
R["avg_dropped_names"] = float(sh[:, 2].mean())
R["avg_p4q_names"] = float(sh[:, 3].mean())
R["n_avg_A3"] = books["A3_q0.2"]["n_avg"]
R["n_avg_P4Q"] = H.baseline("P4Q")["n_avg"]
m_b = H.monthly(B)
m_b = m_b[m_b.index < H.HOLDOUT]
m_a = H.monthly(ra3)
R["P4Q_worst_months"] = {str(k.date()): float(v * 100) for k, v in m_b.nsmallest(4).items()}
R["A3_same_months"] = {str(k.date()): float(m_a.get(k, np.nan) * 100) for k in m_b.nsmallest(4).index}
save()
log("done")
