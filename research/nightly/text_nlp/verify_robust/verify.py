"""Adversarial robustness check of text_nlp V4 (P4+Q ex zero-shot credit-negative doc in (d-90,d-1]). Pre-2026 only."""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from research.nightly import harness as H
from research.nightly.text_nlp import features as TF
HERE = Path(__file__).resolve().parent
T0 = time.time()
RES = {}


def log(*a):
    print(f"[{time.time()-T0:6.0f}s]", *a, flush=True)


def save():
    (HERE / "verify_results.json").write_text(json.dumps(RES, indent=1, default=str), encoding="utf-8")


docs = TF.load_docs()
docs = docs[docs["avail"] < H.HOLDOUT].copy()
P = H.load_panel("M")
keys = P[["cnpj8", "day"]].drop_duplicates().reset_index(drop=True)

# --- recompute zero-shot neg labels at several thresholds and by source
zcols = [c for c in docs.columns if c.startswith("zs_") and c not in ("zs_label", "zs_neg", "zs_routine")
         and not c.startswith("zsl_")]
cls = [c[3:] for c in zcols]
S = docs[zcols].to_numpy(float)
has = np.isfinite(S).all(1)
Sz = np.nan_to_num(S, nan=-9)
arg = Sz.argmax(1)
mx = Sz.max(1)
lab = np.array(cls)[arg]
rout = np.nan_to_num(docs["zs_routine"].to_numpy(float), nan=-9)


def zs_neg_at(T, classes=TF.NEG_EVENTS):
    is_rout = rout >= mx
    full_mx = np.maximum(mx, rout)
    ok = has & (full_mx > T) & ~is_rout
    return ok & np.isin(lab, classes)


RES["zs_neg_recompute_agreement"] = float((zs_neg_at(0.55) == docs["zs_neg"].to_numpy()).mean())
log("agree", RES["zs_neg_recompute_agreement"])
docs["z50"] = zs_neg_at(0.50)
docs["z55"] = zs_neg_at(0.55)
docs["z60"] = zs_neg_at(0.60)
docs["z65"] = zs_neg_at(0.65)
docs["z55_noliab"] = zs_neg_at(0.55, [c for c in TF.NEG_EVENTS if c != "liab_mgmt"])
docs["z55_ipe"] = docs["z55"] & (docs["src"] == "ipe")
docs["z55_news"] = docs["z55"] & (docs["src"] == "news")
docs["one_"] = 1.0
cols = ["z50", "z55", "z60", "z65", "z55_noliab", "z55_ipe", "z55_news", "one_"]
docs[cols] = docs[cols].astype(float)
feat = keys.copy()
for w in (30, 60, 90, 180):
    s = TF._window_sums(docs, cols, keys, w)
    s.columns = [f"{c}_{w}" for c in cols]
    feat = pd.concat([feat, s], axis=1)
P = P.merge(feat, on=["cnpj8", "day"], how="left")
F0 = pd.read_pickle(ROOT / "data/history/nightly/text_nlp/text_signals_M.pkl")[["cnpj8", "day", "zs_neg_90"]]
P = P.merge(F0, on=["cnpj8", "day"], how="left")
RES["z55_90_vs_cached_zs_neg_90_equal_share"] = float((P["z55_90"].fillna(0) == P["zs_neg_90"].fillna(0)).mean())
log("feature eq", RES["z55_90_vs_cached_zs_neg_90_equal_share"])

bP = H.baseline("P4Q")["daily"]


def ex(col):
    return lambda x: (x["p4q"] & (x[col].fillna(0) == 0)).to_numpy()


def st(r, bench=bP):
    s = H.stats(r["daily"], bench=bench)
    out = {}
    for k in ("ann_excess_%", "diff_ann_%", "diff_t_nw", "diff_h1_%", "diff_h2_%", "sharpe", "max_dd_%"):
        v = s.get(k)
        out[k] = round(float(v), 3) if v is not None and np.isfinite(v) else None
    return out


V4 = ex("zs_neg_90")
r4 = H.backtest(V4, panel=P, name="V4")
RES["V4_reproduced"] = st(r4)
RES["V4_n_avg"] = r4["n_avg"]
log("V4", RES["V4_reproduced"])
save()

PERT = {}
for nm, col in {"T0.50_90d": "z50_90", "T0.55_90d(recomp)": "z55_90", "T0.60_90d": "z60_90", "T0.65_90d": "z65_90",
                "T0.55_30d": "z55_30", "T0.55_60d": "z55_60", "T0.55_180d": "z55_180",
                "T0.55_90d_no_liab_mgmt": "z55_noliab_90", "IPE_only_90d": "z55_ipe_90",
                "news_only_90d": "z55_news_90"}.items():
    r = H.backtest(ex(col), panel=P, name=nm)
    PERT[nm] = st(r)
    PERT[nm]["n_avg"] = round(r["n_avg"], 1)
    log(nm, PERT[nm])
RES["perturbations_25bps"] = PERT
save()

ENG = {}
for nm, kw in {"cost50": dict(cost_bps=50), "rec40": dict(scenario="rec40"), "hold63": dict(hold=63),
               "hold252": dict(hold=252), "cap5": dict(issuer_cap=0.05)}.items():
    r = H.backtest(V4, panel=P, **kw)
    b = H.baseline("P4Q", **kw)
    ENG[nm] = st(r, b["daily"])
    log(nm, ENG[nm])
RES["engine_perturbations"] = ENG
save()

m = (H.monthly(r4["daily"]) - H.monthly(bP)).dropna()
m = m[m.index < H.HOLDOUT]


def ann(x):
    return round(float(x.mean() * 12 * 100), 3)


RES["monthly_diff"] = {"all": ann(m), "t": round(H.nw_t(m, 6), 2), "ex2023": ann(m[m.index.year != 2023]),
                       "t_ex2023": round(H.nw_t(m[m.index.year != 2023], 6), 2),
                       "by_year": {int(y): ann(g) for y, g in m.groupby(m.index.year)},
                       "ex_top3_months": ann(m.sort_values().iloc[:-3]), "ex_top5_months": ann(m.sort_values().iloc[:-5]),
                       "share_months_pos": round(float((m > 0).mean()), 3),
                       "top5_months_%": {str(k.date()): round(v * 100, 3) for k, v in m.sort_values().iloc[-5:].items()}}
log("monthly", RES["monthly_diff"])
save()

Pu = P[P["univ"] & P["p4q"] & (P["day"] >= H.START)]
exc = Pu[Pu["zs_neg_90"].fillna(0) > 0]
freq_iss = exc.groupby("cnpj8").size().sort_values(ascending=False)
RES["n_issuers_ever_excluded"] = int(len(freq_iss))
RES["excluded_bond_months"] = int(len(exc))
base_diff = RES["V4_reproduced"]["diff_ann_%"]
contrib = {}
for c8 in freq_iss.index[:40]:
    f = lambda x, c8=c8: (x["p4q"] & ((x["zs_neg_90"].fillna(0) == 0) | (x["cnpj8"] == c8))).to_numpy()
    r = H.backtest(f, panel=P)
    contrib[c8] = round(base_diff - H.stats(r["daily"], bench=bP)["diff_ann_%"], 4)
contrib = dict(sorted(contrib.items(), key=lambda kv: -kv[1]))
RES["loo_issuer_contrib_top10"] = dict(list(contrib.items())[:10])
RES["loo_sum_positive_top40"] = round(sum(v for v in contrib.values() if v > 0), 3)
top = list(contrib)
for k, S_ in (("ex_top1_issuer", top[:1]), ("ex_top3_issuers", top[:3]), ("ex_top5_issuers", top[:5]),
              ("ex_top10_issuers", top[:10])):
    f = lambda x, S_=S_: (x["p4q"] & ((x["zs_neg_90"].fillna(0) == 0) | x["cnpj8"].isin(S_))).to_numpy()
    RES[k] = st(H.backtest(f, panel=P))
    log(k, RES[k])
save()

cnt = exc.groupby("day").size()
iss_cnt = exc.groupby("day")["cnpj8"].nunique()
RES["avg_excluded_names_per_date"] = round(float(cnt.reindex(Pu["day"].unique()).fillna(0).mean()), 1)
rng = np.random.default_rng(7)
Pd = {d: x for d, x in P[P["univ"] & (P["day"] >= H.START)].groupby("day")}


def rand_sig(issuer_level):
    rows = []
    for d, x in Pd.items():
        q = x[x["p4q"]]
        if issuer_level:
            ii = q["cnpj8"].unique()
            k = int(iss_cnt.get(d, 0))
            drop = set(rng.choice(ii, size=min(k, len(ii)), replace=False)) if k else set()
            keep = q.loc[~q["cnpj8"].isin(drop), "codigo"].to_numpy()
        else:
            cand = q["codigo"].to_numpy()
            k = int(cnt.get(d, 0))
            drop = set(rng.choice(cand, size=min(k, len(cand)), replace=False)) if k else set()
            keep = np.array([c for c in cand if c not in drop])
        rows.append(pd.DataFrame({"day": d, "codigo": keep, "select": True}))
    return pd.concat(rows)


for lvl, nm in ((False, "placebo_random_matched_bond_exclusion"), (True, "placebo_random_matched_issuer_exclusion")):
    pl = []
    for i in range(50):
        r = H.backtest(rand_sig(lvl), panel=P)
        pl.append(H.stats(r["daily"], bench=bP)["diff_ann_%"])
    pl = np.array(pl)
    RES[nm] = {"mean": round(float(pl.mean()), 3), "sd": round(float(pl.std()), 3),
               "p95": round(float(np.percentile(pl, 95)), 3), "p_one_sided": round(float((pl >= base_diff).mean()), 3),
               "n": len(pl)}
    log(nm, RES[nm])
    save()

rows = []
for d, x in Pd.items():
    q = x[x["p4q"]]
    k = int(cnt.get(d, 0))
    drop = q.sort_values("one__90", ascending=False)["codigo"].iloc[:k]
    rows.append(pd.DataFrame({"day": d, "codigo": q.loc[~q["codigo"].isin(drop), "codigo"].to_numpy(), "select": True}))
RES["visibility_placebo_drop_most_docs_matched_count"] = st(H.backtest(pd.concat(rows), panel=P))
RES["ex_any_doc_90d"] = st(H.backtest(ex("one__90"), panel=P))
log("visibility", RES["visibility_placebo_drop_most_docs_matched_count"], RES["ex_any_doc_90d"])

if "trades_30d" in P:
    RES["liquidity_trades30d_median"] = {"excluded": float(exc["trades_30d"].median()),
                                         "kept": float(Pu.loc[Pu["zs_neg_90"].fillna(0) == 0, "trades_30d"].median())}
H.plot_curves({"V4 (text overlay)": r4}, str(HERE / "v4_curves.png"), title="verify: V4 vs refs (pre-2026, 25 bps)")
save()
log("done")
