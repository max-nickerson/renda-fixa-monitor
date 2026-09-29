"""Follow-up 2: is the within-DI cshare IC just carry? + variant X7 (P4+Q minus bottom-quintile cshare among DI_SPREAD bonds).
Holm recomputed over all 15 variants."""
import json
import numpy as np, pandas as pd
from research.nightly import harness as H
from research.nightly.fund_flows import features as F
OUT = "research/nightly/fund_flows"
R = json.load(open(f"{OUT}/results.json"))
P = H.load_panel("M"); X = F.signals(P)
P = pd.concat([P.reset_index(drop=True), X.drop(columns=["day", "codigo"]).reset_index(drop=True)], axis=1)
PD = P[P["kind"] == "DI_SPREAD"].copy()
def g(x):
    ok = x["cshare"].notna() & x["cdi_bps"].notna() & x["univ"]
    o = pd.Series(np.nan, index=x.index)
    if ok.sum() > 30:
        a = x.loc[ok, "cshare"].rank(pct=True); b = x.loc[ok, "cdi_bps"].rank(pct=True)
        o[ok] = a - np.polyval(np.polyfit(b, a, 1), b)
    return o
PD["cs_cn"] = PD.groupby("day", group_keys=False).apply(g)
d = {}
for tg in ["fwd_21", "fwd_63", "fwd_126"]:
    r = H.ic(PD, "cs_cn", tg); d[f"DI_cshare_carry_neutral_{tg}"] = {"ic": round(r["mean"], 4), "t": round(r["t_nw"], 2)}
    r = H.ic(PD, "cshare", tg); d[f"DI_cshare_{tg}"] = {"ic": round(r["mean"], 4), "t": round(r["t_nw"], 2)}
print(d)
def x7(x):
    di = (x["kind"] == "DI_SPREAD").to_numpy()
    v = x["cshare"].where(di).rank(pct=True)
    return x["p4q"].to_numpy() & ~(v <= 0.2).fillna(False).to_numpy()
r = H.backtest(x7, panel=P, name="X7")
b = H.baseline("P4Q")
s = H.stats(r["daily"], bench=b["daily"]); su = H.stats(r["daily"], bench=H.baseline("U")["daily"])
r50 = H.backtest(x7, panel=P, cost_bps=50); s50 = H.stats(r50["daily"], bench=H.baseline("P4Q", cost_bps=50)["daily"])
out = {"exCDI_%": s["ann_excess_%"], "exU_%": su["diff_ann_%"], "vs_bench_%": s["diff_ann_%"], "t": s["diff_t_nw"], "p": s["diff_p"],
       "h1": s["diff_h1_%"], "h2": s["diff_h2_%"], "vs_bench_50bps_%": s50["diff_ann_%"], "sharpe": s["sharpe"], "maxdd": s["max_dd_%"],
       "n_avg": r["n_avg"]}
c = {}
for Hh in (21, 63, 126):
    dd = (H.cohort_excess(x7, H=Hh, panel=P) - H.cohort_excess("p4q", H=Hh, panel=P)).dropna()
    c[f"H{Hh}"] = (round(float(dd.mean() / (Hh / 252) * 100), 3), round(float(H.nw_t(dd.to_numpy(), max(1, Hh // 21))), 2))
out["cohort"] = c
print(out)
R["followups"]["X7_ex_low_cshare_DI_only"] = out
R["diagnostics"]["cshare_DI"] = d
pv = {k: v["p_vs_bench"] for k, v in R["table_25bps"].items()}
pv.update({k: v["p"] for k, v in R["followups"].items()})
ks = list(pv); hp = H.holm(np.array([pv[k] for k in ks], dtype=float))
R["holm_all_variants"] = {k: round(float(h), 4) for k, h in zip(ks, hp)}; R["n_variants_total"] = len(ks)
print(R["holm_all_variants"], len(ks))
json.dump(R, open(f"{OUT}/results.json", "w"), indent=1, default=float)
