"""Follow-ups motivated by run.py (counted as extra variants; Holm recomputed over ALL variants):
  F1  weekly decisions, 21-bday tranches: P4+Q minus bottom-quintile press21 (short-lived fire-sale pressure)
      vs P4+Q on the same engine.
  F2  weekly decisions, 21-bday tranches: P4+Q minus top-quintile stress63.
  F3  monthly 126: P4+Q minus bottom-quintile cshare computed WITHIN kind (DI/IPCA/PRE) -> is X4 a kind effect?
Diagnostics: cshare IC within kind; orphan (no fund holder) profile.
"""
import json
import numpy as np
import pandas as pd
from research.nightly import harness as H
from research.nightly.fund_flows import features as F

OUT = "research/nightly/fund_flows"
R = json.load(open(f"{OUT}/results.json"))


def with_feats(P):
    X = F.signals(P)
    return pd.concat([P.reset_index(drop=True), X.drop(columns=["day", "codigo"]).reset_index(drop=True)], axis=1)


PM = with_feats(H.load_panel("M"))
PW = with_feats(H.load_panel("W"))


def ex(f, low=True, within=None):
    def g(x):
        v = x[f].groupby(x[within]).rank(pct=True) if within else x[f].rank(pct=True)
        flag = ((v <= 0.2) if low else (v > 0.8)).fillna(False).to_numpy()
        return x["p4q"].to_numpy() & ~flag
    return g


out = {}
bW = H.backtest("p4q", panel=PW, freq="W", hold=21, name="P4Q_W21")
for nm, sig in [("F1_W21_ex_press21", ex("press21", True)), ("F2_W21_ex_stress63", ex("stress63", False))]:
    r = H.backtest(sig, panel=PW, freq="W", hold=21, name=nm)
    s = H.stats(r["daily"], bench=bW["daily"])
    r50 = H.backtest(sig, panel=PW, freq="W", hold=21, cost_bps=50)
    b50 = H.backtest("p4q", panel=PW, freq="W", hold=21, cost_bps=50)
    s50 = H.stats(r50["daily"], bench=b50["daily"])
    out[nm] = {"exCDI_%": s["ann_excess_%"], "vs_bench_%": s["diff_ann_%"], "t": s["diff_t_nw"], "p": s["diff_p"],
               "h1": s["diff_h1_%"], "h2": s["diff_h2_%"], "turnover": r["turnover_ann"],
               "bench": "P4Q weekly 21-bday tranches", "bench_exCDI_%": H.stats(bW["daily"])["ann_excess_%"],
               "vs_bench_50bps_%": s50["diff_ann_%"], "t_50": s50["diff_t_nw"]}
    print(nm, out[nm])
r = H.backtest(ex("cshare", True, within="kind"), panel=PM, name="F3")
s = H.stats(r["daily"], bench=H.baseline("P4Q")["daily"])
out["F3_ex_low_cshare_within_kind"] = {"exCDI_%": s["ann_excess_%"], "vs_bench_%": s["diff_ann_%"], "t": s["diff_t_nw"],
                                       "p": s["diff_p"], "h1": s["diff_h1_%"], "h2": s["diff_h2_%"], "bench": "P4Q"}
print(out["F3_ex_low_cshare_within_kind"])

# Holm across all variants tried (run.py table + follow-ups)
pv = {k: v["p_vs_bench"] for k, v in R["table_25bps"].items()}
pv.update({k: v["p"] for k, v in out.items()})
ks = list(pv)
hp = H.holm(np.array([pv[k] for k in ks], dtype=float))
R["holm_all_variants"] = {k: round(float(h), 4) for k, h in zip(ks, hp)}
R["n_variants_total"] = len(ks)
R["followups"] = out

# diagnostics
u = PM[PM["univ"]]
diag = {"cshare_by_kind_median": u.groupby("kind")["cshare"].median().round(3).to_dict(),
        "cshare_by_incent_median": u.groupby("incent")["cshare"].median().round(3).to_dict()}
for k, g in PM.groupby("kind"):
    try:
        r_ = H.ic(PM[PM["kind"] == k], "cshare", "fwd_63")
        diag[f"cshare_ic63_{k}"] = {"ic": round(r_["mean"], 4), "t": round(r_["t_nw"], 2), "n_dates": r_["n_dates"]}
    except Exception as e:
        diag[f"cshare_ic63_{k}"] = str(e)[:80]
orph = u["nh"] == 0
diag["orphans"] = {"share_univ": float(orph.mean()), "share_p4q": float(u.loc[u["p4q"], "nh"].eq(0).mean()),
                   "median_cdi_bps_orphan": float(u.loc[orph, "cdi_bps"].median()),
                   "median_cdi_bps_held": float(u.loc[~orph, "cdi_bps"].median()),
                   "trades_30d_orphan": float(u.loc[orph, "trades_30d"].median()) if "trades_30d" in u else None,
                   "trades_30d_held": float(u.loc[~orph, "trades_30d"].median()) if "trades_30d" in u else None,
                   "covered_share_orphan": float(u.loc[orph, "covered"].mean()),
                   "covered_share_held": float(u.loc[~orph, "covered"].mean())}
y = u.dropna(subset=["fwd_126"])
diag["orphans"]["fwd126_mean_orphan_%"] = float(y.loc[y["nh"] == 0, "fwd_126"].mean() * 100)
diag["orphans"]["fwd126_mean_held_%"] = float(y.loc[y["nh"] > 0, "fwd_126"].mean() * 100)
R["diagnostics"] = diag
print(json.dumps(diag, indent=1, default=float))
json.dump(R, open(f"{OUT}/results.json", "w"), indent=1, default=float)
print(R["holm_all_variants"])
