"""Adversarial leakage verification of factor_zoo (pre-2026 only; holdout not re-read)."""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd
from research.nightly import harness as H
from research.nightly.factor_zoo import features as FZ
from research.nightly.factor_zoo.run import composite, screen_fn, nscore

OUT = Path(__file__).parent
spec = json.loads((OUT.parent / "results.json").read_text())["composite_spec"]
P = FZ.attach(H.load_panel("M"))
P = composite(P, spec, "zoo")
res = {}
U = P[P["univ"]]
# --- tie structure of days_since_distress
d = U["days_since_distress"]
res["dsd"] = {"nan_frac": float(d.isna().mean()), "top_value_share": float(d.value_counts(normalize=True).iloc[0]),
              "top_value": float(d.value_counts().index[0]), "describe": d.describe().round(1).to_dict()}
# --- structure of the screen: listed share among dropped vs kept
Q = P[P["p4q"]].copy()
thr = Q.groupby("day")["zoo"].transform(lambda s: s.quantile(0.2))
Q["drop"] = Q["zoo"] <= thr
res["listed_share"] = {"p4q": float(Q["listed"].mean()), "dropped": float(Q.loc[Q["drop"], "listed"].mean()),
                       "kept": float(Q.loc[~Q["drop"], "listed"].mean())}
bench = H.baseline("P4Q")["daily"]
bt = lambda sig, **kw: H.backtest(sig, freq="M", hold=126, panel=P, **kw)
def st(r, lo=None, hi=None):
    s = H.stats(r["daily"], bench=bench)
    out = {k: s.get(k) for k in ("diff_ann_%", "diff_t_nw", "diff_h1_%", "diff_h2_%")}
    out["n_avg"] = r["n_avg"]
    return out
R = {}
R["repro_zoo_screen20"] = bt(screen_fn("zoo", 0.2))
# composite without days_since_distress (tie-bug factor) and without age (staleness)
for nm, drop in (("no_dsd", ["days_since_distress"]), ("no_age", ["age"]), ("no_dsd_age_bondage", ["days_since_distress", "age", "bond_age_y"])):
    P = composite(P, {k: v for k, v in spec.items() if k not in drop}, "z_" + nm)
    R[nm] = bt(screen_fn("z_" + nm, 0.2))
# stale signal: composite from previous monthly decision (1 month older info)
prev = P[["day", "codigo", "zoo"]].copy()
days_sorted = np.sort(P["day"].unique())
nxt = dict(zip(days_sorted[:-1], days_sorted[1:]))
prev["day"] = prev["day"].map(nxt)
P = P.merge(prev.rename(columns={"zoo": "zoo_lag1m"}), on=["day", "codigo"], how="left")
R["lag1m_signal"] = bt(screen_fn("zoo_lag1m", 0.2))
# placebo matched on listed status: drop random 20% but with the same listed/unlisted mix as the actual drop each date
rng = np.random.default_rng(11)
pl = []
P["_one"] = 1.0
for i in range(20):
    P["_r"] = rng.random(len(P))
    def f(x):
        b = x["p4q"].to_numpy()
        z = x["zoo"].where(x["p4q"])
        bad = (z <= z.quantile(0.2)).to_numpy() & z.notna().to_numpy()
        keep = b.copy()
        for lst in (True, False):
            m = b & (x["listed"].to_numpy() == lst)
            k = int((bad & m).sum())
            idx = np.where(m)[0]
            if k > 0:
                drop = idx[np.argsort(x["_r"].to_numpy()[idx])[:k]]
                keep[drop] = False
        return keep
    pl.append(H.stats(bt(f)["daily"], bench=bench)["diff_ann_%"])
res["placebo_listed_matched"] = {"mean": float(np.mean(pl)), "p95": float(np.percentile(pl, 95)), "all": [float(v) for v in pl]}
# rec40 (distressed stop-trading bonds at 40% recovery) for both
r40 = bt(screen_fn("zoo", 0.2), scenario="rec40")
b40 = H.baseline("P4Q", scenario="rec40")["daily"]
s40 = H.stats(r40["daily"], bench=b40)
res["rec40"] = {k: s40.get(k) for k in ("diff_ann_%", "diff_t_nw")}
# look-ahead-flagged distressed stop bonds among dropped vs kept
res["dist_stop_share"] = {"dropped": float(Q.loc[Q["drop"], "dist_stop_LOOKAHEAD"].mean()),
                          "kept": float(Q.loc[~Q["drop"], "dist_stop_LOOKAHEAD"].mean())}
# out-of-selection only: months 2024-01..2025-12 (tranches from decisions >= 2023-07)
for nm, r in R.items():
    res[nm] = st(r)
m = H.monthly(R["repro_zoo_screen20"]["daily"]) - H.monthly(bench)
m = m[(m.index >= "2024-01-01") & (m.index < "2026-01-01")]
res["oos_2024_25"] = {"diff_ann_%": float(m.mean() * 12 * 100), "t_nw": float(H.nw_t(m, 6)), "n": int(len(m))}
m2 = H.monthly(R["repro_zoo_screen20"]["daily"]) - H.monthly(bench)
m2 = m2[(m2.index >= "2022-01-01") & (m2.index < "2024-01-01")]
res["in_sel_2022_23"] = {"diff_ann_%": float(m2.mean() * 12 * 100), "t_nw": float(H.nw_t(m2, 6))}
print(json.dumps(res, indent=1, default=float))
(OUT / "verify_results.json").write_text(json.dumps(res, indent=1, default=float))
