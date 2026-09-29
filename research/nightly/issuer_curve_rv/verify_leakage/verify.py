"""Adversarial leakage checks on issuer_curve_rv (pre-2026 only; no holdout use)."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from research.nightly import harness as H
from research.nightly.issuer_curve_rv import signals as S

OUT = Path(__file__).resolve().parent
R = {}
PM = S.attach(H.load_panel("M"))
dts = np.sort(PM["day"].unique())
prev = pd.Series(dts[:-1], index=dts[1:])

# lagged signals: value from the previous monthly decision date for the same bond (1 month stale)
lagcols = ["e_iss", "resid_bps", "cdi_bps"]
L = PM[["day", "codigo"] + lagcols].copy()
L["day"] = L["day"].map(pd.Series(dts[1:], index=dts[:-1]))
L = L.dropna(subset=["day"]).rename(columns={c: c + "_lag" for c in lagcols})
PM = PM.merge(L, on=["day", "codigo"], how="left")

def make_tilt(lam, col):
    def f(x):
        sc = x["cdi_bps"] + lam * x[col].fillna(0).clip(-300, 300)
        thr = sc.quantile(0.7)
        nz = lambda s, v: s.fillna(v)
        return ((sc >= thr) & ~(nz(x["resid_z"], 0) <= -1.5) & (nz(x["press_neg_30d"], 0) < 1) & ~x["worstQ"]).to_numpy()
    return f

def tilt_ctrl(x):   # same construction with lam=0: isolates the cdi_bps-quantile vs cdi_pct definitional difference
    return make_tilt(0.0, "e_iss")(x)

rng = np.random.default_rng(11)
def shuffled(x):   # placebo: e_iss permuted within date
    v = x["e_iss"].to_numpy().copy(); ok = np.isfinite(v); v[ok] = rng.permutation(v[ok])
    return make_tilt(0.5, "_perm")(x.assign(_perm=v))

BT = {"tilt05": H.backtest(make_tilt(0.5, "e_iss"), panel=PM, name="tilt05"),
      "tilt0_ctrl": H.backtest(tilt_ctrl, panel=PM, name="tilt0"),
      "tilt05_lag1m": H.backtest(make_tilt(0.5, "e_iss_lag"), panel=PM, name="lag"),
      "tilt05_perm": H.backtest(shuffled, panel=PM, name="perm")}
tab = H.compare({**BT, "P4Q": H.baseline("P4Q")}, bench="P4Q")
print(tab[["exCDI_%", "exU_%", "vs_bench_%", "t_vs_bench", "h1_vs_bench", "h2_vs_bench"]].round(3).to_string())
R["books"] = tab[["exCDI_%", "exU_%", "vs_bench_%", "t_vs_bench", "h1_vs_bench", "h2_vs_bench"]].round(4).to_dict("index")
# tilt vs its own lam=0 control (the true marginal value of e_iss)
st = H.stats(BT["tilt05"]["daily"], bench=BT["tilt0_ctrl"]["daily"])
R["tilt05_vs_lam0"] = {k: round(float(st[k]), 4) for k in ("diff_ann_%", "diff_t_nw")}
st = H.stats(BT["tilt05"]["daily"], bench=H.baseline("P4Q", cost_bps=50)["daily"])
# 50bps
b50 = H.backtest(make_tilt(0.5, "e_iss"), panel=PM, cost_bps=50)
st = H.stats(b50["daily"], bench=H.baseline("P4Q", cost_bps=50)["daily"])
R["tilt05_50bps_vs_P4Q"] = {k: round(float(st[k]), 4) for k in ("diff_ann_%", "diff_t_nw")}
print(R)

# ---------------- pair trades: execution-lag / stale-mark / selection checks
def pairs(P, rank_col, tgt="fwd_126", require_exec=True, fresh_today=False):
    Hh = int(tgt.split("_")[1])
    X = P[P["univ"] & P[f"dok_{Hh}"] & P[rank_col].notna()].copy()
    if require_exec:
        X = X[X[tgt].notna()]
    else:
        X[tgt] = X[tgt].fillna(0.0)
    if fresh_today:
        X = X[X["age"] <= 0]
    X = X[X.groupby(["day", "cnpj8"])["codigo"].transform("count") >= 2]
    g = X.groupby(["day", "cnpj8"])
    hi = X.loc[g[rank_col].idxmax()].set_index(["day", "cnpj8"])
    lo = X.loc[g[rank_col].idxmin()].set_index(["day", "cnpj8"])
    d = (hi[tgt] - lo[tgt]).rename("dret").reset_index()
    d = d[(hi["codigo"].to_numpy() != lo["codigo"].to_numpy())]
    s = d.groupby("day")["dret"].mean()
    ann = 252 / Hh
    return {"ann_%": round(float(s.mean() * ann * 100), 3), "t_nw": round(float(H.nw_t(s, Hh // 21)), 2),
            "n": int(len(d)), "h1": round(float(s[s.index < H.SPLIT].mean() * ann * 100), 3),
            "h2": round(float(s[s.index >= H.SPLIT].mean() * ann * 100), 3)}

pt = {}
pt["resid_same_day"] = pairs(PM, "resid_bps")
pt["resid_nonexec_as_cash"] = pairs(PM, "resid_bps", require_exec=False)
pt["resid_lag1m"] = pairs(PM, "resid_bps_lag")
pt["e_iss_same_day"] = pairs(PM, "e_iss")
pt["e_iss_lag1m"] = pairs(PM, "e_iss_lag")
pt["cdi_same_day"] = pairs(PM, "cdi_bps")
pt["cdi_lag1m"] = pairs(PM, "cdi_bps_lag")
if "age" in PM.columns:
    pt["resid_fresh_today_only"] = pairs(PM, "resid_bps", fresh_today=True)
# resid lagged vs same-day disagreement: residual of the lag1m effect net of carry lag
R["pairs"] = pt
print(json.dumps(pt, indent=1))
# entry delay distribution for pair legs
X = PM[PM["univ"] & PM["executed"]]
R["entry_delay_bdays"] = X.eval("entry_pos - dpos").describe().round(2).to_dict()
R["exec_rate_univ"] = round(float(PM.loc[PM["univ"], "executed"].mean()), 3)
json.dump(R, open(OUT / "verify_results.json", "w"), indent=1, default=str)
