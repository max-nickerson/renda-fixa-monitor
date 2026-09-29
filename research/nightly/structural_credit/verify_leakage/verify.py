"""Adversarial leakage verification of structural_credit (A3/A1 avoid rules vs P4+Q). Pre-2026 only."""
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from research.nightly import harness as H
from research.nightly.structural_credit import signals as S
from research.nightly.structural_credit import build_data as B
OUT = Path(__file__).parent
T0 = time.time()
def log(*a): print(f"[{time.time()-T0:5.0f}s]", *a, flush=True)
RES = {}

def worst_q(x, col, q=0.2, low_bad=True):
    v = x[col]; ok = v.notna()
    if ok.sum() < 10: return np.zeros(len(x), bool)
    thr = v[ok].quantile(q if low_bad else 1 - q)
    return (ok & ((v <= thr) if low_bad else (v >= thr))).to_numpy()

def rule(col, low_bad, q=0.2, keep_iss=None):
    def f(x):
        w = worst_q(x, col, q, low_bad)
        if keep_iss is not None: w = w & ~x["cnpj8"].isin(keep_iss).to_numpy()
        return x["p4q"].to_numpy() & ~w
    return f

bench = H.baseline("P4Q")["daily"]
def diff(r):
    s = H.stats(r["daily"], bench=bench)
    return {"diff": round(s["diff_ann_%"], 3), "t": round(s["diff_t_nw"], 2), "h1": round(s["diff_h1_%"], 3), "h2": round(s["diff_h2_%"], 3)}

D0 = S.load_daily()
PM = S.signals(H.load_panel("M"), D0)
# 1 reproduce
for k, (c, lb) in {"A3": ("sc_cg", False), "A1": ("sc_dd", True), "C2": ("eq_dd252", True)}.items():
    RES["repro_" + k] = diff(H.backtest(rule(c, lb), panel=PM)); log(k, RES["repro_" + k])
# 2 lag the structural signal by ~1 week (signal known 5 bdays before decision)
Dl = D0.copy(); Dl["date"] = Dl.groupby("ticker")["date"].shift(-5); Dl = Dl.dropna(subset=["date"])
PL = S.signals(H.load_panel("M"), Dl)
RES["lag5_A3"] = diff(H.backtest(rule("sc_cg", False), panel=PL)); log("lag5 A3", RES["lag5_A3"])
RES["lag5_A1"] = diff(H.backtest(rule("sc_dd", True), panel=PL)); log("lag5 A1", RES["lag5_A1"])
# 3 threshold sensitivity
for q in (0.1, 0.3):
    RES[f"A3_q{q}"] = diff(H.backtest(rule("sc_cg", False, q), panel=PM)); log("q", q, RES[f"A3_q{q}"])
# 4 issuer concentration: which issuers are dropped most; force-keep top issuers one at a time
X = PM[PM["univ"] & (PM["day"] >= H.START) & (PM["day"] < H.HOLDOUT)]
drops = []
for d, x in X.groupby("day"):
    m = rule("sc_cg", False)(x); p = x["p4q"].to_numpy()
    drops.append(x.loc[p & ~m, ["day", "cnpj8", "eq_ticker", "fwd_126"]])
Dr = pd.concat(drops)
top = Dr.groupby(["cnpj8", "eq_ticker"]).agg(n=("day", "size"), fwd=("fwd_126", "mean")).sort_values("n", ascending=False)
RES["dropped_issuers_top"] = top.head(15).reset_index().astype(str).to_dict("records")
worst = Dr.groupby("cnpj8")["fwd_126"].mean().sort_values().head(6).index.tolist()
RES["worst_dropped_issuers"] = Dr[Dr.cnpj8.isin(worst)].groupby(["cnpj8","eq_ticker"])["fwd_126"].agg(["mean","size"]).reset_index().astype(str).to_dict("records")
loo = {}
for c in worst:
    loo[c] = diff(H.backtest(rule("sc_cg", False, keep_iss=[c]), panel=PM)); log("keep", c, loo[c])
loo["keep_worst3"] = diff(H.backtest(rule("sc_cg", False, keep_iss=worst[:3]), panel=PM)); log("keep3", loo["keep_worst3"])
RES["force_keep_issuer"] = loo
# 5 CVM-only debt (drop brapi fundamentals rows, which are today's snapshot with synthetic +45/+90d dates)
log("rebuilding CVM-only debt")
orig = B.debt_pit
def debt_cvm():
    f = pd.read_pickle(B.HIST / "fundamentals_pit.pkl")
    f = f[f["source"] != "brapi"].copy()
    f["avail"] = f["available_date_strict"].astype("datetime64[ns]"); f["period_end"] = f["period_end"].astype("datetime64[ns]")
    f["cnpj8"] = f["cnpj8"].astype(str)
    f = f.sort_values(["cnpj8", "avail", "period_end"])
    f["_mx"] = f.groupby("cnpj8")["period_end"].cummax().groupby(f["cnpj8"]).shift(1)
    f = f[f["_mx"].isna() | (f["period_end"] > f["_mx"])].drop_duplicates(["cnpj8", "avail"], keep="last")
    return f[["cnpj8", "ticker", "is_parent", "avail", "period_end", "gross_debt", "st_debt", "cash", "total_assets", "equity", "ebitda_ltm"]]
B.debt_pit = debt_cvm
P = H.load_panel("W", holdout=True)
tick = sorted(P["eq_ticker"].dropna().unique()); del P
cdi = H.cdi_daily(); cdi_ann = ((1 + cdi.where(cdi > 0).ffill()) ** 252 - 1)
Dc = B.build(tick, cdi_ann)
Dc = Dc.sort_values(["ticker", "date"]).reset_index(drop=True)
g = Dc.groupby("ticker"); Dc["lcg"] = np.log(10 + Dc["cg_spread_bps"])
for n in (21, 63): Dc[f"dd_chg{n}"] = Dc["dd"] - g["dd"].shift(n)
Dc["lcg_chg63"] = Dc["lcg"] - g["lcg"].shift(63)
PC = S.signals(H.load_panel("M"), Dc)
RES["cvm_only_coverage_p4q"] = float(PC.loc[PC.univ & PC.p4q & (PC.day>=H.START), "sc_cg"].notna().mean())
RES["cvmonly_A3"] = diff(H.backtest(rule("sc_cg", False), panel=PC)); log("cvm A3", RES["cvmonly_A3"])
RES["cvmonly_A1"] = diff(H.backtest(rule("sc_dd", True), panel=PC)); log("cvm A1", RES["cvmonly_A1"])
# 6 effective multiple testing: best-of-12 selection. Fraction of A3 gain explained by eq_dd252 control
json.dump(RES, open(OUT / "verify_results.json", "w"), indent=1, default=str)
log("done")
