"""Audit runner: key baseline numbers from the candidate (fixed) harness, plus a decomposition of the fixes.
Run: PYTHONPATH=. .venv/Scripts/python.exe research/nightly/harness_audit/audit_run.py [module]"""
import importlib, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np, pandas as pd

mod = sys.argv[1] if len(sys.argv) > 1 else "research.nightly.harness_audit.harness_candidate"
tag = sys.argv[2] if len(sys.argv) > 2 else "after"
H = importlib.import_module(mod)
OUT = Path(__file__).parent
t0 = time.time()
res = {"module": mod}
C = H._core()
res["core_gap_moved"] = C.get("gap_returns_moved_to_realisation")
if tag == "oldR":   # decomposition: fixed engine but the old (pre-gap) return timing
    old = pd.read_pickle(H.HIST / "nightly" / "harness" / "core.pkl")
    H._MEM[("R64", "R")] = old["R"].astype(np.float64)
    H._MEM[("R64", "R40")] = old["R40"].astype(np.float64)
B = {nm: H.baseline(nm) for nm in ("U", "P4", "P4Q", "P4_live")}
res["baselines"] = H.compare(B, bench="P4Q").round(4).to_dict("index")
for k, v in B.items():
    res["baselines"][k]["cost_ann_%"] = round(v["cost_ann_%"], 4)
sens = {}
for st, kw in {"50bps": dict(cost_bps=50), "rec40": dict(scenario="rec40"), "hold63": dict(hold=63),
               "hold252": dict(hold=252), "weekly_tranche": dict(freq="W")}.items():
    rr = {nm: H.baseline(nm, **kw) for nm in ("U", "P4", "P4Q")}
    sens[st] = {nm: round(H.stats(r["daily"])["ann_excess_%"], 4) for nm, r in rr.items()}
    sens[st]["P4Q-P4"] = round(H.stats(rr["P4Q"]["daily"], bench=rr["P4"]["daily"])["diff_ann_%"], 4)
res["sensitivities"] = sens
var = {"P4 monthly replace-the-book": dict(freq="M", hold=None),
       "P4 monthly sticky (sell when rich)": dict(freq="M", hold=None, exit_signal="rich"),
       "P4 weekly sticky (sell when rich)": dict(freq="W", hold=None, exit_signal="rich"),
       "P4 tranche 126 + IDA overlay": dict(overlay="ida"),
       "P4 tranche 126 + sell when rich (weekly checks)": dict(freq="W", rebalance="M", exit_signal="rich")}
vv = {}
for k, kw in var.items():
    r = H.backtest("p4", **kw)
    s = H.stats(r["daily"], bench=B["P4"]["daily"])
    vv[k] = {"exCDI_%": s["ann_excess_%"], "vs_P4_tranche_%": s["diff_ann_%"], "t": s["diff_t_nw"],
             "turnover": round(r["turnover_ann"], 3), "cost_%": round(r["cost_ann_%"], 3)}
res["engine_variants"] = vv
if tag != "oldR":
    coh = {}
    for hh in (63, 126, 252):
        a = H.cohort_excess("p4", H=hh); q = H.cohort_excess("p4q", H=hh)
        coh[hh] = {"P4_vs_U": round(a.mean() / (hh / 252) * 100, 4), "P4Q_vs_U": round(q.mean() / (hh / 252) * 100, 4),
                   "t_P4Q_vs_P4": round(H.nw_t((q - a.reindex(q.index)).dropna(), hh // 21), 3)}
    res["cohort"] = coh
    BF = {nm: H.baseline(nm, holdout=True) for nm in ("U", "P4", "P4Q")}
    res["reproduction_incl_2026"] = {nm: {"exCDI_%": H.stats(r["daily"], holdout=True)["ann_excess_%"],
                                          "vs_P4_%": H.stats(r["daily"], bench=BF["P4"]["daily"], holdout=True)["diff_ann_%"]}
                                     for nm, r in BF.items()}
    pl = H.placebo("p4q", n=10)
    res["placebo_P4Q_n10"] = {"actual": pl["actual"], "null_mean": round(pl["mean"], 4), "null_p95": round(pl["p95"], 4)}
    PM = H.load_panel("M")
    res["ic_fwd126"] = {f: {k: round(v, 4) if isinstance(v, float) else v for k, v in H.ic(PM, f, "fwd_126").items()
                            if k != "series"} for f in ("cdi_bps", "resid_bps", "f_quality", "eq_r63")}
res["runtime_s"] = round(time.time() - t0, 1)
def clean(o):
    if isinstance(o, dict): return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (np.floating, float)): return None if o != o else round(float(o), 4)
    if isinstance(o, np.integer): return int(o)
    return o
json.dump(clean(res), open(OUT / f"audit_{tag}.json", "w"), indent=1)
print(json.dumps(clean(res), indent=1))
