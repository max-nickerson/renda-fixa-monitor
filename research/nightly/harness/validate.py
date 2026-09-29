"""Validation of the common harness: reproduces the selection-lab / HY-lab baselines, times each call, measures
peak memory. Writes research/nightly/harness/validation.json and baselines.png.

Run:  PYTHONPATH=. PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe research/nightly/harness/validate.py
"""
import ctypes
import json
import sys
import time
from ctypes import wintypes
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np
import pandas as pd

from research.nightly import harness as H

OUT = Path(__file__).parent


def peak_mb():
    try:
        class PMC(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
        c = PMC()
        c.cb = ctypes.sizeof(PMC)
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        k32.K32GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
        k32.K32GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(c), c.cb)
        return {"peak_working_set_MB": round(c.PeakWorkingSetSize / 1e6), "current_MB": round(c.WorkingSetSize / 1e6)}
    except Exception as e:
        return {"error": str(e)}


TM, MEM = {}, {}


def timed(k, f):
    t = time.time()
    r = f()
    TM[k] = round(time.time() - t, 2)
    MEM[k] = peak_mb()["peak_working_set_MB"] if "peak_working_set_MB" in peak_mb() else None
    return r


res = {}
timed("core", H._core)
PM = timed("load_panel_M", lambda: H.load_panel("M"))
PW = timed("load_panel_W", lambda: H.load_panel("W"))
res["panel"] = {"M": {"rows": len(PM), "dates": int(PM["day"].nunique()), "cols": PM.shape[1],
                      "univ_rows": int(PM["univ"].sum()), "MB": round(PM.memory_usage(deep=True).sum() / 1e6)},
                "W": {"rows": len(PW), "dates": int(PW["day"].nunique()), "cols": PW.shape[1],
                      "univ_rows": int(PW["univ"].sum()), "MB": round(PW.memory_usage(deep=True).sum() / 1e6)},
                "coverage_univ_rows": {"fundamentals": float(PM.loc[PM["univ"], "covered"].mean()),
                                       "listed_equity": float(PM.loc[PM["univ"], "listed"].mean()),
                                       "weekly_extras(trades_30d)": float(PM.loc[PM["univ"], "trades_30d"].notna().mean()),
                                       "sector_known": float((PM.loc[PM["univ"], "sector"] != "unknown").mean())},
                "columns": list(PM.columns)}

# ---------------- pre-2026 baselines (the numbers agents compare against)
B = {}
for nm in ("U", "P4", "P4Q", "P4_live"):
    B[nm] = timed(f"baseline_{nm}", lambda nm=nm: H.baseline(nm))
tab = H.compare(B, bench="P4Q")
res["pre2026_baselines_25bps"] = tab.round(4).to_dict("index")
res["pre2026_baselines_vs_P4"] = H.compare({k: B[k] for k in ("U", "P4Q", "P4_live")}, bench="P4").round(4) \
    .to_dict("index")
full = {}
for k, v in B.items():
    s = H.stats(v["daily"], bench=B["U"]["daily"])
    full[k] = {kk: s[kk] for kk in ("ann_excess_%", "vol_%", "sharpe", "max_dd_%", "worst_month_%", "t_nw",
                                    "h1_2022_23_%", "h2_2024_25_%", "diff_ann_%", "diff_t_nw", "n_months", "period")}
    full[k].update(turnover_ann=v["turnover_ann"], cost_ann_pct=v["cost_ann_%"], n_avg=v["n_avg"],
                   avg_exposure=float(v["exposure"].mean()))
res["pre2026_baselines_detail(diff=vs U)"] = full
sens = {}
for tag, kw in {"50bps": dict(cost_bps=50), "rec40": dict(scenario="rec40"), "hold63": dict(hold=63),
                "hold252": dict(hold=252), "weekly_tranche_hold126": dict(freq="W")}.items():
    rr = {nm: timed(f"{nm}_{tag}", lambda nm=nm: H.baseline(nm, **kw)) for nm in ("U", "P4", "P4Q")}
    sens[tag] = {nm: {"exCDI_%": H.stats(r["daily"])["ann_excess_%"],
                      "vs_U_%": H.stats(r["daily"], bench=rr["U"]["daily"])["diff_ann_%"],
                      "P4Q_vs_P4_%": H.stats(rr["P4Q"]["daily"], bench=rr["P4"]["daily"])["diff_ann_%"]}
                 for nm, r in rr.items()}
res["pre2026_sensitivities"] = sens
# P4 engine variants (documented differences between execution designs)
var = {"P4 monthly replace-the-book": dict(freq="M", hold=None),
       "P4 monthly sticky (sell when rich)": dict(freq="M", hold=None, exit_signal="rich"),
       "P4 weekly sticky (sell when rich)": dict(freq="W", hold=None, exit_signal="rich"),
       "P4 tranche 126 + IDA overlay": dict(overlay="ida"),
       "P4 tranche 126 + sell when rich (weekly checks)": dict(freq="W", rebalance="M", exit_signal="rich")}
vv = {}
for k, kw in var.items():
    r = timed(k, lambda kw=kw: H.backtest("p4", **kw))
    s = H.stats(r["daily"], bench=B["P4"]["daily"])
    vv[k] = {"exCDI_%": s["ann_excess_%"], "vs_P4_tranche_%": s["diff_ann_%"], "t": s["diff_t_nw"],
             "turnover": r["turnover_ann"], "cost_%": r["cost_ann_%"], "avg_expo": float(r["exposure"].mean())}
res["pre2026_engine_variants"] = vv

# ---------------- cohort level (selection-lab comparable)
coh = {}
for hh in (63, 126, 252):
    a = timed(f"cohort_P4_{hh}", lambda: H.cohort_excess("p4", H=hh))
    q = H.cohort_excess("p4q", H=hh)
    coh[hh] = {"P4_vs_U_ann_%": a.mean() / (hh / 252) * 100, "P4Q_vs_U_ann_%": q.mean() / (hh / 252) * 100,
               "P4Q_minus_P4_t_nw": H.nw_t((q - a.reindex(q.index)).dropna(), hh // 21), "n_cohorts": len(a)}
res["pre2026_cohort_level"] = coh

# ---------------- placebo and IC sanity
pl = timed("placebo_P4Q_n20", lambda: H.placebo("p4q", n=20))
res["pre2026_placebo_P4Q_random_same_size"] = {"actual": pl["actual"], "null_mean": pl["mean"], "null_p95": pl["p95"]}
res["pre2026_ic_fwd126"] = {f: {k: v for k, v in H.ic(PM, f, "fwd_126").items() if k != "series"}
                            for f in ("cdi_bps", "resid_bps", "f_quality", "eq_r63", "trades_30d")}

# ---------------- reproduction of the published HY-lab / selection-lab numbers (these INCLUDE 2026, which those
# labs already published; no strategy choice is made here)
rep = {}
BF = {nm: H.baseline(nm, holdout=True) for nm in ("U", "P4", "P4Q")}
for nm, r in BF.items():
    s = H.stats(r["daily"], bench=BF["P4"]["daily"], holdout=True)
    rep[nm] = {"exCDI_%": s["ann_excess_%"], "vol_%": s["vol_%"], "sharpe": s["sharpe"], "maxDD_%": s["max_dd_%"],
               "vs_P4_%": s["diff_ann_%"], "t": s["diff_t_nw"], "period": s["period"]}
rep["HY_lab_published"] = {"U": 0.45, "P4": 1.53, "P4Q": 1.94, "P4Q_vs_P4": "+0.41 (t 2.9)", "vol P4": 1.09,
                           "maxDD P4": -1.52}
rep["cohort_P4_vs_U_incl_2026"] = {hh: H.cohort_excess("p4", H=hh, holdout=True).mean() / (hh / 252) * 100
                                   for hh in (63, 126, 252)}
rep["selection_lab_published_P4_vs_U"] = {63: 1.23, 126: 1.10, 252: 1.07}
res["reproduction_incl_2026"] = rep

# ---------------- plot
timed("plot", lambda: H.plot_curves({"P4 (tranche 126)": B["P4"], "P4+Q (tranche 126)": B["P4Q"],
                                     "P4 live (weekly sticky + IDA overlay)": B["P4_live"]},
                                    OUT / "baselines.png", refs=("CDI", "U", "IDADI", "IBOV"),
                                    title="Harness baselines, decisions 2022-01..2025-12 (holdout sealed), 25 bps"))
res["timing_s"] = TM
res["peak_working_set_MB_after_step"] = MEM
res["memory_final"] = peak_mb()


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if o != o else round(float(o), 4)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


json.dump(clean(res), open(OUT / "validation.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print(tab.round(3).to_string())
print(json.dumps(clean({k: res[k] for k in ("pre2026_sensitivities", "pre2026_engine_variants",
                                            "pre2026_cohort_level", "reproduction_incl_2026", "timing_s",
                                            "memory_final", "pre2026_placebo_P4Q_random_same_size",
                                            "pre2026_ic_fwd126")}), indent=1))
