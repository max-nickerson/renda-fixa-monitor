"""Adversarial leakage verification of event_driven's P4Q_exNegEvents (pre-2026 only; holdout not touched)."""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
import pandas as pd
from research.nightly import harness as H
from research.nightly.event_driven import run
from research.nightly.event_driven.events import build

OUT = Path(__file__).resolve().parent
T0 = time.time()
def log(*a): print(f"[{time.time()-T0:6.0f}s]", *a, flush=True)
res = {}
b = H.baseline("P4Q")
def diff(r, bench=b):
    s = H.stats(r["daily"], bench=bench["daily"])
    return {k: s[k] for k in ["ann_excess_%", "diff_ann_%", "diff_t_nw", "diff_h1_%", "diff_h2_%"]}

ref, E = build()
P = H.load_panel("M")
log("panel", P.shape)

# 0. cnpj8 stability per codigo in the panel (issuer remapping after M&A = look-ahead identity)
nun = P.groupby("codigo")["cnpj8"].nunique()
res["codigos_with_multiple_cnpj8"] = int((nun > 1).sum())
regc = ref.set_index("codigo")["cnpj8"]
pc = P.drop_duplicates("codigo").set_index("codigo")["cnpj8"].astype(str)
common = pc.index.intersection(regc.index)
res["panel_vs_registry_cnpj8_mismatch"] = int((pc[common] != regc[common]).sum())

# 1. reproduce with cached signals and with freshly rebuilt signals
PMc = run.load("M")
r_c = H.backtest(run.RULES["P4Q_exNegEvents"], panel=PMc, name="repro_cache")
res["repro_cached"] = diff(r_c)
S = run.signals(P).drop(columns=["cnpj8"])
PMf = P.merge(S, on=["codigo", "day"], how="left")
r_f = H.backtest(run.RULES["P4Q_exNegEvents"], panel=PMf, name="repro_fresh")
res["repro_fresh"] = diff(r_f)
cols = [c for c in S.columns if c.startswith("ev_")]
a = PMc.set_index(["codigo", "day"])[cols].sort_index(); bb = PMf.set_index(["codigo", "day"])[cols].sort_index()
res["cached_vs_fresh_signal_mismatch_cells"] = int(((a.fillna(-1) != bb.fillna(-1))).to_numpy().sum())
log("repro", res["repro_cached"], res["repro_fresh"])

NEG = {"ma": 180, "rating_down": 180, "agd": 180, "resgate": 126, "rj": 365}

def neg_from(Ev, panel, windows=NEG, lo=None):
    """mask of rows with any neg event in (lo, window] days before day (strictly before)."""
    X = panel[["codigo", "day", "cnpj8"]].copy(); X["cnpj8"] = X["cnpj8"].astype(str)
    X["day"] = X["day"].astype("datetime64[ns]"); X["_i"] = np.arange(len(X)); X = X.sort_values("day")
    m_all = np.zeros(len(X), bool)
    for et, w in windows.items():
        e = Ev[Ev["etype"] == et][["cnpj8", "date"]].rename(columns={"date": "_ed"}).sort_values("_ed")
        e["_ed"] = e["_ed"].astype("datetime64[ns]"); e["cnpj8"] = e["cnpj8"].astype(str)
        if lo is None:
            m = pd.merge_asof(X, e, left_on="day", right_on="_ed", by="cnpj8", allow_exact_matches=False)
            d = (m["day"] - m["_ed"]).dt.days.to_numpy()
            m_all |= np.nan_to_num(d, nan=1e9) <= w
        else:  # any event with lo < age <= w  (count events via searchsorted per issuer)
            grp = {k: np.sort(v["_ed"].to_numpy()) for k, v in e.groupby("cnpj8")}
            ok = np.zeros(len(X), bool)
            for i, (c, dday) in enumerate(zip(X["cnpj8"].to_numpy(), X["day"].to_numpy())):
                arr = grp.get(c)
                if arr is None: continue
                hi_ = np.searchsorted(arr, dday - np.timedelta64(lo, "D"), side="left")
                lo_ = np.searchsorted(arr, dday - np.timedelta64(w, "D"), side="left")
                ok[i] = hi_ > lo_
            m_all |= ok
    out = np.zeros(len(X), bool); out[X["_i"].to_numpy()] = m_all
    return out

def run_mask(mask, name):
    Q = P.assign(_neg=mask)
    r = H.backtest(lambda x: x["p4q"].to_numpy() & ~x["_neg"].to_numpy(), panel=Q, name=name)
    return r

# 2. own re-implementation (independent code path) must match
m0 = neg_from(E, P)
r0 = run_mask(m0, "reimpl"); res["reimpl"] = diff(r0); log("reimpl", res["reimpl"])

# 3. delay all events by k calendar days (should decay smoothly if genuine, not collapse)
for k in [3, 10, 30, 60]:
    Ek = E.assign(date=E["date"] + pd.Timedelta(days=k))
    res[f"delay_{k}d"] = diff(run_mask(neg_from(Ek, P), f"delay{k}")); log("delay", k, res[f"delay_{k}d"])

# 4. rating events: conservative date = max(action date, wayback capture date)
rt = pd.read_pickle(H.HIST / "rating_events.pkl")
cap = pd.to_datetime(rt["captured"].astype(str).str[:8], format="%Y%m%d", errors="coerce")
lag = (cap - pd.to_datetime(rt["date"])).dt.days
res["rating_capture_lag_days"] = lag.describe().round(1).to_dict()
rt = rt.assign(d2=np.where(cap.notna() & (cap > rt["date"]), cap, rt["date"]))
rt = rt[rt["direction"] < 0]
Er = E[E["etype"] != "rating_down"]
Er = pd.concat([Er, pd.DataFrame({"cnpj8": rt["cnpj8"].astype(str).str.zfill(8), "date": pd.to_datetime(rt["d2"]).dt.normalize(),
                                  "etype": "rating_down"})], ignore_index=True)
res["rating_captured_dates"] = diff(run_mask(neg_from(Er, P), "ratcap")); log("ratcap", res["rating_captured_dates"])

# 5. drop-one components & per-component alone (already partly in README) and without rating_down entirely
for et in NEG:
    w = {k: v for k, v in NEG.items() if k != et}
    res[f"without_{et}"] = diff(run_mask(neg_from(E, P, w), f"wo_{et}")); log("without", et, res[f"without_{et}"])

# 6. stale-event test: same event types but only events 365-730d old (issuer trait, not event drift)
res["stale_365_730d"] = diff(run_mask(neg_from(E, P, {k: 730 for k in NEG}, lo=365), "stale"))
log("stale", res["stale_365_730d"])
# 'ever had a neg event' (any time before day)
res["ever_before"] = diff(run_mask(neg_from(E, P, {k: 5000 for k in NEG}), "ever")); log("ever", res["ever_before"])

# 7. timing placebo: keep each issuer's events but shift ALL of that issuer's events by a random offset
#    (circular within 2020-06..2025-12) -> same issuers, same event density, wrong timing
rng = np.random.default_rng(7)
lo_, hi_ = pd.Timestamp("2020-06-01"), pd.Timestamp("2025-12-31"); span = (hi_ - lo_).days
En = E[E["etype"].isin(list(NEG))].copy(); En = En[En["date"] < H.HOLDOUT]
tp = []
for s in range(20):
    off = {c: rng.integers(180, span - 180) for c in En["cnpj8"].unique()}
    o = En["cnpj8"].map(off).astype(int)
    d = lo_ + pd.to_timedelta(((En["date"] - lo_).dt.days + o) % span, unit="D")
    tp.append(diff(run_mask(neg_from(En.assign(date=d), P), f"tplac{s}"))["diff_ann_%"])
tp = np.array(tp)
real = res["reimpl"]["diff_ann_%"]
res["timing_placebo"] = {"mean": float(tp.mean()), "p95": float(np.percentile(tp, 95)), "max": float(tp.max()),
                         "share_ge_real": float((tp >= real).mean()), "n": len(tp), "draws": tp.round(3).tolist()}
log("timing placebo", res["timing_placebo"])

# 8. yearly diff vs P4Q
dd = (r0["daily"].reindex(b["daily"].index).fillna(0) - b["daily"])
dd = dd[dd.index < H.HOLDOUT]
res["diff_by_year_%"] = (dd.groupby(dd.index.year).sum() * 100).round(3).to_dict()

(OUT / "verify_results.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
log("done")
