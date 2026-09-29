"""Adversarial leakage/robustness check of macro_cycle's best variant (ov_mom63_inout). Writes only into this folder."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from research.nightly import harness as H
from research.nightly.macro_cycle import run as M
OUT = Path(__file__).resolve().parent
res = {}
F = pd.read_pickle(ROOT / "data/history/nightly/macro_cycle/macro_daily_used.pkl")
base25 = H.baseline("P4Q"); base50 = H.baseline("P4Q", cost_bps=50)
b = base25["daily"]
def st(r, bench=b):
    s = H.stats(r["daily"], bench=bench)
    return {k: s[k] for k in ("ann_excess_%", "vol_%", "max_dd_%", "diff_ann_%", "diff_t_nw", "diff_h1_%", "diff_h2_%")}
# A reproduce
e63 = M.weekly_hold((F["IDADI_x63"] > 0).astype(float).where(F["IDADI_x63"].notna()))
e63 = e63[e63.index >= "2012-01-01"]
r = M.apply_overlay(base25, e63); res["A_repro_25"] = st(r); res["A_avg_expo"] = r["avg_ov"]
res["A_repro_50"] = st(M.apply_overlay(base50, e63, cost_bps=50), base50["daily"])
# B IDA source cross-check (their xlsx vs harness index_levels)
ix = H.index_levels()["IDADI"]; cdi = H._core()["cdi_daily_raw"]
mom_h = (ix.pct_change() - cdi.reindex(ix.index)).rolling(63).sum()
g = H.days()
a1 = F["IDADI_x63"].reindex(g); a2 = mom_h.reindex(mom_h.index.union(g)).ffill().reindex(g)
ok = a1.notna() & a2.notna() & (g >= "2021-06-01")
res["B_sign_agree_harness_vs_theirs"] = float(((a1[ok] > 0) == (a2[ok] > 0)).mean())
res["B_corr"] = float(a1[ok].corr(a2[ok]))
# lead/lag check: which shift of harness mom best matches theirs (0 expected)
res["B_best_shift"] = {int(k): float(a1[ok].corr(a2.shift(k)[ok])) for k in (-2, -1, 0, 1, 2)}
eh = M.weekly_hold((mom_h > 0).astype(float).where(mom_h.notna()))
res["B_harness_ida_mom63"] = st(M.apply_overlay(base25, eh))
# C extra lags / daily (no weekly hold)
for L in (1, 2, 5, 10):
    res[f"C_lag{L}"] = st(M.apply_overlay(base25, e63, extra_lag=L))
# D exclude Americanas Jan-Feb 2023
d = H.monthly(r["daily"][r["daily"].index < H.HOLDOUT]) - H.monthly(b[b.index < H.HOLDOUT])
res["D_monthly_diff_top5"] = {str(k.date()): round(v * 100, 3) for k, v in d.sort_values(ascending=False).head(5).items()}
dx = d[~d.index.isin(pd.to_datetime(["2023-01-01", "2023-02-01"]))]
res["D_diff_ex_JanFeb2023_ann_%"] = float(dx.mean() * 1200); res["D_t_ex"] = H.nw_t(dx, 6)
res["D_diff_ex_2023Q1_ann_%"] = float(d[~d.index.isin(pd.to_datetime(["2023-01-01", "2023-02-01", "2023-03-01"]))].mean() * 1200)
# E placebo circular shift
rng = np.random.default_rng(1)
o = e63.reindex(e63.index.union(g)).ffill().reindex(g)
win = o[(o.index >= "2021-06-01") & (o.index < H.HOLDOUT)].dropna()
act = res["A_repro_25"]["diff_ann_%"]; vals = []
for _ in range(200):
    sh = pd.Series(np.roll(win.to_numpy(), rng.integers(60, len(win) - 60)), index=win.index)
    vals.append(H.stats(M.apply_overlay(base25, sh)["daily"], bench=b)["diff_ann_%"])
vals = np.array(vals)
res["E_placebo"] = {"mean": float(vals.mean()), "p95": float(np.quantile(vals, .95)), "p_ge_actual": float((vals >= act).mean())}
# F neighbourhood windows
rx = F["IDADI_rx1"]
for w in (21, 42, 84, 126):
    ew = M.weekly_hold((rx.rolling(w).sum() > 0).astype(float))
    res[f"F_win{w}"] = st(M.apply_overlay(base25, ew[ew.index >= "2012-01-01"]))
# G implementable entry sizing
res["G_entry_mom63"] = st(M.entry_sized("p4q", e63))
# H switches pre-2026 in harness window
oo = o[(o.index >= "2022-01-01") & (o.index < H.HOLDOUT)]
res["H_switches"] = int((oo.diff().abs() > 0).sum()); res["H_frac_out"] = float((oo == 0).mean())
res["H_out_spells"] = [(str(a.date()), str(bb.date())) for a, bb in
                       [(x.index[0], x.index[-1]) for _, x in oo[oo == 0].groupby((oo != 0).cumsum())]]
(OUT / "verify_results.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
print(json.dumps(res, indent=1, default=str))
