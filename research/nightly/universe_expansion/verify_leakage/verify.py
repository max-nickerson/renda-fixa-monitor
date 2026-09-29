"""Leakage-lens verification of universe_expansion's best mix (80% P4+Q / 20% FIDC-sr P4Q). Pre-2026 only."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from research.nightly import harness as H
from research.nightly.universe_expansion import run as R

D = Path("data/history/nightly/universe_expansion")
OUT = Path("research/nightly/universe_expansion/verify_leakage")
DD, CDI = H.days(), H.cdi_daily()
RAW = pd.read_pickle(D / "fidc_panel.pkl")
RAW = RAW[RAW["FUNDO_EXCLUSIVO"] != "S"]
cdi_tot = (1 + CDI).groupby([DD.year, DD.month]).prod() - 1
cdi_tot.index = pd.to_datetime([f"{y}-{m:02d}-01" for y, m in cdi_tot.index])

def piv_of(F, col):
    return F.pivot_table(index="ref", columns="cls", values=col, aggfunc="first")

Fc = RAW[(RAW["ret"] > -1) & (RAW["ret"] < 0.2) & (RAW["ret"] != 0)].copy()
Fc["ex"] = (1 + Fc["ret"]) / (1 + Fc["ref"].map(cdi_tot)) - 1
PIV_CLEAN = piv_of(Fc, "ex")
Fr = RAW[RAW["ret"] != 0].copy()  # raw target: no knowledge of whether the realised return is a crash / outlier
Fr["ex"] = (1 + Fr["ret"].clip(-1, 5)) / (1 + Fr["ref"].map(cdi_tot)) - 1
PIV_RAW = piv_of(Fr, "ex")
PL = piv_of(Fc, "pl"); INAD = piv_of(Fc, "inad_ratio")
# the "reported at all" matrix (any row incl. zeros): used to know if a class ever reported again
REP = RAW.pivot_table(index="ref", columns="cls", values="ret", aggfunc="first").notna()

def fidc(rule="P4Q", lag_m=2, target="clean", missing="cdi", plaus=None, cost_bps=0.0, end="2025-12-01", diag=None):
    months = pd.date_range("2022-01-01", end, freq="MS")
    tgtpiv = PIV_CLEAN if target == "clean" else PIV_RAW
    mret, prev = {}, set()
    for m in months:
        last_ref = m - pd.DateOffset(months=lag_m)
        hist = PIV_CLEAN.loc[(PIV_CLEAN.index <= last_ref) & (PIV_CLEAN.index > last_ref - pd.DateOffset(months=3))]
        if len(hist) < 3 or m not in PIV_CLEAN.index:
            continue
        carry = hist.mean()
        ok = hist.notna().all() & (PL.reindex([last_ref]).iloc[0] > 50e6).reindex(carry.index).fillna(False) & (hist != 0).all()
        if plaus is not None:  # ex-ante plausibility: a senior quota paying > plaus/month over CDI is a data error
            ok &= (hist.abs() < plaus).all()
        c = carry[ok]
        tgt = tgtpiv.loc[m].reindex(c.index) if m in tgtpiv.index else pd.Series(np.nan, index=c.index)
        cm = float(cdi_tot.get(m, 0.0))
        tgt = tgt.fillna(0.0 if missing == "cdi" else -cm / (1 + cm))
        q = INAD.reindex([last_ref]).iloc[0].reindex(c.index)
        worst = (q >= q.quantile(0.8)).fillna(False)
        top = c >= c.quantile(0.7)
        sel = c.index if rule == "U" else c.index[top & ~worst] if rule == "P4Q" else c.index[top]
        r = float(tgt.reindex(sel).clip(-1, 0.5).mean()) if len(sel) else 0.0
        s = set(sel)
        wn = {k: 1 / len(s) for k in s} if s else {}
        wp = {k: 1 / len(prev) for k in prev} if prev else {}
        turn = sum(abs(wn.get(k, 0) - wp.get(k, 0)) for k in set(wn) | set(wp))  # sum |dw|
        r -= turn * cost_bps / 2 / 1e4
        prev = s
        mret[m] = r
        if diag is not None:
            raw_t = PIV_RAW.loc[m].reindex(sel) if m in PIV_RAW.index else pd.Series(np.nan, index=sel)
            cl_t = PIV_CLEAN.loc[m].reindex(sel) if m in PIV_CLEAN.index else pd.Series(np.nan, index=sel)
            diag.append({"m": m, "n": len(sel), "missing_clean": int(cl_t.isna().sum()),
                         "raw_filtered_out": int((cl_t.isna() & raw_t.notna()).sum()),
                         "raw_filtered_min": float(raw_t[cl_t.isna()].min()) if (cl_t.isna() & raw_t.notna()).any() else np.nan,
                         "never_again": int(sum(1 for k in sel if k in REP.columns and not REP.loc[REP.index > m, k].any())),
                         "max_trailing_ex": float(c.reindex(sel).max()) if len(sel) else np.nan,
                         "n_trailing_gt_2pct": int((c.reindex(sel) > 0.02).sum())})
    ms = pd.Series(mret)
    out = pd.Series(0.0, index=DD)
    for m, r in ms.items():
        idx = (DD.year == m.year) & (DD.month == m.month)
        out[idx] = (1 + r) ** (1 / idx.sum()) - 1
    out = out[(out.index >= H.START) & (out.index < H.HOLDOUT)]
    return out

P4Q = H.baseline("P4Q")["daily"]
P4Q50 = H.baseline("P4Q", cost_bps=50)["daily"]
res = {}
def rep(name, **kw):
    f = fidc(**kw)
    mx = R.mix({"deb": P4Q, "fidc": f}, {"deb": .8, "fidc": .2})
    st = H.stats(mx, bench=P4Q)
    fu = fidc(**{**kw, "rule": "U"})
    sf = H.stats(f, bench=fu)
    res[name] = {"mix_diff_vs_P4Q_%": st["diff_ann_%"], "t": st["diff_t_nw"], "h1": st["diff_h1_%"], "h2": st["diff_h2_%"],
                 "fidc_exCDI_%": H.stats(f)["ann_excess_%"], "fidc_P4Q_vs_fidcU_%": sf["diff_ann_%"], "t_vsU": sf["diff_t_nw"]}
    print(name, res[name], flush=True)

diag = []
fidc(diag=diag)
dg = pd.DataFrame(diag)
res["diag"] = {"avg_n": dg.n.mean(), "avg_missing_target_in_sel": dg.missing_clean.mean(),
               "months_with_target_filtered_out": int((dg.raw_filtered_out > 0).sum()),
               "total_target_rows_filtered_out": int(dg.raw_filtered_out.sum()),
               "min_filtered_target_ret": float(dg.raw_filtered_min.min()),
               "selected_never_report_again": int(dg.never_again.sum()),
               "avg_selected_with_trailing_ex_gt_2pct_month": dg.n_trailing_gt_2pct.mean(),
               "max_trailing_ex_selected": dg.max_trailing_ex.max()}
print(res["diag"], flush=True)
rep("reproduce")
rep("lag3_consistent_with_45d_claim", lag_m=3)
rep("raw_target_no_lookahead_filter", target="raw")
rep("plaus_2pct", plaus=0.02)
rep("plaus_1pct", plaus=0.01)
rep("cost25", cost_bps=25)
rep("lag3_raw_plaus2_cost25", lag_m=3, target="raw", plaus=0.02, cost_bps=25)
rep("lag3_raw_plaus2_cost25_missing0", lag_m=3, target="raw", plaus=0.02, cost_bps=25, missing="zero")
rep("lag2_raw_plaus2_cost25", lag_m=2, target="raw", plaus=0.02, cost_bps=25)
rep("lag3_cost25", lag_m=3, cost_bps=25)
rep("lag3_plaus2", lag_m=3, plaus=0.02)
json.dump(res, open(OUT / "verify_results.json", "w"), indent=1, default=str)
