"""Adversarial verification of w2_primary_market_concession (pre-2026 only; holdout NOT re-read).

Re-runs A2 with the author's engine (imported, nothing written to their folder) and adds:
  V1  reproduction of A2 / A1 / A6 / A7 vs P4+Q and vs U
  V2  never-printed series marked to a REALISED spread-matched cohort (same peer kind_incent, same spread bin at
      fair_day; equal-weight daily mean of the harness patched returns R of those traded bonds) instead of
      carry - dur x d(peer median) -> credit losses of similar-carry bonds are no longer invisible
  V3  never-printed series marked to the issuer's own traded bonds (mean R) when available, else V2
  V4  dur look-ahead: dur_iss uses dur_fp/T_fp from the FIRST PRINT (post-settlement). Recompute selection with the
      default duration factor for every bond (conc shifts through fair value interpolation and IPCA/PRE s_iss)
  V5  carry-neutral: A2 minus its own spread-matched-cohort book (what is left beyond buying carry)
"""
from __future__ import annotations

import json
import numpy as np
import pandas as pd

from research.nightly import harness as H
from research.nightly.w2_primary_market_concession import portfolio as PF
from rfmonitor.ml.selection import BUCKETS

OUTD = H._ROOT / "research" / "nightly" / "verify_w2_primary_market_concession"
SBINS = np.array([-1e9, 100, 175, 250, 350, 500, 1e9])


def cohort_marks():
    """For each (peer, spread bin, day t): members = eligible bonds with cdi_bps in bin on day t.
    Return dict key->(ND,) daily mean R over members defined on fair_day (fixed cohort per position is built later)."""
    g = pd.read_pickle(H.HIST / "lab_daily.pkl")[["codigo", "day", "cdi_bps", "peer", "eligible", "cnpj8"]]
    C = H._core()
    g["pos"] = C["days"].get_indexer(g["day"])
    g["b"] = C["codes"].get_indexer(g["codigo"])
    g = g[(g["pos"] >= 0) & (g["b"] >= 0)]
    return g


def main():
    C = H._core()
    R = H._Rmat("base")
    ND = len(C["days"])
    ni = PF.load(False)
    V = PF.variants(ni)
    b = H.baseline("P4Q")["daily"]
    u = H.baseline("U")["daily"]
    out = {}

    def st(x, bench=b):
        s = H.stats(x["daily"] if isinstance(x, dict) else x, bench=bench)
        return {k: round(float(s[k]), 3) for k in ["ann_excess_%", "diff_ann_%", "diff_t_nw", "diff_p", "diff_h1_%", "diff_h2_%", "vol_%", "max_dd_%"]}

    sel = ni[V["A2_prim_conc50_scr"]].copy()
    a2 = PF.sleeve(sel)
    out["V1_A2_vsP4Q"] = st(a2)
    out["V1_A2_vsU"] = st(a2, u)
    for k in ["A1_prim_conc0_scr", "A6_prim_conc0_scr_noprint_cash"]:
        s1 = ni[V[k]]
        out["V1_" + k] = st(PF.sleeve(s1, noprint="cash" if "cash" in k else "model"))
    print(json.dumps(out, indent=1))

    # ---------------------------------------------------------------- V2/V3 realised-cohort marks
    g = cohort_marks()
    ge = g[g["eligible"] & g["cdi_bps"].notna()]
    by_pos = {p: x for p, x in ge.groupby("pos")}
    gall = {p: x for p, x in g.groupby("pos")}

    def cohort_members(r):
        x = by_pos.get(int(r.fair_pos))
        if x is None:
            return np.array([], int)
        p = x[x["peer"] == f"{r.kind}_{r.incent}"]
        lo, hi = SBINS[np.searchsorted(SBINS, r.s_iss) - 1], SBINS[np.searchsorted(SBINS, r.s_iss)]
        m = p[(p["cdi_bps"] >= lo) & (p["cdi_bps"] < hi)]
        if len(m) < 5:   # fall back to any peer within +-75 bps
            m = x[(x["cdi_bps"] - r.s_iss).abs() < 75]
        return m["b"].unique()

    def issuer_members(r):
        x = gall.get(int(r.fair_pos))
        if x is None:
            return np.array([], int)
        m = x[(x["cnpj8"] == r.cnpj8) & (x["codigo"] != r.codigo)]
        return m["b"].unique()

    def sleeve_cohort(sel, how="cohort", only_unprinted=True, cost_bps=25.0):
        """Same as PF.sleeve primary mode; the never-printed segment (and, if only_unprinted=False, the whole
        position = pure cohort replica) is marked with the realised mean R of the matched members."""
        NR = H._next_row()
        t_end = H._hpos()
        num = np.zeros(ND); den = np.zeros(ND); cost = np.zeros(ND)
        wdeal = 1.0 / sel.groupby(["cnpj8", "emissao"])["codigo"].transform("size").to_numpy()
        for w, r in zip(wdeal, sel.itertuples()):
            s0 = int(r.settle_pos)
            bb = int(r.b) if r.b == r.b else -1
            k = int(r.fp_pos) if (r.on_grid and r.fp_pos >= 0) else -1
            start = s0
            if start >= t_end:
                continue
            e = start + PF.HOLD
            if e < ND and bb >= 0 and k >= 0 and k < e:
                x = int(NR[e, bb]); e = x if x < ND else e
            e = min(e, ND, t_end)
            if e <= start:
                continue
            printed = k >= 0 and k < e
            if printed and only_unprinted:
                carry = r.s_iss / 1e4 / 252
                rr = np.zeros(e - start)
                rr[: k - start] = carry
                jump = -r.dur_fp * (r.cdi_bps_fp - r.s_iss) / 1e4
                rr[max(k - 1 - start, 0)] += jump
                rr[k - start:] = R[k:e, bb]
            else:
                mem = np.array([], int)
                if how == "issuer":
                    mem = issuer_members(r)
                if len(mem) == 0:
                    mem = cohort_members(r)
                if len(mem) == 0:
                    rr = np.full(e - start, r.s_iss / 1e4 / 252)
                else:
                    # buy-and-hold equal-weight cohort from settlement
                    M = R[start:e][:, mem]
                    V_ = np.vstack([np.ones(len(mem)), np.cumprod(1 + M[:-1], axis=0)])
                    rr = (V_ * M).sum(1) / V_.sum(1)
            v = w * np.r_[1.0, np.cumprod(1 + rr[:-1])]
            num[start:e] += v * rr; den[start:e] += v
            if e < ND:
                cost[e - 1] += v[-1] * (1 + rr[-1]) * cost_bps / 2 / 1e4
        gg = np.where(den > 0, num / np.maximum(den, 1e-12), 0.0)
        cc = np.where(den > 0, cost / np.maximum(den, 1e-12), 0.0)
        i0 = int(C["days"].searchsorted(H.START)) + 1
        return pd.Series((gg - cc)[i0:t_end], index=C["days"][i0:t_end])

    a2_coh = sleeve_cohort(sel, "cohort")
    a2_iss = sleeve_cohort(sel, "issuer")
    a2_rep = sleeve_cohort(sel, "cohort", only_unprinted=False)   # pure spread-matched replica of every position
    out["V2_A2_unprinted_cohort_mark_vsP4Q"] = st(a2_coh)
    out["V2_A2_unprinted_cohort_mark_vsU"] = st(a2_coh, u)
    out["V3_A2_unprinted_issuer_else_cohort_vsP4Q"] = st(a2_iss)
    out["V5_A2_cohort_replica_vsP4Q"] = st(a2_rep)
    out["V5_A2_model_minus_cohort_replica"] = st(a2["daily"], a2_rep)
    out["V5_A2_cohortmark_minus_cohort_replica"] = st(a2_coh, a2_rep)
    # A1 too
    s1 = ni[V["A1_prim_conc0_scr"]]
    out["V2_A1_unprinted_cohort_mark_vsP4Q"] = st(sleeve_cohort(s1, "cohort"))
    print(json.dumps({k: v for k, v in out.items() if k[:2] in ("V2", "V3", "V5")}, indent=1))

    # ---------------------------------------------------------------- V4 duration look-ahead
    DUR_F = {"DI_SPREAD": 0.8, "IPCA": 0.6, "PRE": 0.75}
    ni2 = ni.copy()
    ni2["dur_iss0"] = (ni2["T_iss"] * ni2["kind"].map(DUR_F)).clip(lower=0.25)
    cur = pd.read_pickle(H.HIST / "nightly" / "w2_primary_market_concession" / "curves.pkl").sort_index()
    ten = np.array([126, 252, 504, 756, 1260, 1764, 2520]) / 252
    crow = cur.reindex(ni2["fair_day"], method="ffill")

    def interp(curve):
        vals = crow[[f"{curve}_{int(t * 252)}" for t in ten]].to_numpy()
        return np.array([np.interp(y, ten[np.isfinite(v)], v[np.isfinite(v)]) if np.isfinite(v).sum() >= 2 else np.nan
                         for y, v in zip(ni2["dur_iss0"].to_numpy(), vals)])
    pre, dic = interp("PRE"), interp("DIC")
    c = ni2["contract"].to_numpy() / 100
    ni2["s_iss0"] = np.select([ni2["kind"].eq("DI_SPREAD"), ni2["kind"].eq("PRE"), ni2["kind"].eq("IPCA")],
                              [c * 1e4, ((1 + c) / (1 + pre / 100) - 1) * 1e4, ((1 + c) / (1 + dic / 100) - 1) * 1e4], np.nan)
    fair = []
    for r in ni2.itertuples():
        x = by_pos.get(int(r.fair_pos))
        fv = np.nan
        if x is not None:
            p = x[x["peer"] == f"{r.kind}_{r.incent}"]
            # need dur: pull from lab_daily subset
            fair.append((r.Index, p))
        else:
            fair.append((r.Index, None))
    # dur column needed: reload small subset
    gd = pd.read_pickle(H.HIST / "lab_daily.pkl")[["codigo", "day", "dur", "cdi_bps", "peer", "fresh"]]
    gd = gd[gd["fresh"] & gd["day"].isin(set(ni2["fair_day"]))]
    byd = {d: x for d, x in gd.groupby("day")}
    fv0 = []
    for r in ni2.itertuples():
        x = byd.get(r.fair_day)
        if x is None:
            fv0.append(np.nan); continue
        p = x[x["peer"] == f"{r.kind}_{r.incent}"]
        if len(p) >= 8:
            bk = pd.cut(p["dur"], BUCKETS)
            med = p.groupby(bk, observed=True).agg(d=("dur", "median"), s=("cdi_bps", "median")).dropna()
            fv0.append(float(np.interp(r.dur_iss0, med["d"], med["s"])) if len(med) > 1 else float(med["s"].iloc[0]))
        else:
            fv0.append(np.nan)
    ni2["fair0"] = fv0
    ni2["conc0"] = ni2["s_iss0"] - ni2["fair0"] - np.where(ni2["iss_n"] > 0, ni2["iss_eff"], 0.0)
    m4 = ni2["screen_ok"] & (ni2["dur_iss0"] >= 0.5) & ni2["s_iss0"].notna() & ni2["conc0"].notna() \
        & (ni2["s_iss0"].abs() < 2500) & (ni2["settle"] >= H.START) & ~(ni2["on_grid"] & (ni2["fp_pos"] < ni2["settle_pos"])) \
        & (ni2["conc0"] > 50)
    sel4 = ni2[m4].copy()
    sel4["s_iss"] = sel4["s_iss0"]; sel4["dur_iss"] = sel4["dur_iss0"]
    old = set(sel["codigo"]); new = set(sel4["codigo"])
    out["V4_selection_overlap"] = {"n_old": len(old), "n_new": len(new), "common": len(old & new)}
    out["V4_A2_noLA_dur_model_vsP4Q"] = st(PF.sleeve(sel4))
    out["V4_A2_noLA_dur_cohort_vsP4Q"] = st(sleeve_cohort(sel4, "cohort"))
    # unprinted share and conc of the unprinted
    pr = sel["on_grid"] & (sel["fp_pos"] < sel["settle_pos"] + PF.HOLD)
    out["A2_unprinted_share"] = float(1 - pr.mean())
    out["A2_mean_s_iss_printed_vs_unprinted"] = [float(sel.loc[pr, "s_iss"].mean()), float(sel.loc[~pr, "s_iss"].mean())]
    print(json.dumps({k: v for k, v in out.items() if k[:2] in ("V4", "A2")}, indent=1))
    (OUTD / "verify_results.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
