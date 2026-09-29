"""w2_expected_loss_carry / run.py - portfolios on expected-loss-adjusted carry (harness v4).

  PYTHONPATH=. PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/Scripts/python.exe research/nightly/w2_expected_loss_carry/run.py            # pre-2026
  ... run.py --holdout     # reads the sealed holdout ONCE for the frozen best (stored in results.json)

Variants (all monthly decisions, 126-bday tranches, 25 bps, harness execution):
  A_{m}_L{60,40}  rank the universe on EL-carry = cdi_bps - PD_m x LGD x 1e4 (12m PD, horizon 1y); take the top 30%;
                  apply the P4 rich / press rules (p4f).                        m in {logit, lgbm}
  B_{m}           P4 minus the universe-top-quintile PD issuers (PD screen replaces the Q quality screen)
  D_{m}           P4+Q minus the top-quintile PD issuers (Q and PD)
  C_{m}           P7 (combined/p7.py make_signal) with the PD top-quintile names removed from its p4q base
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from research.nightly import harness as H
from research.nightly.combined import p7

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "data" / "history" / "nightly" / "w2_expected_loss_carry"
RD = ROOT / "research" / "nightly" / "w2_expected_loss_carry"
T0 = time.time()
P7KW = dict(as_weights=True, issuer_cap=1.0)


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


def rnd(d, k=3):
    if isinstance(d, dict):
        return {a: rnd(b, k) for a, b in d.items()}
    if isinstance(d, (float, np.floating)):
        return None if not np.isfinite(d) else round(float(d), k)
    if isinstance(d, (np.integer,)):
        return int(d)
    if isinstance(d, (np.bool_,)):
        return bool(d)
    return d


def load(holdout=False) -> pd.DataFrame:
    P = p7.load(holdout=holdout)
    O = pd.read_pickle(OUT / "pd_oos.pkl")[["cnpj8", "day", "pd_logit", "pd_lgbm"]]
    P = P.merge(O, on=["cnpj8", "day"], how="left")
    for m in ("logit", "lgbm"):
        c = f"pd_{m}"
        # top-quintile PD within the universe of each date (NaN PD -> not flagged)
        U = P["univ"]
        thr = P[U].groupby("day")[c].quantile(0.8)
        P[f"pdQ_{m}"] = (P[c] >= P["day"].map(thr)).fillna(False) & U
        P[f"p4q_pd_{m}"] = P["p4q"] & ~P[f"pdQ_{m}"]
    return P


def elc_rule(m: str, lgd: float, top=0.30):
    def fn(x):
        s = x["cdi_bps"].to_numpy(float) - x[f"pd_{m}"].fillna(x[f"pd_{m}"].median()).to_numpy(float) * lgd * 1e4
        r = pd.Series(s).rank(ascending=False, pct=True).to_numpy()
        return (r <= top) & x["p4f"].to_numpy().astype(bool)
    return fn


def variants():
    V = {}
    for m in ("logit", "lgbm"):
        V[f"A_{m}_L60"] = (elc_rule(m, 0.6), {})
        V[f"A_{m}_L40"] = (elc_rule(m, 0.4), {})
        V[f"B_{m}_P4-PDq"] = ((lambda mm: lambda x: (x["p4"] & ~x[f"pdQ_{mm}"]).to_numpy())(m), {})
        V[f"D_{m}_P4Q-PDq"] = ((lambda mm: lambda x: x[f"p4q_pd_{mm}"].to_numpy())(m), {})
        V[f"C_{m}_P7+PDq"] = (p7.make_signal(base=f"p4q_pd_{m}"), P7KW)
    return V


# ------------------------------------------------------------------------------------------------ harsh survivorship
def harsh_R():
    """bias_audit 'survivorship harsh': every bond that stops trading (before 2025-09-30, before maturity) with a last
    mark < 0.98 jumps to 40% of par at its last grid row (same construction as bias_audit/run.py)."""
    from research.nightly.bias_audit import build as B
    lm = B.lab_min()
    g = lm[lm["day"] < H.HOLDOUT]
    last = g.groupby("codigo").agg(last_day=("day", "max"))
    last["last_ratio"] = g.sort_values("day").groupby("codigo")["ratio"].last()
    ref = pd.read_pickle(H.HIST / "nightly" / "event_driven" / "bonds_ref.pkl").drop_duplicates("codigo").set_index("codigo")
    last["maturity"] = ref["maturity"].reindex(last.index)
    stopped = last[(last["last_day"] <= pd.Timestamp("2025-09-30"))
                   & (last["maturity"].isna() | (last["maturity"] > last["last_day"] + pd.Timedelta(days=60)))]
    st = stopped.loc[stopped["last_ratio"] < 0.98, "last_ratio"]
    C = H._core()
    CODES = C["codes"]
    G, _ = pd.read_pickle(H.HIST / "sellab_returns.pkl")
    lastpos = G.groupby("b")["pos"].max()
    R2 = H._Rmat("base").copy()
    hp = H._hpos()
    n = 0
    for c, lr in st.items():
        if c not in CODES or not (lr > 0.4):
            continue
        b = CODES.get_loc(c)
        p = int(lastpos.get(b, -1))
        if p < 0 or p >= hp:
            continue
        R2[p, b] = (1 + R2[p, b]) * (0.4 / lr) - 1
        n += 1
    return R2, n


def harsh_book(sig, kw, P, R2):
    from research.nightly.bias_audit import engine as E
    t_end = H._hpos()
    PP = P[(P["day"] >= pd.Timestamp(H.START)) & (P["dpos"] < t_end)]
    T = H._targets(sig, PP, 0.2, kw.get("issuer_cap", H.ISSUER_CAP), kw.get("as_weights", False), H.MIN_NAMES, "univ")
    return E.tranche_book(T, R=R2)


def screen_placebo(P, flag_col, base_col, n=20, kw=None, is_p7=False):
    """Drop the same NUMBER of base names as the PD screen drops, chosen at random (per date, seeded)."""
    out = []
    for s in range(n):
        Q = P.copy()
        rng = np.random.default_rng(1000 + s)
        drop = np.zeros(len(Q), bool)
        for d, idx in Q.groupby("day").groups.items():
            ii = np.asarray(Q.index.get_indexer(idx))
            base = Q[base_col].to_numpy()[ii].astype(bool)
            k = int((base & Q[flag_col].to_numpy()[ii]).sum())
            cand = ii[base]
            if k and len(cand):
                drop[rng.choice(cand, size=min(k, len(cand)), replace=False)] = True
        Q["_rb"] = Q[base_col].to_numpy().astype(bool) & ~drop
        if is_p7:
            r = H.backtest(p7.make_signal(base="_rb"), panel=Q, **P7KW)
        else:
            r = H.backtest(lambda x: x["_rb"].to_numpy(), panel=Q)
        out.append(r["daily"])
    return out


def main():
    RES = {"slug": "w2_expected_loss_carry", "harness_version": H._VERSION}
    P = load(holdout=False)
    b = H.baseline("P4Q")
    R = {"U": H.baseline("U"), "P4": H.baseline("P4"), "P4Q": b,
         "P7": H.backtest(p7.make_signal(), panel=P, **P7KW, name="P7")}
    V = variants()
    for k, (fn, kw) in V.items():
        R[k] = H.backtest(fn, panel=P, name=k, **kw)
        s = H.stats(R[k]["daily"], bench=b["daily"])
        log(k, round(s["diff_ann_%"], 3), "t", round(s["diff_t_nw"], 2), "n", round(R[k]["n_avg"], 1))
    tab = H.compare({k: R[k] for k in V}, bench="P4Q")
    tab.to_csv(RD / "compare_pre2026.csv")
    log("\n" + tab.round(3).to_string())
    RES["n_variants_tried"] = len(V)
    RES["compare_25bps_vs_P4Q"] = rnd(tab.reset_index().to_dict(orient="records"))
    refs = H.compare({k: R[k] for k in ("P4", "P7")}, bench="P4Q")
    RES["references_vs_P4Q"] = rnd(refs.reset_index().to_dict(orient="records"))
    RES["levels"] = {k: rnd(H.stats(R[k]["daily"], bench=b["daily"])) for k in ["U", "P4", "P4Q", "P7"]}
    # C variants also vs P7, A/B vs P4
    RES["C_vs_P7"] = {k: rnd(H.stats(R[k]["daily"], bench=R["P7"]["daily"])) for k in V if k.startswith("C_")}
    RES["AB_vs_P4"] = {k: rnd(H.stats(R[k]["daily"], bench=R["P4"]["daily"])) for k in V if k[0] in "AB"}
    # 50 bps
    b50 = H.baseline("P4Q", cost_bps=50)
    p750 = H.backtest(p7.make_signal(), panel=P, **P7KW, cost_bps=50)
    RES["vs_P4Q_50bps"] = {}
    R50 = {}
    for k, (fn, kw) in V.items():
        R50[k] = H.backtest(fn, panel=P, cost_bps=50, **kw)
        s = H.stats(R50[k]["daily"], bench=b50["daily"])
        RES["vs_P4Q_50bps"][k] = rnd({"diff_ann_%": s["diff_ann_%"], "t": s["diff_t_nw"], "h1": s["diff_h1_%"], "h2": s["diff_h2_%"]})
    RES["C_vs_P7_50bps"] = {k: rnd(H.stats(R50[k]["daily"], bench=p750["daily"])["diff_ann_%"]) for k in V if k.startswith("C_")}
    log("50bps", {k: v["diff_ann_%"] for k, v in RES["vs_P4Q_50bps"].items()})
    # best (pre-2026, by paired t vs P4+Q at 25 bps) - frozen here
    best = tab["t_vs_bench"].astype(float).idxmax()
    RES["best_variant"] = best
    log("best", best)
    # rec40 and harsh survivorship for best + baselines
    fnb, kwb = V[best]
    rb40 = H.backtest(fnb, panel=P, scenario="rec40", **kwb)
    b40 = H.baseline("P4Q", scenario="rec40")
    RES["best_rec40_vs_P4Q"] = rnd(H.stats(rb40["daily"], bench=b40["daily"]))
    R2, nh = harsh_R()
    hb = harsh_book(fnb, kwb, P, R2)
    hq = harsh_book(lambda x: x["p4q"].to_numpy(), {}, P, R2)
    hu = harsh_book(lambda x: x["univ"].to_numpy(), {}, P, R2)
    hp7 = harsh_book(p7.make_signal(), P7KW, P, R2)
    RES["harsh_survivorship"] = {"bonds_hit": nh, "best_vs_P4Q": rnd(H.stats(hb["daily"], bench=hq["daily"])),
                                 "best_vs_U": rnd(H.stats(hb["daily"], bench=hu["daily"])),
                                 "P4Q_level": rnd(H.stats(hq["daily"])), "P7_vs_P4Q": rnd(H.stats(hp7["daily"], bench=hq["daily"]))}
    for k in V:
        if k != best and k.startswith(("B_", "D_", "C_")):
            pass
    log("harsh", RES["harsh_survivorship"]["best_vs_P4Q"]["diff_ann_%"])
    # placebo
    if best.startswith("A_"):
        pl = H.placebo(fnb, n=20)
        RES["placebo"] = rnd({"kind": "random books of the same size (harness placebo)", "best": H.stats(R[best]["daily"])["ann_excess_%"],
                              "mean": float(np.mean(pl["ann_excess_%"])), "p95": float(np.percentile(pl["ann_excess_%"], 95))})
    else:
        m = best.split("_")[1]
        base_col = "p4" if best.startswith("B_") else "p4q"
        pls = screen_placebo(P, f"pdQ_{m}", base_col, n=20, is_p7=best.startswith("C_"))
        ref = R["P7"]["daily"] if best.startswith("C_") else (R["P4"]["daily"] if base_col == "p4" else b["daily"])
        dif = [H.stats(s, bench=ref)["diff_ann_%"] for s in pls]
        real = H.stats(R[best]["daily"], bench=ref)["diff_ann_%"]
        RES["placebo"] = rnd({"kind": "random screen dropping the same number of base names per date",
                              "real_diff_vs_base": real, "placebo_mean": float(np.mean(dif)),
                              "placebo_p95": float(np.percentile(dif, 95)),
                              "p_value": float((np.sum(np.array(dif) >= real) + 1) / (len(dif) + 1))})
    log("placebo", RES["placebo"])
    # PD screen diagnostics: what share of P4 / P4Q the screen removes, and forward returns of flagged names
    U = P[P["univ"] & (P["day"] >= H.START)]
    diag = {}
    for m in ("logit", "lgbm"):
        f = U[f"pdQ_{m}"]
        diag[m] = rnd({"share_P4_flagged": float(f[U["p4"]].mean()), "share_P4Q_flagged": float(f[U["p4q"]].mean()),
                       "overlap_with_worstQ_in_P4": float((f & U["worstQ"].fillna(False))[U["p4"]].sum() / max(f[U["p4"]].sum(), 1)),
                       "fwd126_flagged_in_P4": float(U.loc[U["p4"] & f, "fwd_126"].mean()),
                       "fwd126_unflagged_in_P4": float(U.loc[U["p4"] & ~f, "fwd_126"].mean()),
                       "carry_flagged_in_P4": float(U.loc[U["p4"] & f, "cdi_bps"].mean()),
                       "carry_unflagged_in_P4": float(U.loc[U["p4"] & ~f, "cdi_bps"].mean()),
                       "corr_pd_carry_spearman": float(U[[f"pd_{m}", "cdi_bps"]].corr("spearman").iloc[0, 1]),
                       "ic_elc60_fwd126": H.ic(U.assign(elc=U["cdi_bps"] - U[f"pd_{m}"] * 6000), "elc")["mean"],
                       "ic_carry_fwd126": H.ic(U, "cdi_bps")["mean"],
                       "ic_negpd_fwd126": H.ic(U.assign(npd=-U[f"pd_{m}"]), "npd")["mean"]})
    RES["screen_diagnostics"] = diag
    log("diag", diag)
    # charts
    show = {best: R[best], "P7": R["P7"]}
    for k in ("A_lgbm_L60", "D_lgbm_P4Q-PDq"):
        if k in R and k != best:
            show[k] = R[k]
    H.plot_curves(show, RD / "equity_pre2026.png", title="EL-carry / PD screens vs P4+Q (pre-2026, 25 bps)")
    json.dump(rnd(RES), open(RD / "results.json", "w"), indent=1, default=str)
    log("done")


def holdout():
    RES = json.load(open(RD / "results.json"))
    best = RES["best_variant"]
    fnb, kwb = variants()[best]
    P = load(holdout=True)
    rh = H.backtest(fnb, panel=P, holdout=True, **kwb)
    bh = H.baseline("P4Q", holdout=True)
    uh = H.baseline("U", holdout=True)
    p7h = H.backtest(p7.make_signal(), panel=P, holdout=True, **P7KW)
    RES["holdout"] = {"best": best, "vs_P4Q": rnd(H.stats(rh["daily"], bench=bh["daily"], holdout="only")),
                      "vs_U": rnd(H.stats(rh["daily"], bench=uh["daily"], holdout="only")),
                      "P7_vs_P4Q": rnd(H.stats(p7h["daily"], bench=bh["daily"], holdout="only")),
                      "P4Q_level": rnd(H.stats(bh["daily"], holdout="only"))}
    H.plot_curves({best: rh, "P7": p7h}, RD / "equity_total_return.png",
                  title=f"{best} incl. sealed 2026 holdout (read once)", holdout=True)
    json.dump(rnd(RES), open(RD / "results.json", "w"), indent=1, default=str)
    log("holdout", RES["holdout"]["vs_P4Q"])


if __name__ == "__main__":
    if "--holdout" in sys.argv:
        holdout()
    else:
        main()
