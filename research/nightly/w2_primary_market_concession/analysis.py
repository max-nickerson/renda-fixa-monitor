"""Stage 2: the new-issue concession and its post-issue path (event study), pre-2026 only.

For each new series (build_data.new_issues.pkl):
  J        = excess over CDI, RATE-HEDGED, from par at settlement to the first SND print:
             tau * s_iss  -  dur * (s_fp - s_iss)        (spread carry  + spread move x duration)
  post_h   = compounded patched return (harness core R) from the first print over h bdays
  rel_h    = post_h minus the equal-weight universe over the same window
  prim_h   = (1+J) * post over [first print, settle+h]  minus universe over [settle, settle+h]   (primary buyer)
  dS_h     = spread change from the issue spread to the mark at first print + h, minus universe median change
Outputs: analysis_results.json, event-study charts.
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

from research.nightly import harness as H

OUT = H.HIST / "nightly" / "w2_primary_market_concession"
RES = H._ROOT / "research" / "nightly" / "w2_primary_market_concession"
HZ = (21, 63, 126)


def universe_daily() -> pd.Series:
    """Equal-weight daily return of the eligible universe (grid rows with univ at t), patched returns."""
    p = OUT / "univ_ew.pkl"
    if p.exists():
        return pd.read_pickle(p)
    C = H._core()
    g = pd.read_pickle(H.HIST / "lab_daily.pkl")[["codigo", "day", "eligible", "cdi_bps"]]
    g = g[g["eligible"] & g["cdi_bps"].notna()]
    pos = C["days"].get_indexer(g["day"])
    b = C["codes"].get_indexer(g["codigo"])
    M = np.zeros(C["R"].shape, bool)
    M[pos, b] = True
    R = H._Rmat()
    u = pd.Series(np.where(M.sum(1) > 0, (R * M).sum(1) / np.maximum(M.sum(1), 1), 0.0), index=C["days"])
    med = g.groupby("day")["cdi_bps"].median()
    pd.to_pickle(u, p)
    pd.to_pickle(med, OUT / "univ_med.pkl")
    return u


def univ_bh() -> dict:
    """Buy-and-hold universe cohort: for each grid pos p and horizon h, the mean compounded return of the bonds that
    are eligible (fresh mark, 0.9-1.1 of par, dur >= 0.5) at p, bought at the p mark and held h bdays (patched R).
    This is the like-for-like benchmark for a bond bought at a fresh print (a daily-rebalanced EW over fresh rows is
    biased: it earns about -4%/yr, print-level bid/ask bounce)."""
    p = OUT / "univ_bh.pkl"
    if p.exists():
        return pd.read_pickle(p)
    C = H._core()
    g = pd.read_pickle(H.HIST / "lab_daily.pkl")[["codigo", "day", "eligible", "cdi_bps"]]
    g = g[g["eligible"] & g["cdi_bps"].notna()]
    pos = C["days"].get_indexer(g["day"])
    b = C["codes"].get_indexer(g["codigo"])
    LC = H._LC()
    ND = len(C["days"])
    out = {"members": {int(pp): b[pos == pp] for pp in np.unique(pos)}}
    for h in (1,) + HZ + (252,):
        v = np.full(ND, np.nan)
        for pp in np.unique(pos):
            if pp + h > ND:
                continue
            bb = b[pos == pp]
            v[pp] = np.mean(np.expm1(LC[min(pp + h, ND), bb] - LC[pp, bb]))
        out[h] = v
    pd.to_pickle(out, p)
    return out


def event_table(ni: pd.DataFrame) -> pd.DataFrame:
    C = H._core()
    R = H._Rmat()
    days, codes = C["days"], C["codes"]
    universe_daily()
    UB = univ_bh()
    LC = H._LC()
    ND = len(days)
    ni = ni.copy()
    ni["b"] = codes.get_indexer(ni["codigo"])
    MB = UB["members"]
    lp_ = pd.read_pickle(H.HIST / "lab_daily.pkl")[["codigo", "peer"]].drop_duplicates("codigo")
    peer_b = np.full(len(codes), "", dtype=object)
    peer_b[codes.get_indexer(lp_["codigo"])] = lp_["peer"].to_numpy()

    def peer_bh(p, h, peer):
        bb = MB.get(int(p))
        if bb is None or p + h > ND:
            return np.nan
        bb = bb[peer_b[bb] == peer]
        if len(bb) < 5:
            return np.nan
        return float(np.mean(np.expm1(LC[min(p + h, ND), bb] - LC[p, bb])))
    tau = ni["bd_to_fp"] / 252
    ni["J"] = tau * ni["s_iss"] / 1e4 - ni["dur_fp"] * (ni["cdi_bps_fp"] - ni["s_iss"]) / 1e4
    ni["J_raw"] = ni["ratio_fp"] - 1
    ni["d_fp_bps"] = ni["cdi_bps_fp"] - ni["s_iss"]
    ni["resid_fp"] = ni["resid_bps_fp"]
    # spread path per bond (lab_daily rows of new issues only)
    g = pd.read_pickle(H.HIST / "lab_daily.pkl")[["codigo", "day", "cdi_bps", "resid_bps"]]
    g = g[g["codigo"].isin(set(ni["codigo"]))]
    g["pos"] = days.get_indexer(g["day"])
    gg = {k: v.sort_values("pos") for k, v in g.groupby("codigo")}
    umed = pd.read_pickle(OUT / "univ_med.pkl").reindex(days).ffill().to_numpy()
    for h in HZ:
        post, rel, prim, ds, rs, relp, primp = [], [], [], [], [], [], []
        for r in ni.itertuples():
            k0, s0, b = r.fp_pos, r.settle_pos, r.b
            peer = f"{r.kind}_{r.incent}"
            if not r.on_grid or b < 0 or k0 < 0 or k0 + h >= ND:
                post.append(np.nan); rel.append(np.nan); prim.append(np.nan); ds.append(np.nan); rs.append(np.nan)
                relp.append(np.nan); primp.append(np.nan)
                continue
            k0 = int(k0)
            lp = LC[k0 + h, b] - LC[k0, b]
            post.append(np.expm1(lp))
            rel.append(np.expm1(lp) - UB[h][k0])
            relp.append(np.expm1(lp) - peer_bh(k0, h, peer))
            e = int(s0) + h
            if k0 < e and np.isfinite(r.J) and e < ND:
                prim.append((1 + r.J) * np.exp(LC[e, b] - LC[k0, b]) - 1 - UB[h][int(s0)])
                primp.append((1 + r.J) * np.exp(LC[e, b] - LC[k0, b]) - 1 - peer_bh(int(s0), h, peer))
            else:
                prim.append(np.nan); primp.append(np.nan)
            x = gg.get(r.codigo)
            row = x[x["pos"] <= k0 + h].tail(1)
            if len(row) and row["pos"].iloc[0] > k0:
                ds.append(row["cdi_bps"].iloc[0] - r.s_iss - (umed[int(row["pos"].iloc[0])] - umed[int(r.fair_pos)]))
                rs.append(row["resid_bps"].iloc[0])
            else:
                ds.append(np.nan); rs.append(np.nan)
        ni[f"post_{h}"], ni[f"rel_{h}"], ni[f"prim_{h}"], ni[f"dS_{h}"], ni[f"resid_{h}"] = post, rel, prim, ds, rs
        ni[f"relp_{h}"], ni[f"primp_{h}"] = relp, primp
    return ni


CLUSTER = {}


def summ(x: pd.Series) -> dict:
    """mean, median, naive t, and t clustered by settlement month (events in the same month share shocks)."""
    x = pd.Series(x).dropna()
    if len(x) < 5:
        return {"n": int(len(x))}
    out = {"n": int(len(x)), "mean": float(x.mean()), "median": float(x.median()),
           "t": float(x.mean() / (x.std() / np.sqrt(len(x)))), "hit": float((x > 0).mean())}
    if "month" in CLUSTER:
        gr = CLUSTER["month"].reindex(x.index)
        m = sm.OLS(x.to_numpy(float), np.ones(len(x))).fit(cov_type="cluster", cov_kwds={"groups": gr.factorize()[0]})
        out["t_cl_month"] = float(m.tvalues[0])
    return out


def main():
    ni = pd.read_pickle(OUT / "new_issues.pkl")
    ev = event_table(ni)
    ev.to_pickle(OUT / "events.pkl")
    hp = H._hpos()
    # pre-2026 sample: settlement before the holdout AND label windows ending before it (126 bdays after first print)
    pre = ev[(ev["settle"] < H.HOLDOUT) & (ev["settle"] >= "2021-06-01")].copy()
    for h in HZ:
        bad = ~(pre["fp_pos"] + h < hp)
        pre.loc[bad, [f"post_{h}", f"rel_{h}", f"dS_{h}", f"resid_{h}"]] = np.nan
        badp = ~(pre["settle_pos"] + h < hp)
        pre.loc[badp, f"prim_{h}"] = np.nan
    pre.loc[pre["fp_pos"] >= hp, ["J", "J_raw", "d_fp_bps"]] = np.nan
    ok = pre["dur_iss"] >= 0.5
    pre = pre[ok]
    CLUSTER["month"] = pre["settle"].dt.to_period("M").astype(str)
    out = {"n_new_series_pre2026": int(len(pre)), "coverage": {
        "on_grid": float(pre["on_grid"].mean()), "fair_peer": float(pre["fair_peer"].notna().mean()),
        "issuer_curve": float((pre["iss_n"] > 0).mean()), "offer_linked": float(pre["off_event"].notna().mean()),
        "first_time": float(pre["first_time"].mean()),
        "bd_settle_to_first_print_median": float(pre["bd_to_fp"].median()),
        "first_print_within_5bd": float((pre["bd_to_fp"] <= 5).mean()),
        "first_print_within_21bd": float((pre["bd_to_fp"] <= 21).mean()),
        "regime": pre["regime"].value_counts().to_dict(), "kind": pre["kind"].value_counts().to_dict(),
        "incent": int(pre["incent"].sum())}}
    # on-grid rows: grid is the observable part
    og = pre[pre["on_grid"] & pre["conc"].notna()]
    out["concession_bps"] = {"conc_peer": summ(pre["conc_peer"]), "conc_iss": summ(pre["conc_iss"]),
                             "conc": summ(pre["conc"]), "iss_resid_raw": summ(pre["iss_resid_raw"])}
    out["first_print"] = {"J_%": summ(og["J"] * 100), "J_raw_%": summ(og["J_raw"] * 100),
                          "spread_move_bps": summ(og["d_fp_bps"]), "resid_at_fp_bps": summ(og["resid_fp"])}
    out["path"] = {}
    for h in HZ:
        out["path"][h] = {"post_%": summ(og[f"post_{h}"] * 100), "rel_univ_%": summ(og[f"rel_{h}"] * 100),
                          "prim_vs_univ_%": summ(og[f"prim_{h}"] * 100),
                          "rel_peer_%": summ(og[f"relp_{h}"] * 100), "prim_vs_peer_%": summ(og[f"primp_{h}"] * 100), "dS_vs_univ_bps": summ(og[f"dS_{h}"]),
                          "resid_bps": summ(og[f"resid_{h}"])}
    # concession quintiles
    og = og.copy()
    og["cq"] = pd.qcut(og["conc"], 5, labels=False, duplicates="drop")
    qt = og.groupby("cq").agg(n=("conc", "size"), conc=("conc", "mean"), J=("J", "mean"), d_fp=("d_fp_bps", "mean"),
                              rel63=("rel_63", "mean"), rel126=("rel_126", "mean"), prim126=("prim_126", "mean"),
                              dS126=("dS_126", "mean"), s_iss=("s_iss", "mean"),
                              relp126=("relp_126", "mean"), primp126=("primp_126", "mean"))
    for c in ["J", "rel63", "rel126", "prim126", "relp126", "primp126"]:
        qt[c] *= 100
    out["by_conc_quintile"] = qt.round(3).reset_index().to_dict("records")
    # regressions: pass-through of concession into first print and into later relative return
    reg = {}
    for y in ["d_fp_bps", "J", "rel_126", "prim_126", "dS_126", "relp_126", "primp_126"]:
        d = og[[y, "conc", "s_iss", "dur_iss", "incent"]].dropna()
        if len(d) > 30:
            X = sm.add_constant(d[["conc", "s_iss", "dur_iss", "incent"]].astype(float))
            m = sm.OLS(d[y].astype(float), X).fit(cov_type="cluster",
                                                  cov_kwds={"groups": og.loc[d.index, "settle"].dt.to_period("M").astype(str).factorize()[0]})
            reg[y] = {"n": int(len(d)), "b_conc": float(m.params["conc"]), "t_conc": float(m.tvalues["conc"]),
                      "b_s_iss": float(m.params["s_iss"]), "t_s_iss": float(m.tvalues["s_iss"]), "r2": float(m.rsquared)}
    out["regressions_month_clustered"] = reg
    # splits
    og["size_t"] = pd.qcut(og["deal_size"].rank(method="first"), 3, labels=["small", "mid", "large"])
    og["hy"] = np.where(og["s_iss"] > og["univ_med"] + 100, "wide(>univ_med+100)", "tight")
    og["mom"] = np.where(og["idadi_x63"] > 0, "idadi_x63>0", "idadi_x63<=0")
    og["flow"] = np.where(og["af21"] > 0, "af21>0", np.where(og["af21"].notna(), "af21<=0", "na"))
    og["year"] = og["settle"].dt.year.astype(str)
    splits = {}
    for col in ["incent", "regime", "first_time", "size_t", "hy", "guarantee", "rat_neg_365d", "mom", "flow", "kind", "year"]:
        t = og.groupby(col, observed=True).agg(n=("conc", "size"), conc=("conc", "mean"), J=("J", "mean"),
                                               d_fp=("d_fp_bps", "mean"), rel126=("rel_126", "mean"),
                                               prim126=("prim_126", "mean"), relp126=("relp_126", "mean"),
                                               primp126=("primp_126", "mean"))
        t[["J", "rel126", "prim126", "relp126", "primp126"]] *= 100
        splits[col] = t.round(3).reset_index().astype({col: str}).to_dict("records")
    out["splits"] = splits
    (RES / "analysis_results.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ["coverage", "concession_bps", "first_print", "path"]}, indent=1, default=str))
    print(pd.DataFrame(out["by_conc_quintile"]))
    print(json.dumps(reg, indent=1))
    for k, v in splits.items():
        print(k); print(pd.DataFrame(v))

    # ---- charts: mean relative path by concession tercile (first print = day 0, plus primary jump)
    C = H._core()
    MB = univ_bh()["members"]
    LC = H._LC()
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    og["ct"] = pd.qcut(og["conc"], 3, labels=["low conc", "mid conc", "high conc"])
    R = H._Rmat()
    for lab, x in og.groupby("ct", observed=True):
        paths = []
        for r in x.itertuples():
            k0 = int(r.fp_pos)
            if k0 + 126 >= hp or r.b < 0:
                continue
            bb = MB.get(k0)
            if bb is None:
                continue
            ub = np.expm1(LC[k0 + 1:k0 + 127][:, bb] - LC[k0, bb]).mean(axis=1)
            rr = np.expm1(LC[k0 + 1:k0 + 127, r.b] - LC[k0, r.b]) - ub
            paths.append(np.r_[0, rr] + (r.J if np.isfinite(r.J) else 0))
        if paths:
            P = np.nanmean(np.vstack(paths), axis=0) * 100
            ax[0].plot(np.arange(len(P)), P, label=f"{lab} (n={len(paths)}, conc {x['conc'].mean():.0f} bps)")
    ax[0].axhline(0, color="grey", lw=0.8)
    ax[0].set_title("Primary buyer: jump to first print + path vs universe (%)")
    ax[0].set_xlabel("bdays after first print")
    ax[0].legend(fontsize=8)
    d = og.dropna(subset=["conc", "d_fp_bps"])
    ax[1].scatter(d["conc"].clip(-300, 300), d["d_fp_bps"].clip(-300, 300), s=6, alpha=0.4)
    ax[1].set_xlabel("concession at issue (bps, vs peer/issuer curve)")
    ax[1].set_ylabel("first-print spread - issue spread (bps)")
    ax[1].axhline(0, color="grey", lw=0.8); ax[1].axvline(0, color="grey", lw=0.8)
    ax[1].set_title("Does the concession tighten at the first print?")
    fig.tight_layout()
    fig.savefig(RES / "event_study.png", dpi=110)


if __name__ == "__main__":
    main()
