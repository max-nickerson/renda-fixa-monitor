"""Stage 3: primary-market sleeve vs P4+Q and P7 (harness v4 statistics).

Sleeve engine (daily grid, own implementation because the harness can only enter a bond at a secondary trade):
  * one position per selected new series, entered at PAR (the bookbuilding / contract rate) at settlement;
    weight 1 / (#selected series of the same deal) so a multi-series deal counts once;
  * settlement -> first SND print: marked at par, accruing the issue spread (carry s_iss/252 per bday), and at the
    first print the rate-hedged price move  -dur x (s_fp - s_iss)  is booked (realised at that mark);
  * from the first print: harness patched returns R (same data as every other study);
  * held 126 bdays from settlement (harness gap rule: a position whose end falls in a no-trade gap closes when the
    bond next trades); a series that never prints is marked at par + carry for the whole hold (optimistic; see
    sensitivity 'noprint_cash');
  * sleeve daily excess = value-weighted mean of live positions (buy-and-hold drift); cash (0) when empty;
  * costs: primary entry free (issuer pays the fees), exit at cost/2; secondary-entry variants pay cost/2 on entry.
Decision information (all known at the close of fair_day = grid day before settlement): issue spread (contract),
peer/issuer fair value, P4 press screen (issuer's grid bonds, press_neg_30d), P4+Q worst-quintile quality and P7's
EQH bottom-20% flag of the issuer on the latest monthly harness panel date <= fair_day (<= 40 days old; issuers not
on the panel = uncovered, kept, as in P4+Q / P7).
"""
from __future__ import annotations

import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from research.nightly import harness as H
from research.nightly.combined import p7

OUT = H.HIST / "nightly" / "w2_primary_market_concession"
RES = H._ROOT / "research" / "nightly" / "w2_primary_market_concession"
HOLD = 126


# ------------------------------------------------------------------------------------------------ data
def issuer_flags(holdout: bool) -> pd.DataFrame:
    P = H.load_panel("M", holdout=holdout)
    rows = []
    for d, x in P[P["univ"]].groupby("day"):
        x = x.reset_index(drop=True)
        eq = p7.flag_eqh(x)
        f = pd.DataFrame({"cnpj8": x["cnpj8"], "eqh_bad": eq, "worstQ": x["worstQ"].fillna(False).astype(bool),
                          "cdi_bps": x["cdi_bps"]})
        g = f.groupby("cnpj8").agg(eqh_bad=("eqh_bad", "max"), worstQ=("worstQ", "max")).reset_index()
        g["pday"] = d
        g["q70"] = x["cdi_bps"].quantile(0.7)
        rows.append(g)
    return pd.concat(rows, ignore_index=True)


def load(holdout: bool = False) -> pd.DataFrame:
    ni = pd.read_pickle(OUT / "new_issues.pkl")
    ev = pd.read_pickle(OUT / "events.pkl")[["codigo", "J", "b"]]
    ni = ni.merge(ev, on="codigo", how="left")
    fl = issuer_flags(holdout)
    ni = ni.sort_values("fair_day")
    fl = fl.sort_values("pday")
    ni = pd.merge_asof(ni, fl, left_on="fair_day", right_on="pday", by="cnpj8", direction="backward",
                       tolerance=pd.Timedelta(days=40))
    # universe carry cut (P4: top 30% CDI+ carry) of the latest monthly date <= fair_day (market-wide)
    q = fl.drop_duplicates("pday")[["pday", "q70"]].sort_values("pday").rename(columns={"pday": "qday", "q70": "q70m"})
    ni = pd.merge_asof(ni, q, left_on="fair_day", right_on="qday", direction="backward")
    ni["eqh_bad"] = ni["eqh_bad"].fillna(False).astype(bool)
    ni["worstQ"] = ni["worstQ"].fillna(False).astype(bool)
    ni["press_ok"] = ~(ni["press_neg_30d"].fillna(0) > 0)
    ni["screen_ok"] = ni["press_ok"] & ~ni["eqh_bad"] & ~ni["worstQ"]
    ni["elig"] = (ni["dur_iss"] >= 0.5) & ni["s_iss"].notna() & ni["conc"].notna() & (ni["s_iss"].abs() < 2500) \
        & (ni["settle"] >= H.START) & ~(ni["on_grid"] & (ni["fp_pos"] < ni["settle_pos"]))
    print("dropped (traded before the registry settlement date):",
          int((ni["on_grid"] & (ni["fp_pos"] < ni["settle_pos"]) & (ni["settle"] >= H.START)).sum()))
    if not holdout:
        ni = ni[ni["settle"] < H.HOLDOUT]
    return ni.reset_index(drop=True)


def peer_curve_moves() -> dict:
    """Daily median CDI+ spread of fresh marks per (peer = kind_incent, duration bucket), forward-filled on the grid:
    used to mark never-printed new issues to model (ANBIMA-style): carry - dur x change in the peer-bucket spread."""
    p = OUT / "peer_bucket_spread.pkl"
    if p.exists():
        return pd.read_pickle(p)
    from rfmonitor.ml.selection import BUCKETS
    g = pd.read_pickle(H.HIST / "lab_daily.pkl")[["day", "peer", "dur", "cdi_bps", "fresh", "eligible"]]
    g = g[g["fresh"] & g["cdi_bps"].notna()]
    g["bk"] = pd.cut(g["dur"], BUCKETS, labels=False)
    m = g.groupby(["peer", "bk", "day"])["cdi_bps"].median()
    dd = H.days()
    out = {}
    for (pe, bk), s in m.groupby(level=[0, 1]):
        s = s.droplevel([0, 1])
        if len(s) > 100:
            out[(pe, int(bk))] = s.reindex(dd).ffill().to_numpy()
    pd.to_pickle(out, p)
    return out


# ------------------------------------------------------------------------------------------------ engine
def sleeve(sel: pd.DataFrame, cost_bps: float = 25.0, mode: str = "primary", holdout: bool = False,
           scenario: str = "base", noprint: str = "model", sec_max_wait: int = 21) -> dict:
    """mode: 'primary' (par at settlement) | 'secondary' (buy at the first print if it comes within sec_max_wait
    bdays of settlement; skipped otherwise)."""
    C = H._core()
    R = H._Rmat(scenario)
    NR = H._next_row()
    dd = C["days"]
    ND = len(dd)
    t_end = ND if holdout else H._hpos()
    num = np.zeros(ND)
    den = np.zeros(ND)
    cost = np.zeros(ND)
    wdeal = 1.0 / sel.groupby(["cnpj8", "emissao"])["codigo"].transform("size").to_numpy()
    PC = peer_curve_moves() if noprint == "model" else None
    from rfmonitor.ml.selection import BUCKETS
    npos = 0
    for w, r in zip(wdeal, sel.itertuples()):
        s0 = int(r.settle_pos)
        b = int(r.b) if r.b == r.b else -1
        k = int(r.fp_pos) if (r.on_grid and r.fp_pos >= 0) else -1
        if mode == "secondary":
            if k < 0 or k - s0 > sec_max_wait or b < 0:
                continue
            start = k
        else:
            start = s0
        if start >= t_end:
            continue
        e = start + HOLD
        if e < ND and b >= 0 and k >= 0 and k < e:
            x = int(NR[e, b])
            e = x if x < ND else e
        e = min(e, ND, t_end)
        if e <= start:
            continue
        rr = np.zeros(e - start)
        if mode == "primary":
            carry = r.s_iss / 1e4 / 252
            if k >= 0 and k < e:
                rr[: k - start] = carry
                jump = -r.dur_fp * (r.cdi_bps_fp - r.s_iss) / 1e4
                if k - 1 - start >= 0:
                    rr[k - 1 - start] += jump
                else:
                    rr[0] += jump
                rr[k - start:] = R[k:e, b]
            else:
                if noprint == "carry":
                    rr[:] = carry
                elif noprint == "cash":
                    rr[:] = 0.0
                else:   # model: carry - dur x daily change of the peer-bucket median spread (realised mark-to-model)
                    bk = int(np.searchsorted(BUCKETS, r.dur_iss) - 1)
                    sp = PC.get((f"{r.kind}_{r.incent}", bk))
                    if sp is None:
                        rr[:] = carry
                    else:
                        seg = sp[start - 1:e]
                        ds = np.nan_to_num(np.diff(seg))
                        rr[:] = carry - r.dur_iss * ds / 1e4
        else:
            rr[:] = R[start:e, b]
        v = w * np.r_[1.0, np.cumprod(1 + rr[:-1])]
        num[start:e] += v * rr
        den[start:e] += v
        exit_c = v[-1] * (1 + rr[-1]) * cost_bps / 2 / 1e4
        if e < ND:
            cost[e - 1] += exit_c
        if mode == "secondary":
            cost[start] += w * cost_bps / 2 / 1e4
        npos += 1
    # costs are expressed as a fraction of the sleeve NAV on that day
    with np.errstate(invalid="ignore", divide="ignore"):
        g = np.where(den > 0, num / den, 0.0)
        c = np.where(den > 0, cost / np.maximum(den, 1e-12), 0.0)
    i0 = int(dd.searchsorted(H.START)) + 1
    idx = dd[i0:t_end]
    net = pd.Series((g - c)[i0:t_end], index=idx)
    return {"daily": net, "gross": pd.Series(g[i0:t_end], index=idx), "n_pos": npos,
            "invested": pd.Series((den > 0)[i0:t_end], index=idx), "live": pd.Series(den[i0:t_end], index=idx)}


def blend(base: pd.Series, sl: dict, a: float) -> pd.Series:
    """Same capital: (1-a) in the base book, a in the sleeve while it holds positions (base otherwise)."""
    s = sl["daily"].reindex(base.index).fillna(0)
    inv = sl["invested"].reindex(base.index).fillna(False).to_numpy()
    return base + np.where(inv, a * (s - base), 0.0)


# ------------------------------------------------------------------------------------------------ variants
def variants(ni: pd.DataFrame) -> dict:
    E = ni["elig"]
    S = ni["screen_ok"]
    top30 = ni["s_iss"] >= ni["q70m"]
    return {
        "A0_prim_all": E,                                           # every new issue (the 'random new issue' base)
        "A1_prim_conc0_scr": E & S & (ni["conc"] > 0),
        "A2_prim_conc50_scr": E & S & (ni["conc"] > 50),
        "A3_prim_top30carry_scr": E & S & top30 & (ni["conc"] > -25),   # P4-like on the primary
        "A4_prim_incent_conc0_scr": E & S & (ni["conc"] > 0) & (ni["incent"] == 1),
        "A5_prim_firsttime_conc0_scr": E & S & (ni["conc"] > 0) & ni["first_time"],
        "A6_prim_conc0_scr_noprint_cash": E & S & (ni["conc"] > 0),     # same selection as A1, never-printed = cash
        "A7_prim_conc0_scr_noprint_carry": E & S & (ni["conc"] > 0),    # same, never-printed = par + carry (optimistic)
        "A8_prim_conc0_scr_printed21": E & S & (ni["conc"] > 0) & (ni["bd_to_fp"] <= 21),   # LOOK-AHEAD diagnostic
        "B1_sec_conc0_scr": E & S & (ni["conc"] > 0),                   # same as A1 but bought at the first print
        "B3_sec_top30carry_scr": E & S & top30 & (ni["conc"] > -25),
    }


def run_all(ni, cost_bps=25.0, holdout=False, scenario="base"):
    out = {}
    for nm, m in variants(ni).items():
        sel = ni[m.to_numpy()]
        mode = "secondary" if nm.startswith("B") else "primary"
        out[nm] = sleeve(sel, cost_bps=cost_bps, mode=mode, holdout=holdout, scenario=scenario,
                         noprint="cash" if "noprint_cash" in nm else ("carry" if "noprint_carry" in nm else "model"))
        out[nm]["n_sel"] = int(m.sum())
    return out


def placebo(ni, target: str, n: int = 50, seed: int = 0, cost_bps: float = 25.0) -> dict:
    """Random new issues: per settlement month, draw the same number of eligible new series as `target` selects."""
    m = variants(ni)[target]
    E = ni[ni["elig"]]
    cnt = ni[m].groupby(ni.loc[m, "settle"].dt.to_period("M")).size()
    rng = np.random.default_rng(seed)
    b = H.baseline("P4Q", cost_bps=cost_bps)["daily"]
    res = []
    for i in range(n):
        picks = []
        for per, c in cnt.items():
            pool = E[E["settle"].dt.to_period("M") == per]
            if len(pool):
                picks.append(pool.sample(n=min(c, len(pool)), random_state=int(rng.integers(1e9))))
        sel = pd.concat(picks)
        st = H.stats(sleeve(sel, cost_bps=cost_bps)["daily"], bench=b)
        res.append((st["ann_excess_%"], st["diff_ann_%"]))
    a = np.array(res)
    return {"exCDI_mean": float(a[:, 0].mean()), "exCDI_p95": float(np.percentile(a[:, 0], 95)),
            "vsP4Q_mean": float(a[:, 1].mean()), "vsP4Q_p95": float(np.percentile(a[:, 1], 95)), "n": n}


def tbl(results: dict, bench: pd.Series, uni: pd.Series) -> pd.DataFrame:
    rows = {}
    for nm, r in results.items():
        s = r["daily"] if isinstance(r, dict) else r
        a = H.stats(s, bench=bench)
        au = H.stats(s, bench=uni)
        rows[nm] = {"exCDI_%": a["ann_excess_%"], "exU_%": au["diff_ann_%"], "t_vsU": au["diff_t_nw"],
                    "vsP4Q_%": a["diff_ann_%"], "t_vsP4Q": a["diff_t_nw"], "p": a["diff_p"], "vol_%": a["vol_%"],
                    "sharpe": a["sharpe"], "maxDD_%": a["max_dd_%"], "h1_vsP4Q": a["diff_h1_%"],
                    "h2_vsP4Q": a["diff_h2_%"], "n_sel": r.get("n_sel") if isinstance(r, dict) else np.nan}
    df = pd.DataFrame(rows).T
    return df


def main(stage: str = "pre"):
    res_path = RES / "results.json"
    R = json.loads(res_path.read_text(encoding="utf-8")) if res_path.exists() else {}
    if stage == "pre":
        ni = load(False)
        print("eligible new series 2022-2025:", int(ni["elig"].sum()), "screen pass", float(ni.loc[ni["elig"], "screen_ok"].mean()))
        b25 = H.baseline("P4Q")["daily"]
        u25 = H.baseline("U")["daily"]
        P7w = p7.signals(p7.load(False))
        p7r = H.backtest(P7w, as_weights=True, issuer_cap=1.0)
        res = run_all(ni)
        # blends on the same capital (P4+Q core + sleeve), fill 100% and 30% of the allocation
        best_guess = "A1_prim_conc0_scr"
        blends = {f"P4Q+20%{best_guess}": blend(b25, res[best_guess], 0.20),
                  f"P4Q+20%x30%fill_{best_guess}": blend(b25, res[best_guess], 0.06),
                  "P4Q+20%A0_prim_all": blend(b25, res["A0_prim_all"], 0.20)}
        allr = {**res, **{k: {"daily": v} for k, v in blends.items()}, "P7": p7r}
        T = tbl(allr, b25, u25)
        fam = [k for k in res]            # Holm across the sleeve variants tried (blends are linear in them)
        T.loc[fam, "p_holm"] = H.holm(T.loc[fam, "p"].astype(float).to_numpy())
        # 50 bps and rec40
        b50 = H.baseline("P4Q", cost_bps=50)["daily"]
        u50 = H.baseline("U", cost_bps=50)["daily"]
        res50 = run_all(ni, cost_bps=50)
        T50 = tbl(res50, b50, u50)
        T["vsP4Q_50bps_%"] = T50["vsP4Q_%"]
        T["t_50bps"] = T50["t_vsP4Q"]
        brec = H.baseline("P4Q", scenario="rec40")["daily"]
        rrec = run_all(ni, scenario="rec40")
        Trec = tbl(rrec, brec, H.baseline("U", scenario="rec40")["daily"])
        T["vsP4Q_rec40_%"] = Trec["vsP4Q_%"]
        # vs P7
        p7d = p7r["daily"]
        for k in res:
            st = H.stats(res[k]["daily"], bench=p7d)
            T.loc[k, "vsP7_%"] = st["diff_ann_%"]
            T.loc[k, "t_vsP7"] = st["diff_t_nw"]
        pd.set_option("display.width", 250)
        print(T.round(3).to_string())
        # primary premium: A1 (par entry) minus B1 (first-print entry), same selection
        st = H.stats(res["A1_prim_conc0_scr"]["daily"], bench=res["B1_sec_conc0_scr"]["daily"])
        prem = {"A1_minus_B1_%": st["diff_ann_%"], "t": st["diff_t_nw"], "h1": st["diff_h1_%"], "h2": st["diff_h2_%"]}
        print("primary vs first-print entry", prem)
        pl = placebo(ni, "A1_prim_conc0_scr", n=40)
        pl3 = placebo(ni, "A3_prim_top30carry_scr", n=40, seed=1)
        print("placebo A1", pl, "\nplacebo A3", pl3)
        inv = {k: float(v["invested"].mean()) for k, v in res.items()}
        R.update({"pre2026_table": T.round(4).reset_index().rename(columns={"index": "variant"}).to_dict("records"),
                  "n_variants_tried": len(fam), "primary_premium_A1_vs_B1": prem,
                  "placebo_random_new_issues": {"A1": pl, "A3": pl3}, "invested_share": inv,
                  "n_positions": {k: v["n_pos"] for k, v in res.items()}})
        res_path.write_text(json.dumps(R, indent=1, default=str), encoding="utf-8")
        H.plot_curves({"A1 primary sleeve (conc>0, screens)": res["A1_prim_conc0_scr"],
                       "A3 primary top-30% carry": res["A3_prim_top30carry_scr"],
                       "A0 all new issues": res["A0_prim_all"],
                       "B1 same, bought at first print": res["B1_sec_conc0_scr"], "P7": p7r},
                      RES / "equity_total_return.png", title="Primary-market sleeves vs P4+Q (pre-2026, 25 bps)")
        # cumulative excess vs P4+Q
        fig, ax = plt.subplots(figsize=(10, 4.5))
        for k in ["A0_prim_all", "A1_prim_conc0_scr", "A3_prim_top30carry_scr", "A5_prim_firsttime_conc0_scr",
                  "B1_sec_conc0_scr"]:
            d = res[k]["daily"]
            ax.plot(d.index, ((1 + d - b25.reindex(d.index).fillna(0)).cumprod() - 1) * 100, label=k)
        ax.plot(p7d.index, ((1 + p7d - b25.reindex(p7d.index).fillna(0)).cumprod() - 1) * 100, "--", label="P7")
        ax.axhline(0, color="grey", lw=0.8)
        ax.set_title("Cumulative excess vs P4+Q (%), pre-2026, 25 bps")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(RES / "cum_vs_p4q.png", dpi=110)
    else:   # sealed holdout, run ONCE at the end
        ni = load(True)
        bh = H.baseline("P4Q", holdout=True)["daily"]
        uh = H.baseline("U", holdout=True)["daily"]
        p7h = H.backtest(p7.signals(p7.load(True), holdout=True), as_weights=True, issuer_cap=1.0, holdout=True)
        res = run_all(ni, holdout=True)
        out = {}
        for k, v in {**res, "P7": p7h}.items():
            s = H.stats(v["daily"], bench=bh, holdout="only")
            su = H.stats(v["daily"], bench=uh, holdout="only")
            out[k] = {"exCDI_%": s["ann_excess_%"], "vsP4Q_%": s["diff_ann_%"], "t": s["diff_t_nw"],
                      "vsU_%": su["diff_ann_%"], "n_months": s["n_months"]}
        out["P4Q"] = {"exCDI_%": H.stats(bh, holdout="only")["ann_excess_%"]}
        print(pd.DataFrame(out).T.round(3))
        R["holdout_2026"] = out
        res_path.write_text(json.dumps(R, indent=1, default=str), encoding="utf-8")
        H.plot_curves({"A1 primary sleeve": res["A1_prim_conc0_scr"], "A3 primary top-30% carry": res["A3_prim_top30carry_scr"],
                       "P7": p7h}, RES / "equity_with_holdout.png", holdout=True,
                      title="Primary sleeves incl. sealed 2026 holdout (read once)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "pre")
