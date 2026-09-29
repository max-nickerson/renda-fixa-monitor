"""FINAL: the frozen model (P7, kept: no wave-2 result verifiably improves it) + PM report charts.

  PYTHONPATH=. PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/Scripts/python.exe research/nightly/final/run.py

Decision rule (fixed before running anything here): build P8 = P7 + X only if a wave-2 result adds to P7 itself
with NW t >= 2 pre-2026 AND survived its verification.  Candidates and their pre-2026 increments vs P7:
  unlisted_issuer_health X6 cascade +0.13 (t 0.88), X1 FRE parents +0.07 (t 0.72), X3 sector proxy +0.02 (t 0.13);
  p7_adversarial flag-vote FV2 +0.10 (t 1.29); expected_loss_carry PD screen -0.12/-0.15 (hurts);
  primary_market A2 sleeve: vs P4+Q only, mark-dependent (-0.29 with unprinted bonds at cash; cohort replica
  +0.51, t 0.98 in verification).  None qualifies -> FINAL = P7 unchanged.  Nothing here is tuned.

The 2026 numbers are the frozen P7 rule applied once more to the sealed months (already read once by `combined`;
no choice here depends on them).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from research.nightly import harness as H
from research.nightly.bias_audit import engine as E
from research.nightly.combined import p7
from research.nightly.combined import run as CR

OUT = Path("research/nightly/final")
OUT.mkdir(parents=True, exist_ok=True)
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)


rnd = CR.rnd


def main():
    RES = {"slug": "final", "harness_version": H._VERSION, "final_model": "P7 (unchanged; no verified wave-2 add-on)"}
    P = p7.load(holdout=True)
    f = p7.make_signal()
    kw = dict(holdout=True)
    S, S50 = {}, {}
    r7 = H.backtest(f, panel=P, as_weights=True, issuer_cap=1.0, **kw)
    r7_50 = H.backtest(f, panel=P, as_weights=True, issuer_cap=1.0, cost_bps=50, **kw)
    S["P7"], S50["P7"] = r7["daily"], r7_50["daily"]
    for b in ("P4Q", "P4", "U"):
        S[b] = H.baseline(b, **kw)["daily"]
        S50[b] = H.baseline(b, cost_bps=50, **kw)["daily"]
    log("backtests done")
    i0 = S["U"].index[0]
    idx = S["U"].index
    cdi = H.cdi_daily().reindex(idx).fillna(0)
    REF = {"CDI": pd.Series(0.0, index=idx),
           "IDA-DI": H.index_excess("IDADI").reindex(idx).fillna(0),
           "Ibovespa": H.index_excess("IBOV").reindex(idx).fillna(0)}
    REF["IDA-DI"].iloc[0] = 0.0
    REF["Ibovespa"].iloc[0] = 0.0

    # --- honest scenario (bias_audit liquidity-bucket costs + harsh survivorship) on all books, full period
    cb = CR.cost_vector(P)
    Rh, nh = CR.harsh_R(holdout=True)
    HON = {}
    HON["P7"] = CR.engine_run(f, P, cost_b=cb, R=Rh, holdout=True)["daily"]
    HON["P4Q"] = CR.engine_run(lambda x: x["p4q"].to_numpy(), P, as_w=False, icap=0.10, cost_b=cb, R=Rh, holdout=True)["daily"]
    HON["P4"] = CR.engine_run(lambda x: x["p4"].to_numpy(), P, as_w=False, icap=0.10, cost_b=cb, R=Rh, holdout=True)["daily"]
    HON["U"] = CR.engine_run(lambda x: np.ones(len(x), bool), P, as_w=False, icap=0.10, cost_b=cb, R=Rh, holdout=True)["daily"]
    log("honest done", nh)

    # --- tables
    def row(s, bq, bu, holdout):
        st = H.stats(s, bench=bq, holdout=holdout)
        su = H.stats(s, bench=bu, holdout=holdout)
        return {"exCDI_%": st["ann_excess_%"], "vol_%": st["vol_%"], "sharpe": st["sharpe"], "max_dd_%": st["max_dd_%"],
                "vs_U_%": su["diff_ann_%"], "vs_U_t": su["diff_t_nw"],
                "vs_P4Q_%": st["diff_ann_%"], "vs_P4Q_t": st["diff_t_nw"],
                "h1_vs_P4Q": st.get("diff_h1_%"), "h2_vs_P4Q": st.get("diff_h2_%"), "months": st["n_months"],
                "cum_%": st["cum_%"]}
    for per, ho in (("pre2026", False), ("holdout2026", "only")):
        tab = {}
        for k in ("P7", "P4Q", "P4", "U"):
            tab[k] = row(S[k], S["P4Q"], S["U"], ho)
            tab[k]["at50_exCDI_%"] = H.stats(S50[k], holdout=ho)["ann_excess_%"]
            tab[k]["at50_vs_P4Q_%"] = H.stats(S50[k], bench=S50["P4Q"], holdout=ho)["diff_ann_%"]
            tab[k]["at50_vs_P4Q_t"] = H.stats(S50[k], bench=S50["P4Q"], holdout=ho)["diff_t_nw"]
            hs = H.stats(HON[k], bench=HON["P4Q"], holdout=ho)
            tab[k]["honest_exCDI_%"] = hs["ann_excess_%"]
            tab[k]["honest_vs_P4Q_%"] = hs["diff_ann_%"]
            tab[k]["honest_vs_P4Q_t"] = hs["diff_t_nw"]
            tab[k]["honest_max_dd_%"] = hs["max_dd_%"]
            tab[k]["honest_sharpe"] = hs["sharpe"]
        for k in ("IDA-DI", "Ibovespa"):
            st = H.stats(REF[k], holdout=ho)
            tab[k] = {"exCDI_%": st["ann_excess_%"], "vol_%": st["vol_%"], "sharpe": st["sharpe"], "max_dd_%": st["max_dd_%"]}
        tab["CDI_ann_%"] = float(H.monthly(cdi[(cdi.index < H.HOLDOUT) if ho is False else (cdi.index >= H.HOLDOUT)]).mean() * 1200)
        RES[per] = tab
    log("tables done")

    # per-year total returns
    yr = {}
    allS = {"CDI": REF["CDI"], "Universe": S["U"], "P4": S["P4"], "P4+Q": S["P4Q"], "P7 (final)": S["P7"],
            "P7 honest": HON["P7"], "P4+Q honest": HON["P4Q"], "IDA-DI": REF["IDA-DI"], "Ibovespa": REF["Ibovespa"]}
    for k, s in allS.items():
        s = s.reindex(idx).fillna(0)
        tr = (1 + cdi + s)
        yr[k] = {str(y): float(tr[tr.index.year == y].prod() - 1) * 100 for y in sorted(set(idx.year))}
    ydf = pd.DataFrame(yr).T
    ydf["P7 - P4+Q"] = np.nan
    yd = pd.DataFrame(yr)
    yd["P7 - P4+Q (pp)"] = yd["P7 (final)"] - yd["P4+Q"]
    yd["P7 - Universe (pp)"] = yd["P7 (final)"] - yd["Universe"]
    RES["per_year_total_return_%"] = rnd(yd.T.to_dict())
    yd.to_csv(OUT / "per_year_total_return.csv", float_format="%.2f")

    # --- charts
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    col = {"P7 (final)": ("#c0392b", 2.6, "-"), "P7 honest": ("#c0392b", 1.4, ":"), "P4+Q": ("#2c3e50", 1.6, "-"),
           "P4+Q honest": ("#2c3e50", 1.0, ":"), "P4": ("#7f8c8d", 1.2, "-"), "Universe": ("#95a5a6", 1.2, "--"),
           "CDI": ("#000000", 1.0, "--"), "IDA-DI": ("#2980b9", 1.2, "-."), "Ibovespa": ("#27ae60", 0.9, "-")}
    end = idx.max()

    def shade(ax):
        ax.axvspan(pd.Timestamp(H.HOLDOUT), end, color="#f1c40f", alpha=0.18, lw=0)
        ax.grid(alpha=0.3)

    # (a) total return
    fig, ax = plt.subplots(figsize=(12, 6.5))
    for k in ("Ibovespa", "CDI", "Universe", "IDA-DI", "P4", "P4+Q", "P4+Q honest", "P7 honest", "P7 (final)"):
        s = allS[k].reindex(idx).fillna(0)
        tr = 100 * (1 + cdi + s).cumprod()
        c, lw, ls = col[k]
        ax.plot(idx, tr, color=c, lw=lw, ls=ls, label=f"{k}  ({tr.iloc[-1] - 100:+.1f}%)")
    ax.set_yscale("log")
    shade(ax)
    ax.text(pd.Timestamp(H.HOLDOUT), ax.get_ylim()[1], "  sealed 2026 holdout", va="top", fontsize=9)
    ax.set_ylabel("total return index, base 100 (log)")
    ax.set_title("Total return 2022-01 .. %s: CDI x (1 + rate-hedged excess), net of 25 bps\n"
                 "'honest' = liquidity-bucket costs + harsh survivorship (bias_audit)" % end.strftime("%Y-%m"), fontsize=11)
    ax.legend(loc="upper left", fontsize=8.5, ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "a_total_return.png", dpi=120)
    plt.close(fig)

    # (b) cumulative excess vs universe (+ vs P4+Q)
    fig, axs = plt.subplots(2, 1, figsize=(12, 7.5), sharex=True, gridspec_kw={"height_ratios": [3, 2]})
    u = S["U"].reindex(idx).fillna(0)
    for k in ("IDA-DI", "P4", "P4+Q", "P7 (final)"):
        s = allS[k].reindex(idx).fillna(0)
        ce = ((1 + s).cumprod() / (1 + u).cumprod() - 1) * 100
        c, lw, ls = col[k]
        axs[0].plot(idx, ce, color=c, lw=lw, ls=ls, label=f"{k} vs universe ({ce.iloc[-1]:+.1f}%)")
    hu = HON["U"].reindex(idx).fillna(0)
    for k, s in (("P7 honest", HON["P7"]), ("P4+Q honest", HON["P4Q"])):
        s = s.reindex(idx).fillna(0)
        ce = ((1 + s).cumprod() / (1 + hu).cumprod() - 1) * 100
        c, lw, ls = col[k]
        axs[0].plot(idx, ce, color=c, lw=lw, ls=ls, label=f"{k} vs honest universe ({ce.iloc[-1]:+.1f}%)")
    axs[0].axhline(0, color="k", lw=0.6)
    axs[0].set_ylabel("cumulative excess vs universe, %")
    axs[0].legend(fontsize=8.5, frameon=False, loc="upper left")
    axs[0].set_title("Cumulative excess return vs the debenture universe (top) and P7 vs P4+Q (bottom)", fontsize=11)
    q = S["P4Q"].reindex(idx).fillna(0)
    d7 = ((1 + S["P7"].reindex(idx).fillna(0)).cumprod() / (1 + q).cumprod() - 1) * 100
    d4 = ((1 + S["P4"].reindex(idx).fillna(0)).cumprod() / (1 + q).cumprod() - 1) * 100
    axs[1].plot(idx, d7, color="#c0392b", lw=2.2, label=f"P7 - P4+Q ({d7.iloc[-1]:+.2f}%)")
    axs[1].plot(idx, d4, color="#7f8c8d", lw=1.2, label=f"P4 - P4+Q ({d4.iloc[-1]:+.2f}%)")
    axs[1].axhline(0, color="k", lw=0.6)
    axs[1].set_ylabel("cumulative, %")
    axs[1].legend(fontsize=8.5, frameon=False, loc="upper left")
    for a in axs:
        shade(a)
    fig.tight_layout()
    fig.savefig(OUT / "b_cum_excess.png", dpi=120)
    plt.close(fig)

    # (c) drawdowns of total return (debenture books + IDA-DI) and of excess over CDI
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    for k in ("Universe", "IDA-DI", "P4", "P4+Q", "P7 (final)", "P7 honest"):
        s = allS[k].reindex(idx).fillna(0)
        c, lw, ls = col[k]
        ex = (1 + s).cumprod()
        dd = (ex / ex.cummax() - 1) * 100
        axs[1].plot(idx, dd, color=c, lw=lw, ls=ls, label=f"{k} (min {dd.min():.2f}%)")
        tr = (1 + cdi + s).cumprod()
        dt = (tr / tr.cummax() - 1) * 100
        axs[0].plot(idx, dt, color=c, lw=lw, ls=ls, label=f"{k} (min {dt.min():.2f}%)")
    axs[0].set_title("Drawdown of total return (daily)", fontsize=10)
    axs[1].set_title("Drawdown of excess over CDI (daily; what a CDI-benchmarked PM sees)", fontsize=10)
    for a in axs:
        shade(a)
        a.legend(fontsize=8, frameon=False, loc="lower left")
        a.set_ylabel("%")
    fig.tight_layout()
    fig.savefig(OUT / "c_drawdowns.png", dpi=120)
    plt.close(fig)

    # (e) per-year table image
    show = yd[["CDI", "Universe", "P4", "P4+Q", "P7 (final)", "P7 - P4+Q (pp)", "P7 honest", "IDA-DI", "Ibovespa"]].copy()
    show.index = [f"{i}{' (Jan-' + end.strftime('%b') + ', holdout)' if i == str(end.year) else ''}" for i in show.index]
    fig, ax = plt.subplots(figsize=(14, 2.6))
    ax.axis("off")
    cells = [[f"{v:+.2f}" if "pp" in c else f"{v:.2f}" for c, v in r.items()] for _, r in show.iterrows()]
    t = ax.table(cellText=cells, rowLabels=list(show.index), colLabels=list(show.columns), loc="center", cellLoc="center")
    t.auto_set_font_size(False)
    t.set_fontsize(9)
    t.scale(1, 1.5)
    for j in range(len(show.columns)):
        t[0, j].set_facecolor("#ecf0f1")
        t[0, j].set_text_props(weight="bold")
        if show.columns[j] == "P7 (final)":
            for i in range(1, len(show) + 1):
                t[i, j].set_facecolor("#fdecea")
    ax.set_title("Calendar-year total return, % (CDI x (1+excess), 25 bps; 'P7 honest' = liquidity costs + harsh survivorship)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "e_per_year_table.png", dpi=130)
    plt.close(fig)

    signals_chart(plt)
    json.dump(rnd(RES), open(OUT / "results.json", "w"), indent=1, default=str)
    log("done")


# (d) one-page "what predicts debenture returns": verified findings, hard-coded from the agents' results.json/README
IC_ROWS = [  # (label, IC, t, status)  6-month (126 bday) forward excess, per-date Spearman, NW t
    ("Carry: CDI+ spread level (universe)", 0.25, 11.2, "verified (ml_ranking)"),
    ("Carry inside P4+Q", 0.33, 11.2, "verified (ml_ranking)"),
    ("Cheapness vs peer curve (resid_z)", 0.23, 14.6, "verified (ml_ranking)"),
    ("LightGBM, peer-demeaned target", 0.16, 4.8, "verified; no better than carry"),
    ("Issuer equity 6m return | carry, RV", 0.107, 6.8, "verified; fragile, OOS 2015-20 flips"),
    ("Merton distance-to-default | carry", 0.054, 5.1, "verified (structural_credit)"),
    ("Issuer equity vol 63d | carry, RV", -0.09, np.nan, "factor_zoo"),
    ("1-week spread widening (ds_5) | carry", -0.048, -5.5, "factor_zoo"),
    ("Credit-fund ownership share | carry", 0.01, 0.8, "null (fund_flows)"),
    ("Fund net buying / holder changes", 0.0, np.nan, "null, |IC|<=0.03 (fund_flows)"),
    ("Text embeddings (CVM + news)", 0.0, np.nan, "null (text_nlp)"),
    ("Expected-loss carry (spread - PD x LGD)", 0.10, np.nan, "worse than raw carry 0.25"),
]
BOOK_ROWS = [  # (label, %/yr vs P4+Q, t, status)
    ("P4+Q vs universe (carry + not rich + quality)", 1.12, 2.4, "OOS 2015-20: +1.83 (t 3.7)"),
    ("P7 = P4+Q - worst equity health, carry wtd, liq-capped", 0.50, 3.3, "holdout +0.68 (t 2.2); iid t 1.7"),
    ("  equity-health screen alone (EW)", 0.38, 2.1, "OOS 2015-20 +0.10"),
    ("  carry tilt + liquidity cap alone", 0.19, 1.5, ""),
    ("Structural credit (CreditGrades) avoid", 0.35, 1.65, "= 3-5 blow-ups"),
    ("Factor-zoo composite screen", 0.36, 2.5, "OOS 2015-20: -0.35"),
    ("LightGBM peer-excess tilt", 0.17, 2.2, "all in 2024"),
    ("Text: credit-negative doc avoid", 0.27, 1.3, "Holm p 1"),
    ("Fund flows: drop low credit-fund share", 0.08, 0.9, "~0 with realistic CDA lag"),
    ("Expected-loss (PD) screen, vs P7", -0.12, -1.04, "hurts P7"),
    ("IDA-DI mom63 in/out overlay", 0.02, 0.05, "holdout -1.04"),
    ("Macro ensemble sizing 0-1.5x", -0.97, -2.7, ""),
    ("P4 regime overlay (IDA 21d)", -0.48, np.nan, "costs + lower exposure"),
    ("Primary-market sleeve A2 (model marks)", 1.31, 2.98, "-0.29 if unprinted at cash"),
]


def signals_chart(plt):
    fig, axs = plt.subplots(1, 2, figsize=(16, 8.2), gridspec_kw={"width_ratios": [1, 1.15]})
    for ax, rows, xl, ttl in ((axs[0], IC_ROWS, "rank IC vs 6-month forward excess return", "Signal strength (rank IC; '|' = after controlling for carry/RV)"),
                              (axs[1], BOOK_ROWS, "%/yr vs P4+Q (pre-2026, 25 bps); first row is vs the universe; hatched = depends on marking assumptions", "What each idea adds to the P4+Q book")):
        labels = [r[0] for r in rows][::-1]
        vals = np.array([r[1] for r in rows][::-1])
        ts = np.array([r[2] for r in rows][::-1])
        c = ["#27ae60" if t == t and abs(t) >= 2 and v > 0 else "#c0392b" if (t == t and abs(t) >= 2 and v < 0) or (t != t and v < -0.2) else "#bdc3c7" for v, t in zip(vals, ts)]
        y = np.arange(len(rows))
        bars = ax.barh(y, vals, color=c, edgecolor="k", lw=0.4)
        for b_, lab in zip(bars, labels):
            if "Primary" in lab:
                b_.set_facecolor("#f5cba7"); b_.set_hatch("//")
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8.5)
        ax.axvline(0, color="k", lw=0.7)
        span = max(abs(vals).max(), 0.05)
        for yi, v, t, r in zip(y, vals, ts, [r[3] for r in rows][::-1]):
            ax.text(v + (0.02 * span if v >= 0 else -0.02 * span), yi, (f"{v:+.2f} (t {t:.1f}) {r}" if t == t else f"{v:+.2f} {r}"), va="center",
                    ha="left" if v >= 0 else "right", fontsize=7.3)
        ax.set_xlim(-span * 1.9, span * (2.9 if ax is axs[0] else 3.4))
        ax.set_xlabel(xl, fontsize=9)
        ax.set_title(ttl, fontsize=10)
        ax.grid(axis="x", alpha=0.3)
    fig.suptitle("What predicts Brazilian debenture returns (2022-25, point-in-time, harness v4). Green/red: |t| >= 2; grey: not significant.\n"
                 "t-stats are Newey-West on smooth marks (inflated); after ~440 variants tried, |t| of ~3.8 is needed for a 5% family-wise test.",
                 fontsize=10.5)
    fig.tight_layout()
    fig.savefig(OUT / "d_what_predicts.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
