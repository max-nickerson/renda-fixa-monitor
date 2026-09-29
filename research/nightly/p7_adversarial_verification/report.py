"""Collect the cached stages of verify.py into results.json and charts.

  PYTHONPATH=. PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe research/nightly/p7_adversarial_verification/report.py
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from research.nightly import harness as H

OUT = Path("research/nightly/p7_adversarial_verification")
CACHE = H.HIST / "nightly" / "p7_adversarial_verification"


def load(name):
    p = CACHE / f"stage_{name}.pkl"
    return pickle.load(open(p, "rb")) if p.exists() else None


def clean(o, k=4):
    if isinstance(o, dict):
        return {str(a): clean(b, k) for a, b in o.items() if not isinstance(b, (pd.Series, pd.DataFrame)) or a in ("top_issuers",)}
    if isinstance(o, pd.DataFrame):
        return clean(o.head(15).reset_index().to_dict(orient="records"), k)
    if isinstance(o, (list, tuple)):
        return [clean(b, k) for b in o]
    if isinstance(o, np.ndarray):
        return None if o.size > 50 else [clean(float(v), k) for v in o]
    if isinstance(o, (float, np.floating)):
        return None if not np.isfinite(o) else round(float(o), k)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def cdi_m():
    c = H.cdi_daily()
    m = (1 + c).groupby([c.index.year, c.index.month]).prod() - 1
    m.index = pd.to_datetime([f"{y}-{mm:02d}-01" for y, mm in m.index])
    return m


def tr(m, cm):
    return (1 + cm.reindex(m.index).fillna(0) + m).cumprod() * 100


def robust_t(S):
    """NW t vs iid t: the monthly diff has strong negative lag-2 autocorrelation (driven by a few outlier months),
    which makes the Bartlett NW variance SMALLER than the iid variance. Report the conservative max-variance t."""
    out = {}
    series = {"P7-P4Q": S["repro"]["d_P7"]}
    if S["placebo2"]:
        for f in ("A_listed_random_carry (P7's own placebo)", "F_ctrl_issuer_q5xsector_carry", "E_ctrl_bond_tercile_carry"):
            series["P7-placebo " + f[:1]] = S["placebo2"][f]["m_rel"]
    if S["honest"]:
        d = H.monthly(S["honest"]["daily"]["P7_honest"]) - H.monthly(S["honest"]["daily"]["P4QL90_honest"])
        series["fully honest P7-P4Q(lag90)"] = d[d.index < H.HOLDOUT].dropna()
    if S["flagvote"]:
        d = H.monthly(S["flagvote"]["daily"]["FV2_carry (primary)"]) - H.monthly(S["repro"]["daily_P7"].reindex(S["flagvote"]["daily"]["FV2_carry (primary)"].index).fillna(0))
        series["FV2 - P7"] = d[d.index < H.HOLDOUT].dropna()
    for k, d in series.items():
        x = d.to_numpy()
        n = len(x)
        tiid = x.mean() / x.std(ddof=1) * np.sqrt(n)
        tnw = H.nw_t(x, 6)
        w = np.clip(x, np.percentile(x, 5), np.percentile(x, 95))
        out[k] = {"ann_%": x.mean() * 1200, "t_nw6": tnw, "t_iid": tiid, "t_conservative": min(tnw, tiid),
                  "acf1": float(pd.Series(x).autocorr(1)), "acf2": float(pd.Series(x).autocorr(2)),
                  "winsor5_ann_%": w.mean() * 1200, "winsor5_t_iid": w.mean() / w.std(ddof=1) * np.sqrt(n),
                  "median_month_%": float(np.median(x) * 100), "hit": float((x > 0).mean())}
    return out


def main():
    S = {k: load(k) for k in ("repro", "placebo", "placebo2", "boot", "decomp", "mult", "flagvote", "sens", "honest", "holdout")}
    res = {"slug": "w2_p7_adversarial_verification", "target": "research/nightly/combined (P7)",
           "harness_version": H._VERSION}
    for k, v in S.items():
        if v is not None:
            res[k] = clean(v)
    res["robust_t"] = clean(robust_t(S))
    json.dump(res, open(OUT / "results.json", "w"), indent=1, default=str)
    cm = cdi_m()
    rp, pl, bo, de, hn = S["repro"], S["placebo"], S["boot"], S["decomp"], S["honest"]

    # ---- chart 1: total return + cumulative excess vs universe (pre-2026)
    fig, ax = plt.subplots(2, 1, figsize=(11, 8.5), sharex=True, gridspec_kw={"height_ratios": [3, 2]})
    curves = {"P7": (rp["m_P7"], "#c0392b", 2.2), "P4+Q": (rp["m_P4Q"], "#2c3e50", 1.6),
              "Universe": (rp["m_U"], "#95a5a6", 1.2)}
    p2 = S["placebo2"]
    if p2:
        curves["Matched control placebo (issuer, non-flagged, mean of 40)"] = (p2["F_ctrl_issuer_q5xsector_carry"]["m_mean_book"], "#e67e22", 1.4)
    if de:
        curves["Carry/liq construction only (no screen)"] = (de["m"]["B_construct_only"], "#8e44ad", 1.1)
        curves["PC top-25 carry @R$500m"] = (de["m"]["PC_top25_carry_fund500m"], "#16a085", 1.1)
    for k, (m, c, lw) in curves.items():
        t = tr(m, cm)
        ax[0].plot(t.index, t.values, label=k, color=c, lw=lw)
        ce = ((1 + m).cumprod() / (1 + rp["m_U"].reindex(m.index).fillna(0)).cumprod() - 1) * 100
        if k != "Universe":
            ax[1].plot(ce.index, ce.values, color=c, lw=lw, label=k)
    t = (1 + cm.reindex(rp["m_P7"].index).fillna(0)).cumprod() * 100
    ax[0].plot(t.index, t.values, "k--", lw=1, label="CDI")
    ax[0].set_ylabel("total return CDI x (1+excess), base 100")
    ax[0].legend(fontsize=8, loc="upper left")
    ax[0].set_title("P7 vs P4+Q, matched placebo and construction-only books (pre-2026, 25 bps, monthly)")
    ax[1].axhline(0, color="k", lw=.6)
    ax[1].set_ylabel("cumulative excess vs universe, %")
    ax[1].legend(fontsize=8, loc="upper left")
    for a in ax:
        a.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(OUT / "equity_total_return.png", dpi=120)
    plt.close(fig)

    # ---- chart 2: placebo distributions
    if pl and p2:
        allp = {**{k: v for k, v in p2.items() if not k.startswith("_") and k != "FV2_carry_ctrl"},
                **{k + " [95% overlap]": pl[k] for k in ("B_strat_bond_carry", "C_strat_issuer_carry")}}
        fams = list(allp)
        pl_ = allp
        fig, axs = plt.subplots(2, 4, figsize=(17, 7.5))
        axs = axs.ravel()
        for a in axs[len(fams):]:
            a.axis("off")
        for a, f in zip(axs, fams):
            o = pl_[f]
            a.hist(o["draws_diff_vs_P4Q"], bins=15, color="#bdc3c7", edgecolor="#7f8c8d")
            a.axvline(o["actual_diff_vs_P4Q"], color="#c0392b", lw=2, label=f"actual {o['actual_diff_vs_P4Q']:+.2f}")
            a.axvline(o["mean"], color="#2c3e50", ls="--", label=f"placebo mean {o['mean']:+.2f}")
            a.set_title(f.replace("_", " ")[:48], fontsize=8)
            a.set_xlabel("diff vs P4+Q, %/yr")
            a.legend(fontsize=7)
        fig.suptitle("Matched placebos: random drops of the same size, same construction", fontsize=10)
        fig.tight_layout()
        fig.savefig(OUT / "placebo_distributions.png", dpi=120)
        plt.close(fig)

    # ---- chart 3: cumulative paired differences / decomposition
    if de:
        fig, ax = plt.subplots(figsize=(11, 4.8))
        m = de["m"]
        A = m["A_P4Q"]
        lines = {"P7 - P4+Q": (m["D_P7"] - A, "#c0392b", 2.2),
                 "construction only - P4+Q": (m["B_construct_only"] - A, "#8e44ad", 1.3),
                 "EQH screen on EW - P4+Q": (m["C_screen_EW"] - A, "#2980b9", 1.3),
                 "P7 - PC top-25 carry @R$500m": (m["D_P7"] - m["PC_top25_carry_fund500m"], "#16a085", 1.3)}
        if p2:
            lines["P7 - matched control placebo F"] = (p2["F_ctrl_issuer_q5xsector_carry"]["m_rel"], "#e67e22", 2.0)
            lines["P7 - P7's own placebo A (40% overlap)"] = (p2["A_listed_random_carry (P7's own placebo)"]["m_rel"], "#d35400", 1.0)
        for k, (d, c, lw) in lines.items():
            d = d.dropna()
            ax.plot(d.index, d.cumsum() * 100, label=f"{k} ({d.mean()*1200:+.2f}%/yr)", color=c, lw=lw)
        ax.axhline(0, color="k", lw=.6)
        ax.set_ylabel("cumulative monthly difference, %")
        ax.legend(fontsize=8)
        ax.grid(alpha=.3)
        ax.set_title("Where P7's edge over P4+Q comes from (pre-2026)")
        fig.tight_layout()
        fig.savefig(OUT / "cum_diff_decomposition.png", dpi=120)
        plt.close(fig)

    # ---- chart 4: concentration
    if bo:
        fig, axs = plt.subplots(1, 3, figsize=(15, 4.2))
        t = bo["top_issuers"]
        tt = pd.concat([t.head(10), t.tail(5)])
        lab = [f"{i} {n[:14]}" for i, n in zip(tt.index, tt["name"])]
        axs[0].barh(range(len(tt))[::-1], tt["ann_%_of_diff"].to_numpy(), color=np.where(tt["ann_%_of_diff"] > 0, "#27ae60", "#c0392b"))
        axs[0].set_yticks(range(len(tt))[::-1])
        axs[0].set_yticklabels(lab, fontsize=7)
        axs[0].set_title("issuer contribution to P7-P4+Q (cohort, %/yr)", fontsize=9)
        d = bo["d"]
        axs[1].bar(d.index, d.to_numpy() * 100, width=20, color=np.where(d > 0, "#27ae60", "#c0392b"))
        axs[1].set_title(f"monthly P7-P4+Q, %  (mean {d.mean()*1200:+.2f}%/yr)", fontsize=9)
        axs[2].hist(bo["block"]["draws"], bins=50, color="#bdc3c7")
        axs[2].hist(bo["issuer"]["draws_issuer"], bins=50, color="#e67e22", alpha=.5)
        axs[2].axvline(0, color="k")
        axs[2].set_title(f"bootstrap of mean diff: 6m-block (grey) p(<=0)={bo['block']['p_one_sided_le0']:.3f};\n"
                         f"issuer-cluster (orange) p(<=0)={bo['issuer']['issuer_boot_p_le0']:.3f}", fontsize=8)
        fig.tight_layout()
        fig.savefig(OUT / "concentration_bootstrap.png", dpi=120)
        plt.close(fig)

    # ---- chart 5: sensitivity summary
    rows = []
    if rp:
        rows.append(("P7 (published spec)", rp["P7_vs_P4Q"]["diff_ann_%"], rp["P7_vs_P4Q"]["diff_t_nw"]))
    if p2:
        for f, v in p2.items():
            if f.startswith("_") or f == "FV2_carry_ctrl":
                continue
            rows.append((f"P7 - placebo {f[:30]}", v["honest_alpha"], v["honest_alpha_t_nw"]))
        rows.append(("FV2 - its control placebo", p2["FV2_carry_ctrl"]["honest_alpha"], p2["FV2_carry_ctrl"]["honest_alpha_t_nw"]))
    if S["sens"]:
        for k, v in S["sens"].items():
            if k.startswith("_") or k == "fund_lag":
                continue
            rows.append((f"EQH {k}", v["diff_ann_%"], v["diff_t_nw"]))
        for k, v in S["sens"]["fund_lag"].items():
            rows.append((f"Q {k} (both books)", v["P7L_vs_P4QL"]["diff_ann_%"], v["P7L_vs_P4QL"]["diff_t_nw"]))
    if hn:
        for k in ("liq_costs", "harsh_surv", "liq+harsh", "liq+harsh+holiday"):
            rows.append((f"honest: {k}", hn[k]["diff_ann_%"], hn[k]["diff_t_nw"]))
        rows.append(("fully honest (+lag90 Q)", hn["fully_honest"]["diff_ann_%"], hn["fully_honest"]["diff_t_nw"]))
        for f, v in hn["fully_honest_placebos"].items():
            rows.append((f"fully honest - placebo {f[:14]}", v["honest_alpha"], v["honest_alpha_t"]))
    if S["flagvote"]:
        for k, v in S["flagvote"]["vs_P4Q"].items():
            rows.append((f"flag-vote {k}", v["diff_ann_%"], v["diff_t_nw"]))
    if rows:
        df = pd.DataFrame(rows, columns=["variant", "diff", "t"])
        fig, ax = plt.subplots(figsize=(10, 0.32 * len(df) + 1.2))
        y = np.arange(len(df))[::-1]
        ax.barh(y, df["diff"], color=np.where(df["t"].abs() >= 2, "#c0392b", "#95a5a6"))
        for yy, dv, tv in zip(y, df["diff"], df["t"]):
            ax.text(dv + (0.01 if dv >= 0 else -0.01), yy, f"{dv:+.2f} (t {tv:.1f})", va="center",
                    ha="left" if dv >= 0 else "right", fontsize=7)
        ax.set_yticks(y)
        ax.set_yticklabels(df["variant"], fontsize=7)
        ax.axvline(0, color="k", lw=.6)
        ax.set_xlabel("%/yr vs P4+Q (or vs placebo), pre-2026; red = |t| >= 2")
        ax.set_title("P7 robustness grid")
        ax.grid(alpha=.3, axis="x")
        fig.tight_layout()
        fig.savefig(OUT / "sensitivity_grid.png", dpi=120)
        plt.close(fig)
        df.to_csv(OUT / "summary_grid.csv", index=False)

    # ---- chart 6: honest-engine total return with the holdout shaded (holdout only recomputed, not used)
    ho = S["holdout"]
    if ho:
        fig, ax = plt.subplots(figsize=(11, 4.5))
        for k, s, c in (("P7", ho["daily_P7"], "#c0392b"), ("P4+Q", ho["daily_P4Q"], "#2c3e50"), ("Universe", ho["daily_U"], "#95a5a6")):
            m = H.monthly(s)
            t = tr(m, cm)
            ax.plot(t.index, t.values, label=k, color=c)
        ax.axvspan(H.HOLDOUT, ho["daily_P7"].index.max(), color="#f1c40f", alpha=.15)
        ax.set_title("P7 vs P4+Q vs universe incl. 2026 sealed holdout (shaded; recomputed only, not used for any choice)", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(alpha=.3)
        fig.tight_layout()
        fig.savefig(OUT / "equity_incl_holdout.png", dpi=120)
        plt.close(fig)
    print("report written")


if __name__ == "__main__":
    main()
