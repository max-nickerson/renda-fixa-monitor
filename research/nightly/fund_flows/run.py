"""Fund flows & holdings (CVM inf_diario + CDA) as timing overlays and cross-sectional filters on top of P4+Q.

Rerun (after download.py and build_data.py):
  PYTHONPATH=. PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/Scripts/python.exe research/nightly/fund_flows/run.py
Add `--holdout` to also print/save the sealed-holdout (>= 2026-01) block (run once, at the end).
"""
import sys, json, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from research.nightly import harness as H
from research.nightly.fund_flows import features as F

OUT = "research/nightly/fund_flows"
HOLDOUT = "--holdout" in sys.argv
t0 = time.time()
RES = {"meta": {"harness_version": getattr(H, "_VERSION", None), "cda_lag_days": F.CDA_LAG_DAYS,
                "inf_lag_bdays": F.INF_LAG, "credit_share": F.CREDIT_SHARE}}


def with_feats(P):
    X = F.signals(P)
    return pd.concat([P.reset_index(drop=True), X.drop(columns=["day", "codigo"]).reset_index(drop=True)], axis=1)


P = with_feats(H.load_panel("M"))
A = F.agg_flows()
dd = H.days()


def on_grid(s):
    s = s.reindex(s.index.union(dd)).ffill().reindex(dd)
    return s


# ------------------------------------------------------------------ descriptive: aggregate flows
u = P[P["univ"]]
RES["coverage"] = {"share_univ_rows_with_fund_holder": float((u["nh"] > 0).mean()),
                   "median_holders": float(u["nh"].median()),
                   "credit_funds_last_pre2026": int(A.loc[:"2025-12-31", "n_credit"].iloc[-1]),
                   "credit_pl_bn_last_pre2026": float(A.loc[:"2025-12-31", "credit_pl"].iloc[-1] / 1e9)}

# time-series predictability (monthly, pre-2026): r_{m+1} on flow at end of m, controlling for r_m
U = H.baseline("U"); B = H.baseline("P4Q")
ts = {}
for nm, s in [("U", U["daily"]), ("P4Q", B["daily"])]:
    m = H.monthly(s)
    m.index = pd.DatetimeIndex(m.index).to_period("M")
    Am = A.loc[:"2025-12-31"].resample("ME").last()
    Am.index = Am.index.to_period("M")
    df = pd.concat([m.rename("r"), Am], axis=1).dropna(subset=["r"])
    df["r_next"] = df["r"].shift(-1)
    df = df.dropna(subset=["r_next", "af21"])
    import statsmodels.api as sm
    row = {}
    for f in ["af5", "af21", "af63", "dcot63"]:
        Xr = sm.add_constant(df[[f, "r"]].astype(float))
        fit = sm.OLS(df["r_next"].astype(float), Xr).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
        row[f] = {"beta": float(fit.params[f]), "t": float(fit.tvalues[f]), "t_r_lag": float(fit.tvalues["r"]),
                  "corr_with_prev_month_ret": float(np.corrcoef(df[f], df["r"])[0, 1]), "n": int(len(df))}
    ts[nm] = row
RES["timeseries_predictability"] = ts

# ------------------------------------------------------------------ ICs (universe, pre-2026)
feats = ["lnh", "dnh3", "dq3", "nb3", "hhi", "cshare", "press21", "press63", "stress63"]
ics = {}
for f in feats:
    ics[f] = {}
    for tg in ["fwd_21", "fwd_63", "fwd_126"]:
        r = H.ic(P, f, tg)
        ics[f][tg] = {"ic": round(r["mean"], 4), "t": round(r["t_nw"], 2)}
# carry-neutral IC: residualise feature on cdi_bps rank within date
def resid_feat(f):
    def g(x):
        ok = x[f].notna() & x["cdi_bps"].notna()
        out = pd.Series(np.nan, index=x.index)
        if ok.sum() > 30:
            a = x.loc[ok, f].rank(pct=True); b = x.loc[ok, "cdi_bps"].rank(pct=True)
            beta = np.polyfit(b, a, 1)
            out[ok] = a - np.polyval(beta, b)
        return out
    return P[P["univ"]].groupby("day", group_keys=False).apply(g)
for f in feats:
    P[f + "_cn"] = np.nan
    P.loc[P["univ"], f + "_cn"] = resid_feat(f)
    r = H.ic(P, f + "_cn", "fwd_126"); r63 = H.ic(P, f + "_cn", "fwd_63")
    ics[f]["carry_neutral_fwd_63"] = {"ic": round(r63["mean"], 4), "t": round(r63["t_nw"], 2)}
    ics[f]["carry_neutral_fwd_126"] = {"ic": round(r["mean"], 4), "t": round(r["t_nw"], 2)}
RES["ic"] = ics
print("ICs done", round(time.time() - t0))


# ------------------------------------------------------------------ variants (pre-registered list)
def q_flag(x, f, low=True, q=0.2):
    """True for the worst quintile of feature f among covered universe rows of the date."""
    v = x[f]
    r = v.rank(pct=True)
    return ((r <= q) if low else (r > 1 - q)).fillna(False).to_numpy()


def ex(f, low=True):
    return lambda x: x["p4q"].to_numpy() & ~q_flag(x, f, low)


def combo(x):
    z = lambda s: (s.rank(pct=True) - 0.5)
    sc = z(x["nb3"]).fillna(0) + z(x["cshare"]).fillna(0) - z(x["stress63"]).fillna(0)
    cov = x[["nb3", "cshare", "stress63"]].notna().any(axis=1)
    r = sc.where(cov).rank(pct=True)
    return x["p4q"].to_numpy() & ~(r <= 0.2).fillna(False).to_numpy()


XS = {
    "X1_ex_stress63": ex("stress63", low=False),   # holders in >10% 63d outflow: fire-sale risk
    "X2_ex_press21": ex("press21", low=True),       # holder-weighted 21d net flow most negative
    "X3_ex_fund_selling": ex("nb3", low=True),      # funds' net selling (CDA buy-sell) over 3m
    "X4_ex_low_credit_share": ex("cshare", low=True),  # holder base least made of credit funds
    "X5_ex_orphans": lambda x: x["p4q"].to_numpy() & (x["nh"] > 0).to_numpy(),   # nobody in a fund holds it
    "X6_ex_combo": combo,
}
ovs = {
    "T1_af21_pos": (on_grid(A["af21"]) > 0).astype(float),
    "T2_af63_pos": (on_grid(A["af63"]) > 0).astype(float),
    "T3_out_if_af21_lt_-2pct": (on_grid(A["af21"]) > -0.02).astype(float),
    "T4_out_if_ida_and_af21_neg": 1.0 - ((H.ida_regime().reindex(dd).fillna(1) < 0.5) & (on_grid(A["af21"]) < 0)).astype(float),
}
RES["overlay_time_in_market"] = {k: float(v.loc["2022-01-01":"2025-12-31"].mean()) for k, v in ovs.items()}

res = {}
for k, f in XS.items():
    res[k] = H.backtest(f, panel=P, name=k)
for k, ov in ovs.items():
    res[k] = H.backtest("p4q", panel=P, overlay=ov, name=k)
res["P4Q_ida_overlay(ref)"] = H.backtest("p4q", panel=P, overlay="ida", name="P4Q_ida")
N_VAR = len(res)
tab = H.compare(res, bench="P4Q")
refs = {"U": H.baseline("U"), "P4": H.baseline("P4"), "P4Q": H.baseline("P4Q")}
tab_ref = H.compare(refs, bench="P4Q")
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print(tab.round(3)); print(tab_ref.round(3))
RES["n_variants"] = N_VAR
RES["table_25bps"] = tab.round(4).to_dict(orient="index")
RES["baselines_25bps"] = tab_ref.round(4).to_dict(orient="index")

# costs 50 bps and rec40
r50 = {}; rrec = {}
for k in res:
    if k.startswith("X"):
        r50[k] = H.backtest(XS[k], panel=P, cost_bps=50)
        rrec[k] = H.backtest(XS[k], panel=P, scenario="rec40")
    elif k.startswith("T"):
        r50[k] = H.backtest("p4q", panel=P, overlay=ovs[k], cost_bps=50)
        rrec[k] = H.backtest("p4q", panel=P, overlay=ovs[k], scenario="rec40")
t50 = H.compare(r50, bench="P4Q", cost_bps=50)
trec = H.compare(rrec, bench=H.baseline("P4Q", scenario="rec40")["daily"],
                 universe=H.baseline("U", scenario="rec40")["daily"])
RES["table_50bps"] = t50[["exCDI_%", "exU_%", "vs_bench_%", "t_vs_bench", "p_holm"]].round(4).to_dict(orient="index")
RES["table_rec40"] = trec[["exCDI_%", "exU_%", "vs_bench_%", "t_vs_bench", "p_holm"]].round(4).to_dict(orient="index")
print(t50.round(3)[["exCDI_%", "vs_bench_%", "t_vs_bench"]]); print(trec.round(3)[["exCDI_%", "vs_bench_%", "t_vs_bench"]])

# cohort-level short horizons for the cross-sectional filters (diagnostic)
coh = {}
for k, f in XS.items():
    c = {}
    for Hh in (21, 63, 126):
        a = H.cohort_excess(f, H=Hh, panel=P); b = H.cohort_excess("p4q", H=Hh, panel=P)
        d = (a - b).dropna()
        c[f"H{Hh}_diff_vs_p4q_ann_%"] = round(float(d.mean() / (Hh / 252) * 100), 3)
        c[f"H{Hh}_t"] = round(float(H.nw_t(d.to_numpy(), max(1, Hh // 21))), 2)
    coh[k] = c
RES["cohort_vs_p4q"] = coh
print(pd.DataFrame(coh).T)

# best variant by paired diff (pre-2026)
best = tab["vs_bench_%"].astype(float).idxmax()
RES["best_variant"] = best
print("best", best, round(time.time() - t0))

# placebo for the best: random exclusion of the same count from P4+Q (XS) / circularly shifted overlay (T)
rng = np.random.default_rng(0)
pl = []
if best.startswith("X"):
    f = XS[best]
    for i in range(20):
        seed = int(rng.integers(1e9))
        def rnd(x, f=f, seed=seed):
            base = x["p4q"].to_numpy(); keep = f(x); ndrop = int(base.sum() - (base & keep).sum())
            r = np.random.default_rng(seed + int(x["dpos"].iloc[0]))
            idx = np.flatnonzero(base)
            m = base.copy()
            if ndrop > 0 and len(idx) > ndrop:
                m[r.choice(idx, ndrop, replace=False)] = False
            return m
        pl.append(H.stats(H.backtest(rnd, panel=P)["daily"], bench=B["daily"])["diff_ann_%"])
elif best.startswith("T"):
    ov = ovs[best]
    for i in range(20):
        sh = int(rng.integers(120, len(ov) - 120))
        pl.append(H.stats(H.backtest("p4q", panel=P, overlay=pd.Series(np.roll(ov.to_numpy(), sh), index=ov.index))["daily"],
                          bench=B["daily"])["diff_ann_%"])
if pl:
    RES["placebo_best"] = {"actual_diff_%": float(tab.loc[best, "vs_bench_%"]), "placebo_mean": float(np.mean(pl)),
                           "placebo_p95": float(np.percentile(pl, 95)),
                           "share_placebo_ge_actual": float(np.mean(np.array(pl) >= float(tab.loc[best, "vs_bench_%"])))}
    print(RES["placebo_best"])

# ------------------------------------------------------------------ charts
def tr(s, idx):
    return H.total_return_curve(s.reindex(idx).fillna(0))

show = {"P4+Q": B["daily"], "P4": refs["P4"]["daily"], "Universe": U["daily"], best: res[best]["daily"]}
second = [k for k in tab["vs_bench_%"].astype(float).sort_values(ascending=False).index if k != best][0]
show[second] = res[second]["daily"]
idx = B["daily"].index
fig, ax = plt.subplots(figsize=(11, 6), dpi=110)
ax.plot(idx, H.total_return_curve(pd.Series(0.0, index=idx)), "k:", label="CDI")
for nm, st in [("IDA-DI", H.index_excess("IDADI")), ("Ibovespa", H.index_excess("IBOV"))]:
    ax.plot(idx, tr(st, idx), "-.", lw=1, label=nm, color="#bcbd22" if nm == "IDA-DI" else "#c7c7c7")
cols = {"Universe": "#7f7f7f", "P4": "#1f77b4", "P4+Q": "#17becf"}
for nm, s in show.items():
    ax.plot(idx, tr(s, idx), "--" if nm in cols else "-", color=cols.get(nm), lw=1.2 if nm in cols else 2, label=nm)
ax.set_yscale("log"); ax.set_title("Total return (CDI x (1+excess)), 2022-01 to 2025-12, 25 bps, pre-holdout")
ax.legend(fontsize=8); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig(f"{OUT}/equity_total_return.png"); plt.close(fig)

fig, ax = plt.subplots(figsize=(11, 6), dpi=110)
uu = U["daily"].reindex(idx).fillna(0)
for nm, s in show.items():
    if nm == "Universe":
        continue
    s = s.reindex(idx).fillna(0)
    ax.plot(idx, ((1 + s - uu).cumprod() - 1) * 100, "--" if nm in cols else "-", color=cols.get(nm), lw=1.2 if nm in cols else 2, label=nm)
ia = H.index_excess("IDADI").reindex(idx).fillna(0)
ax.plot(idx, ((1 + ia - uu).cumprod() - 1) * 100, "-.", color="#bcbd22", lw=1, label="IDA-DI")
ax.axhline(0, color="k", lw=.6); ax.set_ylabel("% cumulative excess vs universe")
ax.set_title("Cumulative excess vs universe (pre-holdout)"); ax.legend(fontsize=8); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig(f"{OUT}/cum_excess.png"); plt.close(fig)

# flows vs spreads chart
med = P.groupby("day")["univ_cdi_med"].first()
fig, ax = plt.subplots(figsize=(11, 5), dpi=110)
a21 = A["af21"].loc["2020-06-01":"2025-12-31"] * 100
ax.fill_between(a21.index, 0, a21, where=a21 < 0, color="#d62728", alpha=.5, label="credit-fund net flow 21d, % PL (outflow)")
ax.fill_between(a21.index, 0, a21, where=a21 >= 0, color="#2ca02c", alpha=.4, label="inflow")
ax.set_ylabel("% of credit-fund PL"); ax2 = ax.twinx()
ax2.plot(med.index, med, "k-", lw=1.5, label="universe median CDI+ spread (bps)")
ax2.set_ylabel("bps"); ax.set_title("Credit-fund flows (CVM inf_diario, PIT-lagged 5d) vs debenture spreads")
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels(); ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left")
fig.tight_layout(); fig.savefig(f"{OUT}/flows_vs_spreads.png"); plt.close(fig)

# ------------------------------------------------------------------ export reusable signal
Xs = P.loc[P["univ"] | P["nh"].notna(), ["day", "codigo", "cnpj8", "nh", "dnh3", "dq3", "nb3", "hhi", "cshare",
                                          "press21", "press63", "stress63"]]
Xs.to_pickle(f"data/history/nightly/fund_flows/bond_signals_M.pkl")
A.to_pickle("data/history/nightly/fund_flows/agg_flows_export.pkl")

# ------------------------------------------------------------------ holdout (once, at the end)
if HOLDOUT:
    PH = with_feats(H.load_panel("M", holdout=True))
    bh = H.baseline("P4Q", holdout=True); uh = H.baseline("U", holdout=True)
    hold = {}
    for k in [best, second]:
        if k.startswith("X"):
            rh = H.backtest(XS[k], panel=PH, holdout=True)
        else:
            rh = H.backtest("p4q", panel=PH, holdout=True, overlay=ovs[k] if k in ovs else "ida")
        s = H.stats(rh["daily"], bench=bh["daily"], holdout="only")
        su = H.stats(rh["daily"], bench=uh["daily"], holdout="only")
        hold[k] = {"exCDI_%": s["ann_excess_%"], "vs_P4Q_%": s["diff_ann_%"], "t": s["diff_t_nw"], "vs_U_%": su["diff_ann_%"]}
    sb = H.stats(bh["daily"], bench=uh["daily"], holdout="only")
    hold["P4Q"] = {"exCDI_%": sb["ann_excess_%"], "vs_U_%": sb["diff_ann_%"]}
    RES["holdout_2026"] = hold
    print("HOLDOUT", hold)

RES["runtime_s"] = round(time.time() - t0)
json.dump(RES, open(f"{OUT}/results.json", "w"), indent=1, default=float)
print("done", RES["runtime_s"])
