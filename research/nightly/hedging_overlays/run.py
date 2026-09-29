"""Hedging & overlays for the P4+Q debenture book (nightly study, harness v4).

Sections
  A  rate / inflation hedges (return transformations of IPCA+/Pre bonds; rates.py)
  B  credit-beta proxy hedges (Ibov futures, SMAL11, bank basket, Brazil-CDS proxy via EMBI+) with PIT rolling betas
  C  tail protection: 1m 10%-OTM Ibovespa puts (Black-Scholes approximation, skewed IV)
  D  portfolio-level overlays on the book (vol targeting, drawdown stop, universe-momentum, Ibov stress, cash buffer,
     CVM-distress stress) - executed at the mark (optimistic), with an execution-lag sensitivity
  E  entry gating (realistic: only NEW tranches go to cash when a gate is on)
  F  bond-level stop-loss exits vs plain tranche holding (weekly flags, monthly decisions)
  G  informational: long-short RV within peer groups (debentures are not practically borrowable)
Every candidate is compared with P4+Q (paired, NW t, Holm across all candidates) at 25 and 50 bps, halves,
crisis window Dec-22..Jun-23; the pre-registered selection rule picks the 'best' overlay; the sealed holdout
(>= 2026-01-01) is evaluated ONCE at the very end for that variant only.

Run:  PYTHONPATH=. PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/Scripts/python.exe \
        research/nightly/hedging_overlays/run.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from research.nightly import harness as H
from research.nightly.hedging_overlays import data as D
from research.nightly.hedging_overlays import rates as RT

OUT = Path(__file__).resolve().parent
T0 = time.time()
LOG = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(f"[{time.time() - T0:6.0f}s] {s}", flush=True)
    LOG.append(s)


dd = H.days()
CDI = H.cdi_daily()
HP = H._hpos()


def on_grid(s: pd.Series, limit=5) -> pd.Series:
    return s.reindex(s.index.union(dd)).ffill(limit=limit).reindex(dd)


def ex_daily(level: pd.Series) -> pd.Series:
    """Daily excess over CDI of a price level on the grid (return from close t-1 to close t booked at t).
    NOTE the harness convention: R[t] is the t->t+1 move booked at t. To align, shift the index return back one
    day: the move close t -> close t+1 is booked at grid position t."""
    lv = on_grid(level)
    r = lv.pct_change().shift(-1)
    return (r - CDI).fillna(0.0)


def crisis(s: pd.Series) -> float:
    m = H.monthly(s[(s.index >= "2022-12-01") & (s.index < "2023-07-01")])
    return float(((1 + m).prod() - 1) * 100)


def add_series(base: pd.Series, extra: pd.Series) -> pd.Series:
    return (base + extra.reindex(base.index).fillna(0.0)).rename(base.name)


# ---------------------------------------------------------------------------------------------------------------
# baselines
# ---------------------------------------------------------------------------------------------------------------
BASE = {c: H.baseline("P4Q", cost_bps=c) for c in (25, 50)}
U = {c: H.baseline("U", cost_bps=c) for c in (25, 50)}
P4 = {c: H.baseline("P4", cost_bps=c) for c in (25, 50)}
b25 = BASE[25]["daily"]
log("baselines ok", H.stats(b25))

CANDS: dict[str, dict] = {}        # name -> {25: result/series, 50: result/series, 'section':..}
INFO: dict[str, object] = {}


def cand(name, section, r25, r50, extra=None):
    CANDS[name] = {"section": section, 25: r25, 50: r50, "extra": extra or {}}
    s = r25["daily"] if isinstance(r25, dict) else r25
    st = H.stats(s, bench=b25)
    log(f"{section} {name}: exCDI {st['ann_excess_%']:+.2f} vs P4Q {st['diff_ann_%']:+.2f} (t {st['diff_t_nw']:.2f}) "
        f"vol {st['vol_%']:.2f} maxDD {st['max_dd_%']:.2f} worst {st['worst_month_%']:.2f} crisis {crisis(s):+.2f}")


# ---------------------------------------------------------------------------------------------------------------
# A) rate / inflation hedges
# ---------------------------------------------------------------------------------------------------------------
log("A: building rate-hedge adjustments")
RA = RT.build()
INFO["A_diag"] = RA["diag"]
pre, real = D.pre_curve(), D.real_curve()
lvl = pd.DataFrame({"real5y": real["DIC_1260"], "pre3y": pre["PRE_756"]})
mlvl = lvl.resample("MS").last().diff()          # month change in level (pp); month m's value = end(m)-end(m-1)
ipca_sig = lambda x: (x["p4q"] & x["kind"].eq("IPCA")).to_numpy()
A_tab = {}
for nm in ("exact(harness)", "raw", "vertex", "di1_ipca", "swap"):
    adj = None if nm == "exact(harness)" else RA["mats"][nm]
    with RT.use_R(adj):
        rb = {c: H.backtest("p4q", cost_bps=c, name=nm) for c in (25, 50)}
        ri = H.backtest(ipca_sig, name=nm + "_ipca")
    row = {}
    for lab, s in (("book", rb[25]["daily"]), ("ipca_subbook", ri["daily"])):
        m = H.monthly(s[s.index < H.HOLDOUT])
        j = pd.concat([m.rename("y"), mlvl], axis=1, join="inner").dropna()
        bet = {}
        for c in ("real5y", "pre3y"):
            X = np.c_[np.ones(len(j)), j[c]]
            bet[c] = float(np.linalg.lstsq(X, j["y"], rcond=None)[0][1] * 100)   # % excess per +1pp
        st = H.stats(s)
        row[lab] = {"exCDI_%": st["ann_excess_%"], "vol_%": st["vol_%"], "maxDD_%": st["max_dd_%"],
                    "worst_month_%": st["worst_month_%"], "sharpe": st["sharpe"],
                    "beta_%_per_pp_real5y": bet["real5y"], "beta_%_per_pp_pre3y": bet["pre3y"],
                    "corr_real5y": float(j["y"].corr(j["real5y"])), "crisis_%": crisis(s)}
    A_tab[nm] = row
    if nm != "exact(harness)":
        cand(f"A_{nm}", "A", rb[25], rb[50])
    log("A", nm, json.dumps({k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in row.items()}))
INFO["A_table"] = A_tab
# hedge cost estimate (DI1/DAP futures): trades = (book turnover + 1 roll) x IPCA/Pre share x duration x half-spread
P = H.load_panel("M")
xs = P[P["p4q"] & (P["day"] >= H.START)]
share = float(xs["kind"].isin(["IPCA", "PRE"]).mean())
durip = float(xs.loc[xs["kind"].isin(["IPCA", "PRE"]), "dur"].mean())
turn = BASE[25]["turnover_ann"]
INFO["A_hedge_cost"] = {"rate_hedged_share": share, "mean_duration": durip, "book_turnover": turn,
                        "book_DV01_years": share * durip,
                        "cost_%/yr_DI1_0.5bp_halfspread": (turn + 1) * share * durip * 0.5 / 100,
                        "cost_%/yr_DAP_2bp_halfspread": (turn + 1) * share * durip * 2 / 100}
log("A hedge cost", INFO["A_hedge_cost"])

# ---------------------------------------------------------------------------------------------------------------
# B) credit-beta proxy hedges (short futures / stocks), PIT rolling Dimson beta on weekly data
# ---------------------------------------------------------------------------------------------------------------
ST = D.stocks()
X = {"IBOV": ex_daily(ST["IBOV"]), "SMAL11": ex_daily(ST["SMAL11"])}
banks = [c for c in ("ITUB4", "BBDC4", "BBAS3", "SANB11", "BPAC11") if c in ST]
X["BANKS"] = pd.concat([ex_daily(ST[c]) for c in banks], axis=1).mean(axis=1)
em = D.embi()
emg = on_grid(em, limit=5)
# long-credit-risk return of a 5y Brazil CDS (sold protection), USD quanto ignored: -4.5 x dSpread + carry
cds_long = (-4.5 * emg.diff().shift(-1) / 1e4 + emg / 1e4 / 252).where(emg.index <= em.index.max()).fillna(0.0)
X["CDS_EMBI"] = cds_long
INFO["B_embi_last_date"] = str(em.index.max().date())
COST_B = {"IBOV": (0.10, 0.0005), "SMAL11": (1.0, 0.001), "BANKS": (1.0, 0.001), "CDS_EMBI": (0.0, 0.0010)}  # (carry %/yr, trade cost per unit)


def weekly(s):
    return s.groupby(pd.Grouper(freq="W-FRI")).sum()


def rolling_beta(book: pd.Series, x: pd.Series, win=52, lo=0.0, hi=0.5) -> pd.Series:
    """Dimson (lag 0 + lag 1) beta of book weekly excess on x weekly excess, estimated on the trailing `win` weeks
    ending at week w, applied to the NEXT week's daily returns. Clipped to [lo, hi] (short hedges only)."""
    bw, xw = weekly(book), weekly(x).reindex(weekly(book).index).fillna(0)
    out = {}
    for i in range(len(bw)):
        if i < 26:
            out[bw.index[i]] = 0.0
            continue
        j0 = max(0, i - win + 1)
        y = bw.iloc[j0:i + 1].to_numpy()
        x0 = xw.iloc[j0:i + 1].to_numpy()
        x1 = np.r_[0.0, x0[:-1]]
        if np.std(x0) == 0:
            out[bw.index[i]] = 0.0
            continue
        Xm = np.c_[np.ones(len(y)), x0, x1]
        c = np.linalg.lstsq(Xm, y, rcond=None)[0]
        out[bw.index[i]] = float(np.clip(c[1] + c[2], lo, hi))
    be = pd.Series(out)
    be.index = be.index + pd.Timedelta(days=1)          # known after Friday close, used from the next week
    return be.reindex(be.index.union(book.index)).ffill().reindex(book.index).fillna(0.0)


def hedge(book_res, x, key, beta=None, fixed=None):
    s = book_res["daily"]
    be = pd.Series(fixed, index=s.index) if fixed is not None else rolling_beta(s, x)
    carry, tc = COST_B[key]
    pnl = -be * x.reindex(s.index).fillna(0)
    cost = be.abs() * carry / 100 / 252 + be.diff().abs().fillna(be.abs()) * tc
    return (s + pnl - cost).rename(key), be


B_info = {}
for key in ("IBOV", "SMAL11", "BANKS", "CDS_EMBI"):
    out = {}
    for c in (25, 50):
        out[c], be = hedge(BASE[c], X[key], key)
    m = H.monthly(b25[b25.index < H.HOLDOUT])
    xm = H.monthly(X[key].reindex(b25.index).fillna(0)[b25.index < H.HOLDOUT])
    B_info[key] = {"avg_beta": float(be[be.index < H.HOLDOUT].mean()), "max_beta": float(be.max()),
                   "corr_monthly_book_vs_proxy": float(m.corr(xm)),
                   "corr_in_book_worst_6_months": float(pd.concat([m, xm], axis=1).loc[m.nsmallest(6).index].corr().iloc[0, 1])}
    cand(f"B_{key}_beta", "B", out[25], out[50])
out = {c: hedge(BASE[c], X["IBOV"], "IBOV", fixed=0.10)[0] for c in (25, 50)}
cand("B_IBOV_fixed10", "B", out[25], out[50])
INFO["B_info"] = B_info
log("B info", B_info)

# ---------------------------------------------------------------------------------------------------------------
# C) tail protection: monthly 1m 90%-strike Ibovespa puts, IV = max(realised 63d, 15%) x 1.25 (skew/vol premium)
# ---------------------------------------------------------------------------------------------------------------
ib = on_grid(ST["IBOV"])
rv = np.log(ib).diff().rolling(63).std() * np.sqrt(252)
cdi_ann = (1 + CDI).pow(252) - 1


def put_overlay(notional=0.25, k=0.90, ivmult=1.25):
    s = pd.Series(0.0, index=dd)
    months = pd.Series(np.arange(len(dd)), index=dd).groupby([dd.year, dd.month]).first().to_numpy()
    rec = []
    for a, b in zip(months[:-1], months[1:]):
        S0, S1 = ib.iloc[a], ib.iloc[b]
        if not np.isfinite(S0) or not np.isfinite(S1):
            continue
        sig = max(float(rv.iloc[a]) if np.isfinite(rv.iloc[a]) else 0.2, 0.15) * ivmult
        T = (b - a) / 252
        r = float(np.log(1 + cdi_ann.iloc[a]))
        K = k * S0
        d1 = (np.log(S0 / K) + (r + sig ** 2 / 2) * T) / (sig * np.sqrt(T))
        d2 = d1 - sig * np.sqrt(T)
        prem = (K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)) / S0
        pay = max(K - S1, 0) / S0
        cdi_m = float((1 + CDI.iloc[a:b]).prod() - 1)
        pnl = notional * (pay - prem * (1 + cdi_m) + 0.0005 * 0)  # premium financed at CDI
        s.iloc[b - 1] += pnl                                        # booked at expiry (R[t] convention)
        rec.append((dd[a], prem, pay))
    return s, pd.DataFrame(rec, columns=["start", "prem", "pay"])


C_info = {}
for nm, N in (("C_put25", 0.25), ("C_put100", 1.0)):
    ov, rec = put_overlay(N)
    rr = {c: add_series(BASE[c]["daily"], ov) for c in (25, 50)}
    r0 = rec[(rec["start"] >= H.START) & (rec["start"] < H.HOLDOUT)]
    C_info[nm] = {"avg_prem_%_of_notional_per_month": float(r0["prem"].mean() * 100),
                  "months_paid": int((r0["pay"] > 0).sum()), "net_%/yr_of_book": float(
                      (N * (r0["pay"] - r0["prem"])).sum() / len(r0) * 12 * 100)}
    cand(nm, "C", rr[25], rr[50])
INFO["C_info"] = C_info
log("C info", C_info)

# ---------------------------------------------------------------------------------------------------------------
# D) portfolio-level overlays (harness overlay: value at close t traded at the t+1 mark)
# ---------------------------------------------------------------------------------------------------------------
bk = b25.reindex(dd).fillna(0.0)
u25 = U[25]["daily"].reindex(dd).fillna(0.0)


def ov_voltarget(floor=0.25):
    sig = bk.rolling(63, min_periods=42).std() * np.sqrt(252)
    tgt = sig.expanding(min_periods=126).median()
    w = (tgt / sig).clip(upper=1.0, lower=floor)
    return w.fillna(1.0)


def ov_ddstop(th=-0.005, low=0.5):
    nav = (1 + bk).cumprod()
    ddn = nav / nav.cummax() - 1
    m21 = bk.rolling(21).sum()
    w, state = [], 1.0
    for d_, m_ in zip(ddn.to_numpy(), m21.fillna(0).to_numpy()):
        if state == 1.0 and d_ < th:
            state = low
        elif state < 1.0 and (m_ > 0 or d_ > th / 2):
            state = 1.0
        w.append(state)
    return pd.Series(w, index=dd)


def ov_umom(low=0.5):
    return pd.Series(np.where(u25.rolling(21).sum() < 0, low, 1.0), index=dd)


def ov_ibovstress(low=0.5):
    x = X["IBOV"].shift(1).rolling(63).sum()     # shift(1): ex_daily books t->t+1 at t; use closes <= t only
    return pd.Series(np.where(x < -0.10, low, 1.0), index=dd)


PW = H.load_panel("W")
mdz = PW.drop_duplicates("day").set_index("day")["mkt_distress_z"].sort_index()


def ov_distress(low=0.5, th=2.0):
    z = on_grid(mdz, limit=10)
    return pd.Series(np.where(z > th, low, 1.0), index=dd)


# book/universe-derived signals: bk[t] is the t->t+1 move (known at close t+1) -> shift(1) to be point-in-time
L1 = lambda s: s.shift(1).fillna(1.0)
OVS = {"D_voltarget": L1(ov_voltarget()), "D_ddstop": L1(ov_ddstop()), "D_umom": L1(ov_umom()), "D_ibovstress": ov_ibovstress(),
       "D_cash10": pd.Series(0.9, index=dd), "D_distress": ov_distress()}
D_info = {}
for nm, ov in OVS.items():
    rr = {c: H.backtest("p4q", cost_bps=c, overlay=ov, name=nm) for c in (25, 50)}
    D_info[nm] = {"avg_exposure_mult": float(ov[(dd >= H.START) & (dd < H.HOLDOUT)].mean()),
                  "switches": int((ov[(dd >= H.START) & (dd < H.HOLDOUT)].diff().abs() > 1e-9).sum())}
    cand(nm, "D", rr[25], rr[50])
INFO["D_info"] = D_info
log("D info", D_info)

# ---------------------------------------------------------------------------------------------------------------
# E) entry gating: the decision's new tranche stays in cash while the gate is on (no forced selling)
# ---------------------------------------------------------------------------------------------------------------
nav = (1 + bk).cumprod()
dd_book = (nav / nav.cummax() - 1).shift(1).fillna(0.0)   # PIT: known at close t
ibx = X["IBOV"].shift(1).rolling(63).sum()
Pm = H.load_panel("M")
med = Pm.drop_duplicates("day").set_index("day")["univ_cdi_med"].sort_index()
dmed = med.diff()


def gate(fn):
    def sig(x):
        d_ = x["day"].iloc[0]
        return np.zeros(len(x), bool) if fn(d_) else x["p4q"].to_numpy()
    return sig


GATES = {"E_gate_dd": lambda d_: dd_book.get(d_, 0) < -0.005,
         "E_gate_ibov": lambda d_: ibx.get(d_, 0) < -0.10,
         "E_gate_widen": lambda d_: dmed.get(d_, 0) > 15,
         "E_gate_tighten": lambda d_: dmed.get(d_, 0) < -15}
E_info = {}
for nm, fn in GATES.items():
    rr = {c: H.backtest(gate(fn), cost_bps=c, name=nm) for c in (25, 50)}
    decs = [d_ for d_ in med.index if H.START <= d_ < H.HOLDOUT]
    E_info[nm] = {"gated_decisions": int(sum(bool(fn(d_)) for d_ in decs)), "of": len(decs)}
    cand(nm, "E", rr[25], rr[50])
INFO["E_info"] = E_info
log("E info", E_info)

# ---------------------------------------------------------------------------------------------------------------
# F) bond-level stops (weekly flags, monthly decisions) vs same-engine P4+Q
# ---------------------------------------------------------------------------------------------------------------
FW = {c: H.backtest("p4q", freq="W", rebalance="M", cost_bps=c, name="P4Q_Wm") for c in (25, 50)}
INFO["F_engine_P4Q_Wm"] = H.stats(FW[25]["daily"], bench=b25)
F_ex = {"F_stop_4w3pct": lambda x: (x["d_ratio_4w"] < -0.03).to_numpy(),
        "F_stop_below95": lambda x: (x["ratio"] < 0.95).to_numpy(),
        "F_exit_rich": "rich"}
F_res = {}
for nm, ex in F_ex.items():
    rr = {c: H.backtest("p4q", freq="W", rebalance="M", exit_signal=ex, cost_bps=c, name=nm) for c in (25, 50)}
    F_res[nm] = H.stats(rr[25]["daily"], bench=FW[25]["daily"])
    cand(nm, "F", rr[25], rr[50], extra={"vs_same_engine": F_res[nm]})
INFO["F_vs_same_engine"] = F_res
log("F vs same engine", {k: (v["diff_ann_%"], v["diff_t_nw"], v["max_dd_%"]) for k, v in F_res.items()})

# ---------------------------------------------------------------------------------------------------------------
# G) informational long-short within peer groups
# ---------------------------------------------------------------------------------------------------------------
def short_leg(x):
    peers = set(x.loc[x["p4q"], "peer"].dropna())
    rk = x.groupby("peer")["cdi_bps"].rank(pct=True)
    return (x["peer"].isin(peers) & (rk <= 0.3) & ~x["p4q"]).to_numpy()


G_info = {}
sl = H.backtest(short_leg, name="short_leg")
for nm, shrt, borrow in (("G_LS_peer_rich_2pct", sl, 2.0), ("G_LS_peer_rich_5pct", sl, 5.0),
                         ("G_LS_vs_universe_2pct", U[25], 2.0)):
    ls = (b25 - shrt["daily"].reindex(b25.index).fillna(0)
          - shrt["exposure"].reindex(b25.index).fillna(0) * borrow / 100 / 252)
    st = H.stats(ls)
    mu = H.monthly(u25.reindex(ls.index)[ls.index < H.HOLDOUT])
    ml = H.monthly(ls[ls.index < H.HOLDOUT])
    G_info[nm] = {**{k: st[k] for k in ("ann_excess_%", "vol_%", "sharpe", "max_dd_%", "t_nw", "h1_2022_23_%",
                                         "h2_2024_25_%")}, "corr_with_universe": float(ml.corr(mu)),
                  "crisis_%": crisis(ls)}
INFO["G_info"] = G_info
INFO["G_short_leg_exCDI"] = H.stats(sl["daily"])["ann_excess_%"]
log("G", G_info)

# ---------------------------------------------------------------------------------------------------------------
# comparison tables (Holm across ALL candidates) + selection rule
# ---------------------------------------------------------------------------------------------------------------
tabs = {}
for c in (25, 50):
    res = {nm: v[c] for nm, v in CANDS.items()}
    res_full = {"P4Q": BASE[c], "P4": P4[c], "U": U[c], **res}
    tab = H.compare(res, bench=BASE[c]["daily"], universe=U[c]["daily"], cost_bps=c)
    ref = H.compare({"P4Q": BASE[c], "P4": P4[c], "U": U[c]}, bench=BASE[c]["daily"], universe=U[c]["daily"])
    ex = {}
    for nm, r in res_full.items():
        s = r["daily"] if isinstance(r, dict) else r
        st = H.stats(s)
        ex[nm] = {"worst_month_%": st["worst_month_%"], "skew": st["skew"], "crisis_%": crisis(s)}
    tab = pd.concat([ref.drop(columns="p_holm"), tab]).join(pd.DataFrame(ex).T)
    tab.insert(0, "section", [("ref" if k in ("P4Q", "P4", "U") else CANDS[k]["section"]) for k in tab.index])
    tabs[c] = tab
pd.set_option("display.width", 250)
log("\n" + tabs[25].round(3).to_string())
log("\n" + tabs[50].round(3).to_string())

# pre-registered rule: among candidates with diff vs P4Q >= -0.15 %/yr (25 bps), pick the largest maxDD improvement
# (tie-break: worst month). Informational G excluded.
t = tabs[25].drop(index=["P4Q", "P4", "U"])
base_dd = tabs[25].loc["P4Q", "maxDD_%"]
ok = t[t["vs_bench_%"] >= -0.15].copy()
ok["dd_gain"] = ok["maxDD_%"] - base_dd
best = ok.sort_values(["dd_gain", "worst_month_%"], ascending=False).index[0] if len(ok) else t["vs_bench_%"].idxmax()
log("BEST (pre-registered rule):", best)

# robustness of the best: rec40, execution lag (for D), placebo
rob = {}
bs = CANDS[best]
if best in OVS:
    ov = OVS[best]
    for lag in (5, 10, 20):
        r = H.backtest("p4q", overlay=ov.shift(lag).fillna(1.0), name=f"{best}_lag{lag}")
        rob[f"exec_lag_{lag}d"] = H.stats(r["daily"], bench=b25)
    r = H.backtest("p4q", overlay=ov, scenario="rec40")
    rob["rec40"] = H.stats(r["daily"], bench=H.baseline("P4Q", scenario="rec40")["daily"])
    rng = np.random.default_rng(0)
    pl = []
    for i in range(40):
        k = int(rng.integers(63, len(ov) - 63))
        ovs = pd.Series(np.roll(ov.to_numpy(), k), index=ov.index)
        r = H.backtest("p4q", overlay=ovs)
        st = H.stats(r["daily"])
        pl.append((st["ann_excess_%"], st["max_dd_%"], st["worst_month_%"]))
    pl = np.array(pl)
    act = H.stats(bs[25]["daily"])
    rob["placebo_circular_shift"] = {"n": len(pl), "ann_mean": float(pl[:, 0].mean()),
                                     "maxDD_mean": float(pl[:, 1].mean()), "maxDD_p95_best": float(np.percentile(pl[:, 1], 95)),
                                     "actual_maxDD": act["max_dd_%"], "actual_ann": act["ann_excess_%"],
                                     "share_placebo_maxDD_better_or_equal": float((pl[:, 1] >= act["max_dd_%"]).mean())}
elif best in GATES:
    fn = GATES[best]
    decs = [d_ for d_ in med.index if H.START <= d_ < H.HOLDOUT]
    ng = sum(bool(fn(d_)) for d_ in decs)
    rng = np.random.default_rng(0)
    pl = []
    for i in range(40):
        pick = set(rng.choice(decs, size=ng, replace=False))
        r = H.backtest(gate(lambda d_, pk=pick: d_ in pk))
        st = H.stats(r["daily"])
        pl.append((st["ann_excess_%"], st["max_dd_%"], st["worst_month_%"]))
    pl = np.array(pl)
    act = H.stats(bs[25]["daily"])
    rob["placebo_random_gates"] = {"n": len(pl), "ann_mean": float(pl[:, 0].mean()), "maxDD_mean": float(pl[:, 1].mean()),
                                   "actual_maxDD": act["max_dd_%"], "actual_ann": act["ann_excess_%"],
                                   "share_placebo_maxDD_better_or_equal": float((pl[:, 1] >= act["max_dd_%"]).mean())}
    r = H.backtest(gate(fn), scenario="rec40")
    rob["rec40"] = H.stats(r["daily"], bench=H.baseline("P4Q", scenario="rec40")["daily"])
INFO["best_robustness"] = rob
log("robustness", json.dumps(rob, default=float)[:1500])

# ---------------------------------------------------------------------------------------------------------------
# charts
# ---------------------------------------------------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

idx = b25.index
show = {"P4+Q (base)": b25}
for nm in [best, "D_voltarget", "D_ddstop", "E_gate_dd", "C_put25", "A_raw", "B_IBOV_beta"]:
    if nm in CANDS and nm not in show:
        v = CANDS[nm][25]
        show[nm] = (v["daily"] if isinstance(v, dict) else v)
refs = {"CDI": pd.Series(0.0, index=idx), "Universe": U[25]["daily"], "P4": P4[25]["daily"],
        "IDA-DI": H.index_excess("IDADI"), "Ibovespa": H.index_excess("IBOV")}
sty = {"CDI": ("k", ":"), "Universe": ("#7f7f7f", "--"), "P4": ("#1f77b4", "--"), "IDA-DI": ("#bcbd22", "-."),
       "Ibovespa": ("#c7c7c7", "-.")}
fig, ax = plt.subplots(figsize=(12, 6), dpi=110)
for nm, s in refs.items():
    ax.plot(idx, H.total_return_curve(s.reindex(idx).fillna(0)), sty[nm][1], color=sty[nm][0], lw=1.1, label=nm)
for nm, s in show.items():
    ax.plot(idx, H.total_return_curve(s.reindex(idx).fillna(0)), lw=2.2 if nm in ("P4+Q (base)", best) else 1.2,
            label=nm + (" [BEST]" if nm == best else ""))
ax.set_yscale("log")
ax.grid(alpha=0.25)
ax.legend(fontsize=8, frameon=False)
ax.set_title("Total return CDI x (1+excess), base 100, net 25 bps, pre-2026 (holdout sealed)")
fig.tight_layout()
fig.savefig(OUT / "equity_total_return.png")
plt.close(fig)

fig, axs = plt.subplots(1, 2, figsize=(15, 5.5), dpi=110)
uu = U[25]["daily"].reindex(idx).fillna(0)
for nm in ("P4", "IDA-DI"):
    s = refs[nm].reindex(idx).fillna(0)
    axs[0].plot(idx, ((1 + s - uu).cumprod() - 1) * 100, sty[nm][1], color=sty[nm][0], lw=1.1, label=nm)
for nm, s in show.items():
    s = s.reindex(idx).fillna(0)
    axs[0].plot(idx, ((1 + s - uu).cumprod() - 1) * 100, lw=2 if nm in ("P4+Q (base)", best) else 1.1, label=nm)
    m = (1 + s).cumprod()
    axs[1].plot(idx, (m / m.cummax() - 1) * 100, lw=2 if nm in ("P4+Q (base)", best) else 1.0, label=nm)
m = (1 + uu).cumprod()
axs[1].plot(idx, (m / m.cummax() - 1) * 100, "--", color="#7f7f7f", lw=1, label="Universe")
axs[0].axhline(0, color="k", lw=0.6)
axs[0].set_title("Cumulative excess vs universe (%)")
axs[1].set_title("Underwater: drawdown of excess-over-CDI (%)")
for a in axs:
    a.grid(alpha=0.25)
    a.legend(fontsize=7, frameon=False)
fig.tight_layout()
fig.savefig(OUT / "cum_excess.png")
plt.close(fig)

# rate-hedge chart: IPCA sub-book under hedge choices
fig, ax = plt.subplots(figsize=(11, 5), dpi=110)
for nm in ("exact(harness)", "raw", "vertex", "di1_ipca", "swap"):
    adj = None if nm == "exact(harness)" else RA["mats"][nm]
    with RT.use_R(adj):
        s = H.backtest(ipca_sig)["daily"]
    ax.plot(s.index, H.total_return_curve(s), label=f"IPCA+ P4+Q sub-book, hedge={nm}")
ax.plot(idx, H.total_return_curve(pd.Series(0.0, index=idx)), "k:", label="CDI")
ax.set_yscale("log"); ax.grid(alpha=0.25); ax.legend(fontsize=8, frameon=False)
ax.set_title("Rate / inflation hedge choice on the IPCA+ part of P4+Q (total return, net 25 bps)")
fig.tight_layout(); fig.savefig(OUT / "rate_hedges_ipca.png"); plt.close(fig)

# ---------------------------------------------------------------------------------------------------------------
# SEALED HOLDOUT - once, best variant only
# ---------------------------------------------------------------------------------------------------------------
hold = {}
bh = H.baseline("P4Q", holdout=True)
if best in OVS:
    ov = OVS[best]
    if best in ("D_voltarget", "D_ddstop"):
        # recompute the overlay from the holdout-inclusive base book (still point-in-time)
        bk_h = bh["daily"].reindex(dd).fillna(0.0)
        bk_saved = bk
        globals()["bk"] = bk_h
        ov = L1(ov_voltarget() if best == "D_voltarget" else ov_ddstop())
        globals()["bk"] = bk_saved
    rh = H.backtest("p4q", overlay=ov, holdout=True)
elif best in GATES:
    bk_h = bh["daily"].reindex(dd).fillna(0.0)
    navh = (1 + bk_h).cumprod()
    ddh = (navh / navh.cummax() - 1).shift(1).fillna(0.0)
    fn = GATES[best] if best != "E_gate_dd" else (lambda d_: ddh.get(d_, 0) < -0.005)
    rh = H.backtest(gate(fn), holdout=True)
else:
    rh = None
if rh is not None:
    hold = {"best": best, "stats_only_2026": H.stats(rh["daily"], bench=bh["daily"], holdout="only"),
            "p4q_only_2026": H.stats(bh["daily"], holdout="only")}
log("HOLDOUT", hold)

res = {"version": "harness_v4", "n_candidates": len(CANDS), "best": best,
       "selection_rule": "max maxDD improvement among candidates with diff vs P4Q >= -0.15%/yr at 25 bps "
                         "(tie: worst month); G long-short excluded (informational)",
       "table_25bps": tabs[25].round(4).reset_index().rename(columns={"index": "variant"}).to_dict("records"),
       "table_50bps": tabs[50].round(4).reset_index().rename(columns={"index": "variant"}).to_dict("records"),
       "info": INFO, "holdout": hold, "runtime_s": round(time.time() - T0)}
(OUT / "results.json").write_text(json.dumps(res, indent=1, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o)))
# reusable PIT overlay signals
sig = pd.DataFrame({k: v for k, v in OVS.items()})
sig.index.name = "day"
sig.to_pickle(D.CACHE / "overlay_signals.pkl")
log("done", round(time.time() - T0), "s")
