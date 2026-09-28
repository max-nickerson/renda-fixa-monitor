"""Lab round 4: do issuer STOCK price/volume, PRODUCTION-CHAIN commodity shocks, RATING actions, REGULATOR
decisions and SECTOR news add anything on top of P4? Weekly, 2022-01 → 2026-09, rate-hedged credit returns,
decision Monday with data to Sunday, execution next week, 25 bps (+ robustness: 2-week lag, 50 bps,
sub-periods), placebo with shuffled exposures, Holm multiple-testing correction.

Outputs research/out/lab3_*.json|png and research/out/LAB3_REPORT.md.
"""
import sys as _sys
from pathlib import Path as _P
_sys.path.insert(0, str(_P(__file__).resolve().parent.parent))  # run from anywhere

import json
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rfmonitor.config import DATA_DIR
from rfmonitor.ml import features_ext as fx, lab, lab_models as lm
from rfmonitor.ml.lab import Rule

warnings.filterwarnings("ignore")
OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)
QUICK = "--quick" in _sys.argv
N_PLACEBO = 5 if QUICK else 20
t0 = time.time()
log = lambda *a: print(f"[{time.time() - t0:5.0f}s]", *a, flush=True)

# ---------------------------------------------------------------- data
g = lm.prepare(pd.read_pickle(DATA_DIR / "history" / "lab_weekly.pkl"), "ret_hedged")
pw = pd.read_pickle(DATA_DIR / "history" / "press_weekly.pkl")
g = g.merge(pw, on=["cnpj8", "week"], how="left")
for c in ["press_neg_7d", "press_neg_30d", "press_neg_90d", "press_spike"]:
    g[c] = g[c].fillna(0.0)
g["cdi_pct"] = lm.weekly_pct(g, "cdi_bps")
weeks = sorted(g["week"].unique())
g["rebal"] = g["week"].isin(weeks[::4])
keys = g[["cnpj8", "week"]].copy()
log("building extra feature blocks")
ext = fx.all_blocks(keys)
g = pd.concat([g, ext], axis=1)
cov = fx.coverage(g)
log("coverage\n" + cov.round(3).to_string())
PRESS = ["press_neg_7d", "press_neg_30d", "press_neg_90d", "press_spike", "mkt_rj_z", "mkt_calote_z"]
EXT = [f for fs in fx.BLOCKS.values() for f in fs]

# ---------------------------------------------------------------- 1. univariate IC (weekly, 4w forward)
el = g[g["eligible"] & (g["week"] >= "2022-01-01") & g["y_fwd"].notna()]
ic_rows = {}
for f in EXT + ["cdi_bps", "resid_z", "d_spread_4w", "press_neg_30d", "n_distress_90d"]:
    sub = el[el[f].notna()]
    vals = [grp[f].rank().corr(grp["y_fwd"].rank()) for _, grp in sub.groupby("week")
            if len(grp) > 30 and grp[f].nunique() > 3]
    vals = np.array([v for v in vals if v == v])
    if len(vals) < 20:
        continue
    t = vals.mean() / (vals.std(ddof=1) / np.sqrt(len(vals)))
    ic_rows[f] = {"IC": vals.mean(), "t": t, "hit_%": (vals > 0).mean() * 100, "weeks": len(vals),
                  "coverage_%": len(sub) / len(el) * 100}
ic = pd.DataFrame(ic_rows).T
# Holm correction (two-sided normal approx; weekly ICs overlap 4w → divide t by 2 as a conservative HAC proxy)
from math import erf, sqrt
ic["t_adj"] = ic["t"] / 2
ic["p"] = ic["t_adj"].abs().map(lambda z: 2 * (1 - 0.5 * (1 + erf(z / sqrt(2)))))
order = ic["p"].sort_values().index
m = len(ic)
holm, run_max = {}, 0.0
for i, f in enumerate(order):
    run_max = max(run_max, min(1.0, (m - i) * ic.loc[f, "p"]))
    holm[f] = run_max
ic["p_holm"] = pd.Series(holm)
ic = ic.sort_values("t_adj")
log("univariate IC\n" + ic.round(3).to_string())

# ---------------------------------------------------------------- 2. event studies
def event_study(mask_fn, name, pre=8, post=13):
    """Average abnormal (vs same-week universe mean) cumulative hedged return and spread change around events."""
    gg = g[(g["week"] >= "2021-06-01")].copy()
    gg["abn"] = gg["ret"] - gg.groupby("week")["ret"].transform("mean")
    gg["dsp"] = gg.groupby("codigo")["cdi_bps"].diff()
    gg["abn_sp"] = gg["dsp"] - gg.groupby("week")["dsp"].transform("median")
    gg = gg.sort_values(["codigo", "week"])
    ev = gg[mask_fn(gg)]
    # first event per bond within 13 weeks (avoid overlapping duplicates)
    ev = ev.sort_values(["codigo", "week"])
    keep, last = [], {}
    for i, r in ev[["codigo", "week"]].iterrows():
        lw = last.get(r["codigo"])
        if lw is None or (r["week"] - lw).days > 91:
            keep.append(i)
            last[r["codigo"]] = r["week"]
    ev = ev.loc[keep]
    by = {c: d.set_index("week") for c, d in gg.groupby("codigo")}
    paths_r, paths_s = [], []
    for _, r in ev.iterrows():
        d = by[r["codigo"]]
        ks = pd.date_range(r["week"] - pd.Timedelta(weeks=pre), r["week"] + pd.Timedelta(weeks=post), freq="7D")
        paths_r.append(d["abn"].reindex(ks).fillna(0).to_numpy())
        paths_s.append(d["abn_sp"].reindex(ks).fillna(0).to_numpy())
    if not paths_r:
        return None
    R, S = np.array(paths_r), np.array(paths_s)
    cr = R.cumsum(axis=1) - R[:, :pre].sum(axis=1, keepdims=True)  # cumulative from event week
    cs = S.cumsum(axis=1) - S[:, :pre].sum(axis=1, keepdims=True)
    post_r = R[:, pre + 1: pre + 1 + post].sum(axis=1)  # weeks +1..+13 = tradeable after the event is known
    post_s = S[:, pre + 1: pre + 1 + post].sum(axis=1)
    pre_r = R[:, :pre + 1].sum(axis=1)
    se = lambda x: x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan
    return {"name": name, "n": len(R), "path_ret_bps": (cr.mean(axis=0) * 1e4).round(1).tolist(),
            "path_spread_bps": cs.mean(axis=0).round(1).tolist(),
            "pre_ret_bps": float(pre_r.mean() * 1e4), "post_ret_bps": float(post_r.mean() * 1e4),
            "post_ret_t": float(post_r.mean() / se(post_r)) if len(post_r) > 1 else np.nan,
            "post_spread_bps": float(post_s.mean()), "post_spread_t": float(post_s.mean() / se(post_s))}


EVENTS = {
    "Ação caiu ≥20% em 4 semanas": lambda d: d["eq_ret_4w"] <= np.log(0.8),
    "Pico de volume da ação (z≥2,5)": lambda d: d["eq_volu_z"] >= 2.5,
    "Choque de commodities contra o emissor (≤−1,5σ, 13s)": lambda d: d["com_shock_13w"] <= -1.5,
    "Choque de commodities a favor (≥+1,5σ, 13s)": lambda d: d["com_shock_13w"] >= 1.5,
    "Rebaixamento de rating (semana)": lambda d: d["rat_days_since_down"] <= 7,
    "Decisão negativa do regulador (30d)": lambda d: d["reg_neg_30d"] >= 1,
    "Notícia setorial negativa (z≥2)": lambda d: d["sec_news_z"] >= 2,
    "Pico de imprensa negativa (referência)": lambda d: (d["press_neg_30d"] >= 3) & (d["press_spike"] >= 1),
}
events = {}
for name, fn in EVENTS.items():
    try:
        r = event_study(fn, name)
    except Exception as e:  # missing block → skip
        log(f"event {name}: {e}")
        r = None
    if r:
        events[name] = r
        log(f"event {name}: n={r['n']} pre {r['pre_ret_bps']:.0f}bps | post {r['post_ret_bps']:.0f}bps "
            f"(t {r['post_ret_t']:.1f}) | post spread {r['post_spread_bps']:+.0f}bps (t {r['post_spread_t']:.1f})")

# ---------------------------------------------------------------- 3. strategies
nz = lambda x, d=0.0: d if x is None or x != x else x
rich = lambda r: nz(r["resid_z"]) <= -1.5
press_any = lambda r: r["press_neg_30d"] >= 1
c3_in = lambda r: nz(r["cdi_pct"], 1) <= 0.3 and not rich(r)
p4_in = lambda r: c3_in(r) and not press_any(r)
reg = g.groupby("week").agg(mom=("mkt_mom_21", "first"))
REGIME = (reg["mom"] > 0).astype(float)

eq_bad = lambda r: nz(r["eq_ret_4w"]) <= -0.15 or nz(r["eq_volu_z"]) >= 2.5
com_bad = lambda r: nz(r["com_shock_13w"]) <= -1.5
rat_bad = lambda r: nz(r["rat_days_since_down"], 9999) <= 180
reg_bad = lambda r: nz(r["reg_neg_30d"]) >= 1
sec_bad = lambda r: nz(r["sec_news_z"]) >= 2

RULES = [
    ("U0 Universo (benchmark)", Rule("U0", lambda r: True, lambda r, st: False), None),
    ("P4 (atual)", Rule("P4", p4_in, lambda r, st: rich(r)), REGIME),
    ("P4 + filtro ação (queda ≥15%/4s ou pico de volume)", Rule("P4eq", lambda r: p4_in(r) and not eq_bad(r), lambda r, st: rich(r)), REGIME),
    ("P4 + saída ação (drawdown ≤−35%)", Rule("P4eqx", p4_in, lambda r, st: rich(r) or nz(r["eq_dd_13w"]) <= -0.35, 26), REGIME),
    ("P4 + filtro commodities (choque ≤−1,5σ)", Rule("P4com", lambda r: p4_in(r) and not com_bad(r), lambda r, st: rich(r)), REGIME),
    ("P4 + saída commodities (choque ≤−2σ)", Rule("P4comx", p4_in, lambda r, st: rich(r) or nz(r["com_shock_13w"]) <= -2, 13), REGIME),
    ("P4 + filtro rating (rebaixado há ≤180d)", Rule("P4rat", lambda r: p4_in(r) and not rat_bad(r), lambda r, st: rich(r)), REGIME),
    ("P4 + saída rating (novo rebaixamento)", Rule("P4ratx", p4_in, lambda r, st: rich(r) or nz(r["rat_days_since_down"], 9999) <= 7, 26), REGIME),
    ("P4 + filtro regulador (decisão negativa 30d)", Rule("P4reg", lambda r: p4_in(r) and not reg_bad(r), lambda r, st: rich(r)), REGIME),
    ("P4 + filtro notícia setorial (z≥2)", Rule("P4sec", lambda r: p4_in(r) and not sec_bad(r), lambda r, st: rich(r)), REGIME),
    ("P4 + todos os filtros", Rule("P4all", lambda r: p4_in(r) and not (eq_bad(r) or com_bad(r) or rat_bad(r) or reg_bad(r) or sec_bad(r)),
                                    lambda r, st: rich(r)), REGIME),
    ("Diag: só ações em queda ≥15%", Rule("Deq", lambda r: nz(r["eq_ret_4w"]) <= -0.15, lambda r, st: nz(r["eq_ret_4w"]) > -0.05, 0, 13), None),
    ("Diag: só choque de commodities ≤−1,5σ", Rule("Dcom", lambda r: com_bad(r), lambda r, st: nz(r["com_shock_13w"]) > -0.5, 0, 13), None),
    ("Diag: só rebaixados (≤180d)", Rule("Drat", lambda r: rat_bad(r), lambda r, st: not rat_bad(r), 0, 26), None),
]

if "--smoke" in _sys.argv:  # pipeline check only
    RULES = RULES[:3]
res, curves, bench = {}, {}, {}
SUBS = {"2022–23": ("2022-01-01", "2023-12-31"), "2024": ("2024-01-01", "2024-12-31"), "2025–26": ("2025-01-01", "2026-12-31")}
for label, rule, ov in RULES:
    row = {}
    for lag, cost in ((1, 25), (2, 25), (1, 50)):
        r = lab.run_rule(g, rule, lag=lag, cost_bps=cost, overlay=ov)
        if label.startswith("U0"):
            bench[(lag, cost)] = r["weekly"]
        st = lab.stats(r["weekly"], bench[(lag, cost)])
        row[f"lag{lag}_{cost}"] = {**st, "avg_bonds": r["avg_n"]}
        if (lag, cost) == (1, 25):
            curves[label] = r["weekly"]
            for sn, (a, b) in SUBS.items():
                w = r["weekly"][a:b]
                row[f"sub_{sn}"] = (w - bench[(1, 25)].reindex(w.index).fillna(0)).mean() * 5200
    res[label] = row
    a = row["lag1_25"]
    log(f"{label:<55} vsU0 {a['vs_bench_ann_%']:6.2f}% | lag2 {row['lag2_25']['vs_bench_ann_%']:6.2f}% | "
        f"50bps {row['lag1_50']['vs_bench_ann_%']:6.2f}% | SR {a['sharpe']:5.2f} DD {a['max_dd_%']:6.2f}% "
        f"n {a['avg_bonds']:4.0f} | subs " + " ".join(f"{row[f'sub_{s}']:+.2f}" for s in SUBS))

# paired t-stat of each P4 variant vs P4 (weekly return differences), Holm-corrected
p4 = curves["P4 (atual)"]
paired = {}
for label in curves:
    if label.startswith("P4 +"):
        d = (curves[label] - p4.reindex(curves[label].index).fillna(0)).dropna()
        paired[label] = {"diff_ann_%": d.mean() * 5200, "t": d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))}
pt = pd.DataFrame(paired).T
pt["p"] = pt["t"].abs().map(lambda z: 2 * (1 - 0.5 * (1 + erf(z / sqrt(2)))))
order = pt["p"].sort_values().index
holm, run_max = {}, 0.0
for i, f in enumerate(order):
    run_max = max(run_max, min(1.0, (len(pt) - i) * pt.loc[f, "p"]))
    holm[f] = run_max
pt["p_holm"] = pd.Series(holm)
log("paired vs P4\n" + pt.round(3).to_string())

# ---------------------------------------------------------------- 4. ML ablation (Ridge, monthly, top20/50, never rich, regime)
abl = {}
base_feats = lm.BASE + lm.NEWS + PRESS
ABL = [("Base (preço, carry, CVM, imprensa)", base_feats),
                    ("+ Ação (EQ)", base_feats + fx.EQ_FEATURES),
                    ("+ Commodities (COM)", base_feats + fx.COM_FEATURES),
                    ("+ Rating (RAT)", base_feats + fx.RAT_FEATURES),
                    ("+ Regulador (REG)", base_feats + fx.REG_FEATURES),
                    ("+ Notícia setorial (SEC)", base_feats + fx.SEC_FEATURES),
                    ("Tudo", base_feats + EXT)]
for name, feats in (ABL[:2] if "--smoke" in _sys.argv else ABL):
    feats = [f for f in feats if f in g and g[f].notna().any()]
    sc = lm.walk_forward_scores(g, feats, kind="rank")
    g["_s"], g["_pct"] = sc, lm.weekly_pct(g.assign(_s=sc), "_s")
    oos = g[g["eligible"] & (g["week"] >= "2022-01-01") & g["_s"].notna() & g["y_fwd"].notna()]
    ics = np.array([grp["_s"].rank().corr(grp["y_fwd"].rank()) for _, grp in oos.groupby("week") if len(grp) > 30])
    rule = Rule("ML", lambda r: r["rebal"] and nz(r["_pct"], 1) <= 0.2 and not rich(r),
                lambda r, st: (r["rebal"] and nz(r["_pct"], 1) > 0.5) or rich(r))
    row = {"IC": ics.mean(), "IC_t": ics.mean() / (ics.std(ddof=1) / np.sqrt(len(ics))) / 2}
    for lag, cost in ((1, 25), (2, 25), (1, 50)):
        r = lab.run_rule(g, rule, lag=lag, cost_bps=cost, overlay=REGIME)
        st = lab.stats(r["weekly"], bench[(lag, cost)])
        row[f"lag{lag}_{cost}"] = st["vs_bench_ann_%"]
        if (lag, cost) == (1, 25):
            row["sharpe"], row["max_dd"] = st["sharpe"], st["max_dd_%"]
    abl[name] = row
    log(f"ML {name:<40} IC {row['IC']:.3f} (t {row['IC_t']:.1f}) | vsU0 {row['lag1_25']:.2f}% lag2 {row['lag2_25']:.2f}% "
        f"50bps {row['lag1_50']:.2f}% | SR {row['sharpe']:.2f}")

# ---------------------------------------------------------------- 5. placebo: shuffle production-chain exposures
plac = {}
real_exp = fx.exposure_table()
if not real_exp.empty and g["com_exposed"].sum() > 0:
    rng = np.random.default_rng(7)
    issuers = sorted(set(g["cnpj8"]))
    real_ic = ic.loc["com_shock_13w", "IC"] if "com_shock_13w" in ic.index else np.nan
    real_rule = res["P4 + filtro commodities (choque ≤−1,5σ)"]["lag1_25"]["vs_bench_ann_%"]
    p_ics, p_rules = [], []
    by_issuer = {c: d[["variable", "weight"]] for c, d in real_exp.groupby("cnpj8")}
    exposed = list(by_issuer)
    for k in range(N_PLACEBO):
        targets = rng.choice(issuers, size=len(exposed), replace=False)  # give each real exposure profile to a random issuer
        perm = pd.concat([by_issuer[src].assign(cnpj8=dst) for src, dst in zip(exposed, targets)])
        cf = fx.commodity_features(keys, perm)
        gp = g.copy()
        gp[["com_shock_4w", "com_shock_13w", "com_exposed"]] = cf.to_numpy()
        e2 = gp[gp["eligible"] & (gp["week"] >= "2022-01-01") & gp["y_fwd"].notna() & gp["com_shock_13w"].notna()]
        v = [grp["com_shock_13w"].rank().corr(grp["y_fwd"].rank()) for _, grp in e2.groupby("week")
             if len(grp) > 30 and grp["com_shock_13w"].nunique() > 3]
        p_ics.append(np.nanmean(v))
        rr = lab.run_rule(gp, next(r for lbl, r, _ in RULES if lbl.startswith("P4 + filtro commodities")), overlay=REGIME)
        p_rules.append(lab.stats(rr["weekly"], bench[(1, 25)])["vs_bench_ann_%"])
        log(f"placebo {k + 1}/{N_PLACEBO}: IC {p_ics[-1]:.3f}  rule vsU0 {p_rules[-1]:.2f}%")
    plac = {"real_ic": real_ic, "placebo_ic": p_ics, "ic_pctile": float((np.array(p_ics) < real_ic).mean() * 100),
            "real_rule": real_rule, "placebo_rule": p_rules,
            "rule_pctile": float((np.array(p_rules) < real_rule).mean() * 100)}
    log(f"placebo: real IC {real_ic:.3f} beats {plac['ic_pctile']:.0f}% of shuffles; real rule {real_rule:.2f}% beats "
        f"{plac['rule_pctile']:.0f}%")

# ---------------------------------------------------------------- outputs
json.dump({"coverage": cov["coverage"].to_dict(), "ic": ic.round(4).to_dict(orient="index"), "events": events,
           "strategies": res, "paired_vs_p4": pt.round(4).to_dict(orient="index"), "ablation": abl, "placebo": plac},
          open(OUT / "lab3_results.json", "w", encoding="utf-8"), indent=1, default=float, ensure_ascii=False)
pd.DataFrame(curves).to_pickle(DATA_DIR / "history" / "lab3_curves.pkl")

fig, ax = plt.subplots(1, 2, figsize=(13, 5), dpi=105)
x = np.arange(-8, 14)
for name, r in events.items():
    ax[0].plot(x, r["path_ret_bps"], label=f"{name} (n={r['n']})", lw=1.6)
    ax[1].plot(x, r["path_spread_bps"], lw=1.6)
for a, t in zip(ax, ["Retorno anormal acumulado (bps, hedgeado)", "Variação anormal do spread CDI+ (bps)"]):
    a.axvline(0, color="#999", lw=0.8)
    a.axhline(0, color="#999", lw=0.6)
    a.set_title(t, fontsize=10)
    a.set_xlabel("semanas em relação ao evento (0 = semana em que o evento fica público)")
    a.grid(alpha=0.25)
ax[0].legend(fontsize=7, frameon=False)
fig.suptitle("Estudos de evento: o crédito anda DEPOIS do sinal? (à direita de 0 = negociável)", fontsize=11)
fig.tight_layout()
fig.savefig(OUT / "lab3_event_studies.png")

fig, ax = plt.subplots(figsize=(12, 5.5), dpi=105)
for label, s in curves.items():
    if label.startswith(("U0", "P4")):
        eq = (1 + s.fillna(0)).cumprod()
        ax.plot(eq.index, (eq - 1) * 100, label=label, lw=2.4 if label == "P4 (atual)" else 1.2,
                ls="--" if label.startswith("U0") else "-")
ax.set_title("P4 e variantes com ação / commodities / rating / regulador / notícia setorial — excesso sobre CDI, "
             "crédito puro, fora da amostra 2022–26", fontsize=10)
ax.grid(alpha=0.25)
ax.legend(fontsize=7, frameon=False, ncol=2)
fig.tight_layout()
fig.savefig(OUT / "lab3_equity_curves.png")
log("done")
