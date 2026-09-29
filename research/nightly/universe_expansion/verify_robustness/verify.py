"""Adversarial robustness check of universe_expansion's MIX_80deb_20fidc claim (pre-2026 only)."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from research.nightly import harness as H

D = Path("data/history/nightly/universe_expansion")
OUT = Path("research/nightly/universe_expansion/verify_robustness")
DD = H.days(); CDI = H.cdi_daily()
cdi_tot = (1 + CDI).groupby([DD.year, DD.month]).prod() - 1
cdi_tot.index = pd.to_datetime([f"{y}-{m:02d}-01" for y, m in cdi_tot.index])

F0 = pd.read_pickle(D / "fidc_panel.pkl")

def fidc(rule="P4Q", lagm=2, missing="cdi", open_only=False, cost_bps=0.0, drop_cls=(), upper=0.2,
         top=0.7, minpl=50e6, ret_contrib=False):
    F = F0[(F0["ret"] > -1) & (F0["ret"] < upper) & (F0["ret"] != 0)]
    F = F[F["FUNDO_EXCLUSIVO"] != "S"]
    if open_only:
        F = F[F["CONDOM"] == "ABERTO"]
    F = F[~F["cls"].isin(drop_cls)].copy()
    F["ex"] = (1 + F["ret"]) / (1 + F["ref"].map(cdi_tot)) - 1
    piv = F.pivot_table(index="ref", columns="cls", values="ex", aggfunc="first")
    pl = F.pivot_table(index="ref", columns="cls", values="pl", aggfunc="first")
    inad = F.pivot_table(index="ref", columns="cls", values="inad_ratio", aggfunc="first")
    months = pd.date_range("2022-01-01", "2025-12-01", freq="MS")
    mret, contrib, prev = {}, {}, set()
    for m in months:
        last_ref = m - pd.DateOffset(months=lagm)
        hist = piv.loc[(piv.index <= last_ref) & (piv.index > last_ref - pd.DateOffset(months=3))]
        if len(hist) < 3 or m not in piv.index:
            continue
        carry = hist.mean()
        ok = hist.notna().all() & (pl.reindex([last_ref]).iloc[0] > minpl).reindex(carry.index).fillna(False) & (hist != 0).all()
        c = carry[ok]
        tgt = piv.loc[m].reindex(c.index)
        if missing == "cdi":
            tgt = tgt.fillna(0.0)
        elif missing == "zero":
            cm = float(cdi_tot.get(m, 0.0)); tgt = tgt.fillna(-cm / (1 + cm))
        elif missing == "last":  # carry forward the last reported excess (neutral)
            tgt = tgt.fillna(hist.iloc[-1].reindex(c.index))
        elif missing == "drop":
            tgt = tgt.dropna()
        q = inad.reindex([last_ref]).iloc[0].reindex(c.index)
        worst = (q >= q.quantile(0.8)).fillna(False)
        tp = c >= c.quantile(top)
        sel = {"U": c.index, "P4": c.index[tp], "P4Q": c.index[tp & ~worst]}[rule]
        sel = [s for s in sel if s in tgt.index]
        r = tgt.reindex(sel).clip(-1, 0.5)
        turn = len(set(sel) ^ prev) / max(len(sel), 1)  # approx one-way*2 turnover (equal weight)
        prev = set(sel)
        mret[m] = float(r.mean()) - turn * cost_bps / 2 / 1e4 if len(sel) else 0.0
        for k, v in r.items():
            contrib[k] = contrib.get(k, 0.0) + v / len(sel)
    ms = pd.Series(mret)
    out = pd.Series(0.0, index=DD)
    for m, r in ms.items():
        idx = (DD.year == m.year) & (DD.month == m.month); n = idx.sum()
        if n: out[idx] = (1 + r) ** (1 / n) - 1
    out = out[(out.index >= H.START) & (out.index < H.HOLDOUT)]
    return (out, pd.Series(contrib).sort_values(ascending=False)) if ret_contrib else out

P4Q = H.baseline("P4Q")["daily"]; P4Q50 = H.baseline("P4Q", cost_bps=50)["daily"]
def mixs(f, deb=P4Q, w=0.2):
    idx = deb.index
    return (1 - w) * deb + w * f.reindex(idx).fillna(0)

def row(s, bench, excl2023=False):
    if excl2023:
        s = s[s.index.year != 2023]; bench = bench[bench.index.year != 2023]
    a = H.stats(s, bench=bench)
    return {k: a[k] for k in ("ann_excess_%", "diff_ann_%", "diff_t_nw", "diff_p", "diff_h1_%", "diff_h2_%")}

res = {}
base, contrib = fidc(ret_contrib=True)
top5 = list(contrib.index[:5])
res["top5_contrib_classes"] = {k: round(float(v), 4) for k, v in contrib.head(5).items()}
fidcU = fidc("U")
V = {
 "reproduce": (mixs(base), P4Q),
 "lag3_strict_avail": (mixs(fidc(lagm=3)), P4Q),
 "missing_last": (mixs(fidc(missing="last")), P4Q),
 "missing_drop": (mixs(fidc(missing="drop")), P4Q),
 "missing_zero": (mixs(fidc(missing="zero")), P4Q),
 "open_only": (mixs(fidc(open_only=True)), P4Q),
 "fidc_cost25": (mixs(fidc(cost_bps=25)), P4Q),
 "at50bps_fidc_cost50": (mixs(fidc(cost_bps=50), deb=P4Q50), P4Q50),
 "drop_top5_classes": (mixs(fidc(drop_cls=top5)), P4Q),
 "upper_winsor_5pct": (mixs(fidc(upper=0.05)), P4Q),
 "top20pct": (mixs(fidc(top=0.8)), P4Q),
 "minpl_200m": (mixs(fidc(minpl=200e6)), P4Q),
 "combo_lag3_last_cost25": (mixs(fidc(lagm=3, missing="last", cost_bps=25)), P4Q),
}
for k, (s, b) in V.items():
    res[k] = row(s, b)
res["reproduce_excl2023"] = row(*V["reproduce"], excl2023=True)
res["combo_excl2023"] = row(*V["combo_lag3_last_cost25"], excl2023=True)
res["fidcP4Q_vs_fidcU"] = row(base, fidcU)
res["fidcP4Q_vs_fidcU_lag3_last"] = row(fidc(lagm=3, missing="last"), fidc("U", lagm=3, missing="last"))
res["fidcP4Q_vs_fidcU_open_only"] = row(fidc(open_only=True), fidc("U", open_only=True))
ps = [res[k]["diff_p"] for k in V]
res["holm_over_perturbations"] = dict(zip(V, [round(float(x), 3) for x in H.holm(ps)]))
json.dump(res, open(OUT / "verify_results.json", "w"), indent=1, default=str)
for k, v in res.items(): print(k, v)
