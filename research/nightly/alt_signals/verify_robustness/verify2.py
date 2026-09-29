"""Part 2: mechanism, PIT perturbations, placebos, capacity."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np, pandas as pd
from research.nightly import harness as H
from research.nightly.alt_signals import signals as S

OUT = Path(__file__).resolve().parent
Q = pd.read_pickle(S.CACHE / "panel_alt_M.pkl")
o = S.load_offers()
res = {}
b = H.baseline("P4Q")

def ds(r, bb=None):
    bb = bb if bb is not None else b
    d = (H.monthly(r["daily"]) - H.monthly(bb["daily"])); d = d[(d.index >= "2022-01-01") & (d.index < "2026-01-01")]
    return {"diff_ann_%": round(float(d.mean()) * 1200, 3), "t_nw": round(H.nw_t(d, 6), 2)}

def mk(bad):
    return lambda x: x["p4q"].to_numpy() & ~bad(x)
ncb = lambda x: x["nc_expo"].to_numpy() < -1

# who flagged Americanas / Oncoclinicas / Dasa
for c in ["00776574", "12104241", "61486650", "34714305", "38482780"]:
    z = Q[(Q.cnpj8 == c) & Q.p4q & Q.univ & (Q.day >= "2022-01-01")]
    fl = z[(z.sup_iss_90d > 0) | (z.nc_expo < -1)]
    res[f"flag_{c}"] = {"p4q_rows": int(len(z)), "flag_rows": int(len(fl)),
                        "by_sup": int((fl.sup_iss_90d > 0).sum()), "by_nc": int((fl.nc_expo < -1).sum()),
                        "days": sorted({str(d.date()) for d in fl.day})[:12], "sector": str(z.sector.iloc[0]) if len(z) else None}
print(json.dumps({k: v for k, v in res.items() if k.startswith("flag_")}, indent=0), flush=True)

# supply leg alone & nowcast leg alone, excluding Americanas from both books
QA = Q[Q.cnpj8 != "00776574"].copy()
bA = H.backtest("p4q", panel=QA)
sup = lambda x: x["sup_iss_90d"].fillna(0).to_numpy() > 0
res["noAmericanas_best"] = ds(H.backtest(mk(lambda x: sup(x) | ncb(x)), panel=QA), bA)
res["noAmericanas_sup_only"] = ds(H.backtest(mk(sup), panel=QA), bA)
res["noAmericanas_nc_only"] = ds(H.backtest(mk(ncb), panel=QA), bA)
print("noAmer", res["noAmericanas_best"], res["noAmericanas_sup_only"], res["noAmericanas_nc_only"], flush=True)

# mechanism: is the excluded bond the NEW issue itself? age column in years or days?
print("age describe", Q["age"].describe().to_dict(), flush=True)
agev = Q["age"].astype(float)
unit_days = agev.median() > 50
young = (agev <= (120 if unit_days else 120 / 365.25))
Q["_young"] = young.fillna(False)
Qp = Q[Q.p4q & Q.univ & (Q.day >= "2022-01-01")]
f = Qp.sup_iss_90d > 0
res["flagged_sup_rows"] = int(f.sum()); res["flagged_sup_young_share"] = round(float(Qp.loc[f, "_young"].mean()), 3)
res["sup_new_bond_only"] = ds(H.backtest(mk(lambda x: sup(x) & x["_young"].to_numpy()), panel=Q))
res["sup_seasoned_only"] = ds(H.backtest(mk(lambda x: sup(x) & ~x["_young"].to_numpy()), panel=Q))
res["young_any_excluded"] = ds(H.backtest(mk(lambda x: x["_young"].to_numpy()), panel=Q))
print("mech", res["flagged_sup_young_share"], res["sup_new_bond_only"], res["sup_seasoned_only"], res["young_any_excluded"], flush=True)

# PIT perturbations of offer availability
def sup_col(oo, w=91):
    return S.issuer_event_counts(Q, oo, {"x": w})["x"].to_numpy()
for lag in (14, 30, 60):
    oo = o.copy(); oo["avail"] = oo["avail"] + pd.Timedelta(days=lag)
    Q["_s"] = sup_col(oo)
    res[f"offers_lag+{lag}d"] = ds(H.backtest(mk(lambda x: (x["_s"].to_numpy() > 0) | ncb(x)), panel=Q))
    print("lag", lag, res[f"offers_lag+{lag}d"], flush=True)
# legacy-only / r160-only
for src in ("legacy", "r160"):
    Q["_s"] = sup_col(o[o.src == src])
    res[f"supply_{src}_only_combo"] = ds(H.backtest(mk(lambda x: (x["_s"].to_numpy() > 0) | ncb(x)), panel=Q))
    print(src, res[f"supply_{src}_only_combo"], flush=True)

# placebo: random offer dates per issuer (same count per issuer, uniform over 2019-2025), 20 draws, combo rule
lo, hi = pd.Timestamp("2019-01-01").value // 86400_000_000_000, pd.Timestamp("2025-12-31").value // 86400_000_000_000
vals = []
for s in range(20):
    rng = np.random.default_rng(100 + s)
    oo = o.copy()
    oo["avail"] = pd.to_datetime(rng.integers(lo, hi, len(oo)), unit="D")
    Q["_s"] = sup_col(oo)
    vals.append(ds(H.backtest(mk(lambda x: (x["_s"].to_numpy() > 0) | ncb(x)), panel=Q))["diff_ann_%"])
vals = np.array(vals)
res["placebo_random_offer_dates_combo"] = {"mean": round(float(vals.mean()), 3), "p95": round(float(np.quantile(vals, .95)), 3),
                                           "max": round(float(vals.max()), 3), "p_ge_real": float((vals >= 0.337).mean()), "vals": vals.round(3).tolist()}
print("placebo dates", res["placebo_random_offer_dates_combo"], flush=True)
# placebo: random issuer exclusion (exclude random P4Q issuers with same count per date, issuer-level) 20 draws
vals = []
for s in range(20):
    rng = np.random.default_rng(500 + s)
    def g(x, rng=rng):
        p = x["p4q"].to_numpy(); m = best_m = p & ~((x["sup_iss_90d"].fillna(0).to_numpy() > 0) | ncb(x))
        iss_bad = set(x.loc[p & ~m, "cnpj8"]); iss = x.loc[p, "cnpj8"].unique()
        k = len(iss_bad)
        pick = set(rng.choice(iss, size=min(k, len(iss)), replace=False)) if k else set()
        return p & ~x["cnpj8"].isin(pick).to_numpy()
    vals.append(ds(H.backtest(g, panel=Q))["diff_ann_%"])
vals = np.array(vals)
res["placebo_random_issuer_exclusion"] = {"mean": round(float(vals.mean()), 3), "p95": round(float(np.quantile(vals, .95)), 3),
                                          "max": round(float(vals.max()), 3), "sd": round(float(vals.std()), 3)}
print("placebo issuers", res["placebo_random_issuer_exclusion"], flush=True)

# capacity / liquidity: trades_30d and issuer counts of kept vs excluded
bad = (Qp.sup_iss_90d > 0) | (Qp.nc_expo < -1)
res["liq"] = {"trades30_kept_med": float(Qp.loc[~bad, "trades_30d"].median()), "trades30_excl_med": float(Qp.loc[bad, "trades_30d"].median()),
              "issuers_per_date_kept": float(Qp[~bad].groupby("day").cnpj8.nunique().mean()),
              "issuers_per_date_p4q": float(Qp.groupby("day").cnpj8.nunique().mean())}
print("liq", res["liq"], flush=True)
(OUT / "verify2.json").write_text(json.dumps(res, indent=1, default=str))
