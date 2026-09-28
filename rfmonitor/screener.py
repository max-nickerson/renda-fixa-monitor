"""Cross-sectional quant screener over the whole ANBIMA-priced debenture universe (+ CRI/CRA with API access).

For every bond with an indicative rate:
  spread_bps       CDI+ equivalent spread (DI+, %DI, Pré and IPCA+ all on one yardstick; see bonds.cdi_equivalent_bps)
  fair_bps         robust spread-vs-duration curve for its peer group (indexer × Lei 12.431 tax status),
                   median per duration bucket
  resid_z          (spread − fair) / group MAD            → cheap vs peers = +      (weight 0.45)
  carry_z          breakeven: spread per year of duration → more cushion = +      (weight 0.20)
  momentum_z       −(5d spread change) / group dispersion  → tightening = +, widening = − (0.15)
  quality_z        + liquidity (% REUNE trades), − dealer dispersion (desvio padrão)  (0.20)
Penalties: material fact / negative CVM filing in the last 15 days (−1.0), widening > 50 bps in 5d (−0.5).
Exclusions: no rate, < 6 months to maturity, price < 90% of par (distressed), top 1% spreads per group
(usually credit events — a "cheap" signal there is mostly default risk).

A wide residual can also mean worse credit quality: free data has no ratings, so the model compares against
peers of the same indexer and duration, not the same rating. Treat the ranking as a shortlist to research.
"""
from __future__ import annotations

import logging
from datetime import date, timedelta

import numpy as np
import pandas as pd

from . import db
from .collect import Market, cdi_spread, debenture_spread
from .sources import anbima_public, cvm, snd

log = logging.getLogger(__name__)

BUCKETS = [0, 1, 2, 3, 4, 5, 7, 10, 40]
WEIGHTS = {"resid_z": 0.45, "carry_z": 0.20, "momentum_z": 0.15, "quality_z": 0.20}
GROUP_LABEL = {"DI_SPREAD": "DI +", "DI_PCT": "% DI", "IPCA": "IPCA +", "PRE": "Pré", "IGPM": "IGP-M"}


def _robust_z(s: pd.Series) -> pd.Series:
    med = s.median()
    mad = (s - med).abs().median() * 1.4826
    return (s - med) / (mad if mad and mad > 0 else (s.std() or 1))


def _fair_curve(g: pd.DataFrame) -> pd.Series:
    """Median spread per duration bucket, linearly interpolated at each bond's duration."""
    b = pd.cut(g["duration"], BUCKETS)
    med = g.groupby(b, observed=True).agg(d=("duration", "median"), s=("spread_bps", "median")).dropna()
    if len(med) == 0:
        return pd.Series(np.nan, index=g.index)
    if len(med) == 1:
        return pd.Series(med["s"].iloc[0], index=g.index)
    return pd.Series(np.interp(g["duration"], med["d"], med["s"]), index=g.index)


def _recent_negative_filings(days: int = 15) -> dict[str, str]:
    """CNPJ digits -> latest material fact title in the last `days` (one CVM file covers every issuer)."""
    out: dict[str, str] = {}
    cutoff = (date.today() - timedelta(days=days)).isoformat()
    try:
        df = cvm._zip_csv("IPE", date.today().year, f"ipe_cia_aberta_{date.today().year}")
    except Exception as e:
        log.warning("CVM IPE unavailable for screener: %s", e)
        return out
    if df.empty:
        return out
    df = df[(df["Data_Entrega"] >= cutoff) & df["Categoria"].str.contains("Fato Relevante", case=False, na=False)]
    for _, r in df.sort_values("Data_Entrega").iterrows():
        out[cvm._digits(r["CNPJ_Companhia"])] = f"{r['Data_Entrega']}: {r.get('Assunto') or 'Fato relevante'}"
    return out


def run(history_days: int = 10) -> dict:
    mkt = Market(max(history_days, 6))
    deb = mkt.deb
    if deb.empty:
        return {"error": "ANBIMA debenture files unavailable"}
    deb = deb.dropna(subset=["taxa_indicativa"]).copy()
    last_date = deb["date"].max()

    # Spreads for every bond/day (history is only used for the 5-day change).
    kinds, spreads, durs, cdis = [], [], [], []
    for _, r in deb.iterrows():
        k, s, d = debenture_spread(r, mkt)
        kinds.append(k)
        spreads.append(s)
        durs.append(d)
        cdis.append(cdi_spread(r, mkt, k, d))
    deb["group"], deb["duration"] = kinds, durs
    # Everything is ranked on the CDI+ equivalent spread (common yardstick); native spread kept for reference.
    deb["spread_native_bps"], deb["spread_bps"] = spreads, cdis

    today = deb[deb["date"] == last_date].copy()
    prev_dates = sorted(deb["date"].unique())
    ref_date = prev_dates[-6] if len(prev_dates) >= 6 else prev_dates[0]
    prev = deb[deb["date"] == ref_date].set_index("codigo")["spread_bps"]
    today["chg_5d_bps"] = (today["spread_bps"] - today["codigo"].map(prev)) if ref_date != last_date else np.nan

    # Registry enrichment: ISIN, CNPJ, Lei 12.431.
    reg = snd.table()
    if not reg.empty:
        reg = reg.drop_duplicates("Codigo do Ativo").set_index("Codigo do Ativo")
        today["isin"] = today["codigo"].map(reg["ISIN"])
        today["cnpj"] = today["codigo"].map(reg["CNPJ"])
        today["incentivada"] = today["codigo"].map(reg["Deb. Incent. (Lei 12.431)"]).eq("S")
    else:
        today["isin"] = today["cnpj"] = None
        today["incentivada"] = False

    # Exclusions.
    total = len(today)
    u = today.dropna(subset=["spread_bps", "duration"])
    u = u[(u["duration"] >= 0.5) & ((u["pct_pu_par"].isna()) | (u["pct_pu_par"] >= 90))]
    u = u[u["group"].isin(GROUP_LABEL)]
    # Peer groups: indexer × tax status. Lei 12.431 (tax-exempt for individuals) bonds price structurally
    # tighter, so comparing them with taxable bonds would make every taxable bond look "cheap".
    u = u.copy()
    u["incentivada"] = u["incentivada"].fillna(False).astype(bool)
    u["peer"] = u["group"].map(GROUP_LABEL) + np.where(u["incentivada"], " · isenta (12.431)", "")
    small = u["peer"].map(u["peer"].value_counts()) < 8
    u.loc[small, "peer"] = u.loc[small, "group"].map(GROUP_LABEL)  # merge tiny groups back
    p99 = u.groupby("peer")["spread_bps"].transform(lambda s: s.quantile(0.99))
    u = u[u["spread_bps"] <= p99].copy()

    # Features within each peer group.
    parts = []
    for g, grp in u.groupby("peer"):
        if len(grp) < 8:
            continue
        grp = grp.copy()
        grp["fair_bps"] = _fair_curve(grp)
        grp["resid_bps"] = grp["spread_bps"] - grp["fair_bps"]
        grp["resid_z"] = _robust_z(grp["resid_bps"]).clip(-3, 3)
        grp["carry_z"] = _robust_z(grp["spread_bps"] / grp["duration"].clip(lower=0.5)).clip(-3, 3)
        grp["momentum_z"] = (-_robust_z(grp["chg_5d_bps"].fillna(0))).clip(-3, 3)
        liq = grp["pct_reune"].fillna(0)
        disp = grp["desvio_padrao"].fillna(grp["desvio_padrao"].median())
        grp["quality_z"] = ((_robust_z(liq).clip(-2, 2) - _robust_z(disp).clip(-2, 2)) / 2)
        parts.append(grp)
    if not parts:
        return {"error": "not enough bonds with rates"}
    u = pd.concat(parts)

    u["score"] = sum(u[k].fillna(0) * w for k, w in WEIGHTS.items())
    flags_by_cnpj = _recent_negative_filings()
    u["flags"] = [[] for _ in range(len(u))]
    for i, r in u.iterrows():
        fl = []
        cnpj = r.get("cnpj")
        fact = flags_by_cnpj.get(cvm._digits(cnpj if isinstance(cnpj, str) else ""))
        if fact:
            fl.append(f"Fato relevante {fact[:90]}")
        if pd.notna(r["chg_5d_bps"]) and r["chg_5d_bps"] > 50:
            fl.append(f"spread +{r['chg_5d_bps']:.0f} bps em 5d")
        u.at[i, "flags"] = fl
    u["score"] -= u["flags"].map(lambda f: sum(1.0 if x.startswith("Fato") else 0.5 for x in f))

    # Validated model (research/run_selection.py): blend of this heuristic and a Ridge model trained on SND
    # trade history; percentile 0 = best. Bonds without a recent trade have no model score.
    from .ml import live
    sel = (live.cached_selection() or {}).get("rows", {})
    u["ml_pred_bps"] = u["codigo"].map(lambda c: (sel.get(c) or {}).get("ml_pred_bps"))
    u["model_pct"] = u["codigo"].map(lambda c: (sel.get(c) or {}).get("blend_pct"))
    u["model_pct"] = pd.to_numeric(u["model_pct"], errors="coerce")
    u = u.sort_values(["model_pct", "score"], ascending=[True, False], na_position="last")
    u["rank"] = range(1, len(u) + 1)
    # Fair value in price terms: converging to the peer curve moves the price by ≈ duration × residual.
    u["upside_pct"] = u["duration"] * u["resid_bps"] / 100
    u["fair_pu"] = u["pu"] * (1 + u["upside_pct"] / 100)
    u["fair_cdi_bps"] = u["fair_bps"]
    u["verdict"] = np.select([u["resid_z"] >= 1, u["resid_z"] <= -1], ["Barato", "Caro"], "Justo")

    # Strategy C3 (research/run_lab2.py — most robust rule, pure-credit OOS 2022–26: +1.7% a.a. vs universe,
    # Sharpe 1.36): hold the top 30% CDI+ carry that is NOT rich vs peers; sell rich (z ≤ −1.5, which lost
    # −4.7% a.a. hedged); whole credit book to CDI when the DI-credit momentum regime is negative.
    regime = (live.cached_regime() or {}).get("series", {})
    di_on = next((v["position"] for k, v in regime.items() if k.startswith("DI")), 1.0) >= 1
    u["cdi_pct"] = u["spread_bps"].rank(pct=True, ascending=False)
    u["rich"] = u["resid_z"] <= -1.5
    u["c3"] = (u["cdi_pct"] <= 0.3) & ~u["rich"]
    # P4 (best risk-adjusted rule, research/run_lab2.py with press): C3 but never BUY an issuer with negative
    # press in the last 30 days (Google News, point-in-time). Existing holdings are not sold on press alone.
    pr = live.cached_press()
    u["press_neg_30d"] = u["codigo"].map(lambda c: (pr.get(c) or {}).get("n"))
    u["press_titles"] = u["codigo"].map(lambda c: (pr.get(c) or {}).get("titles") or [])
    neg = u["press_neg_30d"].fillna(0) > 0
    # P5 risk flags (research/run_lab3b.py): issuer stock −15%+ in ~4 weeks or a rating downgrade ≤180 days.
    rk = live.cached_risk()
    u["eq_ret_4w"] = u["codigo"].map(lambda c: (rk.get(c) or {}).get("eq_ret_4w"))
    u["eq_ticker"] = u["codigo"].map(lambda c: (rk.get(c) or {}).get("ticker"))
    u["downgrade"] = u["codigo"].map(lambda c: (rk.get(c) or {}).get("downgrade"))
    stock_crash = pd.to_numeric(u["eq_ret_4w"], errors="coerce").fillna(0) <= -0.15
    downgraded = u["downgrade"].notna()
    for i in u.index[stock_crash | downgraded]:
        fl = list(u.at[i, "flags"])
        if stock_crash.at[i]:
            fl.append(f"Ação {u.at[i, 'eq_ticker']} {u.at[i, 'eq_ret_4w']:+.0%} em 4 semanas")
        if downgraded.at[i]:
            d = u.at[i, "downgrade"]
            fl.append(f"Rebaixamento {d['date']}: {d['title'][:80]}")
        u.at[i, "flags"] = fl
    risk = stock_crash | downgraded
    u["strategy"] = np.select(
        [u["rich"], u["c3"] & risk, u["c3"] & neg, u["c3"] & di_on, u["c3"] & ~di_on],
        ["VENDER · caro vs pares", "C3 · não comprar (ação caiu / rebaixado)", "C3 · não comprar (imprensa negativa 30d)",
         "C3 · comprar/manter", "C3 · aguardar (regime defensivo → CDI)"], "—")

    cols = ["rank", "codigo", "isin", "nome", "group", "peer", "indice", "vencimento", "duration", "taxa_indicativa",
            "spread_bps", "spread_native_bps", "fair_bps", "fair_cdi_bps", "resid_bps", "chg_5d_bps", "pu",
            "fair_pu", "upside_pct", "verdict", "pct_pu_par", "pct_reune", "desvio_padrao",
            "incentivada", "resid_z", "carry_z", "momentum_z", "quality_z", "score", "flags",
            "ml_pred_bps", "model_pct", "cdi_pct", "rich", "c3", "strategy", "press_neg_30d", "press_titles"]
    out = u[cols].copy()
    out["vencimento"] = out["vencimento"].dt.strftime("%Y-%m-%d")
    out = out.astype(object).where(out.notna(), None)
    rows = out.to_dict(orient="records")
    curves = {}
    for g, grp in u.groupby("peer"):
        pts = grp.sort_values("duration")
        curves[g] = {"x": pts["duration"].round(2).tolist(), "y": pts["spread_bps"].round(1).tolist(),
                     "fair": pts["fair_bps"].round(1).tolist(), "code": pts["codigo"].tolist()}
    result = {"date": last_date.strftime("%Y-%m-%d"), "universe": total, "ranked": len(rows),
              "weights": WEIGHTS, "rows": rows, "curves": curves, "regime_on": bool(di_on),
              "c3_n": int(u["c3"].sum()), "rich_n": int(u["rich"].sum())}
    db.save_screener(result)
    return result


def latest() -> dict | None:
    return db.load_screener()


def ensure_fresh() -> dict | None:
    """Re-run when ANBIMA has published a newer file than the stored ranking."""
    cur = latest()
    d, _ = anbima_public.latest(anbima_public.debentures)
    if cur and d and cur.get("date") >= d.isoformat():
        return cur
    try:
        return run()
    except Exception:
        log.exception("screener failed")
        return cur
