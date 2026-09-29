"""Build the point-in-time new-issue table (one row per new debenture series that settles 2021-04 .. 2026-09).

Sources (all cached / already in the repo):
  * SND registry (rfmonitor.sources.snd.table): issue date, start of accrual (= settlement of the primary),
    contract rate (= bookbuilding rate for DI+/IPCA+/Pre series), indexer, quantity x nominal (size), CVM
    registration (ICVM 400 / ICVM 476 / RCVM 160 automatic), lead manager, guarantee, Lei 12.431 flag.
  * CVM offer registry  data/history/nightly/alt_signals/offers_deb.pkl (cnpj8, event, avail, value, incent, src).
  * lab_daily grid (first print, same-day peer curve, issuer's other bonds) and B3/NTN-B curves.

Point-in-time rule:  the decision ("participate in the book at the clearing rate") is taken at the close of the
grid day BEFORE settlement (fair_day).  The peer and issuer fair values use only lab_daily rows of fair_day (fresh
marks); curves are the last curve on/before fair_day.  Entry = par at settlement.  Everything after settlement is
outcome only.

Output: data/history/nightly/w2_primary_market_concession/new_issues.pkl
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from rfmonitor.ml.selection import BUCKETS
from rfmonitor.sources import snd
from research.nightly import harness as H

OUT = H.HIST / "nightly" / "w2_primary_market_concession"
OUT.mkdir(parents=True, exist_ok=True)
DUR_F = {"DI_SPREAD": 0.8, "IPCA": 0.6, "PRE": 0.75}   # fallback duration / maturity factors (as selection.py)


def br_float(s):
    try:
        return float(str(s).replace(".", "").replace(",", ".")) if s not in (None, "-", "") else np.nan
    except Exception:
        return np.nan


def _dt(s):
    d = pd.to_datetime(s, format="%d/%m/%Y", errors="coerce")
    d = d.where((d.dt.year < 2090) & (d.dt.year > 1990))
    return d.astype("datetime64[ns]")


def registry() -> pd.DataFrame:
    t = snd.table().drop_duplicates("Codigo do Ativo").copy()
    idx = t["indice"].astype(str).str.upper()
    pct = t["Percentual Multiplicador/Rentabilidade"].map(br_float)
    rate = t["Juros Criterio Novo - Taxa"].map(br_float)
    kind = np.select([idx.eq("DI") & pct.fillna(100).eq(100), idx.eq("IPCA"), idx.str.startswith("PR")],
                     ["DI_SPREAD", "IPCA", "PRE"], "OTHER")
    reg = t["Registro CVM da Emissao"].astype(str).str.upper()
    regime = np.select([reg.str.contains("476"), reg.str.startswith("AUT/"), reg.str.startswith("CVM/SRE")],
                       ["ICVM476", "RCVM160", "ICVM400"], "other")
    qty = t["Quantidade Emitida"].map(br_float)
    vne = t["Valor Nominal na Emissao"].map(br_float)
    d = pd.DataFrame({
        "codigo": t["Codigo do Ativo"].astype(str), "issuer": t["Empresa"].astype(str),
        "cnpj8": t["CNPJ"].astype(str).str.replace(r"\D", "", regex=True).str[:8],
        "kind": kind, "contract": rate, "incent": t["Deb. Incent. (Lei 12.431)"].eq("S").astype(int),
        "issue_date": _dt(t["Data de Emissao"]), "accr_start": _dt(t["Data do Inicio da Rentabilidade"]),
        "dist_start": _dt(t["Data do Inicio da Distribuicao"]), "maturity": _dt(t["Data de Vencimento"]),
        "regime": regime, "size_brl": qty * vne, "lead": t["Coordenador Lider"].astype(str),
        "guarantee": t["Garantia/Especie"].astype(str), "emissao": t["Emissao"].astype(str),
        "serie": t["Serie"].astype(str), "classe": t["Classe"].astype(str),
    })
    return d


def main():
    ref = registry()
    # settlement = start of accrual (when present and >= issue date), else distribution start, else issue date
    st = ref["accr_start"].where(ref["accr_start"].notna(), ref["dist_start"]).fillna(ref["issue_date"])
    ref["settle"] = st.where(st >= ref["issue_date"], ref["issue_date"])
    # repeat issuer: another series of the same issuer issued >= 30 days before this one (registry, incl. excluded)
    first_iss = ref.groupby("cnpj8")["issue_date"].transform("min")
    ref["first_time"] = (ref["issue_date"] - first_iss).dt.days < 30
    n_prior = []
    g_iss = {k: np.sort(v.dropna().to_numpy()) for k, v in ref.groupby("cnpj8")["issue_date"]}
    for c, d in zip(ref["cnpj8"], ref["issue_date"]):
        arr = g_iss.get(c)
        n_prior.append(int(np.searchsorted(arr, np.datetime64(d - pd.Timedelta(days=30)))) if arr is not None and d == d else 0)
    ref["n_prior_series"] = n_prior
    ni = ref[(ref["settle"] >= "2021-04-01") & (ref["settle"] <= "2026-09-15")
             & ref["kind"].isin(["DI_SPREAD", "IPCA", "PRE"]) & ref["contract"].notna()
             & ref["classe"].str.startswith("Simples")].copy()
    print("new-issue candidates", len(ni), ni["kind"].value_counts().to_dict(), ni["regime"].value_counts().to_dict())

    # ---- grid
    g = pd.read_pickle(H.HIST / "lab_daily.pkl")[["codigo", "day", "date", "ratio", "cdi_bps", "dur", "kind", "incent",
                                                   "cnpj8", "fresh", "eligible", "resid_bps", "peer", "bench_rate",
                                                   "press_neg_30d", "rat_days_since_down"]]
    days = H.days()
    first = g.sort_values("day").groupby("codigo").head(1).set_index("codigo")
    ni = ni.join(first[["day", "date", "ratio", "cdi_bps", "dur", "resid_bps", "eligible", "bench_rate"]]
                 .add_suffix("_fp"), on="codigo")
    ni["on_grid"] = ni["day_fp"].notna()
    # decision / fair-value day: last grid day strictly before settlement
    sp = days.searchsorted(ni["settle"].to_numpy())          # first grid pos >= settle
    ni["settle_pos"] = sp
    ni["fair_pos"] = sp - 1
    ni = ni[(ni["fair_pos"] >= 5) & (ni["settle_pos"] < len(days))].copy()
    ni["fair_day"] = days[ni["fair_pos"].to_numpy()]
    ni["fp_pos"] = days.get_indexer(ni["day_fp"])
    ni["bd_to_fp"] = np.where(ni["on_grid"], ni["fp_pos"] - ni["settle_pos"], np.nan)

    # ---- duration at issue and CDI+-equivalent issue spread
    T_iss = (ni["maturity"] - ni["settle"]).dt.days / 365.25
    T_fp = (ni["maturity"] - ni["date_fp"]).dt.days / 365.25
    fac = (ni["dur_fp"] / T_fp).where(ni["on_grid"], ni["kind"].map(DUR_F))
    ni["dur_iss"] = (T_iss * fac).clip(lower=0.25)
    ni["T_iss"] = T_iss
    cur = pd.read_pickle(OUT / "curves.pkl").sort_index()
    ten = np.array([126, 252, 504, 756, 1260, 1764, 2520]) / 252
    crow = cur.reindex(ni["fair_day"], method="ffill")

    def interp(curve):
        vals = crow[[f"{curve}_{int(t * 252)}" for t in ten]].to_numpy()
        return np.array([np.interp(y, ten[np.isfinite(v)], v[np.isfinite(v)]) if np.isfinite(v).sum() >= 2 else np.nan
                         for y, v in zip(ni["dur_iss"].to_numpy(), vals)])
    pre, dic = interp("PRE"), interp("DIC")
    c = ni["contract"].to_numpy() / 100
    ni["bench_iss"] = np.select([ni["kind"].eq("IPCA"), ni["kind"].eq("PRE")], [dic, pre], np.nan)
    ni["s_iss"] = np.select([ni["kind"].eq("DI_SPREAD"), ni["kind"].eq("PRE"), ni["kind"].eq("IPCA")],
                            [c * 1e4, ((1 + c) / (1 + pre / 100) - 1) * 1e4, ((1 + c) / (1 + dic / 100) - 1) * 1e4],
                            np.nan)

    # ---- same-day peer fair value (resid framework) and issuer effect, from fair_day fresh marks only
    gf = g[g["fresh"] & g["day"].isin(set(ni["fair_day"]))]
    fair, n_peer, iss_eff, iss_n, iss_raw = [], [], [], [], []
    by_day = {d: x for d, x in gf.groupby("day")}
    for r in ni.itertuples():
        x = by_day.get(r.fair_day)
        peer = f"{r.kind}_{r.incent}"
        if x is None:
            fair.append(np.nan); n_peer.append(0); iss_eff.append(np.nan); iss_n.append(0); iss_raw.append(np.nan)
            continue
        p = x[x["peer"] == peer]
        if len(p) >= 8:
            b = pd.cut(p["dur"], BUCKETS)
            med = p.groupby(b, observed=True).agg(d=("dur", "median"), s=("cdi_bps", "median")).dropna()
            fv = float(np.interp(r.dur_iss, med["d"], med["s"])) if len(med) > 1 else float(med["s"].iloc[0])
        else:
            fv = np.nan
        fair.append(fv); n_peer.append(len(p))
        o = x[(x["cnpj8"] == r.cnpj8) & (x["codigo"] != r.codigo) & x["resid_bps"].notna()]
        n = len(o)
        iss_n.append(n)
        iss_raw.append(float(o["resid_bps"].mean()) if n else np.nan)
        iss_eff.append(float(o["resid_bps"].mean()) * n / (n + 1) if n else np.nan)   # shrunk, k = 1
    ni["fair_peer"] = fair
    ni["n_peer"] = n_peer
    ni["iss_eff"] = iss_eff
    ni["iss_n"] = iss_n
    ni["iss_resid_raw"] = iss_raw
    ni["conc_peer"] = ni["s_iss"] - ni["fair_peer"]
    ni["conc_iss"] = ni["conc_peer"] - ni["iss_eff"]             # vs issuer curve (defined if issuer has fresh bonds)
    ni["conc"] = ni["conc_iss"].where(ni["iss_n"] > 0, ni["conc_peer"])   # best available fair value

    # ---- issuer screens known at fair_day: press (any bond of the issuer, lab_daily), negative rating 365d
    pr = pd.read_pickle(H.HIST / "press_items.pkl")
    # press_items has no sentiment: use lab_daily's press_neg_30d of the issuer's bonds on fair_day (P4's screen);
    # first-time issuers have no grid rows -> screened with the harness-style 'uncovered kept' convention.
    gp = g[g["day"].isin(set(ni["fair_day"]))].groupby(["day", "cnpj8"])["press_neg_30d"].max()
    ni["press_neg_30d"] = [gp.get((d, c), np.nan) for d, c in zip(ni["fair_day"], ni["cnpj8"])]
    ni["press_any_30d"] = 0
    prc = pr.groupby("cnpj8")["date"].apply(lambda s: np.sort(s.to_numpy()))
    for i, (c, d) in enumerate(zip(ni["cnpj8"], ni["fair_day"])):
        a = prc.get(c)
        if a is not None:
            lo, hi = np.searchsorted(a, np.datetime64(d - pd.Timedelta(days=30))), np.searchsorted(a, np.datetime64(d), "right")
            ni.iloc[i, ni.columns.get_loc("press_any_30d")] = int(hi - lo)
    rt = pd.read_pickle(H.HIST / "rating_events.pkl")
    rt = rt[rt["direction"] < 0]
    rtc = rt.groupby("cnpj8")["date"].apply(lambda s: np.sort(s.to_numpy()))
    neg = []
    for c, d in zip(ni["cnpj8"], ni["fair_day"]):
        a = rtc.get(c)
        neg.append(int(a is not None and ((a <= np.datetime64(d)) & (a > np.datetime64(d - pd.Timedelta(days=365)))).any()))
    ni["rat_neg_365d"] = neg

    # ---- offer registry link: nearest offer event of the issuer in [settle-180d, settle+10d]
    of = pd.read_pickle(H.HIST / "nightly" / "alt_signals" / "offers_deb.pkl")
    ofg = {k: v.sort_values("event") for k, v in of.groupby("cnpj8")}
    cols = {"off_event": [], "off_avail": [], "off_value": [], "off_incent": [], "off_src": []}
    for c, s in zip(ni["cnpj8"], ni["settle"]):
        o = ofg.get(c)
        hit = None
        if o is not None:
            m = o[(o["event"] >= s - pd.Timedelta(days=180)) & (o["event"] <= s + pd.Timedelta(days=10))]
            if len(m):
                hit = m.iloc[(m["event"] - s).abs().argsort().iloc[0]]
        for k, f in zip(cols, ["event", "avail", "value", "incent", "src"]):
            cols[k].append(hit[f] if hit is not None else (pd.NaT if f in ("event", "avail") else np.nan))
    for k, v in cols.items():
        ni[k] = v
    ni["deal_size"] = ni.groupby(["cnpj8", "emissao"])["size_brl"].transform("sum")

    # ---- market regime at fair_day (known at close): IDA-DI excess momentum, credit-fund flows (lag 5 bdays)
    ida = H.index_excess("IDADI")
    x21 = ida.rolling(21).sum().reindex(days).ffill()
    x63 = ida.rolling(63).sum().reindex(days).ffill()
    ni["idadi_x21"] = x21.reindex(ni["fair_day"]).to_numpy()
    ni["idadi_x63"] = x63.reindex(ni["fair_day"]).to_numpy()
    af = pd.read_pickle(H.HIST / "nightly" / "fund_flows" / "agg_flows_export.pkl")
    af21 = af["af21"].reindex(days).ffill().shift(5)
    ni["af21"] = af21.reindex(ni["fair_day"]).to_numpy()
    # universe median spread (known at fair_day)
    um = g[g["eligible"]].groupby("day")["cdi_bps"].median()
    ni["univ_med"] = um.reindex(ni["fair_day"]).to_numpy()
    ni = ni.reset_index(drop=True)
    ni.to_pickle(OUT / "new_issues.pkl")
    print(ni[["s_iss", "fair_peer", "conc_peer", "conc_iss", "conc", "bd_to_fp", "n_peer", "iss_n"]].describe().T)
    print("on grid", ni["on_grid"].mean(), "offer linked", ni["off_event"].notna().mean(), "first-time", ni["first_time"].mean())


if __name__ == "__main__":
    main()
