"""Coverage / feasibility numbers for sleeves that could NOT be backtested properly:
  * bank paper (LF/CDB/LCI/LCA) and CRI/CRA holdings in CVM fund portfolios (CDA BLC_5 / BLC_6), one month, read-only
    from the fund_flows agent's raw cache (no market prices: positions carry contract rate and fund marks, no codes
    for CRI/CRA in BLC_6 -> no clean security id).
  * offshore Brazil corporates: FRED ICE BofA EM / LatAm corporate indices (only a rolling ~3y window is public).
  * CRI/CRA securitisation reports (CVM) and FIDC informe mensal coverage.
Writes research/nightly/universe_expansion/feasibility.json
"""
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path("data/history/nightly/universe_expansion/raw")
CDA = Path("data/history/nightly/fund_flows/raw/cda_fi_202506.zip")
D = Path("data/history/nightly/universe_expansion")
out = {}

# ---- CDA bank paper + CRI/CRA holdings
if CDA.exists():
    z = zipfile.ZipFile(CDA)
    b5 = pd.read_csv(z.open("cda_fi_BLC_5_202506.csv"), sep=";", encoding="latin1", dtype=str, on_bad_lines="skip",
                     usecols=["TP_ATIVO", "CNPJ_EMISSOR", "VL_MERC_POS_FINAL", "PR_INDEXADOR_POSFX", "PR_CUPOM_POSFX",
                              "CD_INDEXADOR_POSFX", "DT_VENC"])
    b5["v"] = pd.to_numeric(b5["VL_MERC_POS_FINAL"], errors="coerce")
    g = b5.groupby("TP_ATIVO").agg(n_pos=("v", "size"), vol_bn=("v", lambda x: x.sum() / 1e9),
                                   issuers=("CNPJ_EMISSOR", "nunique"))
    lf = b5[b5["TP_ATIVO"].str.contains("Letra Financeira", na=False) & (b5["CD_INDEXADOR_POSFX"] == "DI1")]
    cup = pd.to_numeric(lf["PR_CUPOM_POSFX"], errors="coerce")
    pct = pd.to_numeric(lf["PR_INDEXADOR_POSFX"], errors="coerce")
    out["cda_202506_blc5_bank_paper"] = json.loads(g.sort_values("vol_bn", ascending=False).round(2).to_json(orient="index"))
    out["lf_di_plus_spread_pct_quantiles"] = cup[(pct == 100) & (cup > 0)].quantile([.1, .25, .5, .75, .9]).round(3).to_dict()
    b6 = pd.read_csv(z.open("cda_fi_BLC_6_202506.csv"), sep=";", encoding="latin1", dtype=str, on_bad_lines="skip",
                     usecols=["TP_ATIVO", "CPF_CNPJ_EMISSOR", "VL_MERC_POS_FINAL", "DT_VENC"])
    b6["v"] = pd.to_numeric(b6["VL_MERC_POS_FINAL"], errors="coerce")
    g6 = b6.groupby("TP_ATIVO").agg(n_pos=("v", "size"), vol_bn=("v", lambda x: x.sum() / 1e9),
                                    issuers=("CPF_CNPJ_EMISSOR", "nunique"))
    out["cda_202506_blc6_credit"] = json.loads(g6.sort_values("vol_bn", ascending=False).head(12).round(2).to_json(orient="index"))

# ---- FRED ICE BofA
fr = {}
for sid in ["BAMLEMCBPITRIV", "BAMLEMRLCRPILATRIV", "BAMLEMHBHYCRPITRIV", "BAMLEMIBHGCRPITRIV", "BAMLEMRLCRPILAOAS"]:
    p = RAW / f"fred_{sid}.csv"
    if p.exists():
        s = pd.read_csv(p, index_col=0, parse_dates=True).iloc[:, 0]
        s = pd.to_numeric(s, errors="coerce").dropna()
        fr[sid] = {"first": str(s.index.min().date()), "last": str(s.index.max().date()), "n": int(len(s))}
        if sid.endswith("TRIV"):
            y = s[(s.index >= "2024-01-01") & (s.index < "2026-01-01")]
            fr[sid]["usd_tr_2024_25_ann_%"] = round(float((y.iloc[-1] / y.iloc[0]) ** (252 / len(y)) - 1) * 100, 2)
out["fred_ice_bofa"] = fr
out["fred_note"] = ("ICE BofA series on FRED are truncated to a rolling ~3y window (first obs 2023-09-29): no 2022-23 "
                    "history -> no pre-2024 backtest of an offshore sleeve is possible from free data; bond-level "
                    "eurobond prices (TRACE does not cover 144A/RegS EM corporates well) are not free.")

# ---- CRI/CRA + FIDC coverage
C = pd.read_pickle(D / "crisec_panel.pkl")
C = C[C["ref"] >= "2022-01-01"]
lag = (C["avail"] - C["ref"]).dt.days
out["crisec"] = {"certificates_2022_26": int(C["Codigo_Identificacao_Certificado"].nunique()),
                 "by_year_kind": json.loads(C.groupby([C["ref"].dt.year.astype(str), "kind"])["Codigo_Identificacao_Certificado"].nunique().unstack().to_json()),
                 "median_filing_lag_days": float(lag.median()),
                 "share_rows_in_arrears": round(float(C["arrears"].mean()), 4),
                 "share_rows_value_unchanged_vs_prev_month": None,
                 "note": "Values are the securitiser's book (curve) values, often stale for months; 'Rentabilidade' "
                         "is unusable (garbage units). No market price -> no return-based backtest. Useful as a "
                         "credit-event registry (arrears) with a ~58-day filing lag."}
C2 = C.sort_values("ref")
chg = C2.groupby(["Codigo_Identificacao_Certificado", "Classe", "Numero_Serie"])["Valor_Certificados"].diff()
out["crisec"]["share_rows_value_unchanged_vs_prev_month"] = round(float((chg == 0).mean()), 3)
F = pd.read_pickle(D / "fidc_panel.pkl")
F = F[F["ref"] >= "2022-01-01"]
out["fidc"] = {"senior_classes_2022_26": int(F["cls"].nunique()),
               "share_rows_ret_not_reported(0)": round(float((F["ret"] == 0).mean()), 3),
               "share_closed_end": round(float((F["CONDOM"] == "FECHADO").mean()), 3),
               "share_exclusive": round(float((F["FUNDO_EXCLUSIVO"] == "S").mean()), 3),
               "note": "Self-reported monthly quota returns; closed-end quotas mostly untradeable; availability "
                       "assumed ref month-end + 45d (no delivery date in open data)."}
P = pd.read_pickle(D / "fii_panel.pkl")
P = P[(P["day"] >= "2022-01-01") & (P["day"] < "2026-01-01")]
out["fii"] = {"tickers_ever": int(P["ticker"].nunique()),
              "avg_liquid_names(adtv>=500k)": round(float(P[P["adtv"] >= 5e5].groupby("day").size().mean()), 1),
              "pnav_coverage": round(float(P["pnav"].notna().mean()), 3),
              "note": "B3 closes + distributions from brapi; NAV from CVM FII informe mensal by Data_Entrega. "
                      "Universe = tickers still resolvable on brapi today (dead/delisted funds missing)."}
cand = pd.read_csv(RAW / "cvm_credit_fii_candidates.csv")
out["fii"]["cvm_credit_fii_listed_candidates"] = int(len(cand))
json.dump(out, open("research/nightly/universe_expansion/feasibility.json", "w"), indent=1, default=str)
print(json.dumps(out, indent=1, default=str)[:4000])
