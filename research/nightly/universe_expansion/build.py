"""Build point-in-time datasets for the universe-expansion study.

Outputs (data/history/nightly/universe_expansion/):
  index_sleeves.pkl   daily excess over CDI on the harness grid for ANBIMA indices (IMA-B5, IMA-B, IMA-B5+, IRF-M,
                      IMA-S, IDKA-IPCA-2A, IDA-IPCA, IDA-DI, IDA-Geral). Index closes <= t.
  fii_daily.pkl       {'ret': DataFrame grid x ticker of daily total return (close + distributions reinvested at the
                      ex-date close), 'px': close, 'adtv': 63d median traded value}
  fii_panel.pkl       monthly decision panel (day, ticker, dy12, pnav, nav_chg6, nav_chg12, mom126, vol63, adtv, crish)
                      PIT rule: distributions counted by ex-date <= day; NAV from CVM FII monthly report is used
                      only when Data_Entrega <= day (delivery date, ~15 days after month end).
  fidc_panel.pkl      monthly senior-class FIDC panel (CVM informe mensal): ref month, class, monthly return, PL,
                      delinquency ratio, subordination; available_date = ref month end + 45 days (conservative:
                      open data has no delivery date; regulatory deadline is 15 days after month end).
  crisec_panel.pkl    CRI/CRA monthly securitisation reports (CVM): certificate, class, rate text, value, status
                      (Adimplente / Em atraso), Data_Entrega (PIT date) + debtor/cedent CNPJs.
"""
from __future__ import annotations

import pickle
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from research.nightly import harness as H  # noqa: E402

RAW = Path("data/history/nightly/universe_expansion/raw")
OUT = Path("data/history/nightly/universe_expansion")


def _num(s):
    return pd.to_numeric(s, errors="coerce")


# ------------------------------------------------------------------ ANBIMA index sleeves
def index_sleeves() -> pd.DataFrame:
    dd = H.days()
    cdi = H.cdi_daily()
    out = {}
    for nm in ["IMAB5", "IMAB", "IMAB5MAIS", "IRFM", "IMAS", "IDKAIPCA2A", "IDAIPCA", "IDADI", "IDAGERAL"]:
        df = pd.read_excel(RAW / f"anbima_{nm}.xls")
        df.columns = [str(c).strip() for c in df.columns]
        dc = next(c for c in df.columns if "Data" in c)
        ic = next(c for c in df.columns if "mero" in c)
        lv = pd.Series(_num(df[ic]).to_numpy(), index=pd.to_datetime(df[dc], dayfirst=True, errors="coerce"))
        lv = lv[lv.index.notna()].dropna().sort_index()
        lv = lv[~lv.index.duplicated()]
        lv = lv.reindex(lv.index.union(dd)).ffill(limit=5).reindex(dd)
        out[nm] = (lv.pct_change() - cdi).fillna(0)
    return pd.DataFrame(out)


# ------------------------------------------------------------------ listed CRI funds (FII de papel)
def _fii_cvm():
    G, C, A = [], [], []
    for y in range(2020, 2027):
        z = zipfile.ZipFile(RAW / f"inf_mensal_fii_{y}.zip")
        for n in z.namelist():
            df = pd.read_csv(z.open(n), sep=";", encoding="latin1", dtype=str, on_bad_lines="skip")
            (G if "geral" in n else A if "ativo" in n else C).append(df)
    G, C, A = pd.concat(G), pd.concat(C), pd.concat(A)
    key = ["CNPJ_Fundo_Classe", "Data_Referencia"]
    for d in (G, C, A):
        d["Versao"] = _num(d["Versao"])
    # PIT: keep the FIRST version delivered (restated versions arrive later; we use the delivery date of v1)
    G = G.sort_values("Versao").drop_duplicates(key, keep="first")
    C = C.sort_values("Versao").drop_duplicates(key, keep="first")
    A = A.sort_values("Versao").drop_duplicates(key, keep="first")
    m = G[key + ["Data_Entrega", "Codigo_ISIN", "Segmento_Atuacao"]].merge(
        C[key + ["Valor_Patrimonial_Cotas", "Patrimonio_Liquido"]], on=key, how="inner").merge(
        A[key + ["CRI", "CRI_CRA", "Total_Investido"]], on=key, how="left")
    m["ref"] = pd.to_datetime(m["Data_Referencia"])
    m["avail"] = pd.to_datetime(m["Data_Entrega"], errors="coerce")
    m["nav"] = _num(m["Valor_Patrimonial_Cotas"])
    m["pl"] = _num(m["Patrimonio_Liquido"])
    m["crish"] = (_num(m["CRI"]).fillna(0) + _num(m["CRI_CRA"]).fillna(0)) / _num(m["Total_Investido"])
    m["tick"] = m["Codigo_ISIN"].str[2:6] + "11"
    return m[["CNPJ_Fundo_Classe", "tick", "ref", "avail", "nav", "pl", "crish", "Segmento_Atuacao"]]


def fii_build():
    raw = pickle.load(open(RAW / "listed_credit_brapi.pkl", "rb"))
    dd = H.days()
    rets, pxs, vals, divs = {}, {}, {}, {}
    for t, v in raw.items():
        if not v.get("hist") or v.get("group") in ("CRA", "INFRA"):
            continue  # Fiagro history < 1y on brapi; FI-Infra distributions incomplete on brapi -> excluded
        h = pd.DataFrame(v["hist"])
        h["date"] = pd.to_datetime(h["date"], unit="s").dt.tz_localize(None).dt.normalize()
        h = h.dropna(subset=["close"]).drop_duplicates("date", keep="last").set_index("date").sort_index()
        h = h[h["close"] > 0]
        if len(h) < 250:
            continue
        dv = pd.DataFrame(v["divs"])
        s_div = pd.Series(0.0, index=h.index)
        if len(dv):
            dv = dv[dv["label"].fillna("").str.upper().str.contains("REND|DIVID|JCP|AMORT")].copy()
            ex = pd.to_datetime(dv["lastDatePrior"].str[:10], errors="coerce")
            dv["ex"] = ex
            dv = dv.dropna(subset=["ex"])
            # ex-date = first trading day AFTER the last date with rights
            pos = h.index.searchsorted(dv["ex"].to_numpy(), side="right")
            dv = dv[pos < len(h)]
            pos = pos[pos < len(h)]
            add = pd.Series(dv["rate"].astype(float).to_numpy(), index=h.index[pos]).groupby(level=0).sum()
            s_div = s_div.add(add, fill_value=0)
        # brapi closes are split-adjusted for most tickers but distributions are not: a distribution worth >3% of
        # the price in one payment is rescaled by the split factor that brings it closest to the ticker's typical
        # (median) distribution yield.
        yl = (s_div / h["close"]).where(s_div > 0)
        typ = yl[yl < 0.03].median() if (yl < 0.03).sum() >= 3 else 0.01
        for dt_ in yl.index[(yl > 0.03).fillna(False).to_numpy()]:
            f = min((10, 5, 4, 3, 2, 100, 1000), key=lambda k: abs(np.log(yl[dt_] / k / typ)))
            if abs(np.log(yl[dt_] / f / typ)) < abs(np.log(yl[dt_] / typ)):
                s_div[dt_] /= f
        c = h["close"]
        # splits / reverse splits (brapi closes are not split-adjusted): a >2.5x one-day jump. Put everything in
        # today's units: prices and distributions before the split date are multiplied by the jump ratio.
        jr = (c / c.shift(1)).fillna(1.0)
        spl = (jr < 0.4) | (jr > 2.5)
        fac = pd.Series(1.0, index=c.index)
        for dt_ in c.index[spl.to_numpy()]:
            fac[c.index < dt_] *= jr[dt_]
        r = (c + s_div) / c.shift(1) - 1
        r[spl] = 0.0
        c = c * fac
        s_div = s_div * fac
        r[(r.abs() > 0.25)] = np.nan  # obvious bad prints / unadjusted splits
        rets[t] = r
        pxs[t] = c
        vals[t] = (h["close"] * h["volume"]).rolling(63, min_periods=20).median()
        divs[t] = s_div
    R = pd.DataFrame(rets).reindex(dd)
    # drop duplicated listings (identical return series, e.g. BTCR11/BTCI11 after a merger)
    dup = R.T.duplicated(keep="last")
    R = R.loc[:, ~dup.to_numpy()]
    pxs = {k: v for k, v in pxs.items() if k in R.columns}
    PX = pd.DataFrame(pxs).reindex(dd).ffill(limit=5)
    V = pd.DataFrame(vals).reindex(dd).ffill(limit=5)
    D = pd.DataFrame(divs).reindex(dd).fillna(0)
    cdi = H.cdi_daily()
    X = R.sub(cdi, axis=0)  # daily excess over CDI (NaN when not trading / not listed)
    return {"ret": R, "ex": X, "px": PX, "adtv": V, "div": D}


def fii_panel(F, cvm):
    dd = H.days()
    dec = [dd[i] for i in H._decision_positions("M")]
    R, PX, V, D = F["ret"], F["px"], F["adtv"], F["div"]
    tr = (1 + R.fillna(0)).cumprod()
    div365 = D.rolling(252, min_periods=200).sum()
    vol63 = R.rolling(63, min_periods=40).std() * np.sqrt(252)
    rows = []
    cvm = cvm.dropna(subset=["avail", "nav"])
    cvm = cvm[cvm["nav"] > 0]
    for d in dec:
        i = dd.get_loc(d)
        known = cvm[cvm["avail"] <= d].sort_values("ref")
        last = known.groupby("tick").tail(13)
        navs = last.groupby("tick")["nav"].apply(list)
        crish = last.groupby("tick")["crish"].last()
        pl = last.groupby("tick")["pl"].last()
        for t in R.columns:
            px = PX[t].iloc[i]
            if not np.isfinite(px) or not np.isfinite(V[t].iloc[i]):
                continue
            nv = navs.get(t)
            row = {"day": d, "ticker": t, "px": px, "adtv": V[t].iloc[i], "dy12": div365[t].iloc[i] / px,
                   "mom126": tr[t].iloc[i] / tr[t].iloc[max(i - 126, 0)] - 1 if i >= 126 else np.nan,
                   "vol63": vol63[t].iloc[i], "crish": crish.get(t, np.nan), "pl": pl.get(t, np.nan)}
            if nv:
                nv = list(nv)
                for k in range(len(nv) - 1, 0, -1):  # NAV per quota splits: rescale older NAVs
                    q = nv[k] / nv[k - 1] if nv[k - 1] else np.nan
                    if q < 0.4 or q > 2.5:
                        nv[:k] = [x * q for x in nv[:k]]
                pn = px / nv[-1]
                row["pnav"] = pn if 0.3 < pn < 2.0 else np.nan
                row["nav_chg6"] = nv[-1] / nv[-7] - 1 if len(nv) >= 7 else np.nan
                row["nav_chg12"] = nv[-1] / nv[-13] - 1 if len(nv) >= 13 else np.nan
            rows.append(row)
    P = pd.DataFrame(rows)
    return P


# ------------------------------------------------------------------ FIDC (CVM informe mensal)
def fidc_build():
    files = sorted(RAW.glob("inf_mensal_fidc_*.zip"))
    X3, X2, IV, I = [], [], [], []
    for f in files:
        z = zipfile.ZipFile(f)
        for n in z.namelist():
            tab = re.search(r"tab_([IVX_0-9]+)_\d{6}", n)
            if not tab:
                continue
            tb = tab.group(1)
            if tb not in ("X_3", "X_2", "IV", "I"):
                continue
            if tb == "I":
                cols = ["CNPJ_FUNDO_CLASSE", "CNPJ_FUNDO", "DT_COMPTC", "CONDOM", "FUNDO_EXCLUSIVO", "TAB_I2_VL_CARTEIRA",
                        "TAB_I2A_VL_DIRCRED_RISCO", "TAB_I2A2_VL_CRED_VENC_INAD", "TAB_I2A3_VL_CRED_INAD",
                        "TAB_I2A11_VL_REDUCAO_RECUP"]
                df = pd.read_csv(z.open(n), sep=";", encoding="latin1", dtype=str,
                                 usecols=lambda c: c in cols, on_bad_lines="skip")
            else:
                df = pd.read_csv(z.open(n), sep=";", encoding="latin1", dtype=str, on_bad_lines="skip")
            if "CNPJ_FUNDO" in df.columns and "CNPJ_FUNDO_CLASSE" not in df.columns:
                df = df.rename(columns={"CNPJ_FUNDO": "CNPJ_FUNDO_CLASSE"})
            {"X_3": X3, "X_2": X2, "IV": IV, "I": I}[tb].append(df)
        print(f.name, flush=True)
    X3, X2, IV, I = (pd.concat(x, ignore_index=True) for x in (X3, X2, IV, I))
    for d in (X3, X2, IV, I):
        d["ref"] = pd.to_datetime(d["DT_COMPTC"]).dt.to_period("M").dt.to_timestamp()
    X3["ret"] = _num(X3["TAB_X_VL_RENTAB_MES"]) / 100
    cls = X3["TAB_X_CLASSE_SERIE"].fillna("")
    X3["senior"] = cls.str.contains(r"S.nior|Senior", case=False, regex=True)
    X2["val"] = _num(X2["TAB_X_QT_COTA"]) * _num(X2["TAB_X_VL_COTA"])
    X2["senior"] = X2["TAB_X_CLASSE_SERIE"].fillna("").str.contains(r"S.nior|Senior", case=False, regex=True)
    X2["val_sen"] = X2["val"].where(X2["senior"], 0.0)
    sub = X2.groupby(["CNPJ_FUNDO_CLASSE", "ref"])[["val_sen", "val"]].sum().rename(columns={"val": "val_tot"})
    sub["subord"] = 1 - sub["val_sen"] / sub["val_tot"].replace(0, np.nan)
    IV["pl"] = _num(IV["TAB_IV_A_VL_PL"])
    I["cart"] = _num(I["TAB_I2_VL_CARTEIRA"])
    I["dc"] = _num(I["TAB_I2A_VL_DIRCRED_RISCO"])
    I["inad"] = _num(I["TAB_I2A2_VL_CRED_VENC_INAD"]).fillna(0) + _num(I["TAB_I2A3_VL_CRED_INAD"]).fillna(0)
    I["pdd"] = _num(I["TAB_I2A11_VL_REDUCAO_RECUP"]).abs()
    I["inad_ratio"] = I["inad"] / I["dc"].replace(0, np.nan)
    I["pdd_ratio"] = I["pdd"] / I["dc"].replace(0, np.nan)
    key = ["CNPJ_FUNDO_CLASSE", "ref"]
    sen = X3[X3["senior"]].groupby(key + ["TAB_X_CLASSE_SERIE"])["ret"].first().reset_index()
    P = sen.merge(IV.drop_duplicates(key)[key + ["pl"]], on=key, how="left") \
        .merge(I.drop_duplicates(key)[key + ["CONDOM", "FUNDO_EXCLUSIVO", "inad_ratio", "pdd_ratio", "cart"]],
               on=key, how="left") \
        .merge(sub.reset_index()[key + ["subord"]], on=key, how="left")
    P["avail"] = P["ref"] + pd.offsets.MonthEnd(0) + pd.Timedelta(days=45)
    P["cls"] = P["CNPJ_FUNDO_CLASSE"] + "|" + P["TAB_X_CLASSE_SERIE"].fillna("")
    return P


# ------------------------------------------------------------------ CRI/CRA securitisation reports
def crisec_build():
    C, G, D = [], [], []
    for kind in ("cri", "cra"):
        for y in range(2020, 2027):
            p = RAW / f"inf_mensal_{kind}_{y}.zip"
            if not p.exists():
                continue
            z = zipfile.ZipFile(p)
            for n in z.namelist():
                if "_classe_" in n:
                    df = pd.read_csv(z.open(n), sep=";", encoding="latin1", dtype=str, on_bad_lines="skip"); df["kind"] = kind; C.append(df)
                elif "_geral_" in n:
                    df = pd.read_csv(z.open(n), sep=";", encoding="latin1", dtype=str, on_bad_lines="skip",
                                     usecols=lambda c: c in ("Codigo_Identificacao_Certificado", "Data_Referencia",
                                                             "Versao", "Data_Entrega", "Companhia_Emissora",
                                                             "Tipo_Segmento", "Detalhamento_Lastro"))
                    df["kind"] = kind; G.append(df)
                elif "cedente_devedor" in n:
                    df = pd.read_csv(z.open(n), sep=";", encoding="latin1", dtype=str, on_bad_lines="skip"); df["kind"] = kind; D.append(df)
    C, G, D = pd.concat(C, ignore_index=True), pd.concat(G, ignore_index=True), pd.concat(D, ignore_index=True)
    key = ["Codigo_Identificacao_Certificado", "Data_Referencia"]
    for d in (C, G):
        d["Versao"] = _num(d["Versao"])
    G = G.sort_values("Versao").drop_duplicates(key, keep="first")
    C = C.sort_values("Versao").drop_duplicates(key + ["Classe", "Numero_Serie"], keep="first")
    C = C.merge(G[key + ["Data_Entrega", "Tipo_Segmento"]], on=key, how="left")
    C["ref"] = pd.to_datetime(C["Data_Referencia"])
    C["avail"] = pd.to_datetime(C["Data_Entrega"], errors="coerce")
    for c in ("Valor_Certificados", "Rendimentos", "Amortizacoes", "Quantidade_Certificados"):
        C[c] = _num(C[c])
    C["arrears"] = C["Situacao"].fillna("").str.contains("atraso", case=False)
    D["cnpj8"] = D["CNPJ"].fillna("").str.replace(r"\D", "", regex=True).str.zfill(14).str[:8]
    D["ref"] = pd.to_datetime(D["Data_Referencia"])
    return C, D


def main(parts=("index", "fii", "fidc", "crisec")):
    OUT.mkdir(parents=True, exist_ok=True)
    if "index" in parts:
        print("index sleeves", flush=True); IS = index_sleeves(); IS.to_pickle(OUT / "index_sleeves.pkl")
    if "fii" in parts:
        print("fii", flush=True); F = fii_build(); pickle.dump(F, open(OUT / "fii_daily.pkl", "wb"))
        cvm = _fii_cvm(); cvm.to_pickle(OUT / "fii_cvm.pkl")
        P = fii_panel(F, cvm); P.to_pickle(OUT / "fii_panel.pkl")
        print("fii panel", P.shape, P["day"].nunique(), flush=True)
    if "fidc" in parts:
        print("fidc", flush=True); FD = fidc_build(); FD.to_pickle(OUT / "fidc_panel.pkl")
        print("fidc", FD.shape, FD["cls"].nunique(), flush=True)
    if "crisec" in parts:
        print("crisec", flush=True); C, D = crisec_build()
        C.to_pickle(OUT / "crisec_panel.pkl"); D.to_pickle(OUT / "crisec_parties.pkl")
        print("crisec", C.shape, C["Codigo_Identificacao_Certificado"].nunique(), D.shape, flush=True)


if __name__ == "__main__":
    main(tuple(sys.argv[1:]) or ("index", "fii", "fidc", "crisec"))
