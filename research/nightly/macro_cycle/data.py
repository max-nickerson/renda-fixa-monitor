"""PIT macro / credit-cycle panel for the macro_cycle nightly study.

Builds a DAILY business-day panel (2009 -> today) where every value in row d is known at the CLOSE of d
(availability-dated, not reference-dated). Cached at data/history/nightly/macro_cycle/macro_daily.pkl.

Sources (all public, keyless):
  * Yahoo chart API: ^BVSP, EWZ, ^VIX, HYG, IEF, EMB, BRL=X            (US listings: shifted +1 bday)
  * FRED fredgraph csv: VIXCLS, BAA10Y, DGS10, DGS2, DCOILBRENTEU, BAMLH0A0HYM2 (3y only)  (+1 bday)
  * BCB SGS: 1 PTAX (same day, published ~13h), 432 Selic target (effective date), 12 CDI
  * IPCA actual: research commodity panel (data/history/commodities.pkl, available_date = IBGE release + 1d)
  * BCB Olinda Focus: monthly IPCA median expectations (survey 'Data' = Friday, published Monday -> +3 cal days)
  * Tesouro Direto (Tesouro Transparente) morning rates: LTN/NTN-F -> nominal 1y/5y, NTN-B -> real 5y (+1 day)
  * CVM offers registry (oferta_distribuicao + oferta_resolucao_160): debenture issuance R$ by registration date
  * ANBIMA IDA-DI / IDA-IPCA / IDA-Geral (rfmonitor.history.ida, read-only cached xlsx) -> excess momentum
  * Market press counts (data/history/press_market.pkl: 'mkt_rj' judicial recovery news, 'mkt_calote')
"""
from __future__ import annotations

import io
import json
import time
import zipfile
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
CACHE = ROOT / "data" / "history" / "nightly" / "macro_cycle"
RAW = CACHE / "raw"
RAW.mkdir(parents=True, exist_ok=True)
HDR = {"User-Agent": "Mozilla/5.0"}


def _get(url, tries=4, timeout=60):
    last = None
    for k in range(tries):
        try:
            r = httpx.get(url, headers=HDR, timeout=timeout, follow_redirects=True)
            if r.status_code == 200:
                return r
            last = r.status_code
        except Exception as e:  # noqa
            last = type(e).__name__
        time.sleep(3 * (k + 1))
    raise RuntimeError(f"GET failed {url[:80]}: {last}")


def _cached(name, fetch, max_age_h=48):
    p = RAW / name
    if p.exists() and time.time() - p.stat().st_mtime < max_age_h * 3600:
        return p.read_bytes()
    b = fetch()
    p.write_bytes(b)
    return b


# ------------------------------------------------------------------ fetchers
def yahoo(sym: str) -> pd.Series:
    def f():
        u = (f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}?period1=1199145600"
             f"&period2={int(time.time())}&interval=1d")
        return _get(u).content
    j = json.loads(_cached(f"yahoo_{sym.replace('^', '_').replace('=', '_')}.json", f))
    r = j["chart"]["result"][0]
    ts = pd.to_datetime(r["timestamp"], unit="s").normalize()
    ind = r["indicators"]
    px = ind["adjclose"][0]["adjclose"] if "adjclose" in ind else ind["quote"][0]["close"]
    s = pd.Series(px, index=ts, dtype=float).dropna()
    return s[~s.index.duplicated(keep="last")].sort_index()


def fred(sid: str) -> pd.Series:
    b = _cached(f"fred_{sid}.csv", lambda: _get(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}").content)
    d = pd.read_csv(io.BytesIO(b))
    d.columns = ["date", "v"]
    return pd.Series(pd.to_numeric(d["v"], errors="coerce").to_numpy(), index=pd.to_datetime(d["date"])).dropna()


def sgs(code: int, start="01/01/2008") -> pd.Series:
    def f():
        rows = []
        for y0 in range(2008, pd.Timestamp.today().year + 1, 9):
            y1 = min(y0 + 8, pd.Timestamp.today().year)
            u = (f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{code}/dados?formato=json"
                 f"&dataInicial=01/01/{y0}&dataFinal=31/12/{y1}")
            for k in range(5):
                try:
                    rows += _get(u).json()
                    break
                except Exception:
                    time.sleep(4 * (k + 1))
        return json.dumps(rows).encode()
    rows = json.loads(_cached(f"sgs_{code}.json", f))
    d = pd.DataFrame(rows)
    s = pd.Series(pd.to_numeric(d["valor"], errors="coerce").to_numpy(), index=pd.to_datetime(d["data"], dayfirst=True))
    return s[~s.index.duplicated()].sort_index().dropna()


def focus_ipca_monthly() -> pd.DataFrame:
    def f():
        out, skip = [], 0
        while True:
            u = ("https://olinda.bcb.gov.br/olinda/servico/Expectativas/versao/v1/odata/ExpectativaMercadoMensais?"
                 "$filter=Indicador%20eq%20'IPCA'%20and%20baseCalculo%20eq%200%20and%20Data%20ge%20'2008-06-01'"
                 f"&$select=Data,DataReferencia,Mediana&$format=json&$top=20000&$skip={skip}")
            v = _get(u, timeout=120).json()["value"]
            out += v
            if len(v) < 20000:
                break
            skip += 20000
        return json.dumps(out).encode()
    d = pd.DataFrame(json.loads(_cached("focus_ipca_m.json", f, max_age_h=24 * 7)))
    d["Data"] = pd.to_datetime(d["Data"])
    d["ref"] = pd.to_datetime("01/" + d["DataReferencia"], format="%d/%m/%Y")
    return d


def tesouro() -> pd.DataFrame:
    url = ("https://www.tesourotransparente.gov.br/ckan/dataset/df56aa42-484a-4a59-8184-7676580c81e3/"
           "resource/796d2059-14e9-44e3-80c9-2d9e30b405c1/download/PrecoTaxaTesouroDireto.csv")
    b = _cached("tesouro_direto.csv", lambda: _get(url, timeout=300).content, max_age_h=24 * 7)
    df = pd.read_csv(io.BytesIO(b), sep=";", decimal=",", encoding="latin1")
    df.columns = ["tipo", "venc", "date", "taxa_c", "taxa_v", "pu_c", "pu_v", "pu_base"][:len(df.columns)] + \
        list(df.columns[8:])
    df["venc"] = pd.to_datetime(df["venc"], format="%d/%m/%Y")
    df["date"] = pd.to_datetime(df["date"], format="%d/%m/%Y")
    return df


def issuance() -> pd.DataFrame:
    b = _cached("oferta_distribuicao.zip",
                lambda: _get("https://dados.cvm.gov.br/dados/OFERTA/DISTRIB/DADOS/oferta_distribuicao.zip", timeout=300).content,
                max_age_h=24 * 7)
    z = zipfile.ZipFile(io.BytesIO(b))
    a = pd.read_csv(z.open("oferta_distribuicao.csv"), sep=";", encoding="latin1", low_memory=False)
    a = a[a["Tipo_Ativo"].astype(str).str.upper().str.contains("DEB")]
    da = pd.to_datetime(a["Data_Registro_Oferta"], errors="coerce").fillna(
        pd.to_datetime(a["Data_Dispensa_Oferta"], errors="coerce")).fillna(pd.to_datetime(a["Data_Inicio_Oferta"], errors="coerce"))
    A = pd.DataFrame({"date": da, "value": pd.to_numeric(a["Valor_Total"], errors="coerce")})
    c = pd.read_csv(z.open("oferta_resolucao_160.csv"), sep=";", encoding="latin1", low_memory=False)
    c = c[c["Valor_Mobiliario"].astype(str).str.upper().str.contains("DEB")]
    c = c[~c["Status_Requerimento"].astype(str).str.upper().str.contains("INDEFER|CANCEL|DESIST")]
    C = pd.DataFrame({"date": pd.to_datetime(c["Data_Registro"], errors="coerce"),
                      "value": pd.to_numeric(c["Valor_Total_Registrado"], errors="coerce")})
    out = pd.concat([A, C]).dropna()
    return out[(out["value"] > 0) & (out["value"] < 5e10)]


# ------------------------------------------------------------------ builders
def _bday_index(start="2008-06-01"):
    return pd.bdate_range(start, pd.Timestamp.today().normalize())


def _on(s: pd.Series, idx, lag_bdays=0):
    """align a dated series on business days (ffill), then delay by lag_bdays."""
    s = s[~s.index.duplicated(keep="last")].sort_index()
    x = s.reindex(s.index.union(idx)).ffill().reindex(idx)
    return x.shift(lag_bdays) if lag_bdays else x


def _curve_rates(td: pd.DataFrame) -> pd.DataFrame:
    rows = {}
    nom = td[td["tipo"].isin(["Tesouro Prefixado", "Tesouro Prefixado com Juros Semestrais"])]
    real = td[td["tipo"].isin(["Tesouro IPCA+", "Tesouro IPCA+ com Juros Semestrais"])]
    for nm, g0, ts in (("pre", nom, (1, 3, 5)), ("real", real, (5,))):
        for d, g in g0.groupby("date"):
            t = ((g["venc"] - d).dt.days / 365.25).to_numpy()
            y = g["taxa_c"].to_numpy(float)
            ok = (t > 0.2) & np.isfinite(y) & (y > 0)
            if ok.sum() < 2:
                continue
            o = np.argsort(t[ok])
            for T in ts:
                rows.setdefault(d, {})[f"{nm}{T}y"] = float(np.interp(T, t[ok][o], y[ok][o]))
    return pd.DataFrame.from_dict(rows, orient="index").sort_index()


def build(force=False) -> pd.DataFrame:
    path = CACHE / "macro_daily.pkl"
    if path.exists() and not force and time.time() - path.stat().st_mtime < 24 * 3600:
        return pd.read_pickle(path)
    idx = _bday_index()
    F = pd.DataFrame(index=idx)
    src = {}

    # --- ANBIMA IDA (published after close of d -> known at close d for the next-day trade; harness lags overlay 1d)
    def ida(nm):   # read the cached ANBIMA xlsx read-only (no refresh -> no writes outside our folder)
        x = pd.read_excel(ROOT / "data" / "history" / f"{nm}.xlsx")
        x.columns = [str(c) for c in x.columns]
        dc = next(c for c in x.columns if "Data" in c)
        ic = next(c for c in x.columns if "mero" in c)
        return pd.DataFrame({"index": pd.to_numeric(x[ic], errors="coerce").to_numpy()},
                            index=pd.to_datetime(x[dc], dayfirst=True, errors="coerce")).dropna().sort_index()
    print("ida/cdi...", flush=True)
    cdi = sgs(12)                                   # CDI % per day
    cdi_d = _on(cdi / 100, idx).fillna(0)
    F["cdi_ann"] = ((1 + cdi_d) ** 252 - 1) * 100
    for nm in ("IDADI", "IDAIPCA", "IDAGERAL"):
        try:
            lv = ida(nm)["index"]
        except Exception:
            continue
        lvl = _on(lv, idx)
        rx = lvl.pct_change(fill_method=None).fillna(0) - cdi_d
        F[f"{nm}_rx1"] = rx
        for w in (5, 21, 63, 126, 252):
            F[f"{nm}_x{w}"] = rx.rolling(w).sum()
        cum = (1 + rx).cumprod()
        F[f"{nm}_dd252"] = cum / cum.rolling(252, min_periods=20).max() - 1
        F[f"{nm}_vol63"] = rx.rolling(63).std() * np.sqrt(252)
        src[nm] = str(lv.index.min().date())

    # --- Brazil markets
    ib = yahoo("^BVSP")
    ibx = _on(ib, idx).pct_change(fill_method=None).fillna(0) - cdi_d
    for w in (21, 63, 126):
        F[f"ibov_x{w}"] = ibx.rolling(w).sum()
    F["ibov_dd252"] = _on(ib, idx) / _on(ib, idx).rolling(252, min_periods=20).max() - 1
    F["ibov_vol21"] = ibx.rolling(21).std() * np.sqrt(252)
    ptax = sgs(1)
    fx = _on(ptax, idx)
    F["usdbrl"] = fx
    for w in (21, 63):
        F[f"usdbrl_r{w}"] = np.log(fx).diff(w)
    F["usdbrl_vol21"] = np.log(fx).diff().rolling(21).std() * np.sqrt(252)
    sel = sgs(432)
    F["selic"] = _on(sel, idx)
    F["selic_ch126"] = F["selic"].diff(126)
    F["selic_ch252"] = F["selic"].diff(252)

    td = tesouro()
    cr = _curve_rates(td)
    for c in cr.columns:
        F[c] = _on(cr[c], idx, lag_bdays=1)
    F["slope_5y1y"] = F["pre5y"] - F["pre1y"]
    F["pre1y_m_selic"] = F["pre1y"] - F["selic"]          # priced hikes (+) / cuts (-) over 1y
    F["breakeven5y"] = F["pre5y"] - F["real5y"]
    for c in ("pre1y", "pre5y", "real5y", "slope_5y1y"):
        F[f"{c}_ch21"] = F[c].diff(21)
        F[f"{c}_ch63"] = F[c].diff(63)

    # --- IPCA surprise vs Focus median (latest survey published before release)
    comm = pd.read_pickle(ROOT / "data" / "history" / "commodities.pkl")
    ip = comm[comm["series"] == "ipca"][["period_date", "available_date", "value"]].copy()
    ip["period_date"] = pd.to_datetime(ip["period_date"]).dt.to_period("M").dt.to_timestamp()
    # pre-2018 IPCA from SGS 433, availability = 15th of the following month (IBGE releases ~8-12th; conservative)
    s433 = sgs(433)
    old = pd.DataFrame({"period_date": s433.index, "value": s433.to_numpy()})
    old["available_date"] = old["period_date"] + pd.DateOffset(months=1) + pd.Timedelta(days=14)
    old = old[old["period_date"] < ip["period_date"].min()]
    ip = pd.concat([old, ip], ignore_index=True).sort_values("period_date")
    fo = focus_ipca_monthly()
    fo["pub"] = fo["Data"] + pd.Timedelta(days=3)
    surp = []
    for _, r in ip.iterrows():
        g = fo[(fo["ref"] == r["period_date"]) & (fo["pub"] < pd.Timestamp(r["available_date"]))]
        if len(g):
            exp = g.sort_values("Data")["Mediana"].iloc[-1]
            surp.append((pd.Timestamp(r["available_date"]), r["value"] - exp, r["value"]))
    su = pd.DataFrame(surp, columns=["date", "s", "v"]).set_index("date").sort_index()
    F["ipca_surp"] = _on(su["s"], idx)
    F["ipca_surp_3m"] = _on(su["s"].rolling(3).sum(), idx)
    F["ipca_12m"] = _on(((1 + su["v"] / 100).rolling(12).apply(np.prod, raw=True) - 1) * 100, idx)
    # Focus 12m-ahead IPCA path: sum of the next 12 monthly medians of the latest published survey
    fo_s = fo.sort_values("Data")
    e12 = {}
    for d, g in fo_s.groupby("pub"):
        m0 = (d - pd.offsets.MonthBegin(1)).normalize()
        gg = g[(g["ref"] > m0) & (g["ref"] <= m0 + pd.DateOffset(months=12))]
        if len(gg) >= 10:
            e12[d] = gg["Mediana"].sum()
    F["focus_ipca12"] = _on(pd.Series(e12), idx)
    F["real_policy"] = F["selic"] - F["focus_ipca12"]
    F["focus_ipca12_ch63"] = F["focus_ipca12"].diff(63)
    src["focus"] = str(fo["Data"].min().date())

    # --- global (US close is after the B3 close -> +1 bday)
    F["vix"] = _on(yahoo("^VIX"), idx, 1)
    F["vix_ch21"] = F["vix"].diff(21)
    for s in ("BAA10Y", "DGS10", "DGS2"):
        F[s.lower()] = _on(fred(s), idx, 1)
    F["baa10y_ch63"] = F["baa10y"].diff(63)
    F["ust2_ch63"] = F["dgs2"].diff(63)
    br = _on(fred("DCOILBRENTEU"), idx, 1)
    F["brent_r63"] = np.log(br).diff(63)
    hyg, ief, emb, ewz = (_on(yahoo(t), idx, 1) for t in ("HYG", "IEF", "EMB", "EWZ"))
    F["hyg_ief_63"] = np.log(hyg / ief).diff(63)                      # US HY credit-risk proxy (total-return)
    F["hyg_ief_21"] = np.log(hyg / ief).diff(21)
    F["emb_ief_63"] = np.log(emb / ief).diff(63)                      # EM sovereign spread proxy
    F["ewz_r63"] = np.log(ewz).diff(63)
    try:
        F["us_hy_oas"] = _on(fred("BAMLH0A0HYM2"), idx, 1)          # only ~3y history on fredgraph
    except Exception:
        pass
    # Brazil sovereign-risk proxy (no free CDS): nominal 5y - UST 5y-ish - (Focus inflation - 2%)
    F["br_country_proxy"] = F["pre5y"] - F["dgs10"] - (F["focus_ipca12"] - 2.0)
    F["br_country_proxy_ch63"] = F["br_country_proxy"].diff(63)

    # --- commodity index (daily, equal-weight log returns of brent, cattle, sugar_br)
    cm = []
    for s in ("brent", "cattle_br", "sugar_br", "gas_hh"):
        g = comm[comm["series"] == s][["available_date", "value"]].dropna()
        if len(g):
            cm.append(np.log(_on(g.set_index(pd.to_datetime(g["available_date"]))["value"], idx)).diff())
    F["commod_r63"] = pd.concat(cm, axis=1).mean(axis=1).rolling(63).sum()

    # --- debenture issuance (CVM registry, by registration date)
    iss = issuance().groupby("date")["value"].sum()
    iss_d = iss.reindex(iss.index.union(idx)).fillna(0).rolling("91D").sum().reindex(idx).ffill()
    F["iss_91d_bn"] = iss_d / 1e9
    F["iss_yoy"] = np.log((iss_d + 1e8) / (iss_d.shift(252) + 1e8))
    src["issuance_first"] = str(iss.index.min().date())

    # --- market-wide credit news (Google News counts, publication-dated), 2020-10 on
    pm = pd.read_pickle(ROOT / "data" / "history" / "press_market.pkl")
    for s in ("mkt_rj", "mkt_calote"):
        c = pm[pm["series"] == s].groupby("date").size()
        c = c.reindex(c.index.union(idx)).fillna(0).rolling("30D").sum().reindex(idx)
        F[f"{s}_30d"] = c.where(idx >= pd.Timestamp("2020-11-01"))
    F["mkt_rj_z"] = (F["mkt_rj_30d"] - F["mkt_rj_30d"].rolling(252, min_periods=60).mean()) / \
        F["mkt_rj_30d"].rolling(252, min_periods=60).std()

    F.attrs["sources"] = src
    F = F.astype(float)
    tmp = path.with_suffix(".tmp")
    F.to_pickle(tmp)
    tmp.replace(path)
    return F


if __name__ == "__main__":
    F = build(force=True)
    print(F.shape, F.index.min(), F.index.max())
    cov = F.notna().idxmax().astype(str)
    print(pd.DataFrame({"first": cov, "nn": F.notna().sum()}).to_string())
