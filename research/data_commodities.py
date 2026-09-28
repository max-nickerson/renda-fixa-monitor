"""Point-in-time commodity / macro panel for the debenture production-chain study.

Re-runnable, cached downloader + parser.  Output:
    data/history/commodities.pkl  -- long DataFrame
        [series, period_date, available_date, value, unit, source]

Every row carries `available_date` = the first date on which the value can be
assumed PUBLIC (conservative).  Values are latest vintage (no ALFRED-style
vintages); availability rules are deliberately conservative.  See
research/data/COMMODITIES_README.md.

Run:  .venv\\Scripts\\python.exe research\\data_commodities.py [--refresh]
"""
from __future__ import annotations

import io
import json
import re
import subprocess
import sys
import time
import contextlib
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay, MonthBegin, MonthEnd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "history" / "commod_raw"
OUT = ROOT / "data" / "history" / "commodities.pkl"
RAW.mkdir(parents=True, exist_ok=True)

START = pd.Timestamp("2018-01-01")          # 2018 kept as warm-up for YoY / z-scores
TODAY = pd.Timestamp.today().normalize()
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")
REFRESH = "--refresh" in sys.argv
LIVE_MAX_AGE_H = 20                          # re-download "live" files older than this
_client = httpx.Client(headers={"User-Agent": UA}, follow_redirects=True, timeout=90)
_last_req = [0.0]


# --------------------------------------------------------------------------- utils
def _xlrd():
    """xlrd is needed for legacy .xls (CEPEA, EIA). Vendored under commod_raw/_pylib."""
    lib = RAW / "_pylib"
    if str(lib) not in sys.path:
        sys.path.insert(0, str(lib))
    try:
        import xlrd  # noqa
    except ImportError:
        print("  installing xlrd into", lib)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet",
                               "--target", str(lib), "xlrd"])
        import xlrd  # noqa
    return xlrd


def fetch(url: str, fname: str, live: bool = True, timeout: float = 90,
          validate=None, tries: int = 1) -> Path:
    """Download url -> RAW/fname with caching. live=False => cache forever.
    validate(bytes)->bool rejects soft errors (e.g. BCB WAF HTML with status 200)."""
    p = RAW / fname
    if p.exists() and p.stat().st_size > 0 and (validate is None or validate(p.read_bytes())):
        age_h = (time.time() - p.stat().st_mtime) / 3600
        if not live or (age_h < LIVE_MAX_AGE_H and not REFRESH):
            return p
    for k in range(tries):
        wait = 1.0 + 4.0 * k - (time.time() - _last_req[0])   # politeness + backoff
        if wait > 0:
            time.sleep(wait)
        _last_req[0] = time.time()
        r = _client.get(url, timeout=timeout)
        r.raise_for_status()
        if validate is None or validate(r.content):
            p.write_bytes(r.content)
            return p
    raise RuntimeError(f"invalid response after {tries} tries: {url}")


def _is_json(b: bytes) -> bool:
    return b.lstrip()[:1] in (b"[", b"{")


def read_xls(path: Path) -> list[list]:
    xlrd = _xlrd()
    with contextlib.redirect_stdout(io.StringIO()):
        wb = xlrd.open_workbook(str(path), ignore_workbook_corruption=True,
                                logfile=io.StringIO())
    return wb


def next_bday(d: pd.Series, n: int = 1) -> pd.Series:
    return pd.to_datetime(d) + BDay(n)


def frame(series, period, avail, value, unit, source) -> pd.DataFrame:
    df = pd.DataFrame({"series": series,
                       "period_date": pd.to_datetime(period).values,
                       "available_date": pd.to_datetime(avail).values,
                       "value": pd.to_numeric(pd.Series(value).values, errors="coerce"),
                       "unit": unit, "source": source})
    df = df.dropna(subset=["value", "period_date"])
    return df[df["period_date"] >= START]


# --------------------------------------------------------------------------- FRED / EIA
FRED = {"gas_hh": ("DHHNGSP", "ng/hist_xls/RNGWHHDd.xls", "USD/MMBtu"),
        "brent": ("DCOILBRENTEU", "pet/hist_xls/RBRTEd.xls", "USD/bbl"),
        "wti": ("DCOILWTICO", "pet/hist_xls/RWTCd.xls", "USD/bbl"),
        # extras (EIA only): refined-product / NGL proxies for naphtha & diesel
        "ulsd_ny": (None, "pet/hist_xls/EER_EPD2DXL0_PF4_Y35NY_DPGd.xls", "USD/gal"),
        "gasoline_nyh": (None, "pet/hist_xls/EER_EPMRU_PF4_Y35NY_DPGd.xls", "USD/gal"),
        "propane_mb": (None, "pet/hist_xls/EER_EPLLPA_PF4_Y44MB_DPGd.xls", "USD/gal")}
_fred_dead = [False]


def load_fred_eia(key):
    fred_id, eia_path, unit = FRED[key]
    if fred_id and not _fred_dead[0]:
        try:
            p = fetch(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={fred_id}",
                      f"fred_{fred_id}.csv", timeout=20)
            d = pd.read_csv(p)
            d.columns = ["date", "v"]
            d["v"] = pd.to_numeric(d["v"], errors="coerce")
            return frame(key, d["date"], next_bday(d["date"]), d["v"], unit, f"FRED {fred_id}")
        except Exception as e:  # FRED often blocks/timeouts from some networks
            print(f"  FRED {fred_id} failed ({type(e).__name__}); falling back to EIA")
            _fred_dead[0] = True
    p = fetch(f"https://www.eia.gov/dnav/{eia_path}", "eia_" + eia_path.split("/")[-1])
    sh = read_xls(p).sheet_by_name("Data 1")
    rows = [sh.row_values(i) for i in range(3, sh.nrows)]
    d = pd.DataFrame(rows, columns=["date", "v"])
    d = d[pd.to_numeric(d["date"], errors="coerce").notna()]
    d["date"] = pd.to_datetime("1899-12-30") + pd.to_timedelta(d["date"].astype(float), "D")
    src = f"EIA {eia_path.split('/')[-1][:-5]}" + (f" (=FRED {fred_id})" if fred_id else "")
    return frame(key, d["date"], next_bday(d["date"]), d["v"], unit, src)


# --------------------------------------------------------------------------- World Bank
WB = {"gas_eu": "Natural gas, Europe", "gas_us_wb": "Natural gas, US",
      "lng_japan": "Liquefied natural gas, Japan",
      "brent_wb": "Crude oil, Brent", "urea": "Urea", "dap": "DAP",
      "potash": "Potassium chloride", "tsp": "TSP", "phosphate_rock": "Phosphate rock",
      "sugar": "Sugar, world", "soy": "Soybeans", "soymeal": "Soybean meal",
      "soyoil": "Soybean oil", "corn": "Maize", "wheat": "Wheat, US HRW",
      "iron_ore": "Iron ore, cfr spot", "coal": "Coal, Australian",
      "coal_sa": "Coal, South African", "beef_wb": "Beef", "chicken_wb": "Chicken",
      "coffee_arabica": "Coffee, Arabica", "cotton": "Cotton, A Index",
      "aluminum": "Aluminum", "copper": "Copper",
      "nickel": "Nickel", "zinc": "Zinc", "rubber": "Rubber, TSR20", "palm_oil": "Palm oil"}


def wb_url():
    p = fetch("https://www.worldbank.org/en/research/commodity-markets", "wb_page.html")
    m = re.findall(r'https?://[^"\']*CMO-Historical-Data-Monthly\.xlsx', p.read_text("utf-8", "ignore"))
    if not m:
        raise RuntimeError("Pink Sheet monthly xlsx link not found on WB page")
    return m[0]


def load_worldbank():
    p = fetch(wb_url(), "wb_CMO-Historical-Data-Monthly.xlsx")
    x = pd.read_excel(p, sheet_name="Monthly Prices", header=None)
    names = x.iloc[4].astype(str).str.replace(r"\*", "", regex=True).str.strip()
    units = x.iloc[5].astype(str).str.strip("() ")
    body = x.iloc[6:]
    per = pd.to_datetime(body[0].astype(str).str.replace("M", "-"), format="%Y-%m", errors="coerce")
    # Pink Sheet: released ~2nd business day of M+1; rule = 1st bday of M+1 + 3 bdays
    first_bd = (per + MonthBegin(1)).map(lambda d: d if d.weekday() < 5 else d + BDay(1))
    avail = first_bd + BDay(3)
    out = []
    for key, nm in WB.items():
        if nm is None:
            continue
        idx = [i for i, n in enumerate(names) if n == nm]
        if not idx:
            print(f"  WB column not found: {nm}")
            continue
        j = idx[0]
        v = pd.to_numeric(body[j].replace("…", np.nan), errors="coerce")
        out.append(frame(key, per, avail, v, units.iloc[j], f"World Bank Pink Sheet '{nm}'"))
    return pd.concat(out)


# --------------------------------------------------------------------------- CEPEA
CEPEA = {"sugar_br": ("acucar.aspx?id=53", "BRL/50kg", "CEPEA acucar cristal SP (daily)", 1),
         "cattle_br": ("boi-gordo.aspx?id=2", "BRL/@", "CEPEA boi gordo (daily)", 1),
         "ethanol_br": ("etanol.aspx?id=103", "BRL/L",
                        "CEPEA etanol hidratado combustivel SP (weekly, Fri)", 1),
         "ethanol_anidro_br": ("etanol.aspx?id=104", "BRL/L",
                               "CEPEA etanol anidro SP (weekly, Fri)", 1)}


def load_cepea(key):
    path, unit, src, lag = CEPEA[key]
    p = fetch(f"https://www.cepea.org.br/br/indicador/series/{path}",
              "cepea_" + re.sub(r"\W", "_", path) + ".xls")
    sh = read_xls(p).sheet_by_index(0)
    rows = [sh.row_values(i) for i in range(sh.nrows)]
    d = pd.DataFrame([r[:2] for r in rows if re.match(r"\d{2}/\d{2}/\d{4}", str(r[0]))],
                     columns=["date", "v"])
    d["date"] = pd.to_datetime(d["date"], format="%d/%m/%Y")
    return frame(key, d["date"], next_bday(d["date"], lag), d["v"], unit, src)


# --------------------------------------------------------------------------- ONS
ONS_S3 = "https://ons-aws-prod-opendata.s3.amazonaws.com/dataset"


def _ons_years(dataset, prefix, first_year):
    frames = []
    for y in range(first_year, TODAY.year + 1):
        fn = f"{prefix}_{y}.csv"
        try:
            p = fetch(f"{ONS_S3}/{dataset}/{fn}", "ons_" + fn, live=(y == TODAY.year))
        except httpx.HTTPStatusError as e:
            print(f"  ONS {fn}: {e.response.status_code}")
            continue
        frames.append(pd.read_csv(p, sep=";", decimal="."))
    return pd.concat(frames, ignore_index=True)


def load_reservoir():
    d = _ons_years("ear_subsistema_di", "EAR_DIARIO_SUBSISTEMA", START.year)
    d["id_subsistema"] = d["id_subsistema"].str.strip()
    d["date"] = pd.to_datetime(d["ear_data"])
    se = d[d["id_subsistema"] == "SE"]
    out = [frame("reservoir", se["date"], se["date"] + pd.Timedelta(days=1),
                 se["ear_verif_subsistema_percentual"], "% EARmax",
                 "ONS EAR diario subsistema SE/CO")]
    g = d.groupby("date")[["ear_verif_subsistema_mwmes", "ear_max_subsistema"]].sum()
    out.append(frame("reservoir_br", g.index, g.index + pd.Timedelta(days=1),
                     100 * g.iloc[:, 0] / g.iloc[:, 1], "% EARmax",
                     "ONS EAR diario, SIN aggregate (sum verif / sum max)"))
    return pd.concat(out)


def load_cmo():
    out = []
    # daily mean of semi-hourly CMO (DESSEM) SE/CO -- PLD proxy, 2020+
    d = _ons_years("cmo_tm", "CMO_SEMIHORARIO", max(START.year, 2020))
    d = d[d["id_subsistema"].str.strip() == "SE"]
    d["date"] = pd.to_datetime(d["din_instante"]).dt.normalize()
    g = d.groupby("date")["val_cmo"].mean()
    out.append(frame("pld", g.index, g.index + pd.Timedelta(days=1), g.values, "BRL/MWh",
                     "ONS CMO semi-horario SE/CO daily mean (PLD proxy; unclipped by ANEEL floor/cap)"))
    # weekly DECOMP CMO SE/CO (covers 2019, before semi-hourly data exist)
    w = _ons_years("cmo_se", "CMO_SEMANAL", START.year)
    w = w[w["id_subsistema"].str.strip() == "SE"]
    # din_instante = Friday that ENDS the Sat-Fri operative week; DECOMP is published the
    # Thu/Fri before the week starts -> period_date = week start (Sat), available = same Sat.
    w["date"] = pd.to_datetime(w["din_instante"]) - pd.Timedelta(days=6)
    out.append(frame("cmo_se_weekly", w["date"], w["date"],
                     w["val_cmomediasemanal"], "BRL/MWh",
                     "ONS CMO semanal SE/CO (DECOMP ex-ante; period=operative week start Sat)"))
    return pd.concat(out)


# --------------------------------------------------------------------------- ANP
ANP_URL = ("https://www.gov.br/anp/pt-br/assuntos/precos-e-defesa-da-concorrencia/precos/"
           "precos-revenda-e-de-distribuicao-combustiveis/shlp/semanal/semanal-brasil-desde-2013.xlsx")
ANP = {"diesel_br": "OLEO DIESEL", "diesel_s10_br": "OLEO DIESEL S10",
       "gasoline_br": "GASOLINA COMUM", "ethanol_retail_br": "ETANOL HIDRATADO",
       "glp_br": "GLP", "gnv_br": "GNV"}


def load_anp():
    p = fetch(ANP_URL, "anp_semanal_brasil_desde_2013.xlsx")
    x = pd.read_excel(p, header=None)
    h = x.index[x[0].astype(str).str.strip() == "DATA INICIAL"][0]
    x.columns = [str(c).strip() for c in x.iloc[h]]
    x = x.iloc[h + 1:]
    x["end"] = pd.to_datetime(x["DATA FINAL"], errors="coerce")
    out = []
    for key, prod in ANP.items():
        s = x[x["PRODUTO"].astype(str).str.strip() == prod]
        out.append(frame(key, s["end"], s["end"] + pd.Timedelta(days=3),
                         s["PREÇO MÉDIO REVENDA"], str(s["UNIDADE DE MEDIDA"].iloc[-1]),
                         f"ANP SLP semanal Brasil, revenda media '{prod}' (period=week end Sat)"))
    return pd.concat(out)


# --------------------------------------------------------------------------- BCB / IBGE
def bcb(code, fname):
    p = fetch(f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{code}/dados?formato=json"
              f"&dataInicial={START:%d/%m/%Y}&dataFinal={TODAY:%d/%m/%Y}", fname,
              validate=_is_json, tries=5)
    d = pd.DataFrame(json.loads(p.read_text("utf-8")))
    d["date"] = pd.to_datetime(d["data"], format="%d/%m/%Y")
    d["v"] = pd.to_numeric(d["valor"], errors="coerce")
    return d


def ipca_release_dates():
    p = fetch("https://servicodados.ibge.gov.br/api/v3/calendario/9256?qtd=500"
              f"&de=01-01-{START.year}&ate=31-12-{TODAY.year + 1}", "ibge_cal_ipca.json",
              validate=_is_json, tries=3)
    items = json.loads(p.read_text("utf-8"))["items"]
    m = {}
    for it in items:
        per = pd.Timestamp(it["ano_referencia_inicio"], it["mes_referencia_inicio"], 1)
        m[per] = pd.to_datetime(it["data_divulgacao"], format="%d/%m/%Y %H:%M:%S").normalize()
    return m


def load_bcb():
    out = []
    d = bcb(1, "bcb_sgs1_ptax.json")
    out.append(frame("usdbrl", d["date"], next_bday(d["date"]), d["v"], "BRL/USD", "BCB SGS 1 PTAX venda"))
    d = bcb(432, "bcb_sgs432_selic_meta.json")
    out.append(frame("selic", d["date"], next_bday(d["date"]), d["v"], "% a.a.", "BCB SGS 432 Selic meta"))
    d = bcb(433, "bcb_sgs433_ipca.json")
    try:
        cal = ipca_release_dates()
    except Exception as e:
        print("  IBGE calendar failed:", e)
        cal = {}
    fallback = d["date"] + MonthBegin(1) + pd.Timedelta(days=11)       # ~12th of M+1
    av = [cal.get(p, fb) for p, fb in zip(d["date"], fallback)]
    # release at 09:00 BRT -> usable same day for EOD signals; we keep +1 day to be safe
    av = pd.to_datetime(av) + pd.Timedelta(days=1)
    n_exact = sum(p in cal for p in d["date"])
    out.append(frame("ipca", d["date"], av, d["v"], "% m/m",
                     f"BCB SGS 433; avail = IBGE calendar release+1d ({n_exact} exact dates)"))
    d = bcb(24363, "bcb_sgs24363_ibcbr.json")
    av = next_bday(d["date"] + MonthEnd(0) + pd.Timedelta(days=55), 0)
    out.append(frame("ibc_br", d["date"], av, d["v"], "index 2002=100 (NSA)",
                     "BCB SGS 24363 IBC-Br NSA; avail = month end + 55d (revised)"))
    d = bcb(24364, "bcb_sgs24364_ibcbr_sa.json")
    av = next_bday(d["date"] + MonthEnd(0) + pd.Timedelta(days=55), 0)
    out.append(frame("ibc_br_sa", d["date"], av, d["v"], "index 2002=100 (SA)",
                     "BCB SGS 24364 IBC-Br SA; avail = month end + 55d (revised, SA re-estimated)"))
    return pd.concat(out)


# --------------------------------------------------------------------------- main
def main():
    jobs = [(k, (lambda k=k: load_fred_eia(k))) for k in FRED]
    jobs += [("worldbank", load_worldbank)]
    jobs += [(k, (lambda k=k: load_cepea(k))) for k in CEPEA]
    jobs += [("reservoir", load_reservoir), ("cmo/pld", load_cmo), ("anp", load_anp),
             ("bcb", load_bcb)]
    parts, fails = [], []
    for name, fn in jobs:
        print(f"[{name}]")
        try:
            parts.append(fn())
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}")
            fails.append((name, f"{type(e).__name__}: {e}"))
    df = pd.concat(parts, ignore_index=True)
    df = df[df["available_date"] <= TODAY + pd.Timedelta(days=90)]
    df["series"] = df["series"].astype(str)
    df["unit"] = df["unit"].astype(str)
    df["source"] = df["source"].astype(str)
    df["value"] = df["value"].astype(float)
    df = (df.drop_duplicates(["series", "period_date"], keep="last")
            .sort_values(["series", "period_date"]).reset_index(drop=True))
    exante = df["series"].eq("cmo_se_weekly")        # ex-ante weekly plan: avail == week start
    assert (df.loc[~exante, "available_date"] > df.loc[~exante, "period_date"]).all()
    assert (df["available_date"] >= df["period_date"]).all()
    df.to_pickle(OUT)
    summ = df.groupby("series").agg(n=("value", "size"), first=("period_date", "min"),
                                    last=("period_date", "max"),
                                    last_avail=("available_date", "max"),
                                    lag_days_med=("available_date",
                                                  lambda s: np.nan))
    lag = (df["available_date"] - df["period_date"]).dt.days
    summ["lag_days_med"] = lag.groupby(df["series"]).median()
    pd.set_option("display.width", 200)
    print(summ.to_string())
    print(f"\nsaved {len(df):,} rows, {df['series'].nunique()} series -> {OUT}")
    if fails:
        print("FAILURES:", *fails, sep="\n  ")


if __name__ == "__main__":
    main()
