"""Point-in-time issuer fundamentals panel for debenture issuers.

Output
  data/history/fundamentals_pit.parquet (or .pkl if no parquet engine)
      one row per (cnpj8, available_date); available_date = date the numbers became public.
  data/history/cvm_fin/*.zip          raw CVM ITR/DFP yearly zips (cached)
  data/history/brapi_fund/<T>.json    raw brapi quarterly fundamentals (cached)

Sources (priority): CVM ITR/DFP open data (DT_RECEB = filing receipt date) -> brapi quarterly
history (available_date = endDate + 45d for Q1-Q3, + 90d for Q4) for mapped tickers of issuers
with no CVM filings (or for periods before the issuer's first CVM filing).  Fundamentus is only
used as a spot cross-check (--fundamentus).  See research/data/FUNDAMENTALS_README.md.

Run:
  PYTHONPATH=. PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe research/data_fundamentals.py [--fundamentus]
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import time
import unicodedata
import zipfile
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
HIST = ROOT / "data" / "history"
CVM_DIR = HIST / "cvm_fin"
BRAPI_DIR = HIST / "brapi_fund"
LAB = HIST / "lab_daily.pkl"
MAP_CSV = ROOT / "research" / "data" / "equity_map.csv"
OUT = HIST / "fundamentals_pit.parquet"
YEARS = range(2019, 2027)
CVM_URL = "https://dados.cvm.gov.br/dados/CIA_ABERTA/DOC/{K}/DADOS/{k}_cia_aberta_{y}.zip"
UA = "Mozilla/5.0 (research; renda-fixa-monitor)"
_http = httpx.Client(headers={"User-Agent": UA}, follow_redirects=True, timeout=120)

# ------------------------------------------------------------------------------------------
# CVM account map (standard non-financial DFP/ITR template)
# ------------------------------------------------------------------------------------------
STOCK_CODES = {"1", "1.01.01", "1.01.02", "2.01.04", "2.02.01", "2.03"}
FLOW_CODES_DRE = {"3.01", "3.05", "3.06", "3.06.01", "3.06.02", "3.11", "3.09"}
DA_RE = re.compile(r"deprec|amortiz|exaust")
DA_EXCL = re.compile(r"custo[s]? de (?:transa|capta)|juros|encargo|emprest|financiament|debent|"
                     r"desagio|ajuste a valor presente|premio|agio na|variac")


def _ascii(s: str) -> str:
    return unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()


def _get(url: str, retries: int = 4, **kw) -> httpx.Response:
    for a in range(retries + 1):
        try:
            r = _http.get(url, **kw)
            if r.status_code in (429, 500, 502, 503, 504) and a < retries:
                time.sleep(5 * (a + 1))
                continue
            return r
        except httpx.HTTPError:
            if a == retries:
                raise
            time.sleep(5 * (a + 1))
    raise RuntimeError("unreachable")


def cvm_zip(kind: str, year: int, refresh_current: bool) -> Path | None:
    CVM_DIR.mkdir(parents=True, exist_ok=True)
    p = CVM_DIR / f"{kind}_cia_aberta_{year}.zip"
    stale = refresh_current and year >= pd.Timestamp.today().year - 1 and \
        (time.time() - p.stat().st_mtime > 86400 if p.exists() else True)
    if p.exists() and p.stat().st_size > 0 and not stale:
        return p
    r = _get(CVM_URL.format(K=kind.upper(), k=kind, y=year))
    if r.status_code == 404:
        return p if p.exists() else None
    r.raise_for_status()
    p.write_bytes(r.content)
    return p


def _read(z: zipfile.ZipFile, name: str, cols: list[str]) -> pd.DataFrame:
    if name not in z.namelist():
        return pd.DataFrame(columns=cols)
    return pd.read_csv(z.open(name), sep=";", encoding="latin1", dtype=str,
                       usecols=lambda c: c in cols)


def load_cvm(universe: set[str], refresh: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (index of filings with first/last DT_RECEB, long table of statement lines)."""
    idx, lines = [], []
    stmt_cols = ["CNPJ_CIA", "DT_REFER", "ESCALA_MOEDA", "ORDEM_EXERC", "DT_INI_EXERC",
                 "DT_FIM_EXERC", "CD_CONTA", "DS_CONTA", "VL_CONTA"]
    for kind in ("itr", "dfp"):
        for y in YEARS:
            p = cvm_zip(kind, y, refresh)
            if p is None:
                continue
            z = zipfile.ZipFile(p)
            i = _read(z, f"{kind}_cia_aberta_{y}.csv", ["CNPJ_CIA", "DT_REFER", "VERSAO", "DT_RECEB"])
            i["cnpj8"] = i.CNPJ_CIA.str.replace(r"\D", "", regex=True).str[:8]
            i = i[i.cnpj8.isin(universe)]
            i["doc"] = kind
            idx.append(i)
            for st in ("BPA", "BPP", "DRE", "DFC_MI"):
                for basis in ("con", "ind"):
                    d = _read(z, f"{kind}_cia_aberta_{st}_{basis}_{y}.csv", stmt_cols)
                    d["cnpj8"] = d.CNPJ_CIA.str.replace(r"\D", "", regex=True).str[:8]
                    d = d[d.cnpj8.isin(universe)]
                    if st == "DFC_MI":
                        d = d[d.CD_CONTA.str.startswith("6.01.01.") & (d.CD_CONTA.str.count(r"\.") == 3)]
                    elif st == "DRE":
                        d = d[d.CD_CONTA.isin(FLOW_CODES_DRE)]
                    else:
                        d = d[d.CD_CONTA.isin(STOCK_CODES)]
                    d = d.assign(doc=kind, st=st, basis=basis)
                    lines.append(d)
            print(f"  loaded {kind} {y}", flush=True)
    idx = pd.concat(idx, ignore_index=True)
    idx["VERSAO"] = idx.VERSAO.astype(int)
    idx["DT_RECEB"] = pd.to_datetime(idx.DT_RECEB)
    idx = (idx.groupby(["cnpj8", "CNPJ_CIA", "DT_REFER", "doc"])
              .agg(first_receb=("DT_RECEB", "min"), last_receb=("DT_RECEB", "max"),
                   versao=("VERSAO", "max")).reset_index())
    L = pd.concat(lines, ignore_index=True)
    scale = np.where(L.ESCALA_MOEDA.str.upper().str.startswith("MIL"), 1e-3, 1e-6)
    L["val"] = pd.to_numeric(L.VL_CONTA, errors="coerce") * scale  # -> R$ millions
    L["ordem"] = np.where(L.ORDEM_EXERC.map(_ascii).str.startswith("ult"), "U", "P")
    L["desc"] = L.DS_CONTA.map(_ascii)
    return idx, L.drop(columns=["VL_CONTA", "ORDEM_EXERC", "DS_CONTA", "ESCALA_MOEDA"])


def _pick_basis(L: pd.DataFrame) -> pd.DataFrame:
    """Per filing (CNPJ_CIA, DT_REFER, doc): consolidated if the con balance sheet has total assets."""
    ta = L[(L.CD_CONTA == "1") & (L.st == "BPA") & (L.ordem == "U")]
    has_con = set(map(tuple, ta[ta.basis == "con"][["CNPJ_CIA", "DT_REFER", "doc"]].values))
    key = list(zip(L.CNPJ_CIA, L.DT_REFER, L.doc))
    want = np.array(["con" if k in has_con else "ind" for k in key])
    return L[L.basis.values == want]


def cvm_filings(idx: pd.DataFrame, L: pd.DataFrame) -> pd.DataFrame:
    L = _pick_basis(L)
    K = ["CNPJ_CIA", "DT_REFER", "doc"]
    basis = L.groupby(K).basis.first()

    # --- balance sheet (ÚLTIMO) --------------------------------------------------------------
    B = L[L.st.isin(["BPA", "BPP"]) & (L.ordem == "U")]
    ok = ((B.CD_CONTA.isin(["1", "2.03"])) |
          ((B.CD_CONTA == "1.01.01") & B.desc.str.contains("caixa")) |
          ((B.CD_CONTA == "1.01.02") & B.desc.str.contains("aplica")) |
          (B.CD_CONTA.isin(["2.01.04", "2.02.01"]) & B.desc.str.contains("emprest|financ")))
    B = B[ok].pivot_table(index=K, columns="CD_CONTA", values="val", aggfunc="first")
    bs = pd.DataFrame(index=B.index)
    bs["total_assets"] = B.get("1")
    bs["equity"] = B.get("2.03")
    bs["cash"] = B.get("1.01.01", 0).fillna(0) + B.get("1.01.02", 0).fillna(0)
    bs.loc[B.get("1.01.01").isna() & B.get("1.01.02").isna(), "cash"] = np.nan
    bs["st_debt"] = B.get("2.01.04")
    bs["gross_debt"] = B.get("2.01.04", 0).fillna(0) + B.get("2.02.01", 0).fillna(0)
    bs.loc[B.get("2.01.04").isna() & B.get("2.02.01").isna(), "gross_debt"] = np.nan

    # --- flows: year-to-date (earliest DT_INI) for ÚLTIMO and PENÚLTIMO -------------------------
    F = L[L.st.isin(["DRE", "DFC_MI"])].copy()
    F["field"] = None
    F.loc[F.CD_CONTA == "3.01", "field"] = "revenue"
    F.loc[F.CD_CONTA == "3.05", "field"] = "ebit"
    F.loc[F.CD_CONTA.isin(["3.06.01", "3.06.02"]) & F.desc.str.contains("despesa"), "field"] = "fin_exp"
    F.loc[F.CD_CONTA.isin(["3.06.01", "3.06.02"]) & F.desc.str.contains("receita"), "field"] = "fin_inc"
    F.loc[(F.CD_CONTA == "3.06") & F.desc.str.contains("financ"), "field"] = "fin_net"
    F.loc[(F.CD_CONTA == "3.11") & F.desc.str.contains("lucro|prejuizo|resultado"), "field"] = "ni"
    F.loc[(F.CD_CONTA == "3.09") & F.desc.str.contains("lucro|prejuizo"), "field"] = "ni9"
    isda = (F.st == "DFC_MI") & F.desc.str.contains(DA_RE) & ~F.desc.str.contains(DA_EXCL)
    F.loc[isda, "field"] = "da"
    F = F[F.field.notna()]
    F["DT_INI_EXERC"] = pd.to_datetime(F.DT_INI_EXERC)
    F["DT_FIM_EXERC"] = pd.to_datetime(F.DT_FIM_EXERC)
    first_ini = F.groupby(K + ["ordem"]).DT_INI_EXERC.transform("min")
    F = F[F.DT_INI_EXERC == first_ini]
    agg = F.groupby(K + ["ordem", "field"]).val.sum(min_count=1).unstack("field")
    dates = F.groupby(K + ["ordem"]).agg(ini=("DT_INI_EXERC", "min"), fim=("DT_FIM_EXERC", "max"))
    agg = agg.join(dates)
    if "ni9" in agg:
        agg["ni"] = agg.get("ni").fillna(agg["ni9"]) if "ni" in agg else agg["ni9"]
    for c in ("revenue", "ebit", "fin_exp", "fin_inc", "fin_net", "ni", "da"):
        if c not in agg:
            agg[c] = np.nan
    # some filers (e.g. Sabesp) leave 3.06.01/3.06.02 at zero and report only the net result 3.06:
    # fall back to the net financial result as the expense proxy and flag it
    agg["fin_exp_is_net"] = (agg.fin_exp.fillna(0) == 0) & (agg.fin_inc.fillna(0) == 0) & (agg.fin_net < 0)
    agg["fin_exp"] = -np.where(agg.fin_exp_is_net, agg.fin_net, agg.fin_exp)  # positive = expense
    U = agg.xs("U", level="ordem")
    P = agg.xs("P", level="ordem").drop(columns=["fin_exp_is_net"])[["revenue", "ebit", "fin_exp", "ni", "da", "ini"]].add_suffix("_py")

    f = bs.join(U, how="outer").join(P, how="left").join(basis).reset_index()
    f = f.merge(idx, on=K, how="left")
    f["period_end"] = pd.to_datetime(f.DT_REFER)
    f["months"] = (f.fim.dt.year - f.ini.dt.year) * 12 + f.fim.dt.month - f.ini.dt.month + 1
    # DFP overrides an ITR with the same period end (non-calendar filers)
    f = f.sort_values("doc").drop_duplicates(["cnpj8", "period_end"], keep="first")  # 'dfp' < 'itr'
    f = f.sort_values(["cnpj8", "period_end", "CNPJ_CIA"]).drop_duplicates(["cnpj8", "period_end"])

    # --- LTM = YTD + prior annual - prior-year YTD (prior annual from the earlier DFP) --------
    ann = f[f.months >= 12][["cnpj8", "period_end", "revenue", "ebit", "fin_exp", "ni", "da",
                             "first_receb", "last_receb"]]
    ann = ann.rename(columns={c: c + "_fy" for c in ann.columns if c not in ("cnpj8", "period_end")})
    f["prev_fy_end"] = f.ini - pd.Timedelta(days=1)
    f = f.merge(ann.rename(columns={"period_end": "prev_fy_end"}), on=["cnpj8", "prev_fy_end"], how="left")
    full = f.months >= 12
    for c in ("revenue", "ebit", "fin_exp", "ni", "da"):
        f[c + "_ltm"] = np.where(full, f[c], f[c] + f[c + "_fy"] - f[c + "_py"])
    f["available_date"] = np.where(full, f.first_receb, f[["first_receb", "first_receb_fy"]].max(axis=1))
    f["available_date_strict"] = np.where(full, f.last_receb, f[["last_receb", "last_receb_fy"]].max(axis=1))
    # when LTM is unavailable (no prior DFP), the filing's own receipt date still dates the balance sheet
    f["available_date"] = pd.to_datetime(f.available_date).fillna(f.first_receb)
    f["available_date_strict"] = pd.to_datetime(f.available_date_strict).fillna(f.last_receb)
    f["source"] = "cvm_" + f.doc + "_" + f.basis.fillna("na")
    f["is_parent"] = False
    f["cvm_versao"] = f.versao
    return f


# ------------------------------------------------------------------------------------------
# brapi
# ------------------------------------------------------------------------------------------
BRAPI_MODULES = "balanceSheetHistoryQuarterly,incomeStatementHistoryQuarterly,financialDataHistoryQuarterly"


def fetch_brapi(ticker: str, token: str, max_age_h: float = 24 * 7) -> dict | None:
    BRAPI_DIR.mkdir(parents=True, exist_ok=True)
    p = BRAPI_DIR / f"{ticker}.json"
    if p.exists() and time.time() - p.stat().st_mtime < max_age_h * 3600:
        return json.loads(p.read_text()) or None
    time.sleep(0.6)
    r = _get(f"https://brapi.dev/api/quote/{ticker}", params={"modules": BRAPI_MODULES},
             headers={"Authorization": f"Bearer {token}"})  # token in header, never in URL/logs
    if r.status_code != 200:
        print(f"  brapi {ticker}: HTTP {r.status_code}")
        p.write_text("{}")
        return None
    res = (r.json().get("results") or [{}])[0]
    p.write_text(json.dumps(res))
    return res


def brapi_frame(ticker: str, res: dict) -> pd.DataFrame:
    def df(k):
        x = pd.DataFrame(res.get(k) or [])
        if x.empty:
            return x
        x["period_end"] = pd.to_datetime(x.endDate).dt.normalize()
        return x.drop_duplicates("period_end").set_index("period_end").sort_index()
    bs, inc, fd = df("balanceSheetHistoryQuarterly"), df("incomeStatementHistoryQuarterly"), \
        df("financialDataHistoryQuarterly")
    if bs.empty or inc.empty:
        return pd.DataFrame()
    g = lambda d, c: pd.to_numeric(d[c], errors="coerce") / 1e6 if c in d else pd.Series(np.nan, index=d.index)  # noqa: E731
    o = pd.DataFrame(index=bs.index.union(inc.index))
    o["total_assets"] = g(bs, "totalAssets")
    o["equity"] = g(bs, "shareholdersEquity").fillna(g(bs, "totalAssets") - g(bs, "totalLiab"))
    o["cash"] = g(bs, "cash").fillna(0) + g(bs, "shortTermInvestments").fillna(0)
    o.loc[g(bs, "cash").reindex(o.index).isna(), "cash"] = np.nan
    o["st_debt"] = g(bs, "loansAndFinancing")
    o["gross_debt"] = g(bs, "loansAndFinancing").fillna(0) + g(bs, "longTermLoansAndFinancing").fillna(0)
    nodebt = g(bs, "loansAndFinancing").isna() & g(bs, "longTermLoansAndFinancing").isna()
    o.loc[nodebt.reindex(o.index).fillna(True), "gross_debt"] = np.nan
    if not fd.empty:  # fallback: brapi totalDebt / totalCash
        o["gross_debt"] = o.gross_debt.fillna(g(fd, "totalDebt"))
        o["cash"] = o.cash.fillna(g(fd, "totalCash"))
    q = pd.DataFrame({"revenue": g(inc, "totalRevenue"), "ebit": g(inc, "ebit").fillna(g(inc, "operatingIncome")),
                      "fin_exp": -g(inc, "financialExpenses"), "ni": g(inc, "netIncome")}).reindex(o.index)
    # LTM only over 4 consecutive quarter ends
    contiguous = pd.Series(o.index, index=o.index).diff(3).dt.days.between(260, 290)
    for c in q:
        o[c + "_ltm"] = q[c].rolling(4, min_periods=4).sum().where(contiguous)
    o["da_ltm"] = np.nan
    o["ebitda_brapi"] = g(fd, "ebitda").reindex(o.index) if not fd.empty else np.nan
    o = o.reset_index()
    q4 = o.period_end.dt.month == 12
    o["available_date"] = o.period_end + pd.to_timedelta(np.where(q4, 90, 45), unit="D")
    o["available_date_strict"] = o.available_date
    o["source"] = "brapi"
    o["ticker"] = ticker
    return o


# ------------------------------------------------------------------------------------------
# ratios, PIT ordering
# ------------------------------------------------------------------------------------------
COLS = ["cnpj8", "period_end", "available_date", "available_date_strict", "source", "is_parent", "ticker",
        "cvm_versao", "total_assets", "equity", "gross_debt", "cash", "net_debt", "st_debt",
        "revenue_ltm", "ebit_ltm", "da_ltm", "ebitda_ltm", "ebitda_is_ebit", "fin_exp_ltm", "fin_exp_is_net", "net_income_ltm",
        "net_debt_ebitda", "gross_debt_equity", "interest_coverage", "cash_to_st_debt", "equity_ratio",
        "revenue_growth_yoy", "ebitda_margin", "d_net_debt_ebitda_4q", "d_interest_coverage_4q"]


def finish(f: pd.DataFrame) -> pd.DataFrame:
    f = f.copy()
    f["net_debt"] = f.gross_debt - f.cash
    if "ebitda_brapi" not in f:
        f["ebitda_brapi"] = np.nan
    has_da = f.da_ltm.notna() & (f.da_ltm != 0)
    f["ebitda_ltm"] = np.where(has_da, f.ebit_ltm + f.da_ltm, f.ebitda_brapi.fillna(f.ebit_ltm))
    f["ebitda_is_ebit"] = ~has_da & f.ebitda_brapi.isna() & f.ebit_ltm.notna()
    f["net_income_ltm"] = f.ni_ltm
    f["fin_exp_is_net"] = f.get("fin_exp_is_net", pd.Series(False, index=f.index)).fillna(False).astype(bool)
    f["fin_exp_ltm"] = f.fin_exp_ltm
    f["net_debt_ebitda"] = f.net_debt / f.ebitda_ltm
    f["gross_debt_equity"] = f.gross_debt / f.equity
    f["interest_coverage"] = f.ebitda_ltm / f.fin_exp_ltm
    f["cash_to_st_debt"] = f.cash / f.st_debt
    f["equity_ratio"] = f.equity / f.total_assets
    f["ebitda_margin"] = f.ebitda_ltm / f.revenue_ltm
    # year-ago values: exact period_end - 1 year (same source stream)
    lag = f[["cnpj8", "source_grp", "period_end", "revenue_ltm", "net_debt_ebitda", "interest_coverage"]].copy()
    lag["period_end"] = lag.period_end + pd.DateOffset(years=1)
    lag["period_end"] = lag.period_end + pd.offsets.MonthEnd(0)
    f = f.merge(lag.drop_duplicates(["cnpj8", "source_grp", "period_end"]), on=["cnpj8", "source_grp", "period_end"],
                how="left", suffixes=("", "_4q"))
    f["revenue_growth_yoy"] = f.revenue_ltm / f.revenue_ltm_4q - 1
    f["d_net_debt_ebitda_4q"] = f.net_debt_ebitda - f.net_debt_ebitda_4q
    f["d_interest_coverage_4q"] = f.interest_coverage - f.interest_coverage_4q
    num = f.select_dtypes("number").columns
    f[num] = f[num].replace([np.inf, -np.inf], np.nan)
    for c in COLS:
        if c not in f:
            f[c] = np.nan
    return f[COLS]


def pit_order(p: pd.DataFrame) -> pd.DataFrame:
    """One row per (cnpj8, available_date): drop rows that are not newer than an already-public
    period (late re-filings / back-filled quarters), keep the latest period when several arrive
    on the same day."""
    p = p.sort_values(["cnpj8", "available_date", "period_end"])
    p = p.drop_duplicates(["cnpj8", "available_date"], keep="last")
    prev_max = p.groupby("cnpj8").period_end.transform(lambda s: s.cummax().shift())
    return p[prev_max.isna() | (p.period_end > prev_max)].reset_index(drop=True)


# ------------------------------------------------------------------------------------------
# Fundamentus spot check
# ------------------------------------------------------------------------------------------
def _num(s: str) -> float:
    s = s.strip().replace(".", "").replace(",", ".").replace("%", "")
    try:
        return float(s)
    except ValueError:
        return np.nan


def fundamentus(ticker: str) -> dict:
    r = _get("https://www.fundamentus.com.br/detalhes.php", params={"papel": ticker},
             headers={"User-Agent": "Mozilla/5.0"})
    txt = r.content.decode("latin1", "ignore")
    cells = re.findall(r'<span class="txt">(.*?)</span>', txt, flags=re.S)
    cells = [re.sub(r"<.*?>", "", c).strip() for c in cells]
    out = {}
    for k, v in zip(cells[:-1], cells[1:]):  # label -> next cell; first occurrence = last 12 months
        out.setdefault(_ascii(k), v)
    g = lambda k: _num(out.get(k, "")) / 1e6  # noqa: E731
    return {"ticker": ticker, "last_bal": out.get("ult balanco processado"),
            "gross_debt": g("div. bruta"), "net_debt": g("div. liquida"), "cash": g("disponibilidades"),
            "equity": g("patrim. liq"), "ebit_ltm": g("ebit"), "revenue_ltm": g("receita liquida"),
            "net_income_ltm": g("lucro liquido")}


# ------------------------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="re-download current-year CVM zips if >1 day old")
    ap.add_argument("--fundamentus", action="store_true", help="print Fundamentus spot cross-check")
    ap.add_argument("--no-brapi", action="store_true")
    a = ap.parse_args()
    load_dotenv(ROOT / ".env")

    lab = pd.read_pickle(LAB)
    el = lab[lab.eligible == True][["cnpj8", "date"]].dropna()  # noqa: E712
    U = set(el.cnpj8)
    print(f"universe: {len(U)} issuers, {len(el):,} eligible bond-days")

    print("CVM ITR/DFP ...")
    idx, L = load_cvm(U, a.refresh)
    cv = cvm_filings(idx, L)
    cv["source_grp"] = "cvm"
    print(f"  CVM filings: {len(cv):,} rows, {cv.cnpj8.nunique()} issuers")

    parts = [cv]
    emap = pd.read_csv(MAP_CSV, dtype=str)
    emap = emap[emap.cnpj8.isin(U) & emap.ticker.notna()].drop_duplicates("cnpj8")
    if not a.no_brapi:
        token = os.environ.get("BRAPI_TOKEN", "")
        first_cvm = cv.groupby("cnpj8").available_date.min()
        need = emap[~emap.cnpj8.isin(first_cvm.index)]
        xcheck = emap[emap.cnpj8.isin(first_cvm.index) & (emap.mapping_type == "direct")]
        tickers = sorted(set(need.ticker) | set(xcheck.ticker))
        print(f"brapi: {len(tickers)} tickers ({need.ticker.nunique()} gap-fill, rest cross-check/pre-CVM history)")
        frames = {}
        for t in tickers:
            res = fetch_brapi(t, token)
            if res:
                bf = brapi_frame(t, res)
                if not bf.empty:
                    frames[t] = bf
        rows = []
        for _, m in emap.iterrows():
            bf = frames.get(m.ticker)
            if bf is None:
                continue
            bf = bf.assign(cnpj8=m.cnpj8, is_parent=(m.mapping_type == "parent"), source_grp="brapi")
            if m.cnpj8 in first_cvm.index:
                if m.mapping_type != "direct":
                    continue
                bf = bf[bf.available_date < first_cvm[m.cnpj8]]  # pre-CVM history only
            rows.append(bf)
        if rows:
            br = pd.concat(rows, ignore_index=True)
            parts.append(br)
            print(f"  brapi rows used: {len(br):,}, issuers {br.cnpj8.nunique()}")
        # cross-check brapi vs CVM on shared period ends (direct tickers)
        xs = []
        for _, m in emap[emap.cnpj8.isin(first_cvm.index) & (emap.mapping_type == "direct")].iterrows():
            bf = frames.get(m.ticker)
            if bf is None:
                continue
            c = cv[cv.cnpj8 == m.cnpj8].set_index("period_end")
            j = bf.set_index("period_end").join(c[["total_assets", "gross_debt", "revenue_ltm", "equity"]],
                                                rsuffix="_cvm", how="inner")
            j = j[j.index >= "2021-01-01"]
            for col in ("total_assets", "gross_debt", "revenue_ltm", "equity"):
                r = (j[col] / j[col + "_cvm"] - 1).abs()
                xs.append(pd.DataFrame({"field": col, "err": r.values}))
        if xs:
            xs = pd.concat(xs).dropna()
            print("  brapi vs CVM (direct tickers, same period, 2021+): share within 2% / median abs rel err")
            print(xs.groupby("field").err.agg(within2=lambda s: (s < 0.02).mean(), med="median").round(4).to_string())

    allp = pd.concat(parts, ignore_index=True)
    pan = finish(allp)
    pan = pit_order(pan)
    for c in ("available_date", "available_date_strict", "period_end"):
        pan[c] = pd.to_datetime(pan[c]).astype("datetime64[ns]")
    pan = pan[pan.period_end >= "2018-01-01"]
    pan = pan.sort_values(["cnpj8", "available_date"]).reset_index(drop=True)
    try:
        pan.to_parquet(OUT, index=False)
        out = OUT
    except ImportError:
        out = OUT.with_suffix(".pkl")
        pan.to_pickle(out)
    print(f"wrote {out.relative_to(ROOT)}: {len(pan):,} rows, {pan.cnpj8.nunique()} issuers")

    # ---------------- validation ----------------
    e = el.assign(date=el.date.astype("datetime64[ns]")).sort_values("date").reset_index(drop=True)
    m = pd.merge_asof(e, pan.sort_values("available_date")[["available_date", "cnpj8", "source", "period_end",
                                                              "net_debt_ebitda"]],
                      left_on="date", right_on="available_date", by="cnpj8", direction="backward")
    has = m.available_date.notna()
    stale = (m.date - m.period_end).dt.days
    print(f"\ncoverage: {m[has].cnpj8.nunique()}/{len(U)} issuers; {has.mean():.1%} of eligible bond-days "
          f"(CVM {m.source.str.startswith('cvm').mean():.1%}, brapi {(m.source == 'brapi').mean():.1%}); "
          f"with net_debt_ebitda {m.net_debt_ebitda.notna().mean():.1%}; median staleness "
          f"(day - period_end) {stale.median():.0f}d, p90 {stale.quantile(.9):.0f}d")
    print(m.assign(y=m.date.dt.year, h=has).groupby("y").h.mean().round(3).to_string())
    rc = ["net_debt_ebitda", "gross_debt_equity", "interest_coverage", "cash_to_st_debt", "equity_ratio",
          "revenue_growth_yoy", "ebitda_margin"]
    print("\nmedian ratios by period-end year:")
    print(pan.assign(y=pan.period_end.dt.year).groupby("y")[rc].median().round(2).to_string())
    print(f"ebitda_is_ebit share: {pan.ebitda_is_ebit.mean():.1%}")
    spot = {"16670085": "Localiza", "00864214": "Energisa", "43776517": "Sabesp", "04423567": "Eneva",
            "02998611": "ISA Energia"}
    show = ["period_end", "available_date", "source", "gross_debt", "net_debt", "ebitda_ltm", "revenue_ltm",
            "net_debt_ebitda", "interest_coverage", "ebitda_margin"]
    for c, n in spot.items():
        s = pan[pan.cnpj8 == c]
        print(f"\n{n} ({c}):")
        print(s[show].tail(3).round(1).to_string(index=False))

    if a.fundamentus:
        print("\nFundamentus spot check (R$ mn):")
        for c, t in [("16670085", "RENT3"), ("43776517", "SBSP3"), ("04423567", "ENEV3"),
                     ("00864214", "ENGI11"), ("02998611", "ISAE4")]:
            try:
                fu = fundamentus(t)
            except Exception as ex:  # noqa: BLE001
                print(f"  {t}: failed {ex!r}")
                continue
            s = pan[(pan.cnpj8 == c) & pan.source.str.startswith("cvm")].iloc[-1]
            print(f"  {t} fundamentus bal {fu['last_bal']}: gross {fu['gross_debt']:.0f} net {fu['net_debt']:.0f} "
                  f"ebit {fu['ebit_ltm']:.0f} rev {fu['revenue_ltm']:.0f} | ours {s.period_end.date()}: gross "
                  f"{s.gross_debt:.0f} net {s.net_debt:.0f} ebit {s.ebit_ltm:.0f} rev {s.revenue_ltm:.0f}")
            time.sleep(1)


if __name__ == "__main__":
    main()
