"""CVM open data (dados.cvm.gov.br): company registry, material facts (IPE) and ITR/DFP filings."""
from __future__ import annotations

import io
import re
import unicodedata
import zipfile
from datetime import date
from functools import lru_cache

import pandas as pd

from ..http import cached_bytes

BASE = "https://dados.cvm.gov.br/dados/CIA_ABERTA"


def _digits(cnpj: str | None) -> str:
    return re.sub(r"\D", "", cnpj or "")


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().upper()
    return re.sub(r"[^A-Z0-9 ]", " ", s)


@lru_cache(maxsize=1)
def registry() -> pd.DataFrame:
    raw = cached_bytes(f"{BASE}/CAD/DADOS/cad_cia_aberta.csv", max_age_hours=24, name="cvm_cad.csv")
    df = pd.read_csv(io.BytesIO(raw), sep=";", encoding="latin1", dtype=str)
    df["cnpj_digits"] = df["CNPJ_CIA"].map(_digits)
    return df


def company_by_cnpj(cnpj: str) -> dict | None:
    df = registry()
    hit = df[df["cnpj_digits"] == _digits(cnpj)]
    if hit.empty:
        return None
    hit = hit.sort_values("SIT").iloc[0]  # prefer ATIVO
    return {"cnpj": hit["CNPJ_CIA"], "name": hit["DENOM_SOCIAL"], "cvm_code": hit["CD_CVM"],
            "status": hit["SIT"], "sector": hit.get("SETOR_ATIV")}


_NOISE = {"SA", "S", "A", "BV", "NV", "LTD", "LTDA", "INC", "LLC", "PLC", "CORP", "FINANCE", "FINANCIAL",
          "NETHERLANDS", "INTERNATIONAL", "INTL", "OVERSEAS", "GLOBAL", "CAYMAN", "AUSTRIA", "LUXEMBOURG",
          "HOLDING", "HOLDINGS", "COMPANY", "CO", "GMBH", "SARL", "AMERICA", "USA", "TRADING", "ISLAND"}


def core_name(name: str) -> str:
    words = [w for w in _norm(name).split() if w not in _NOISE]
    return " ".join(words)


def company_by_name(name: str) -> dict | None:
    """Heuristic match of a (possibly foreign subsidiary) issuer name to its Brazilian parent."""
    core = core_name(name)
    if not core:
        return None
    df = registry()
    names = df["DENOM_SOCIAL"].map(core_name)
    first = core.split()[0]
    for cond in (names == core, names.str.startswith(core), names.str.split().str[0] == first):
        hit = df[cond & (df["SIT"] == "ATIVO")]
        if not hit.empty:
            r = hit.iloc[0]
            return {"cnpj": r["CNPJ_CIA"], "name": r["DENOM_SOCIAL"], "cvm_code": r["CD_CVM"],
                    "status": r["SIT"], "sector": r.get("SETOR_ATIV")}
    return None


def _zip_csv(kind: str, year: int, member_prefix: str) -> pd.DataFrame:
    url = f"{BASE}/DOC/{kind}/DADOS/{kind.lower()}_cia_aberta_{year}.zip"
    # Current-year files are refreshed daily by CVM.
    raw = cached_bytes(url, max_age_hours=6 if year == date.today().year else 24 * 30,
                       name=f"cvm_{kind.lower()}_{year}.zip")
    if not raw:
        return pd.DataFrame()
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        member = next((n for n in z.namelist() if n.startswith(member_prefix)), None)
        if not member:
            return pd.DataFrame()
        return pd.read_csv(z.open(member), sep=";", encoding="latin1", dtype=str)


def material_facts(cnpj: str, year: int | None = None) -> list[dict]:
    """IPE documents: fato relevante, comunicado ao mercado, assembleias, rating reports etc."""
    year = year or date.today().year
    df = _zip_csv("IPE", year, f"ipe_cia_aberta_{year}")
    if df.empty:
        return []
    df = df[df["CNPJ_Companhia"].map(_digits) == _digits(cnpj)]
    out = []
    for _, r in df.iterrows():
        title = " — ".join(x for x in [r.get("Categoria"), r.get("Tipo"), r.get("Assunto")] if isinstance(x, str) and x)
        out.append({"date": r["Data_Entrega"], "title": title, "url": r["Link_Download"],
                    "category": r.get("Categoria") or "", "uid": f"ipe:{r['Protocolo_Entrega']}:{r.get('Versao')}"})
    return out


def financial_statements(cnpj: str, year: int | None = None) -> list[dict]:
    """ITR (quarterly) and DFP (annual) submissions = 'fundamentals published' events."""
    year = year or date.today().year
    out = []
    for kind in ("ITR", "DFP"):
        df = _zip_csv(kind, year, f"{kind.lower()}_cia_aberta_{year}.csv")
        if df.empty:
            continue
        df = df[df["CNPJ_CIA"].map(_digits) == _digits(cnpj)]
        for _, r in df.iterrows():
            out.append({
                "date": r["DT_RECEB"], "url": r.get("LINK_DOC"),
                "title": f"{kind} {r['DT_REFER']} (versão {r['VERSAO']}) publicado",
                "uid": f"{kind.lower()}:{r['ID_DOC']}:{r['VERSAO']}",
            })
    return out
