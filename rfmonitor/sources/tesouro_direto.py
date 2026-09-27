"""Tesouro Direto daily rates/prices since 2002 (Tesouro Transparente open data, no key).

Used for (a) history of federal bonds and (b) NTN-B / pre curves for spread history when
ANBIMA files for that date are not available (ANBIMA public files only cover ~10 business days).
"""
from __future__ import annotations

import io
from functools import lru_cache

import pandas as pd

from ..http import cached_bytes

URL = ("https://www.tesourotransparente.gov.br/ckan/dataset/df56aa42-484a-4a59-8184-7676580c81e3/"
       "resource/796d2059-14e9-44e3-80c9-2d9e30b405c1/download/PrecoTaxaTesouroDireto.csv")

TITULO = {
    "Tesouro IPCA+ com Juros Semestrais": "NTN-B",
    "Tesouro IPCA+": "NTN-B Principal",
    "Tesouro Prefixado": "LTN",
    "Tesouro Prefixado com Juros Semestrais": "NTN-F",
    "Tesouro Selic": "LFT",
    "Tesouro IGPM+ com Juros Semestrais": "NTN-C",
}


@lru_cache(maxsize=1)
def table() -> pd.DataFrame:
    raw = cached_bytes(URL, max_age_hours=12, name="tesouro_direto.csv")
    if not raw:
        return pd.DataFrame()
    df = pd.read_csv(io.BytesIO(raw), sep=";", decimal=",", encoding="latin1")
    df = df.rename(columns={"Tipo Titulo": "nome", "Data Vencimento": "vencimento", "Data Base": "date",
                            "Taxa Compra Manha": "taxa_indicativa", "PU Base Manha": "pu"})
    df["titulo"] = df["nome"].map(TITULO)
    df = df.dropna(subset=["titulo"])
    df["vencimento"] = pd.to_datetime(df["vencimento"], format="%d/%m/%Y")
    df["date"] = pd.to_datetime(df["date"], format="%d/%m/%Y")
    return df[["date", "titulo", "nome", "vencimento", "taxa_indicativa", "pu"]]


def curve_like_tpf(start: pd.Timestamp) -> pd.DataFrame:
    """Rows shaped like ANBIMA TPF (titulo, vencimento, taxa_indicativa, pu, date) from `start`."""
    df = table()
    if df.empty:
        return df
    out = df[df["date"] >= start].copy()
    out.loc[out["titulo"] == "NTN-B Principal", "titulo"] = "NTN-B"  # same real curve
    return out.drop_duplicates(["date", "titulo", "vencimento"])
