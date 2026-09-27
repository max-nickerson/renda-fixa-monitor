"""SND (debentures.com.br) registry export: ISIN -> CETIP code, issuer CNPJ, indexer, maturity."""
from __future__ import annotations

import io
from functools import lru_cache

import pandas as pd

from ..http import cached_bytes

URL = ("https://www.debentures.com.br/exploreosnd/consultaadados/emissoesdedebentures/"
       "caracteristicas_e.asp?tip_deb=publicas&op_exc=Nada")


@lru_cache(maxsize=1)
def table() -> pd.DataFrame:
    raw = cached_bytes(URL, max_age_hours=24, name="snd_caracteristicas.tsv")
    if not raw:
        return pd.DataFrame()
    lines = raw.decode("latin1").splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith("Codigo do Ativo"))
    df = pd.read_csv(io.StringIO("\n".join(lines[start:])), sep="\t", dtype=str)
    df.columns = [c.strip() for c in df.columns]
    return df.apply(lambda s: s.astype("string").str.strip())


def by_isin(isin: str) -> dict | None:
    df = table()
    if df.empty:
        return None
    hit = df[df["ISIN"] == isin]
    if hit.empty:
        return None
    r = hit.iloc[0]
    return {
        "cetip_code": r["Codigo do Ativo"],
        "issuer": r["Empresa"],
        "cnpj": r.get("CNPJ"),
        "status": r.get("Situacao"),
        "maturity": pd.to_datetime(r.get("Data de Vencimento"), format="%d/%m/%Y", errors="coerce"),
        "index": r.get("indice"),
        "index_pct": r.get("Percentual Multiplicador/Rentabilidade"),
        "coupon_rate": r.get("Juros Criterio Novo - Taxa"),
        "law_12431": r.get("Deb. Incent. (Lei 12.431)"),
        "series": r.get("Serie"),
        "issue": r.get("Emissao"),
    }


def issues_of(cnpj: str) -> pd.DataFrame:
    """All registered (active) debentures of the same issuer — used for peer spreads."""
    df = table()
    if df.empty or not cnpj:
        return pd.DataFrame()
    return df[(df["CNPJ"] == cnpj) & (df["Situacao"].str.startswith("Registrado", na=False))]
