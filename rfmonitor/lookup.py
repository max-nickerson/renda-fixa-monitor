"""Turn a pasted Excel column (ISINs, debenture codes, B3 stock tickers) into bond ISINs."""
from __future__ import annotations

import re
from functools import lru_cache

import pandas as pd

from . import db, isin as isin_mod
from .config import ROOT
from .sources import snd

_TICKER = re.compile(r"^[A-Z]{4}(3|4|5|6|11)$")  # B3 share classes / units, not debenture codes like EQTL13


@lru_cache(maxsize=1)
def ticker_by_cnpj8() -> dict[str, tuple[str, bool]]:
    """Issuer CNPJ root → (B3 ticker of the issuer or its listed parent, is_direct) from research/data/equity_map.csv."""
    p = ROOT / "research" / "data" / "equity_map.csv"
    if not p.exists():
        return {}
    m = pd.read_csv(p, dtype=str).dropna(subset=["ticker"])
    return {c.zfill(8): (t.upper(), mt == "direct") for c, t, mt in zip(m["cnpj8"], m["ticker"], m["mapping_type"])}


def registry() -> pd.DataFrame:
    """SND registry reduced to código, ISIN, issuer name and issuer ticker."""
    t = snd.table()
    if t.empty:
        return pd.DataFrame(columns=["codigo", "isin", "empresa", "ticker", "ativo"])
    r = pd.DataFrame({
        "codigo": t["Codigo do Ativo"].str.upper(), "isin": t["ISIN"].str.upper(), "empresa": t["Empresa"],
        "ativo": t["Situacao"].str.startswith("Registrado", na=False),
        "cnpj8": t["CNPJ"].fillna("").str.replace(r"\D", "", regex=True).str.zfill(14).str[:8],
    })
    tk = ticker_by_cnpj8()
    r["ticker"] = r["cnpj8"].map(lambda c: tk.get(c, (None,))[0])
    r["direct"] = r["cnpj8"].map(lambda c: tk.get(c, (None, False))[1])
    return r.drop_duplicates("codigo")


def tokens(text: str) -> list[str]:
    seen, out = set(), []
    for t in re.split(r"[\s;,|]+", (text or "").upper()):
        t = t.strip().strip('"\'')
        if t and t not in seen and t not in {"ISIN", "CODIGO", "CÓDIGO", "TICKER", "ATIVO"}:
            seen.add(t)
            out.append(t)
    return out


def resolve_tokens(text: str) -> tuple[list[str], list[str]]:
    """Returns (ISINs to add, tokens not found). A stock ticker adds the ANBIMA-priced bonds of that listed
    company itself, or, if it has none, of its subsidiaries; a ticker with no priced bond is reported as not found."""
    reg = registry()
    by_code = reg.set_index("codigo")["isin"].to_dict()
    priced = {r["codigo"] for r in (db.load_screener() or {"rows": []})["rows"]}
    found, missing = [], []
    for t in tokens(text):
        try:
            found.append(isin_mod.normalize(t))
            continue
        except ValueError:
            pass
        if t in by_code and isinstance(by_code[t], str):
            found.append(by_code[t])
            continue
        if _TICKER.match(t):
            grp = reg[reg["ticker"].fillna("").str[:4].eq(t[:4]) & reg["ativo"] & reg["codigo"].isin(priced)]
            hit = grp[grp["direct"].fillna(False).astype(bool)]  # PETR3 and PETR4 are the same issuer
            hit = hit if not hit.empty else grp
            if not hit.empty:
                found += hit["isin"].dropna().tolist()
                continue
        missing.append(t)
    return list(dict.fromkeys(found)), missing
