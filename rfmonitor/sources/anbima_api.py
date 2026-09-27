"""ANBIMA Feed API (needs ANBIMA_CLIENT_ID / ANBIMA_CLIENT_SECRET) — used for CRI/CRA prices."""
from __future__ import annotations

import base64
import json
import time
from datetime import date

import pandas as pd

from ..config import CACHE_DIR, settings
from ..http import client

_token: dict = {}


def _base() -> str:
    return "https://api-sandbox.anbima.com.br" if settings.anbima_env == "sandbox" else "https://api.anbima.com.br"


def _access_token() -> str:
    if _token.get("value") and _token["exp"] > time.time() + 60:
        return _token["value"]
    basic = base64.b64encode(f"{settings.anbima_client_id}:{settings.anbima_client_secret}".encode()).decode()
    r = client().post(f"{_base()}/oauth/access-token", json={"grant_type": "client_credentials"},
                      headers={"Authorization": f"Basic {basic}", "Content-Type": "application/json"})
    r.raise_for_status()
    body = r.json()
    _token.update(value=body["access_token"], exp=time.time() + int(body.get("expires_in", 3600)))
    return _token["value"]


def feed(path: str, **params) -> list[dict]:
    if not settings.anbima_api_enabled:
        raise RuntimeError("ANBIMA API credentials not configured (.env ANBIMA_CLIENT_ID/SECRET)")
    # Past dates never change: cache them on disk so backfills run once.
    cache = None
    if params.get("data") and params["data"] < date.today().isoformat():
        cache = CACHE_DIR / f"anbima_api_{path.replace('/', '_')}_{params['data']}.json"
        if cache.exists():
            return json.loads(cache.read_text(encoding="utf-8"))
    r = client().get(f"{_base()}/feed/precos-indices/v1/{path}", params=params,
                     headers={"client_id": settings.anbima_client_id, "access_token": _access_token()})
    if r.status_code == 404:
        rows: list = []
    else:
        r.raise_for_status()
        data = r.json()
        rows = data if isinstance(data, list) else data.get("content") or data.get("data") or []
    if cache is not None:
        cache.write_text(json.dumps(rows), encoding="utf-8")
    return rows


def _grupo_to_indice(g: str) -> str:
    g = (g or "").upper()
    if "IPCA" in g:
        return "IPCA"
    if "%" in g or "PERCENT" in g:
        return "% do DI"
    if "DI" in g:
        return "DI +"
    if "PR" in g:
        return "PRE"
    return g


def debentures(d: date) -> pd.DataFrame | None:
    """Same columns as anbima_public.debentures(), for any past date."""
    df = pd.DataFrame(feed("debentures/mercado-secundario", data=d.isoformat()))
    if df.empty:
        return None
    out = pd.DataFrame({
        "codigo": df.get("codigo_ativo", pd.Series(dtype=str)).astype(str).str.strip(),
        "nome": df.get("emissor"),
        "vencimento": pd.to_datetime(df.get("data_vencimento"), errors="coerce"),
        "indice": df.get("grupo", pd.Series("", index=df.index)).map(_grupo_to_indice),
        "taxa_indicativa": pd.to_numeric(df.get("taxa_indicativa"), errors="coerce"),
        "pu": pd.to_numeric(df.get("pu"), errors="coerce"),
        "pct_pu_par": pd.to_numeric(df.get("percentual_pu_par"), errors="coerce") if "percentual_pu_par" in df else None,
        "duration_du": pd.to_numeric(df.get("duration"), errors="coerce"),
        "ref_ntnb": pd.to_datetime(df.get("referencia_ntnb"), errors="coerce") if "referencia_ntnb" in df else pd.NaT,
    })
    out["date"] = pd.Timestamp(d)
    return out


def tpf(d: date) -> pd.DataFrame | None:
    df = pd.DataFrame(feed("titulos-publicos/mercado-secundario-TPF", data=d.isoformat()))
    if df.empty:
        return None
    out = pd.DataFrame({
        "titulo": df.get("tipo_titulo").astype(str).str.strip(),
        "vencimento": pd.to_datetime(df.get("data_vencimento"), errors="coerce"),
        "taxa_indicativa": pd.to_numeric(df.get("taxa_indicativa"), errors="coerce"),
        "pu": pd.to_numeric(df.get("pu"), errors="coerce"),
    })
    out["date"] = pd.Timestamp(d)
    return out


def cri_cra(d: date | None = None) -> pd.DataFrame:
    rows = feed("cri-cra/mercado-secundario", **({"data": d.isoformat()} if d else {}))
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # Normalise the field names we rely on; keep everything else as-is.
    ren = {"codigo_ativo": "codigo", "data_referencia": "date", "data_vencimento": "vencimento"}
    df = df.rename(columns={k: v for k, v in ren.items() if k in df.columns})
    for c in ("taxa_indicativa", "taxa_compra", "taxa_venda", "pu", "duration"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df
