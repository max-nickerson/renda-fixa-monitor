"""ANBIMA public daily files (no credentials needed).

- Debentures secondary market: /informacoes/merc-sec-debentures/arqs/dbYYMMDD.txt
- Federal bonds (títulos públicos): /informacoes/merc-sec/arqs/msYYMMDD.txt
Published every business day after ~20h (Brasília).
"""
from __future__ import annotations

import io
from datetime import date, timedelta

import pandas as pd

from ..bonds import br_float
from ..http import cached_bytes

BASE = "https://www.anbima.com.br/informacoes"

DEB_COLS = ["codigo", "nome", "vencimento", "indice", "taxa_compra", "taxa_venda", "taxa_indicativa",
            "desvio_padrao", "intervalo_min", "intervalo_max", "pu", "pct_pu_par", "duration_du",
            "pct_reune", "ref_ntnb"]
TPF_COLS = ["titulo", "data_referencia", "codigo_selic", "data_base", "vencimento", "taxa_compra",
            "taxa_venda", "taxa_indicativa", "pu", "desvio_padrao", "inf_d0", "sup_d0", "inf_d1",
            "sup_d1", "criterio"]


def _read(url: str, name: str, header_token: str, cols: list[str], is_today: bool) -> pd.DataFrame | None:
    # Past files never change: cache for a year. Today's may not exist yet: re-check hourly.
    raw = cached_bytes(url, max_age_hours=1 if is_today else 24 * 365, name=name)
    if not raw:
        return None
    text = raw.decode("latin1")
    lines = text.splitlines()
    try:
        start = next(i for i, l in enumerate(lines) if header_token in l)
    except StopIteration:
        return None
    body = "\n".join(l for l in lines[start + 1:] if l.count("@") >= len(cols) - 2)
    df = pd.read_csv(io.StringIO(body), sep="@", header=None, dtype=str, engine="python")
    df = df.iloc[:, : len(cols)]
    df.columns = cols[: df.shape[1]]
    return df


def debentures(d: date) -> pd.DataFrame | None:
    url = f"{BASE}/merc-sec-debentures/arqs/db{d:%y%m%d}.txt"
    df = _read(url, f"anbima_db_{d:%Y%m%d}.txt", "@Nome@", DEB_COLS, d >= date.today() - timedelta(days=1))
    if df is None:
        return None
    for c in ["taxa_compra", "taxa_venda", "taxa_indicativa", "desvio_padrao", "pu", "pct_pu_par", "duration_du"]:
        df[c] = df[c].map(br_float)
    df["codigo"] = df["codigo"].str.strip()
    df["vencimento"] = pd.to_datetime(df["vencimento"], format="%d/%m/%Y", errors="coerce")
    df["ref_ntnb"] = pd.to_datetime(df["ref_ntnb"], format="%d/%m/%Y", errors="coerce")
    df["date"] = pd.Timestamp(d)
    return df


def tpf(d: date) -> pd.DataFrame | None:
    url = f"{BASE}/merc-sec/arqs/ms{d:%y%m%d}.txt"
    df = _read(url, f"anbima_ms_{d:%Y%m%d}.txt", "Titulo@Data Referencia", TPF_COLS,
               d >= date.today() - timedelta(days=1))
    if df is None:
        return None
    for c in ["taxa_compra", "taxa_venda", "taxa_indicativa", "pu"]:
        df[c] = df[c].map(br_float)
    df["titulo"] = df["titulo"].str.strip()
    df["vencimento"] = pd.to_datetime(df["vencimento"], format="%Y%m%d", errors="coerce")
    df["date"] = pd.Timestamp(d)
    return df


def business_days_back(n: int, end: date | None = None) -> list[date]:
    end = end or date.today()
    out, d = [], end
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d -= timedelta(days=1)
    return sorted(out)


def latest(fetch, max_lookback: int = 7):
    """Most recent available file (skips weekends/holidays and not-yet-published days)."""
    d = date.today()
    for _ in range(max_lookback + 1):
        if d.weekday() < 5:
            df = fetch(d)
            if df is not None and not df.empty:
                return d, df
        d -= timedelta(days=1)
    return None, None


def history(fetch, days: int) -> pd.DataFrame:
    frames = []
    for d in business_days_back(days):
        try:
            df = fetch(d)
        except Exception:
            df = None
        if df is not None and not df.empty:
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def parse_indexer(indice: str) -> tuple[str, float | None]:
    """'DI + 1,6%' -> ('DI_SPREAD', 1.6); '108% do DI' -> ('DI_PCT', 108); 'IPCA + 6,5%' -> ('IPCA', 6.5)."""
    s = (indice or "").upper().replace(" ", "")
    num = br_float("".join(ch for ch in s.split("+")[-1] if ch.isdigit() or ch in ",.")) if s else None
    if s.startswith("DI+"):
        return "DI_SPREAD", num
    if "DODI" in s or s.endswith("DI"):
        pct = br_float(s.split("%")[0])
        return "DI_PCT", pct
    if s.startswith("IPCA"):
        return "IPCA", num
    if s.startswith("IGP"):
        return "IGPM", num
    if "PR" in s[:3]:
        return "PRE", num
    return "OTHER", num
