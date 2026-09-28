"""Point-in-time news/event features per issuer (CNPJ root) and market-wide.

Source 1 — CVM IPE (every filing of every CVM-registered issuer, with delivery date, 2021→today):
  fact      Fato Relevante (material fact)
  distress  judicial/extrajudicial recovery, restructuring, waiver, early maturity, default, covenant,
            renegotiation, downgrade (keywords in subject) or the 'Recuperação Judicial' category
  deb_mtg   debenture-holder meetings / notices (Assembleia de debenturistas, Aviso aos Debenturistas)
  rating    rating reports/mentions
  oficio    CVM/B3 inquiry (ofício / esclarecimentos) — usually triggered by press news or odd price moves
Other sources (GDELT etc.) plug in through `external_counts()` with the same (cnpj8, date, flag) shape.

Everything is keyed by the filing's delivery date, and features at date d only use events dated ≤ d−1.
"""
from __future__ import annotations

import re
from datetime import date

import numpy as np
import pandas as pd

from ..sources import cvm

DISTRESS = r"recupera[cç][aã]o (judicial|extrajudicial)|reestrutura|waiver|vencimento antecipado|inadimpl|" \
           r"covenant|renegocia|rebaix|default|descumprimento|pedido de prote[cç][aã]o|mediação|morat[óo]ria"
FLAGS = ("fact", "distress", "deb_mtg", "rating", "oficio")


def _root(cnpj) -> str:
    d = re.sub(r"\D", "", str(cnpj or ""))
    return d[:8] if len(d) >= 8 else ""


def cvm_events(start_year: int = 2021) -> pd.DataFrame:
    frames = []
    for y in range(start_year, date.today().year + 1):
        df = cvm._zip_csv("IPE", y, f"ipe_cia_aberta_{y}")
        if not df.empty:
            frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    cat = df["Categoria"].fillna("").str.lower()
    subj = (df["Assunto"].fillna("") + " " + df["Tipo"].fillna("")).str.lower()
    out = pd.DataFrame({
        "cnpj8": df["CNPJ_Companhia"].map(_root),
        "date": pd.to_datetime(df["Data_Entrega"], errors="coerce"),
        "fact": cat.str.contains("fato relevante"),
        "distress": subj.str.contains(DISTRESS, regex=True) | cat.str.contains("recupera"),
        "deb_mtg": (cat.str.contains("assembleia") & subj.str.contains("debentur")) | cat.str.contains("debenturistas"),
        "rating": subj.str.contains("rating") | cat.str.contains("rating"),
        "oficio": subj.str.contains(r"of[íi]cio|esclarecimento", regex=True),
        "title": df["Categoria"].fillna("") + " — " + df["Assunto"].fillna(""),
    })
    return out.dropna(subset=["date"]).query("cnpj8 != ''")


def features_at(events: pd.DataFrame, keys: pd.DataFrame, windows=(30, 90)) -> pd.DataFrame:
    """For each row of `keys` (cnpj8, asof) → counts of each flag in (asof−w, asof−1] and days since the
    last distress event. Vectorised with searchsorted per issuer."""
    out = pd.DataFrame(index=keys.index)
    for f in FLAGS:
        for w in windows:
            out[f"n_{f}_{w}d"] = 0.0
    out["days_since_distress"] = 9999.0
    ev = {k: g for k, g in events.groupby("cnpj8")}
    for cnpj8, idx in keys.groupby("cnpj8").groups.items():
        g = ev.get(cnpj8)
        if g is None:
            continue
        asof = keys.loc[idx, "asof"].to_numpy(dtype="datetime64[ns]")
        for f in FLAGS:
            d = np.sort(g.loc[g[f], "date"].to_numpy(dtype="datetime64[ns]"))
            if not len(d):
                continue
            hi = np.searchsorted(d, asof - np.timedelta64(1, "D"), side="right")
            for w in windows:
                lo = np.searchsorted(d, asof - np.timedelta64(w, "D"), side="right")
                out.loc[idx, f"n_{f}_{w}d"] = hi - lo
            if f == "distress":
                last = np.where(hi > 0, d[np.maximum(hi - 1, 0)], np.datetime64("NaT"))
                days = (asof - last) / np.timedelta64(1, "D")
                out.loc[idx, "days_since_distress"] = np.where(np.isnan(days), 9999, days)
    return out


def market_gauge(events: pd.DataFrame, index: pd.DatetimeIndex, window: int = 30) -> pd.Series:
    """Market-wide distress pulse: # issuers with a distress filing in the trailing window, z-scored vs 1y."""
    d = events[events["distress"]].drop_duplicates(["cnpj8", "date"]).set_index("date").sort_index()
    daily = d.groupby(level=0)["cnpj8"].nunique().reindex(pd.date_range(index.min() - pd.Timedelta(days=400),
                                                                          index.max()), fill_value=0)
    roll = daily.rolling(window).sum().shift(1)
    z = (roll - roll.rolling(365).mean()) / roll.rolling(365).std()
    return z.reindex(index, method="ffill")
