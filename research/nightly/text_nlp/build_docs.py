"""Step 1 - build the point-in-time document table for TEXT & NLP.

Sources
  * CVM IPE filings 2021-2026 (data/cache/cvm_ipe_YYYY.zip, already cached by rfmonitor.sources.cvm):
    one row per filing; text = Categoria | Tipo | Especie | Assunto. Availability = Data_Entrega (delivery date).
    Re-presentations of the same filing keep the FIRST delivery date (a restated doc cannot move information back).
  * Google News headlines (data/history/press_items.pkl, negative-keyword queries per brand) -> availability = pubDate.
  * Rating-action headlines (data/history/rating_events.pkl) are NOT re-used as text (they are already a feature elsewhere).

PIT rule used downstream: a document with availability date a is usable for a decision at the close of day d only if
a < d (i.e. a <= d-1).  Filings delivered on d may arrive after the close; news pubDate is a date without a time.

Output: data/history/nightly/text_nlp/docs.pkl  [doc_id, src, cnpj8, avail, cat, tipo, text, link]
"""
from __future__ import annotations

import re
import sys
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "data" / "history" / "nightly" / "text_nlp"
OUT.mkdir(parents=True, exist_ok=True)

# categories that carry no credit information (insider position reports, voting ballots, bylaws, policies, ...)
DROP_CAT = re.compile(r"valores mobili.rios negociados|c.digo de conduta|regimento interno|pol.tica de|"
                      r"estatuto social|contratos de indenidade|relat.rio de sustentabilidade|calend.rio de eventos",
                      re.I)
DROP_ESP = re.compile(r"boletim de voto|mapa (sint.tico|de vota)", re.I)


def load_ipe() -> pd.DataFrame:
    fr = []
    for y in range(2021, 2027):
        p = ROOT / "data" / "cache" / f"cvm_ipe_{y}.zip"
        if not p.exists():
            continue
        z = zipfile.ZipFile(p)
        fr.append(pd.read_csv(z.open(z.namelist()[0]), sep=";", encoding="latin1", dtype=str))
    d = pd.concat(fr, ignore_index=True)
    d["cnpj8"] = d["CNPJ_Companhia"].str.replace(r"\D", "", regex=True).str[:8]
    d["avail"] = pd.to_datetime(d["Data_Entrega"], errors="coerce")
    d = d.dropna(subset=["avail"])
    for c in ("Categoria", "Tipo", "Especie", "Assunto"):
        d[c] = d[c].fillna("").str.replace("&amp", "&", regex=False).str.strip()
    d = d[~d["Categoria"].str.contains(DROP_CAT) & ~d["Especie"].str.contains(DROP_ESP)]
    # first delivery of each filing (protocol without version suffix), then collapse exact duplicate texts per issuer-day
    d["proto"] = d["Protocolo_Entrega"].str.replace(r"-\d+$", "", regex=True)
    d = d.sort_values("avail").drop_duplicates(["cnpj8", "proto"])
    d["text"] = (d["Categoria"] + " | " + d["Tipo"] + " | " + d["Especie"] + " | "
                 + d["Assunto"].str.replace("||", "; ", regex=False)).str.replace(r"\s+", " ", regex=True)
    d = d.drop_duplicates(["cnpj8", "avail", "text"])
    return pd.DataFrame({"src": "ipe", "cnpj8": d["cnpj8"], "avail": d["avail"], "cat": d["Categoria"],
                         "tipo": d["Tipo"], "assunto": d["Assunto"], "text": d["text"], "link": d["Link_Download"]})


def load_press() -> pd.DataFrame:
    p = pd.read_pickle(ROOT / "data" / "history" / "press_items.pkl")
    p = p.dropna(subset=["date"])
    t = p["title"].fillna("").str.replace(r"\s+-\s+[^-]{2,60}$", "", regex=True).str.strip()  # drop " - Outlet"
    out = pd.DataFrame({"src": "news", "cnpj8": p["cnpj8"], "avail": pd.to_datetime(p["date"]).dt.normalize(),
                        "cat": "news", "tipo": p["brand"], "assunto": t, "text": t, "link": ""})
    return out.drop_duplicates(["cnpj8", "avail", "text"])


def main():
    from research.nightly import harness as H
    P = H.load_panel("W")          # pre-2026 panel, only used to get the set of issuers that ever appear on the grid
    issuers = set(P["cnpj8"].dropna())
    ipe = load_ipe()
    ipe = ipe[ipe["cnpj8"].isin(issuers)]
    news = load_press()
    docs = pd.concat([ipe, news], ignore_index=True).sort_values(["avail", "cnpj8"]).reset_index(drop=True)
    docs["doc_id"] = range(len(docs))
    docs.to_pickle(OUT / "docs.pkl")
    print(docs.groupby([docs["src"], docs["avail"].dt.year]).size().unstack(0))
    print("issuers:", docs.groupby("src")["cnpj8"].nunique().to_dict(), "unique texts:", docs["text"].nunique())


if __name__ == "__main__":
    main()
