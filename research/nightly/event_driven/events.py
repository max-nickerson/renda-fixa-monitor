"""Point-in-time event sets for the event-driven study.

Sources (all already cached locally, no new downloads needed):
  * SND registry export  data/cache/snd_caracteristicas.tsv  (bond terms, distribution start, exit date, call flag)
  * CVM IPE filings      data/cache/cvm_ipe_{2021..2026}.zip   (dated by Data_Entrega = publication date)
  * rating actions       data/history/rating_events.pkl        (dated by action / capture date)

PIT rules
  * IPE event is usable from the first decision whose day is STRICTLY AFTER Data_Entrega (filings have no time stamp).
  * New issue (SND): usable from the bond's first secondary trade, which is after its distribution start.
  * Issuer supply event (SND): a new bond of the same issuer starts distribution on 'Data do Inicio da Distribuicao'
    (public via ANBIMA/CVM announcements at that date); usable strictly after it.
  * Early exit (SND 'Data de Saida' < maturity - 30d) is EX-POST; used only for descriptive event studies, and
    for a call-probability check that uses only bond terms known in advance.
  * Static-registry caveat: terms come from today's registry snapshot (repactuated terms leak back in time).

Outputs (data/history/nightly/event_driven/):
  bonds_ref.pkl      codigo-level terms
  events_issuer.pkl  (cnpj8, date, etype, text)
"""
from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
CACHE = ROOT / "data" / "cache"
OUT = ROOT / "data" / "history" / "nightly" / "event_driven"
OUT.mkdir(parents=True, exist_ok=True)


def _d(s):
    return pd.to_datetime(s, format="%d/%m/%Y", errors="coerce")


def snd_registry() -> pd.DataFrame:
    raw = (CACHE / "snd_caracteristicas.tsv").read_bytes()
    lines = raw.decode("latin1").splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith("Codigo do Ativo"))
    df = pd.read_csv(io.StringIO("\n".join(lines[start:])), sep="\t", dtype=str)
    df.columns = [c.strip() for c in df.columns]
    df = df.apply(lambda s: s.str.strip())
    out = pd.DataFrame({
        "codigo": df["Codigo do Ativo"],
        "cnpj8": df["CNPJ"].str.replace(r"\D", "", regex=True).str.zfill(14).str[:8],
        "status": np.where(df["Situacao"].str.startswith("Exclu"), "excluded", "registered"),
        "issue_date": _d(df["Data de Emissao"]),
        "dist_start": _d(df["Data do Inicio da Distribuicao"]),
        "maturity": _d(df["Data de Vencimento"]),
        "exit_date": _d(df["Data de Saida / Novo Vencimento"]),
        "exit_reason": df["Motivo de Saida"].str.split(" - ", n=1).str[1].str.upper(),
        "callable": df["Resgate Antecipado"].eq("S"),
        "icvm476": df["Registro CVM da Emissao"].str.upper().str.contains("476", na=False),
        "guarantee": df["Garantia/Especie"],
        "incent": df["Deb. Incent. (Lei 12.431)"].eq("S"),
        "qty_issued": pd.to_numeric(df["Quantidade Emitida"], errors="coerce"),
        "vn_issue": pd.to_numeric(df["Valor Nominal na Emissao"].str.replace(",", "."), errors="coerce"),
        "amort_start": _d(df["Amortizacao - Carencia"]),
        "index": df["indice"],
    })
    out["size_brl"] = out["qty_issued"] * out["vn_issue"]
    out["dist_start"] = out["dist_start"].fillna(out["issue_date"])
    ex = out["status"].eq("excluded")
    out["early_exit"] = ex & out["exit_date"].notna() & out["maturity"].notna() & \
        (out["exit_date"] < out["maturity"] - pd.Timedelta(days=30))
    # duplicates: keep the most recent issue per code (codes are re-used after old bonds die)
    out = out.sort_values(["codigo", "issue_date"]).drop_duplicates("codigo", keep="last")
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------------------------------------------------
RX = {
    # issuer announces early redemption / redemption offer / extraordinary amortisation of its debentures
    "resgate": r"resgate\s+antecipad|oferta\s+de\s+resgate|resgate\s+(total|facultativ|obrigat|parcial)|amortiza[çc][ãa]o\s+extraordin",
    # issuer buys back its own debentures in the market
    "deb_buyback": r"aquisi[çc][ãa]o\s+facultativa|recompra\s+de\s+deb|plano\s+de\s+recompra\s+de\s+deb",
    # equity raise (creditor-positive deleveraging, or a sign of cash need)
    "equity_raise": r"aumento\s+de\s+capital|oferta\s+p[úu]blica.{0,60}a[çc][õo]es|follow.?on|distribui[çc][ãa]o\s+prim[áa]ria.{0,60}a[çc][õo]es|subscri[çc][ãa]o\s+de\s+a[çc][õo]es",
    # M&A / corporate reorganisation / control change / tender offer
    "ma": r"incorpora[çc][ãa]o|fus[ãa]o|cis[ãa]o|aliena[çc][ãa]o\s+d[eo]\s+controle|transfer[êe]ncia\s+de\s+controle|\bopa\b|oferta\s+p[úu]blica\s+de\s+aquisi|reorganiza[çc][ãa]o\s+societ|combina[çc][ãa]o\s+de\s+neg|aquisi[çc][ãa]o\s+d[aeo]\s+(totalidade|controle|participa)",
    # judicial / extrajudicial recovery (control; distress)
    "rj": r"recupera[çc][ãa]o\s+(judicial|extrajudicial)",
}
WAIVER_RX = r"waiver|perd[ãa]o|anu[êe]ncia|n[ãa]o\s+declara|vencimento\s+antecipad|ren[úu]ncia|n[ãa]o\s+cumprimento|[íi]ndice[s]?\s+financeir|covenant"


def ipe_events() -> pd.DataFrame:
    frames = []
    for y in range(2021, 2027):
        p = CACHE / f"cvm_ipe_{y}.zip"
        if not p.exists():
            continue
        z = zipfile.ZipFile(p)
        df = pd.read_csv(z.open(z.namelist()[0]), sep=";", encoding="latin1", dtype=str)
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df["cnpj8"] = df["CNPJ_Companhia"].str.replace(r"\D", "", regex=True).str.zfill(14).str[:8]
    df["date"] = pd.to_datetime(df["Data_Entrega"], errors="coerce")
    df = df.dropna(subset=["date"])
    txt = (df["Categoria"].fillna("") + " | " + df["Tipo"].fillna("") + " | " + df["Assunto"].fillna("")).str.lower()
    cat = df["Categoria"].fillna("")
    ev = []
    # issuer-level corporate events: only from material-fact / market communication / offer / debenture notices
    rel = cat.str.contains("Fato Relevante|Comunicado ao Mercado|Aviso aos Debenturistas|Documentos de Oferta|"
                           "Reunião da Administração|Aviso aos Acionistas|Recuperação", regex=True)
    for k, rx in RX.items():
        m = txt.str.contains(rx, regex=True) & rel
        if k == "equity_raise":   # do not count debenture offer documents as equity raises
            m &= ~txt.str.contains(r"deb[êe]ntur|notas?\s+comerciai|cri\b|certificados", regex=True) | \
                 txt.str.contains(r"a[çc][õo]es", regex=True) & cat.str.contains("Fato Relevante")
        if k == "rj":
            m = (txt.str.contains(rx, regex=True) & cat.str.contains("Fato Relevante|Comunicado")) | \
                cat.str.contains("Informações de Companhias em Recuperação")
        ev.append(pd.DataFrame({"cnpj8": df.loc[m, "cnpj8"], "date": df.loc[m, "date"], "etype": k,
                                "text": df.loc[m, "Assunto"].fillna(df.loc[m, "Categoria"]).str[:160]}))
    # debenture-holder meetings (AGD): any AGDEB filing; the waiver subset by text
    agd = df["Tipo"].fillna("").eq("AGDEB") | txt.str.contains(r"assembleia\s+geral\s+de\s+debenturistas", regex=True)
    ev.append(pd.DataFrame({"cnpj8": df.loc[agd, "cnpj8"], "date": df.loc[agd, "date"], "etype": "agd",
                            "text": df.loc[agd, "Assunto"].fillna("").str[:160]}))
    w = agd & txt.str.contains(WAIVER_RX, regex=True)
    ev.append(pd.DataFrame({"cnpj8": df.loc[w, "cnpj8"], "date": df.loc[w, "date"], "etype": "agd_waiver",
                            "text": df.loc[w, "Assunto"].fillna("").str[:160]}))
    # a new debenture offer of a LISTED issuer (IPE): Anúncio de Início for debentures
    ni = cat.str.contains("Documentos de Oferta") & txt.str.contains(r"in[íi]cio", regex=True) & \
        txt.str.contains(r"deb[êe]ntur", regex=True)
    ev.append(pd.DataFrame({"cnpj8": df.loc[ni, "cnpj8"], "date": df.loc[ni, "date"], "etype": "ipe_new_deb",
                            "text": df.loc[ni, "Assunto"].fillna("").str[:160]}))
    E = pd.concat(ev, ignore_index=True)
    E["source"] = "cvm_ipe"
    return E


def rating_events() -> pd.DataFrame:
    r = pd.read_pickle(ROOT / "data" / "history" / "rating_events.pkl")
    r = r[r["direction"] != 0]
    return pd.DataFrame({"cnpj8": r["cnpj8"].astype(str).str.zfill(8), "date": pd.to_datetime(r["date"]).dt.normalize(),
                         "etype": np.where(r["direction"] > 0, "rating_up", "rating_down"),
                         "text": r["title"].astype(str).str[:160], "source": "ratings"})


def supply_events(ref: pd.DataFrame) -> pd.DataFrame:
    """Issuer taps the market: a new bond of the issuer starts distribution (SND). One event per issuer-day."""
    s = ref.dropna(subset=["dist_start"])
    s = s[s["dist_start"] >= "2020-06-01"]
    g = s.groupby(["cnpj8", "dist_start"]).agg(n=("codigo", "size"), size=("size_brl", "sum"),
                                               codes=("codigo", lambda v: ",".join(sorted(v)[:6]))).reset_index()
    return pd.DataFrame({"cnpj8": g["cnpj8"], "date": g["dist_start"], "etype": "supply_new_issue",
                         "text": g["codes"] + " size=" + (g["size"] / 1e6).round(0).astype(str) + "mm",
                         "source": "snd_registry"})


def build(force: bool = False):
    p1, p2 = OUT / "bonds_ref.pkl", OUT / "events_issuer.pkl"
    if p1.exists() and p2.exists() and not force:
        return pd.read_pickle(p1), pd.read_pickle(p2)
    ref = snd_registry()
    E = pd.concat([ipe_events(), rating_events(), supply_events(ref)], ignore_index=True)
    E = E.drop_duplicates(["cnpj8", "date", "etype"]).sort_values(["date", "cnpj8"]).reset_index(drop=True)
    ref.to_pickle(p1)
    E.to_pickle(p2)
    return ref, E


if __name__ == "__main__":
    ref, E = build(force=True)
    print(ref.shape, E.shape)
    print(E.groupby("etype").agg(n=("date", "size"), issuers=("cnpj8", "nunique"), first=("date", "min"),
                                 last=("date", "max")))
    print("early exits 2021+:", int((ref["early_exit"] & (ref["exit_date"] >= "2021-01-01")).sum()))
