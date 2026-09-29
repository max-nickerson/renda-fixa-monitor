"""w2_expected_loss_carry / build_data.py

Builds the point-in-time inputs of the issuer distress-hazard model:

1. events.pkl        issuer-level distress events (cnpj8, date, type) from
                       a) research/nightly/bias_audit/distress_events_pit.csv (first DI+ mark < 0.85 or public filing)
                       b) severe rating actions in data/history/rating_events.pkl (downgrade to CCC / CC / C / D / RD,
                          'calote', 'default', RJ-driven downgrades; the file has no rating levels, so a '>= 3 notch'
                          rule cannot be computed and is approximated by the text rule)
                       c) bond loss episodes: patched harness return path (v4, gap moves booked at realisation)
                          falling > 10% below its trailing 126-bday high (dated the day the mark shows it)
                       d) CVM IPE filings: first 'Informações de Companhias em Recuperação Judicial ou Extrajudicial'
                          document or a Fato Relevante / Comunicado whose subject announces an RJ/RE request or a
                          payment default (dated by Data_Entrega), 2019-2026 for EVERY CVM issuer (labels for the
                          broad fundamentals training panel)
   Each issuer's events are collapsed into episodes: an event within 365 days of the previous one of the same
   issuer continues the episode (only the episode start is an 'onset').
2. fund_all.pkl      CVM ITR/DFP fundamentals for ALL CVM filers (not only the debenture universe), built with the
                     unmodified functions of research/data_fundamentals.py (imported, not edited), then merged with
                     data/history/fundamentals_pit.pkl (which adds brapi rows for the universe).  PIT key =
                     available_date_strict.
Caches: data/history/nightly/w2_expected_loss_carry/
"""
from __future__ import annotations

import io
import re
import sys
import time
import zipfile
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

from research.nightly import harness as H

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "data" / "history" / "nightly" / "w2_expected_loss_carry"
OUT.mkdir(parents=True, exist_ok=True)
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


# ---------------------------------------------------------------------------------------------------- IPE
def ipe_zip(y: int) -> Path | None:
    p = ROOT / "data" / "cache" / f"cvm_ipe_{y}.zip"
    if p.exists():
        return p
    q = OUT / f"cvm_ipe_{y}.zip"
    if q.exists():
        return q
    url = f"https://dados.cvm.gov.br/dados/CIA_ABERTA/DOC/IPE/DADOS/ipe_cia_aberta_{y}.zip"
    log("download", url)
    r = httpx.get(url, timeout=120, follow_redirects=True, headers={"User-Agent": "Mozilla/5.0 (research)"})
    r.raise_for_status()
    q.write_bytes(r.content)
    return q


RJ_RE = re.compile(r"recupera[cç][aã]o\s+(judicial|extrajudicial)|pedido de recupera|inadimpl|n[aã]o pagamento|"
                   r"default|fal[eê]ncia|reestrutura[cç][aã]o de d[ií]vida", re.I)
RJ_NEG = re.compile(r"encerramento|sa[ií]da da recupera|homologa[cç][aã]o do plano|aprova[cç][aã]o do plano|"
                    r"cumprimento|t[eé]rmino", re.I)


def ipe_events() -> pd.DataFrame:
    rows = []
    for y in range(2019, 2027):
        p = ipe_zip(y)
        z = zipfile.ZipFile(p)
        df = pd.read_csv(z.open(z.namelist()[0]), sep=";", encoding="latin1", dtype=str)
        df["cnpj8"] = df["CNPJ_Companhia"].str.replace(r"\D", "", regex=True).str[:8]
        df["date"] = pd.to_datetime(df["Data_Entrega"], errors="coerce").dt.normalize()
        cat = df["Categoria"].fillna("")
        subj = (df["Tipo"].fillna("") + " " + df["Assunto"].fillna(""))
        a = cat.str.contains("Recupera", case=False)
        b = cat.isin(["Fato Relevante", "Comunicado ao Mercado"]) & subj.str.contains(RJ_RE) & ~subj.str.contains(RJ_NEG)
        e = df[a | b][["cnpj8", "date", "Nome_Companhia"]].copy()
        e["type"] = np.where(a[a | b], "ipe_rj_doc", "ipe_rj_fact")
        rows.append(e)
    E = pd.concat(rows, ignore_index=True).dropna(subset=["date"])
    # A Fato Relevante mentioning RJ / default is often about a COUNTERPARTY (Vale, Klabin, Engie, Natura...
    # disclosing exposure to someone else's RJ).  Keep a 'fact' only when the issuer itself files RJ documents
    # within 400 days of it, or its registered name carries 'RECUPERA' / 'FALID' / 'LIQUIDA'.
    doc = E[E["type"] == "ipe_rj_doc"].groupby("cnpj8")["date"].apply(list)
    nm = E["Nome_Companhia"].fillna("").str.upper()
    own = nm.str.contains("RECUPERA|FALID|LIQUIDA")
    def near(r):
        return any(abs((d - r["date"]).days) <= 400 for d in doc.get(r["cnpj8"], []))
    keep = (E["type"] == "ipe_rj_doc") | own | E.apply(near, axis=1)
    return E[keep]


# ---------------------------------------------------------------------------------------------------- ratings
SEV_RE = re.compile(r"(to|para)[-\s]*['‘\"“]?(ccc|cc|c|d|rd|sd)(\+|-)?[-\s(]|(to|para)[-\s]*['‘\"“]?(ccc|cc|c|d|rd)(\(bra\)|-bra|\.br)|"
                    r"calote|default|recupera[cç][aã]o (judicial|extrajudicial)|alto risco|n[aã]o pagamento", re.I)


def rating_events() -> pd.DataFrame:
    r = pd.read_pickle(H.HIST / "rating_events.pkl")
    d = r[(r["direction"] < 0) & r["title"].astype(str).str.contains(SEV_RE)].copy()
    d["date"] = pd.to_datetime(d["date"]).dt.normalize()
    d["type"] = "rating_severe"
    return d[["cnpj8", "date", "type", "title"]]


# ---------------------------------------------------------------------------------------------------- bond losses
def bond_loss_events(thr: float = -0.10, win: int = 126) -> pd.DataFrame:
    C = H._core()
    dd = C["days"]
    iss = H._issuer_of_b()
    # market-neutral path: R minus the same-day median R of on-grid bonds of the same kind (IPCA / PRE rate-hedged
    # returns carry hedge noise and curve moves; a raw -10% rule flags 236 IPCA issuers, mostly rates)
    R = H._Rmat("base")
    Pw = H.load_panel("W", holdout=True)
    kind = Pw.drop_duplicates("b", keep="last").set_index("b")["kind"].reindex(range(R.shape[1])).fillna("?").to_numpy()
    on = C["TD"] >= 0
    Rr = R.copy()
    for k in np.unique(kind):
        m = kind == k
        X = np.where(on[:, m], R[:, m], np.nan)
        med = np.nan_to_num(np.nanmedian(X, axis=1)) if np.isfinite(X).any() else 0
        Rr[:, m] = np.where(on[:, m], R[:, m] - med[:, None], R[:, m])
    del Pw
    LC = np.vstack([np.zeros((1, R.shape[1])), np.cumsum(np.log1p(np.clip(Rr, -0.99, None)), axis=0)])
    ND1, NB = LC.shape
    rows = []
    # rolling max over the trailing window (inclusive) using a strided approach per block of bonds
    s = pd.DataFrame(LC)
    rm = s.rolling(win + 1, min_periods=1).max().to_numpy()
    dd_ = LC - rm
    # DI+ floaters: -10% (spec).  IPCA / PRE: -20% market-neutral, because at -10% only 10% of the IPCA issuers
    # flagged have any hard distress event (rate-hedge noise on long real-rate duration), vs 62% for DI+.
    thr_b = np.where(kind == "DI_SPREAD", thr, 2 * thr)
    hit = dd_ < np.log1p(thr_b)[None, :]
    for b in np.flatnonzero(hit.any(axis=0)):
        ps = np.flatnonzero(hit[:, b])
        # episodes: new episode when the previous hit is > 63 bdays earlier
        starts = ps[np.r_[True, np.diff(ps) > 63]]
        for p in starts:
            p = min(int(p), len(dd) - 1)
            rows.append((iss[b], dd[p], C["codes"][b], float(np.expm1(dd_[p, b])), kind[b]))
    E = pd.DataFrame(rows, columns=["cnpj8", "date", "codigo", "dd", "kind"])
    E["type"] = "bond_loss"
    return E


# ---------------------------------------------------------------------------------------------------- fundamentals
def fund_all() -> pd.DataFrame:
    path = OUT / "fund_all.pkl"
    if path.exists():
        return pd.read_pickle(path)
    sys.path.insert(0, str(ROOT / "research"))
    import data_fundamentals as DF          # imported, never edited
    # universe = every CVM filer: collect cnpj8s from the ITR index files
    U = set()
    for y in DF.YEARS:
        for kind in ("itr", "dfp"):
            p = DF.CVM_DIR / f"{kind}_cia_aberta_{y}.zip"
            if not p.exists():
                continue
            z = zipfile.ZipFile(p)
            i = DF._read(z, f"{kind}_cia_aberta_{y}.csv", ["CNPJ_CIA"])
            U |= set(i.CNPJ_CIA.str.replace(r"\D", "", regex=True).str[:8])
    log("CVM filers", len(U))
    idx, L = DF.load_cvm(U, refresh=False)
    cv = DF.cvm_filings(idx, L)
    del L
    cv["source_grp"] = "cvm"
    f = DF.finish(cv)
    f = DF.pit_order(f)
    f.to_pickle(path)
    log("fund_all rows", len(f), "issuers", f.cnpj8.nunique())
    return f


def episodes(E: pd.DataFrame, gap_days: int = 365) -> pd.DataFrame:
    E = E.sort_values(["cnpj8", "date"]).reset_index(drop=True)
    prev = E.groupby("cnpj8")["date"].shift()
    E["onset"] = prev.isna() | ((E["date"] - prev).dt.days > gap_days)
    return E


def main():
    ev = []
    d = pd.read_csv(ROOT / "research/nightly/bias_audit/distress_events_pit.csv", dtype={"cnpj8": str})
    d["date"] = pd.to_datetime(d["event_date"])
    d["type"] = "bias_audit_distress"
    ev.append(d[["cnpj8", "date", "type"]])
    log("bias_audit events", len(d))
    r = rating_events()
    log("severe rating events", len(r), r["cnpj8"].nunique())
    ev.append(r[["cnpj8", "date", "type"]])
    b = bond_loss_events()
    log("bond loss events", len(b), b["cnpj8"].nunique())
    b.to_pickle(OUT / "bond_loss_events.pkl")
    ev.append(b[["cnpj8", "date", "type"]])
    ip = ipe_events()
    log("ipe events", len(ip), ip["cnpj8"].nunique())
    ip.to_pickle(OUT / "ipe_events_raw.pkl")
    ev.append(ip[["cnpj8", "date", "type"]])
    E = pd.concat(ev, ignore_index=True)
    E["cnpj8"] = E["cnpj8"].astype(str).str.zfill(8)
    E = E[E["cnpj8"].str.fullmatch(r"\d{8}")]
    E = episodes(E)
    E.to_pickle(OUT / "events.pkl")
    log("events total", len(E), "onsets", int(E["onset"].sum()))
    F = fund_all()
    log("fund_all", F.shape)


if __name__ == "__main__":
    main()
