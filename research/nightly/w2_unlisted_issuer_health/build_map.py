"""Point-in-time parent / controller map for debenture issuers that have no own (or harness-mapped) equity series.

Sources (all public CVM, dados.cvm.gov.br, cached in data/history/nightly/unlisted_issuer_health/raw):
  * FRE (Formulario de Referencia), per filer and version, available from DT_RECEB (receipt date of that version):
      - fre_cia_aberta_participacao_sociedade: the FILER's controlled / affiliated companies with CNPJ and % held
        -> edge child -> filer when % >= 50 (control).
      - fre_cia_aberta_posicao_acionaria: the FILER's shareholders (and their holders, via ID_Acionista_Relacionado)
        flagged Acionista_Controlador = 'S', legal persons with CNPJ -> edge filer -> controller.
  * FCA valor_mobiliario (read-only from structural_credit's cache): CNPJ -> B3 share tickers (listed companies).
  * cad_cia_aberta: CVM registration category (A/B) and status, used for descriptive coverage only.
  * Fallback: issuer-name / group-string match against the names of issuers that already have a harness ticker
    (research/data/issuer_sectors.csv names, research/data/equity_map.csv). Static (names are today's), flagged
    as map_type 'name' and reported separately.

Output: data/history/nightly/unlisted_issuer_health/parent_map_pit.pkl with rows
  (cnpj8, day, parent_cnpj8, parent_ticker, route, depth, avail)  for every monthly panel day (incl. 2026 so that
  the frozen rule can be evaluated on the holdout), only for issuers with no harness equity on that day.
"""
from __future__ import annotations

import io
import re
import unicodedata
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from research.nightly import harness as H

OUT = Path("research/nightly/w2_unlisted_issuer_health")
CACHE = H.HIST / "nightly" / "unlisted_issuer_health"
RAW = CACHE / "raw"
FCA = H.HIST / "nightly" / "structural_credit"


def c8(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace(r"\D", "", regex=True).str.zfill(14).str[:8]


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().upper()
    return re.sub(r"[^A-Z0-9 ]", " ", s)


def _read(z: zipfile.ZipFile, name: str) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(z.read(name)), sep=";", encoding="latin1", dtype=str, on_bad_lines="skip")


def fre_edges() -> pd.DataFrame:
    """child_cnpj8 -> parent_cnpj8 control edges with availability date (DT_RECEB of the FRE version)."""
    rows = []
    for f in sorted(RAW.glob("fre_cia_aberta_20*.zip")):
        y = f.stem[-4:]
        z = zipfile.ZipFile(f)
        docs = _read(z, f"fre_cia_aberta_{y}.csv")[["ID_DOC", "DT_RECEB", "DT_REFER"]].rename(columns={"ID_DOC": "ID_Documento"})
        ps = _read(z, f"fre_cia_aberta_participacao_sociedade_{y}.csv")
        ps["pct"] = pd.to_numeric(ps["Participacao_Emissor"], errors="coerce")
        ps = ps[(ps["pct"] >= 50) & ps["CNPJ"].notna()]
        e1 = pd.DataFrame({"child": c8(ps["CNPJ"]), "parent": c8(ps["CNPJ_Companhia"]), "pct": ps["pct"],
                           "ID_Documento": ps["ID_Documento"], "filer": c8(ps["CNPJ_Companhia"]), "src": "participacao"})
        pa = _read(z, f"fre_cia_aberta_posicao_acionaria_{y}.csv")
        pa = pa[(pa["Acionista_Controlador"] == "S") & (pa["Tipo_Pessoa_Acionista"] == "PJ") & pa["CPF_CNPJ_Acionista"].notna()]
        child = np.where(pa["CPF_CNPJ_Acionista_Relacionado"].notna() & (pa["Tipo_Pessoa_Acionista_Relacionado"] == "PJ"),
                         pa["CPF_CNPJ_Acionista_Relacionado"].fillna(""), pa["CNPJ_Companhia"])
        e2 = pd.DataFrame({"child": c8(pd.Series(child, index=pa.index)), "parent": c8(pa["CPF_CNPJ_Acionista"]),
                           "pct": pd.to_numeric(pa["Percentual_Total_Acoes_Circulacao"], errors="coerce"),
                           "ID_Documento": pa["ID_Documento"], "filer": c8(pa["CNPJ_Companhia"]), "src": "posicao"})
        e = pd.concat([e1, e2], ignore_index=True).merge(docs, on="ID_Documento", how="left")
        rows.append(e)
    E = pd.concat(rows, ignore_index=True)
    E["avail"] = pd.to_datetime(E["DT_RECEB"], errors="coerce")
    # control only: >= 50% (members of a controlling group with small stakes, e.g. Itausa 8.5% of NTS, are dropped)
    E = E[E["pct"].fillna(0) >= 50]
    E = E[E["avail"].notna() & (E["child"] != E["parent"]) & (E["parent"] != "00000000") & (E["child"] != "00000000")]
    return E.drop_duplicates(["child", "parent", "ID_Documento", "src"])


def listed_tickers() -> pd.DataFrame:
    """cnpj8 -> share tickers from FCA valor_mobiliario (any year), with listing end date."""
    rows = []
    for f in sorted(FCA.glob("fca_20*.zip")):
        y = f.stem[-4:]
        z = zipfile.ZipFile(f)
        v = _read(z, f"fca_cia_aberta_valor_mobiliario_{y}.csv")
        v = v[v["Codigo_Negociacao"].notna() & v["Valor_Mobiliario"].str.contains("A..es|Units|Unit", regex=True, na=False)]
        rows.append(pd.DataFrame({"cnpj8": c8(v["CNPJ_Companhia"]), "ticker": v["Codigo_Negociacao"].str.strip().str.upper(),
                                  "end": v["Data_Fim_Negociacao"]}))
    return pd.concat(rows, ignore_index=True).drop_duplicates(["cnpj8", "ticker"])


def harness_ticker_owners() -> dict:
    """cnpj8 -> ticker for the equity series the harness actually has (direct mappings in equity_map + FCA)."""
    eq = pd.read_pickle(H.HIST / "equity_daily.pkl")
    have = set(eq["ticker"].unique())
    mp = pd.read_csv(H.RDATA / "equity_map.csv", dtype=str)
    d = mp[(mp["mapping_type"] == "direct") & mp["ticker"].isin(have)]
    own = {k: g["ticker"].tolist() for k, g in d.groupby("cnpj8")}
    lt = listed_tickers()
    lt = lt[lt["ticker"].isin(have)]
    for k, g in lt.groupby("cnpj8"):
        own.setdefault(k, [])
        for t in g["ticker"]:
            if t not in own[k]:
                own[k].append(t)
    return own, have


GENERIC = set("""CIA COMPANHIA CENTRAIS CENTRAL CONCESSIONARIA ENERGIA ENERGETICA TRANSMISSORA TRANSMISSAO BANCO BCO
EMPRESA EMPRESAS GRUPO HOLDING PARTICIPACOES PARTICIPACAO BRASIL BRASILEIRA SERVICOS DISTRIBUIDORA GERACAO
ELETRICA ELETRICAS RODOVIAS RODOVIA SANEAMENTO AGUAS INVESTIMENTOS INFRAESTRUTURA SOCIEDADE NACIONAL ESTADO
SUL NORTE NORDESTE LESTE OESTE SAO PAULO RIO JANEIRO MINAS GERAIS PARANA BAHIA GOIAS SOLAR EOLICA EOLICAS
LOGISTICA TELECOM COMERCIO INDUSTRIA AGRICOLA ALIMENTOS SAUDE HOSPITAL REDE LOCACAO DISTRIBUICAO ENERGIAS
PRODUTOS SOLUCOES VALE CEARA ENGENHARIA PETROLEO LUIZ LOJAS PORTO AZUL ALIANCA MINERACAO RENOVAVEIS ENERGETICO
ENERGETICA ELETRICIDADE TRANSPORTADORA CONCESSOES CONSTRUCOES EMPREENDIMENTOS DESENVOLVIMENTO GERAL NOVA NOVO
AMBIENTAL COMPLEXO VENTOS SANTA SANTO AGRO ATLAS SERVICOS CORRETAGEM SEGUROS MOBILIDADE REALTY PAPEL CELULOSE
TERMINAL TERMINAIS PORTUARIO FERROVIA FERROVIARIA BIOENERGIA ACUCAR ALCOOL MEDICOS OPERACOES""".split())
# names whose token match was checked by hand and found wrong (manual-check sample, see README)
NAME_REJECT = {"08439659"}
# manual check of all 22 first-token candidates (2026-09-29): 10 correct, 12 wrong (e.g. LINHA -> Vamos, BRAZIL -> Cyrela,
# IMOBILIARIOS -> Multiplan, TENDA ATACADO -> Construtora Tenda).  Only the hand-verified group tokens are accepted.
NAME_ACCEPT = {"NEOENERGIA", "ECORODOVIAS", "SERENA", "RAIZEN", "AXIA", "LOCALIZA", "UNIDAS", "CORPOREOS"}   # CPFL Energias Renovaveis: controlled by CPFL (CPFE3), not matched correctly by tokens


def name_tokens(name: str) -> list[str]:
    return [t for t in norm(name).split() if len(t) >= 4 and t not in GENERIC and not t.isdigit()
            and t not in {"LTDA", "S A", "SPE", "EIRELI"}]


def build(panel_days: pd.DatetimeIndex, issuers_by_day: pd.DataFrame) -> pd.DataFrame:
    E = fre_edges()
    own, have = harness_ticker_owners()
    E.to_pickle(CACHE / "fre_edges.pkl")
    # PIT graph per decision day: for each (filer, src) keep the latest version received <= d
    out = []
    E = E.sort_values("avail")
    for d in panel_days:
        e = E[E["avail"] <= d]
        if e.empty:
            continue
        lastdoc = e.groupby(["filer", "src"])["avail"].transform("max")
        e = e[e["avail"] == lastdoc]
        # parents per child: prefer 'posicao' (child's own filing) over 'participacao', then larger pct
        e = e.assign(pr=(e["src"] == "posicao").astype(int)).sort_values(["pr", "pct"], ascending=[False, False])
        up = e.groupby("child")[["parent", "avail"]].apply(lambda g: list(zip(g["parent"], g["avail"]))).to_dict()
        todo = issuers_by_day.loc[issuers_by_day["day"] == d, "cnpj8"].unique()
        for c in todo:
            # BFS up the control chain until a cnpj8 that owns a harness equity series
            frontier, seen, hit = [(c, 0, pd.NaT)], {c}, None
            while frontier and hit is None:
                nxt = []
                for node, dep, av in frontier:
                    for par, a in up.get(node, []):
                        if par in seen or dep >= 4:
                            continue
                        seen.add(par)
                        a2 = max(a, av) if pd.notna(av) else a
                        if par in own:
                            hit = (par, own[par], dep + 1, a2)
                            break
                        nxt.append((par, dep + 1, a2))
                    if hit:
                        break
                frontier = nxt
            if hit:
                out.append((c, d, hit[0], "|".join(hit[1]), "fre", hit[2], hit[3]))
    F = pd.DataFrame(out, columns=["cnpj8", "day", "parent_cnpj8", "parent_tickers", "route", "depth", "avail"])
    return F


def name_map(unl: pd.DataFrame) -> pd.DataFrame:
    """Static fallback: distinctive group token of a harness-listed company inside the unlisted issuer's name."""
    own, have = harness_ticker_owners()
    names = pd.read_csv(H.RDATA / "issuer_sectors.csv", dtype=str).drop_duplicates("cnpj8").set_index("cnpj8")["issuer_name"]
    mp = pd.read_csv(H.RDATA / "equity_map.csv", dtype=str).drop_duplicates("cnpj8").set_index("cnpj8")["issuer_name"]
    names = names.combine_first(mp)
    tok_owner: dict[str, set] = {}
    for k in own:
        if k in names.index:
            for t in name_tokens(names[k]):               # any distinctive token of a listed owner's name
                tok_owner.setdefault(t, set()).add(k)
    tok_owner = {t: v for t, v in tok_owner.items() if len(v) == 1}
    rows = []
    for c in unl["cnpj8"].unique():
        if c not in names.index or c in own or c in NAME_REJECT:
            continue
        toks = name_tokens(names[c])[:1]                  # only the issuer's FIRST distinctive token (its group name)
        for t in toks:
            if t in tok_owner:
                p = next(iter(tok_owner[t]))
                rows.append((c, names[c], p, names.get(p, ""), "|".join(own[p]), t))
                break
    N = pd.DataFrame(rows, columns=["cnpj8", "issuer_name", "parent_cnpj8", "parent_name", "parent_tickers", "token"])
    N["accepted"] = N["token"].isin(NAME_ACCEPT)
    return N


if __name__ == "__main__":
    P = pd.concat([H.load_panel("M", holdout=True)], ignore_index=True)
    P = P[P["univ"]]
    unl = P.loc[P["eq_ticker"].isna(), ["cnpj8", "day"]].drop_duplicates()
    days = pd.DatetimeIndex(sorted(unl["day"].unique()))
    F = build(days, unl)
    F.to_pickle(CACHE / "parent_map_fre_pit.pkl")
    N = name_map(unl)
    N.to_pickle(CACHE / "name_map_static.pkl")
    N.to_csv(OUT / "name_map_static.csv", index=False)
    names = pd.read_csv(H.RDATA / "issuer_sectors.csv", dtype=str).drop_duplicates("cnpj8").set_index("cnpj8")["issuer_name"]
    s = F.drop_duplicates("cnpj8").copy()
    s["issuer_name"] = s["cnpj8"].map(names)
    s["parent_name"] = s["parent_cnpj8"].map(names)
    s.to_csv(OUT / "fre_map_sample.csv", index=False)
    print("unlisted issuers", unl["cnpj8"].nunique(), "fre-mapped issuers", F["cnpj8"].nunique(), "name-mapped", len(N))
    print(s.head(60).to_string())
    print(N.to_string())
