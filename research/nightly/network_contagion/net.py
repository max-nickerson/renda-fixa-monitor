"""Network construction (point-in-time where the source allows) for the network_contagion agent.

Networks between issuers (cnpj8):
  * group   : corporate group. Union of (a) equity_map ticker roots (direct + parent mappings, static hand map),
              (b) curated brand tokens in the issuer name (static), (c) CVM FRE controlling shareholders
              (PIT: an edge exists from the FRE document's receipt date DT_RECEB on).
  * sector  : research/data/issuer_sectors.csv (static).
  * expo    : cosine similarity of sector/issuer macro-commodity exposure vectors (research/data/sector_exposures.csv,
              issuer_overrides.csv) between issuers of DIFFERENT sectors (supply-chain / common-driver link).
  * holders : common fund holders from CVM CDA (fund portfolio disclosures, BLC_4 'Debentures'). A quarter-end
              snapshot is used only from reference month end + 100 days (covers the 90-day confidentiality lag).
Caches: data/history/nightly/network_contagion/.
"""
from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
RD = ROOT / "research" / "data"
CD = ROOT / "data" / "history" / "nightly" / "network_contagion"
CD.mkdir(parents=True, exist_ok=True)

BRANDS = {  # brand token in issuer name -> group key (curated from tokens appearing in >= 2 issuer names)
    "ENERGISA": "ENGI", "REENERGISA": "ENGI", "EDP": "ENBR", "AXS": "AXS", "BRK": "BRK", "EQUATORIAL": "EQTL",
    "MEZ": "MEZ", "ATHON": "ATHON", "EVOLTZ": "EVOLTZ", "ARGO": "ARGO", "COPEL": "CPLE", "SERENA": "SRNA",
    "AUREN": "AURE", "CPFL": "CPFE", "ENEL": "ENEL", "ECOVIAS": "ECOR", "ECORODOVIAS": "ECOR", "NEOENERGIA": "NEOE",
    "IGUA": "IGUA", "EPR": "EPR", "THOPEN": "THOPEN", "COCAL": "COCAL", "LOCALIZA": "RENT", "ENGIE": "EGIE",
    "AXIA": "AXIA", "VAMOS": "VAMO", "ALGAR": "ALGAR", "CEMIG": "CMIG", "RUMO": "RAIL", "UNIDAS": "RENT",
    "CELESC": "CLSC", "ORIZON": "ORVR", "BTG": "BPAC", "RAIZEN": "RAIZ", "TIM": "TIMS",
    "TENDA": "TEND", "CIMED": "CIMED", "GNA": "GNA", "HELEXIA": "HELEXIA", "SOLFACIL": "SOLFACIL",
    "AZUL": "AZUL", "QLUZ": "QLUZ", "FORTBRASIL": "FORTBRASIL", "IPIRANGA": "UGPA",
    "AGEO": "AGEO", "ATLAS": "ATLAS", "RIALMA": "RIALMA", "AEGEA": "AEGEA", "SABESP": "SBSP", "COPASA": "CSMG",
    "SIMPAR": "SIMH", "JSL": "SIMH", "MOVIDA": "SIMH", "CCR": "MOTV", "MOTIVA": "MOTV", "ELETROBRAS": "AXIA",
    "FURNAS": "AXIA", "CHESF": "AXIA", "ELETRONORTE": "AXIA", "ELETROSUL": "AXIA", "COSAN": "CSAN",
    "COMGAS": "CSAN", "ALUPAR": "ALUP", "TAESA": "TAEE", "LIGHT": "LIGT", "OMEGA": "SRNA", "ENEVA": "ENEV",
    "PRIO": "PRIO", "HAPVIDA": "HAPV", "REDE D OR": "RDOR", "DASA": "DASA", "VIA": None,
}
GOV_WORDS = ("GOVERNO", "ESTADO", "MUNICIPIO", "TESOURO", "UNIAO", "FUNDO", "PREVI", "FUNDACAO", "BNDES",
             "PREFEITURA", "SECRETARIA", "FAZENDA", "CAIXA DE PREV", "PETROS", "FUNCEF", "FII", "FIP")


def _c8(s):
    return re.sub(r"\D", "", str(s)).zfill(14)[:8] if pd.notna(s) and re.sub(r"\D", "", str(s)) else None


def issuers() -> pd.DataFrame:
    s = pd.read_csv(RD / "issuer_sectors.csv", dtype=str)[["cnpj8", "issuer_name", "sector"]].drop_duplicates("cnpj8")
    s["cnpj8"] = s["cnpj8"].str.zfill(8)
    return s.reset_index(drop=True)


class UF:
    def __init__(self):
        self.p = {}

    def f(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def u(self, a, b):
        ra, rb = self.f(a), self.f(b)
        if ra != rb:
            self.p[rb] = ra


def static_group_edges() -> list[tuple[str, str, str]]:
    """(cnpj8, group-key, source) edges from equity_map ticker roots and brand tokens."""
    iss = issuers()
    edges = []
    em = pd.read_csv(RD / "equity_map.csv", dtype=str)
    em["cnpj8"] = em["cnpj8"].str.zfill(8)
    for c8, t in em.dropna(subset=["ticker"])[["cnpj8", "ticker"]].itertuples(index=False):
        edges.append((c8, "G:" + t.strip()[:4], "equity_map"))
    for c8, nm in iss[["cnpj8", "issuer_name"]].itertuples(index=False):
        toks = set(re.sub(r"[^A-Z0-9 ]", " ", str(nm).upper()).split())
        for k, g in BRANDS.items():
            if g and k in toks:
                edges.append((c8, "G:" + g, "brand"))
    return edges


def fre_edges() -> pd.DataFrame:
    """Controller edges from CVM FRE: company cnpj8 -> controlling PJ shareholder cnpj8, with receipt date."""
    path = CD / "fre_edges.pkl"
    if path.exists():
        return pd.read_pickle(path)
    rows = []
    for z in sorted((CD / "fre").glob("fre_cia_aberta_*.zip")):
        y = z.stem[-4:]
        zf = zipfile.ZipFile(z)
        try:
            meta = pd.read_csv(zf.open(f"fre_cia_aberta_{y}.csv"), sep=";", encoding="latin1", dtype=str)
            pa = pd.read_csv(zf.open(f"fre_cia_aberta_posicao_acionaria_{y}.csv"), sep=";", encoding="latin1",
                             dtype=str)
        except KeyError:
            continue
        meta = meta[["ID_DOC", "DT_RECEB"]].rename(columns={"ID_DOC": "ID_Documento"})
        pa = pa[(pa["Acionista_Controlador"] == "S") & (pa["Tipo_Pessoa_Acionista"] == "PJ")
                & pa["ID_Acionista_Relacionado"].isna()]          # direct shareholders only (no nested chain rows)
        pa = pa.merge(meta, on="ID_Documento", how="left")
        for r in pa.itertuples(index=False):
            a, b = _c8(r.CNPJ_Companhia), _c8(r.CPF_CNPJ_Acionista)
            if a and b and a != b:
                pct = pd.to_numeric(r.Percentual_Total_Acoes_Circulacao, errors="coerce")
                rows.append((a, b, str(r.Acionista).upper(), pd.to_datetime(r.DT_RECEB, errors="coerce"), pct))
    E = pd.DataFrame(rows, columns=["cnpj8", "ctrl8", "ctrl_name", "recv", "pct"]).dropna(subset=["recv"])
    E = E.sort_values(["recv", "pct"]).groupby(["cnpj8", "ctrl8", "recv"], as_index=False).last()
    import unicodedata
    E["ctrl_name"] = E["ctrl_name"].apply(lambda s: unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode())
    bad = E["ctrl_name"].apply(lambda s: any(w in s for w in GOV_WORDS))
    E = E[~bad]
    ncon = E.groupby("ctrl8")["cnpj8"].nunique()
    E = E[E["ctrl8"].map(ncon) <= 25]
    # keep one row per (company, controller, receipt date): later documents may change the stake
    E.to_pickle(path)
    return E


def groups_asof(day: pd.Timestamp, use_fre: bool = True, min_pct: float = 50.0) -> pd.Series:
    """cnpj8 -> group id valid at `day` (static maps + FRE majority-controller edges whose parent is itself an
    issuer; the latest FRE document received <= day decides the stake)."""
    uf = UF()
    iss = issuers()
    for c8 in iss["cnpj8"]:
        uf.f("C:" + c8)
    for c8, g, _ in static_group_edges():
        uf.u("C:" + c8, g)
    if use_fre:
        E = fre_edges()
        E = E[(E["recv"] <= day) & E["ctrl8"].isin(set(iss["cnpj8"]))]   # parent is itself an issuer
        E = E.sort_values("recv").groupby(["cnpj8", "ctrl8"]).last().reset_index()
        E = E[E["pct"] >= min_pct]
        for a, b in zip(E["cnpj8"], E["ctrl8"]):
            uf.u("C:" + a, "C:" + b)
    out = {c8: uf.f("C:" + c8) for c8 in iss["cnpj8"]}
    return pd.Series(out, name="group")


def affiliates_asof(day: pd.Timestamp, lo: float = 20.0, hi: float = 50.0) -> list[tuple[str, str]]:
    """JV / significant-stake links (FRE controlling-block holder with lo <= stake < hi), parent must be an issuer."""
    iss = set(issuers()["cnpj8"])
    E = fre_edges()
    E = E[(E["recv"] <= day) & E["ctrl8"].isin(iss) & E["cnpj8"].isin(iss)]
    E = E.sort_values("recv").groupby(["cnpj8", "ctrl8"]).last().reset_index()
    E = E[(E["pct"] >= lo) & (E["pct"] < hi)]
    return list(zip(E["cnpj8"], E["ctrl8"]))


def exposure_sim() -> pd.DataFrame:
    """Issuer x issuer cosine similarity of exposure vectors (sector row, replaced by issuer overrides)."""
    iss = issuers()
    se = pd.read_csv(RD / "sector_exposures.csv").dropna(subset=["variable"])
    ov = pd.read_csv(RD / "issuer_overrides.csv", dtype={"cnpj8": str}).dropna(subset=["variable"])
    ov["cnpj8"] = ov["cnpj8"].str.zfill(8)
    S = se.pivot_table(index="sector", columns="variable", values="weight", aggfunc="sum").fillna(0)
    O = ov.pivot_table(index="cnpj8", columns="variable", values="weight", aggfunc="sum").fillna(0)
    cols = sorted(set(S.columns) | set(O.columns))
    S, O = S.reindex(columns=cols, fill_value=0), O.reindex(columns=cols, fill_value=0)
    V = S.reindex(iss["sector"]).fillna(0).to_numpy()
    V = pd.DataFrame(V, index=iss["cnpj8"], columns=cols)
    V.loc[O.index.intersection(V.index)] = O.loc[O.index.intersection(V.index)].to_numpy()
    X = V.to_numpy()
    nrm = np.linalg.norm(X, axis=1, keepdims=True)
    Xn = np.divide(X, nrm, out=np.zeros_like(X), where=nrm > 0)
    return pd.DataFrame(Xn @ Xn.T, index=V.index, columns=V.index)


# ------------------------------------------------------------------------------------------------ CDA fund holdings
def _read_cda_month(zf: zipfile.ZipFile, ym: str) -> pd.DataFrame | None:
    names = [n for n in zf.namelist() if n.startswith(f"cda_fi_BLC_4_{ym}")]
    if not names:
        return None
    use = ["CNPJ_FUNDO_CLASSE", "CNPJ_FUNDO", "TP_APLIC", "CD_ATIVO", "VL_MERC_POS_FINAL"]
    d = pd.read_csv(zf.open(names[0]), sep=";", encoding="latin1", dtype=str,
                    usecols=lambda c: c in use)
    d = d.rename(columns={"CNPJ_FUNDO": "CNPJ_FUNDO_CLASSE"})
    d = d[d["TP_APLIC"].str.startswith("Deb", na=False)]
    d["v"] = pd.to_numeric(d["VL_MERC_POS_FINAL"], errors="coerce")
    d = d[d["v"] > 0]
    d["fund"] = d["CNPJ_FUNDO_CLASSE"].str.replace(r"\D", "", regex=True)
    d["codigo"] = d["CD_ATIVO"].str.strip().str.upper()
    d["ref"] = pd.Timestamp(f"{ym[:4]}-{ym[4:]}-01") + pd.offsets.MonthEnd(0)
    return d[["fund", "codigo", "v", "ref"]]


def cda_holdings() -> pd.DataFrame:
    """Quarter-end fund x bond holdings (debentures), with avail = ref + 100 days."""
    path = CD / "cda_holdings.pkl"
    if path.exists():
        return pd.read_pickle(path)
    out = []
    for z in sorted((CD / "cda").glob("cda_fi_*.zip")):
        zf = zipfile.ZipFile(z)
        tag = z.stem.split("_")[-1]
        if len(tag) == 4:   # annual history file: one csv with all months -> keep quarter-end months
            nm = [n for n in zf.namelist() if n.startswith(f"cda_fi_BLC_4_{tag}")][0]
            use = ["CNPJ_FUNDO_CLASSE", "CNPJ_FUNDO", "DT_COMPTC", "TP_APLIC", "CD_ATIVO", "VL_MERC_POS_FINAL"]
            for ch in pd.read_csv(zf.open(nm), sep=";", encoding="latin1", dtype=str, usecols=lambda c: c in use,
                                  chunksize=500_000):
                ch = ch.rename(columns={"CNPJ_FUNDO": "CNPJ_FUNDO_CLASSE"})
                ch = ch[ch["TP_APLIC"].str.startswith("Deb", na=False)]
                dt = pd.to_datetime(ch["DT_COMPTC"], errors="coerce")
                ch = ch[dt.dt.month.isin([3, 6, 9, 12])]
                ch["v"] = pd.to_numeric(ch["VL_MERC_POS_FINAL"], errors="coerce")
                ch = ch[ch["v"] > 0]
                ch["fund"] = ch["CNPJ_FUNDO_CLASSE"].str.replace(r"\D", "", regex=True)
                ch["codigo"] = ch["CD_ATIVO"].str.strip().str.upper()
                ch["ref"] = pd.to_datetime(ch["DT_COMPTC"]) + pd.offsets.MonthEnd(0)
                out.append(ch[["fund", "codigo", "v", "ref"]])
            print("cda", tag, flush=True)
        else:
            d = _read_cda_month(zf, tag)
            if d is not None:
                out.append(d)
                print("cda", tag, len(d), flush=True)
    H = pd.concat(out, ignore_index=True)
    H = H.groupby(["ref", "fund", "codigo"], as_index=False)["v"].sum()
    H["avail"] = H["ref"] + pd.Timedelta(days=100)
    H.to_pickle(path)
    return H
