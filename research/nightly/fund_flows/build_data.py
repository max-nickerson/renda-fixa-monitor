"""Parse CVM CDA (debenture holdings, BLC_4) + fund PL, and inf_diario (daily flows) into compact caches.

Outputs (data/history/nightly/fund_flows/):
  cda_deb.pkl : fund, ref (month-end DT_COMPTC), codigo, qty, val, buy, sell   (debenture positions, BLC_4)
  cda_pl.pkl  : fund, ref, pl, name                                         (fund net assets per CDA month)
  inf.pkl     : fund, day, pl, captc, resg, ncot  (daily, only funds that ever held debentures or credit-named funds)
Point-in-time: CDA rows carry a reference month; availability is applied later (ref + 60 calendar days).
inf_diario rows are used with a 5-business-day publication lag later.
"""
import os, re, zipfile, glob, io
import numpy as np, pandas as pd
D = "data/history/nightly/fund_flows"; RAW = f"{D}/raw"

def _cn(cols):
    return "CNPJ_FUNDO_CLASSE" if "CNPJ_FUNDO_CLASSE" in cols else "CNPJ_FUNDO"

def read_member(z, name, usecols_fn, filt=None, chunks=500_000):
    hdr = z.open(name).readline().decode("latin1").strip().split(";")
    cn = _cn(hdr)
    cols = usecols_fn(cn)
    out = []
    for ch in pd.read_csv(z.open(name), sep=";", encoding="latin1", usecols=cols, chunksize=chunks,
                          dtype=str, on_bad_lines="skip"):
        if filt is not None:
            ch = filt(ch)
        ch = ch.rename(columns={cn: "fund"})
        out.append(ch)
    return pd.concat(out) if out else pd.DataFrame()

def build_cda():
    deb, pl = [], []
    files = sorted(glob.glob(f"{RAW}/cda_fi_20*.zip"))
    for f in files:
        z = zipfile.ZipFile(f)
        for n in z.namelist():
            if "BLC_4" in n:
                d = read_member(z, n, lambda cn: [cn, "DT_COMPTC", "TP_APLIC", "CD_ATIVO", "QT_POS_FINAL",
                                                  "VL_MERC_POS_FINAL", "VL_AQUIS_NEGOC", "VL_VENDA_NEGOC"],
                                filt=lambda c: c[c["TP_APLIC"].str.startswith("Deb", na=False)])
                deb.append(d.drop(columns="TP_APLIC"))
            elif "_PL_" in n:
                d = read_member(z, n, lambda cn: [cn, "DENOM_SOCIAL", "DT_COMPTC", "VL_PATRIM_LIQ"])
                pl.append(d)
        print(f, sum(len(x) for x in deb), flush=True)
    deb = pd.concat(deb)
    deb = deb.rename(columns={"DT_COMPTC": "ref", "CD_ATIVO": "codigo", "QT_POS_FINAL": "qty", "VL_MERC_POS_FINAL": "val",
                              "VL_AQUIS_NEGOC": "buy", "VL_VENDA_NEGOC": "sell"})[["fund", "ref", "codigo", "qty", "val", "buy", "sell"]]
    for c in ["qty", "val", "buy", "sell"]:
        deb[c] = pd.to_numeric(deb[c], errors="coerce")
    deb["ref"] = pd.to_datetime(deb["ref"]).dt.to_period("M").dt.to_timestamp("M")
    deb["codigo"] = deb["codigo"].str.strip().str.upper()
    deb = deb.groupby(["fund", "ref", "codigo"], as_index=False)[["qty", "val", "buy", "sell"]].sum(min_count=1)
    pl = pd.concat(pl).rename(columns={"DENOM_SOCIAL": "name", "DT_COMPTC": "ref", "VL_PATRIM_LIQ": "pl"})[["fund", "name", "ref", "pl"]]
    pl["pl"] = pd.to_numeric(pl["pl"], errors="coerce")
    pl["ref"] = pd.to_datetime(pl["ref"]).dt.to_period("M").dt.to_timestamp("M")
    pl = pl.drop_duplicates(["fund", "ref"], keep="last")
    deb.to_pickle(f"{D}/cda_deb.pkl"); pl.to_pickle(f"{D}/cda_pl.pkl")
    print("cda", deb.shape, pl.shape, deb.ref.min(), deb.ref.max())
    return deb, pl

CREDIT_RX = re.compile(r"CR[EÉ]D(ITO)?\.? ?PRIV|CRED PRIV| CP\b|INCENTIVAD|DEB[EÊ]NTURE|INFRA|HIGH ?YIELD|12\.?431", re.I)

def build_inf(funds_keep):
    out = []
    files = sorted(glob.glob(f"{RAW}/inf_diario_fi_20*.zip"))
    for f in files:
        z = zipfile.ZipFile(f)
        for n in z.namelist():
            d = read_member(z, n, lambda cn: [cn, "DT_COMPTC", "VL_PATRIM_LIQ", "CAPTC_DIA", "RESG_DIA", "NR_COTST"],
                            filt=None)
            d = d[d["fund"].isin(funds_keep)]
            out.append(d)
        print(f, len(out[-1]), flush=True)
    inf = pd.concat(out)
    inf = inf.rename(columns={"DT_COMPTC": "day", "VL_PATRIM_LIQ": "pl", "CAPTC_DIA": "captc", "RESG_DIA": "resg", "NR_COTST": "ncot"})[["fund", "day", "pl", "captc", "resg", "ncot"]]
    inf["day"] = pd.to_datetime(inf["day"])
    for c in ["pl", "captc", "resg", "ncot"]:
        inf[c] = pd.to_numeric(inf[c], errors="coerce").astype("float64")
    # CVM175: several subclass rows per class possible -> aggregate
    inf = inf.groupby(["fund", "day"], as_index=False).agg(pl=("pl", "sum"), captc=("captc", "sum"),
                                                          resg=("resg", "sum"), ncot=("ncot", "sum"))
    inf.to_pickle(f"{D}/inf.pkl"); print("inf", inf.shape)
    return inf

if __name__ == "__main__":
    import sys
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("all", "cda"):
        deb, pl = build_cda()
    if what in ("all", "inf"):
        deb = pd.read_pickle(f"{D}/cda_deb.pkl"); pl = pd.read_pickle(f"{D}/cda_pl.pkl")
        keep = set(deb["fund"].unique())
        cad = pd.read_csv(f"{RAW}/cad_fi.csv", sep=";", encoding="latin1", usecols=["CNPJ_FUNDO", "DENOM_SOCIAL"], dtype=str)
        keep |= set(cad.loc[cad["DENOM_SOCIAL"].fillna("").str.contains(CREDIT_RX), "CNPJ_FUNDO"])
        z = zipfile.ZipFile(f"{RAW}/registro_fundo_classe.zip")
        rc = pd.read_csv(z.open("registro_classe.csv"), sep=";", encoding="latin1", dtype=str, usecols=["CNPJ_Classe", "Denominacao_Social"])
        keep |= set(rc.loc[rc["Denominacao_Social"].fillna("").str.contains(CREDIT_RX), "CNPJ_Classe"])
        print("funds kept", len(keep))
        build_inf(keep)
