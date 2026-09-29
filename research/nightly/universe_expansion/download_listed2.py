"""Second pass: every FII registered at CVM with median CRI/CRA share of investments > 60% and B3 listing (ticker
derived from ISIN), fetched from brapi if not already present. Reduces (not removes) survivorship bias."""
import pickle, time, zipfile, pandas as pd
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(".env")
from rfmonitor.sources import brapi
RAW = Path("data/history/nightly/universe_expansion/raw")
A, G = [], []
for y in range(2021, 2027):
    z = zipfile.ZipFile(RAW / f"inf_mensal_fii_{y}.zip")
    for n in z.namelist():
        if "ativo" in n: A.append(pd.read_csv(z.open(n), sep=";", encoding="latin1", dtype=str))
        if "geral" in n: G.append(pd.read_csv(z.open(n), sep=";", encoding="latin1", dtype=str))
A, G = pd.concat(A), pd.concat(G)
num = lambda s: pd.to_numeric(s, errors="coerce").fillna(0)
A["sh"] = (num(A.CRI) + num(A.CRI_CRA)) / pd.to_numeric(A.Total_Investido, errors="coerce")
g = G.drop_duplicates("CNPJ_Fundo_Classe", keep="last").set_index("CNPJ_Fundo_Classe")
g["crish"] = A.groupby("CNPJ_Fundo_Classe").sh.median()
cand = g[(g.crish > 0.6) & (g.Mercado_Negociacao_Bolsa == "S")].copy()
cand["tick"] = cand.Codigo_ISIN.str[2:6] + "11"
ticks = sorted(t for t in cand.tick.dropna().unique() if t[0].isalpha())
cand[["tick", "crish", "Nome_Fundo_Classe", "Codigo_ISIN"]].to_csv(RAW / "cvm_credit_fii_candidates.csv")
out = pickle.load(open(RAW / "listed_credit_brapi.pkl", "rb"))
for t in ticks:
    if t in out: continue
    try:
        q = brapi._get(f"/quote/{t}", range="max", interval="1d", dividends="true")["results"][0]
        out[t] = {"group": "CVM", "hist": q.get("historicalDataPrice") or [],
                  "divs": (q.get("dividendsData") or {}).get("cashDividends") or [], "name": q.get("longName")}
        print(t, len(out[t]["hist"]), len(out[t]["divs"]))
    except Exception as e:
        out[t] = {"group": "CVM", "hist": [], "divs": [], "err": repr(e)[:80]}; print(t, "ERR")
    time.sleep(0.4)
pickle.dump(out, open(RAW / "listed_credit_brapi.pkl", "wb"))
