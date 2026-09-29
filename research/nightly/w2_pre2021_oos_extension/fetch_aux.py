"""Auxiliary public data for the pre-2021 extension (cached in data/history/nightly/pre2021_oos_extension/):
CVM IPE filings 2013-2020 (dated by Data_Entrega = delivery date, PIT), BCB SGS 12 (CDI) and 433 (IPCA) from 2012."""
import time
from pathlib import Path
import httpx, pandas as pd

OUT = Path("data/history/nightly/pre2021_oos_extension")
for y in range(2013, 2021):
    p = OUT / f"cvm_ipe_{y}.zip"
    if not p.exists():
        r = httpx.get(f"https://dados.cvm.gov.br/dados/CIA_ABERTA/DOC/IPE/DADOS/ipe_cia_aberta_{y}.zip", timeout=120)
        r.raise_for_status(); p.write_bytes(r.content); print("ipe", y, len(r.content)); time.sleep(1)
for code in (12, 433):
    p = OUT / f"bcb_{code}.csv"
    if p.exists():
        continue
    fr = []
    for a, b in [("01/01/2012", "31/12/2021"), ("01/01/2022", "31/12/2026")]:
        r = httpx.get(f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{code}/dados?formato=json&dataInicial={a}&dataFinal={b}", timeout=60)
        fr.append(pd.DataFrame(r.json()))
    s = pd.concat(fr)
    s["date"] = pd.to_datetime(s["data"], format="%d/%m/%Y"); s["valor"] = s["valor"].astype(float)
    s[["date", "valor"]].to_csv(p, index=False); print("bcb", code, len(s))
