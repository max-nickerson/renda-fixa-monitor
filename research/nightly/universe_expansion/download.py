"""Download free public data for the universe-expansion study into data/history/nightly/universe_expansion/raw.
Sources: ANBIMA index history (S3 public), CVM securitization monthly reports (CRI/CRA), CVM FIDC monthly reports,
FRED ICE BofA EM corporate indices (only a rolling ~3y window is public), Tesouro Direto prices/rates."""
import time, httpx
from pathlib import Path
RAW = Path("data/history/nightly/universe_expansion/raw"); RAW.mkdir(parents=True, exist_ok=True)
c = httpx.Client(timeout=120, follow_redirects=True, headers={"User-Agent": "Mozilla/5.0"})
def get(url, name, force=False):
    p = RAW / name
    if p.exists() and p.stat().st_size > 0 and not force:
        return p
    for k in range(3):
        try:
            r = c.get(url)
            if r.status_code == 200:
                p.write_bytes(r.content); print("ok", name, len(r.content)); return p
            print("status", r.status_code, url); return None
        except Exception as e:
            print("err", url, repr(e)[:80]); time.sleep(3 * (k + 1))
S3 = "https://s3-data-prd-use1-precos.s3.us-east-1.amazonaws.com/arquivos/indices-historico/{}-HISTORICO.xls"
for ix in ["IMAB5", "IMAB", "IMAB5MAIS", "IRFM", "IMAS", "IMAGERAL", "IDKAIPCA2A", "IDADI", "IDAIPCA", "IDAGERAL"]:
    get(S3.format(ix), f"anbima_{ix}.xls")
B = "https://dados.cvm.gov.br/dados"
for kind in ["CRI", "CRA"]:
    for y in range(2020, 2027):
        get(f"{B}/SECURIT/DOC/INF_MENSAL_{kind}/DADOS/inf_mensal_{kind.lower()}_{y}.zip", f"inf_mensal_{kind.lower()}_{y}.zip")
    get(f"{B}/SECURIT/DOC/INF_MENSAL_{kind}/META/meta_inf_mensal_{kind.lower()}.zip", f"meta_inf_mensal_{kind.lower()}.zip")
for y in range(2020, 2025):
    get(f"{B}/FIDC/DOC/INF_MENSAL/DADOS/HIST/inf_mensal_fidc_{y}.zip", f"inf_mensal_fidc_{y}.zip")
for y in (2025, 2026):
    for m in range(1, 13):
        if y == 2026 and m > 8: break
        get(f"{B}/FIDC/DOC/INF_MENSAL/DADOS/inf_mensal_fidc_{y}{m:02d}.zip", f"inf_mensal_fidc_{y}{m:02d}.zip")
for sid in ["BAMLEMCBPITRIV", "BAMLEMRLCRPILATRIV", "BAMLEMHBHYCRPITRIV", "BAMLEMIBHGCRPITRIV", "BAMLEMCBPIOAS",
            "BAMLEMRLCRPILAOAS", "BAMLC0A0CMTRIV", "BAMLHYH0A0HYM2TRIV", "DGS5", "DEXBZUS"]:
    get(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}", f"fred_{sid}.csv", force=True); time.sleep(1)
get("https://www.tesourotransparente.gov.br/ckan/dataset/df56aa42-484a-4a59-8184-7676580c81e3/resource/796d2059-14e9-44e3-80c9-2d9e30b405c1/download/PrecoTaxaTesouroDireto.csv", "tesouro_direto.csv")
