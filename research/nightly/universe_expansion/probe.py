"""Data-feasibility probes: which public sources for CRI/CRA, bank paper, FIDC, NTN-B, eurobonds respond, and how far back."""
import json, httpx, datetime as dt
from pathlib import Path
OUT = Path("research/nightly/universe_expansion/probe_results.json")
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124 Safari/537.36"}
c = httpx.Client(headers=UA, timeout=30, follow_redirects=True)
res = {}
def probe(name, url, n=300):
    try:
        r = c.get(url)
        res[name] = {"url": url, "status": r.status_code, "bytes": len(r.content), "ctype": r.headers.get("content-type"),
                     "head": r.content[:n].decode("latin1", "replace")}
    except Exception as e:
        res[name] = {"url": url, "error": repr(e)[:200]}
    print(name, res[name].get("status"), res[name].get("bytes"), res[name].get("error", ""))
S3 = "https://s3-data-prd-use1-precos.s3.us-east-1.amazonaws.com/arquivos/indices-historico/{}-HISTORICO.xls"
for ix in ["IMAB5", "IMAB", "IMAB5MAIS", "IRFM", "IMAS", "IMAGERAL", "IDKAIPCA2A", "IDADI", "IDAIPCA", "IHFA"]:
    probe(f"anbima_idx_{ix}", S3.format(ix))
# ANBIMA public daily files (debentures known to work ~ recent only). Try CRI/CRA guesses + old dates.
for d in [dt.date(2026, 9, 25), dt.date(2025, 6, 2), dt.date(2023, 6, 1)]:
    probe(f"anbima_deb_{d}", f"https://www.anbima.com.br/informacoes/merc-sec-debentures/arqs/db{d:%y%m%d}.txt")
    probe(f"anbima_tpf_{d}", f"https://www.anbima.com.br/informacoes/merc-sec/arqs/ms{d:%y%m%d}.txt")
probe("anbima_cri_cra_page", "https://www.anbima.com.br/informacoes/merc-sec-cri-cra/default.asp")
probe("anbima_cri_cra_arq", "https://www.anbima.com.br/informacoes/merc-sec-cri-cra/arqs/cr250602.txt")
probe("anbima_data_cri", "https://data.anbima.com.br/certificado-de-recebiveis?view=precos")
# CVM open data
B = "https://dados.cvm.gov.br/dados"
probe("cvm_fidc_inf_mensal_2024", f"{B}/FIDC/DOC/INF_MENSAL/DADOS/inf_mensal_fidc_202406.zip")
probe("cvm_fidc_dir", f"{B}/FIDC/DOC/INF_MENSAL/DADOS/", 2000)
probe("cvm_securit_dir", f"{B}/SECURIT/DOC/", 2000)
probe("cvm_ofertas", f"{B}/OFERTA/DISTRIB/DADOS/", 2000)
# FRED: ICE BofA EM / Brazil corporate
for sid in ["BAMLEMCBPITRIV", "BAMLEMCBPIOAS", "BAMLEMRLCRPILATRIV", "BAMLEMRLCRPILAOAS", "BAMLEMHBHYCRPITRIV", "BAMLEMIBHGCRPITRIV", "BAMLHYH0A0HYM2TRIV", "BAMLC0A0CMTRIV", "DGS5"]:
    probe(f"fred_{sid}", f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}", 120)
# B3 private fixed income trades (guesses)
probe("b3_rf_privada", "https://www.b3.com.br/pt_br/market-data-e-indices/servicos-de-dados/market-data/consultas/mercado-de-balcao/renda-fixa/", 500)
probe("bcb_sgs_cdb_4389", "https://api.bcb.gov.br/dados/serie/bcdata.sgs.4389/dados/ultimos/3?formato=json")
OUT.write_text(json.dumps(res, indent=1, ensure_ascii=True), encoding="utf-8")
