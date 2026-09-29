"""Download CVM open data (fund daily reports, CDA holdings, registry) into data/history/nightly/fund_flows/raw."""
import os, sys, time, httpx
RAW = "data/history/nightly/fund_flows/raw"
os.makedirs(RAW, exist_ok=True)
B = "https://dados.cvm.gov.br/dados/FI"
def get(url, dst):
    if os.path.exists(dst) and os.path.getsize(dst) > 1000:
        return
    for k in range(4):
        try:
            with httpx.stream("GET", url, timeout=300, follow_redirects=True) as r:
                if r.status_code != 200:
                    print("HTTP", r.status_code, url); return
                with open(dst + ".tmp", "wb") as f:
                    for ch in r.iter_bytes(1 << 20):
                        f.write(ch)
            os.replace(dst + ".tmp", dst); print("ok", dst, os.path.getsize(dst) >> 20, "MB", flush=True); return
        except Exception as e:
            print("retry", url, e); time.sleep(5)
what = sys.argv[1] if len(sys.argv) > 1 else "all"
months = [f"{y}{m:02d}" for y in range(2021, 2027) for m in range(1, 13) if f"{y}{m:02d}" <= "202609"]
if what in ("all", "cad"):
    get(f"{B}/CAD/DADOS/cad_fi.csv", f"{RAW}/cad_fi.csv")
    get(f"{B}/CAD/DADOS/registro_fundo_classe.zip", f"{RAW}/registro_fundo_classe.zip")
if what in ("all", "cda"):
    for y in (2020, 2021, 2022):
        get(f"{B}/DOC/CDA/DADOS/HIST/cda_fi_{y}.zip", f"{RAW}/cda_fi_{y}.zip")
    for m in months:
        if m >= "202301" and m <= "202608":
            get(f"{B}/DOC/CDA/DADOS/cda_fi_{m}.zip", f"{RAW}/cda_fi_{m}.zip")
if what in ("all", "inf"):
    get(f"{B}/DOC/INF_DIARIO/DADOS/HIST/inf_diario_fi_2020.zip", f"{RAW}/inf_diario_fi_2020.zip")
    for m in months:
        get(f"{B}/DOC/INF_DIARIO/DADOS/inf_diario_fi_{m}.zip", f"{RAW}/inf_diario_fi_{m}.zip")
