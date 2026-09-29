"""Download CVM CDA annual history (2021-22) and fund daily reports (inf_diario) 2021-01..2026-09."""
import httpx, pathlib
D = pathlib.Path("data/history/nightly/network_contagion")
jobs = [(f"https://dados.cvm.gov.br/dados/FI/DOC/CDA/DADOS/HIST/cda_fi_{y}.zip", D / "cda" / f"cda_fi_{y}.zip") for y in (2021, 2022)]
for y in range(2021, 2027):
    for m in range(1, 13):
        ym = f"{y}{m:02d}"
        if ym <= "202609":
            jobs.append((f"https://dados.cvm.gov.br/dados/FI/DOC/INF_DIARIO/DADOS/inf_diario_fi_{ym}.zip", D / "infd" / f"inf_diario_fi_{ym}.zip"))
with httpx.Client(timeout=600, follow_redirects=True) as c:
    for u, p in jobs:
        if p.exists() and p.stat().st_size > 1000:
            continue
        try:
            with c.stream("GET", u) as r:
                if r.status_code != 200:
                    print("miss", u, r.status_code, flush=True); continue
                tmp = p.with_suffix(".part")
                with open(tmp, "wb") as f:
                    for ch in r.iter_bytes(1 << 20):
                        f.write(ch)
                tmp.replace(p)
            print("ok", p.name, p.stat().st_size, flush=True)
        except Exception as e:
            print("err", u, e, flush=True)
