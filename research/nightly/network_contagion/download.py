"""Download CVM public data for the network agent: fund holdings (CDA, quarter-end months) and FRE (shareholders)."""
import httpx, pathlib, sys, time
D = pathlib.Path("data/history/nightly/network_contagion")
jobs = []
for y in range(2020, 2027):
    for m in (3, 6, 9, 12):
        ym = f"{y}{m:02d}"
        if "202012" <= ym <= "202606":
            jobs.append((f"https://dados.cvm.gov.br/dados/FI/DOC/CDA/DADOS/cda_fi_{ym}.zip", D / "cda" / f"cda_fi_{ym}.zip"))
for y in range(2020, 2027):
    jobs.append((f"https://dados.cvm.gov.br/dados/CIA_ABERTA/DOC/FRE/DADOS/fre_cia_aberta_{y}.zip", D / "fre" / f"fre_cia_aberta_{y}.zip"))
with httpx.Client(timeout=300, follow_redirects=True) as c:
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
