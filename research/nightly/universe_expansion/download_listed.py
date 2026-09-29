"""Listed credit vehicles (B3): FII de papel (CRI), Fiagro (CRA) and FI-Infra (incentivised debentures) daily prices
+ cash distributions from brapi, and CVM FII monthly reports (NAV per quota, with delivery date) for P/NAV.
Ticker list = hand-built from vehicles listed today => SURVIVORSHIP-BIASED (dead/merged funds missing)."""
import json, time, pickle, httpx
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(".env")
from rfmonitor.sources import brapi
RAW = Path("data/history/nightly/universe_expansion/raw")
TICK = {
 "CRI": ["KNCR11", "KNIP11", "KNSC11", "KNHY11", "MXRF11", "HGCR11", "IRDM11", "RECR11", "CPTS11", "VGIR11", "RBRR11",
         "RBRY11", "MCCI11", "VRTA11", "BCRI11", "HABT11", "DEVA11", "XPCI11", "CVBI11", "URPR11", "VGHF11", "HCTR11",
         "RZAK11", "AFHI11", "OUJP11", "BTCI11", "CACR11", "SNCI11", "PORD11", "NCHB11", "VCJR11", "KCRE11", "MCHY11",
         "RBHY11", "HSAF11", "VSLH11", "ARRI11", "RZAT11", "SADI11", "WHGR11"],
 "CRA": ["KNCA11", "VGIA11", "RZAG11", "XPCA11", "CPTR11", "GCRA11", "SNAG11", "OIAG11", "FGAA11", "VCRA11", "RURA11",
         "EGAF11", "NCRA11", "AGRX11", "BBGO11", "ECOO11", "HGAG11", "JGPX11", "KNRI11X"],
 "INFRA": ["JURO11", "KDIF11", "CPTI11", "BDIF11", "IFRA11", "BIDB11", "XPID11", "SNID11", "CDII11", "IFRI11",
           "RBIF11", "BODB11", "JMBI11"],
}
out = {}
for grp, ts in TICK.items():
    for t in ts:
        try:
            r = brapi._get(f"/quote/{t}", range="max", interval="1d", dividends="true")
            q = r["results"][0]
            out[t] = {"group": grp, "hist": q.get("historicalDataPrice") or [],
                      "divs": (q.get("dividendsData") or {}).get("cashDividends") or [],
                      "name": q.get("longName") or q.get("shortName")}
            print(t, grp, len(out[t]["hist"]), len(out[t]["divs"]))
        except Exception as e:
            print(t, "ERR", repr(e)[:120])
        time.sleep(0.4)
pickle.dump(out, open(RAW / "listed_credit_brapi.pkl", "wb"))
c = httpx.Client(timeout=120, follow_redirects=True, headers={"User-Agent": "Mozilla/5.0"})
for y in range(2020, 2027):
    p = RAW / f"inf_mensal_fii_{y}.zip"
    if not p.exists():
        r = c.get(f"https://dados.cvm.gov.br/dados/FII/DOC/INF_MENSAL/DADOS/inf_mensal_fii_{y}.zip")
        print("fii", y, r.status_code, len(r.content))
        if r.status_code == 200: p.write_bytes(r.content)
