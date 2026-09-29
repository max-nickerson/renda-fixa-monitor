"""(a) B3 COTAHIST 2013-2020 spot-market closes for the equity_map tickers (+ verified predecessors), cached as
    data/history/nightly/pre2021_oos_extension/cotahist_<Y>.csv.gz (unadjusted closes / fatcot; PIT: close of day).
(b) CVM ITR/DFP 2011-2020 zips into the same folder (fundamentals built later with research/data_fundamentals.py
    functions, available_date_strict = receipt date of the latest version)."""
import io, shutil, time, zipfile
from pathlib import Path
import httpx, pandas as pd

OUT = Path("data/history/nightly/pre2021_oos_extension")
mp = pd.read_csv("research/data/equity_map.csv", dtype=str)
PRED = {"AXIA3": ["ELET3"], "AZZA3": ["ARZZ3"], "BHIA3": ["VIIA3", "VVAR3"], "BRAV3": ["RRRP3"], "BRST3": ["BRIT3"],
        "DXCO3": ["DTEX3"], "IGTI11": ["IGTA3"], "ISAE4": ["TRPL4"], "MBRF3": ["MRFG3"], "MOTV3": ["CCRO3"],
        "NATU3": ["NTCO3"], "RIAA3": ["GUAR3"], "SBFG3": ["CNTO3"], "TIMS3": ["TIMP3"], "VBBR3": ["BRDT3"],
        "WIZC3": ["WIZS3"], "AMER3": ["BTOW3"], "ALOS3": ["ALSO3"], "SRNA3": ["OMGE3"], "ZAMP3": ["BKBR3"]}
want = set(mp["ticker"].dropna()) | {p for v in PRED.values() for p in v}
cli = httpx.Client(headers={"User-Agent": "Mozilla/5.0"}, follow_redirects=True, timeout=300, verify=False)
for y in range(2013, 2021):
    p = OUT / f"cotahist_{y}.csv.gz"
    if p.exists():
        continue
    r = cli.get(f"https://bvmf.bmfbovespa.com.br/InstDados/SerHist/COTAHIST_A{y}.ZIP")
    r.raise_for_status()
    rows = []
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        with z.open(z.namelist()[0]) as f:
            for raw in f:
                ln = raw.decode("latin1")
                if not ln.startswith("01") or ln[24:27] != "010":
                    continue
                t = ln[12:24].strip()
                if t not in want:
                    continue
                rows.append((t, ln[2:10], int(ln[108:121]) / 100, int(ln[152:170]), int(ln[170:188]) / 100,
                             int(ln[210:217])))
    df = pd.DataFrame(rows, columns=["ticker", "date", "close", "volume", "fin_volume", "fatcot"])
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    df["close"] = df["close"] / df["fatcot"]
    df.drop(columns="fatcot").to_csv(p, index=False, compression="gzip")
    print("cotahist", y, len(df), df.ticker.nunique(), flush=True)
    time.sleep(2)
for kind in ("itr", "dfp"):
    for y in range(2011, 2021):
        p = OUT / f"{kind}_cia_aberta_{y}.zip"
        if p.exists():
            continue
        src = Path("data/history/cvm_fin") / p.name
        if src.exists():
            shutil.copy(src, p); continue
        r = cli.get(f"https://dados.cvm.gov.br/dados/CIA_ABERTA/DOC/{kind.upper()}/DADOS/{kind}_cia_aberta_{y}.zip")
        if r.status_code == 200:
            p.write_bytes(r.content); print(kind, y, len(r.content), flush=True)
        time.sleep(1)
