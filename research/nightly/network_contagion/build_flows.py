"""Daily NAV / subscriptions / redemptions (CVM inf_diario) for every fund that ever held debentures in CDA.
Output data/history/nightly/network_contagion/fund_flows.pkl: fund, day, nav, captc, resg (daily).
PIT: inf_diario is published daily (D+1..D+2); features built from it lag 2 business days."""
import sys, zipfile
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
import net
funds = set(net.cda_holdings()["fund"].unique())
out = []
for z in sorted((net.CD / "infd").glob("inf_diario_fi_*.zip")):
    zf = zipfile.ZipFile(z)
    for nm in zf.namelist():
        for ch in pd.read_csv(zf.open(nm), sep=";", encoding="latin1", dtype=str, chunksize=1_000_000,
                              usecols=lambda c: c in ("CNPJ_FUNDO", "CNPJ_FUNDO_CLASSE", "ID_SUBCLASSE", "DT_COMPTC",
                                                      "VL_PATRIM_LIQ", "CAPTC_DIA", "RESG_DIA")):
            ch = ch.rename(columns={"CNPJ_FUNDO": "CNPJ_FUNDO_CLASSE"})
            if "ID_SUBCLASSE" in ch:
                ch = ch[ch["ID_SUBCLASSE"].isna()]
            ch["fund"] = ch["CNPJ_FUNDO_CLASSE"].str.replace(r"\D", "", regex=True)
            ch = ch[ch["fund"].isin(funds)]
            out.append(pd.DataFrame({"fund": ch["fund"], "day": pd.to_datetime(ch["DT_COMPTC"]),
                                     "nav": pd.to_numeric(ch["VL_PATRIM_LIQ"], errors="coerce"),
                                     "captc": pd.to_numeric(ch["CAPTC_DIA"], errors="coerce"),
                                     "resg": pd.to_numeric(ch["RESG_DIA"], errors="coerce")}))
    print(z.name, flush=True)
F = pd.concat(out, ignore_index=True).drop_duplicates(["fund", "day"], keep="last")
F["fund"] = F["fund"].astype("category")
F.to_pickle(net.CD / "fund_flows.pkl")
print(F.shape, F["fund"].nunique(), F["day"].min(), F["day"].max())
