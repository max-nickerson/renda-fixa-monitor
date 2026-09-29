"""Leakage-lens verification of alt_signals best rule (pre-2026 only; holdout not touched)."""
import json, zipfile
from pathlib import Path
import numpy as np, pandas as pd
from research.nightly import harness as H
from research.nightly.alt_signals import signals as S

OUT = Path(__file__).resolve().parent
Q = pd.read_pickle(S.CACHE / "panel_alt_M.pkl")
assert Q["day"].max() < H.HOLDOUT
B = H.baseline("P4Q")["daily"]
res = {}

def ev(f, name):
    r = H.backtest(f, panel=Q)
    s = H.stats(r["daily"], bench=B)
    out = {k: s[k] for k in ["ann_excess_%", "diff_ann_%", "diff_t_nw", "diff_h1_%", "diff_h2_%"]}
    out["n_avg"] = round(r["n_avg"], 1)
    res[name] = out; print(name, out, flush=True)
    return r

def rule(sup, nc_col="nc_expo", thr=-1.0):
    def f(x):
        return x["p4q"].to_numpy() & ~((x[sup].fillna(0).to_numpy() > 0) | (x[nc_col].to_numpy() < thr))
    return f
def sup_only(sup):
    return lambda x: x["p4q"].to_numpy() & ~(x[sup].fillna(0).to_numpy() > 0)

# 0 reproduce
ev(rule("sup_iss_90d"), "repro_best")
ev(sup_only("sup_iss_90d"), "repro_supply_only")

# 1 alternative availability dates for offers
z = zipfile.ZipFile(S.RAW / "oferta_distribuicao.zip")
a = pd.read_csv(z.open("oferta_distribuicao.csv"), sep=";", encoding="latin-1", low_memory=False)
a = a[a["Tipo_Ativo"].str.contains("DEB", na=False)].copy()
b = pd.read_csv(z.open("oferta_resolucao_160.csv"), sep=";", encoding="latin-1", low_memory=False)
b = b[b["Valor_Mobiliario"].str.startswith("Deb", na=False)].copy()
st = pd.to_datetime(a["Data_Inicio_Oferta"], errors="coerce")
com = pd.to_datetime(a["Data_Comunicado"], errors="coerce")
reg400 = pd.to_datetime(a["Data_Registro_Oferta"], errors="coerce")
rq = pd.to_datetime(b["Data_requerimento"], errors="coerce")
rg = pd.to_datetime(b["Data_Registro"], errors="coerce")
enc = pd.to_datetime(b["Data_Encerramento"], errors="coerce")
c8a = S._cnpj8(a["CNPJ_Emissor"]); c8b = S._cnpj8(b["CNPJ_Emissor"])
notexp = ~b["Status_Requerimento"].isin(["Requerimento Expirado"]).values

def offers(la, lb, keep_b=notexp):
    o = pd.concat([pd.DataFrame({"cnpj8": c8a.values, "avail": la.values}),
                   pd.DataFrame({"cnpj8": c8b.values[keep_b], "avail": lb.values[keep_b]})]).dropna()
    return o.drop_duplicates()

variants = {
  "orig": (st + pd.Timedelta(days=7), rq + pd.Timedelta(days=1), notexp),
  "legacy_at_last_comunicado": (com + pd.Timedelta(days=1), rq + pd.Timedelta(days=1), notexp),
  "r160_at_registro": (st + pd.Timedelta(days=7), rg.fillna(rq + pd.Timedelta(days=14)) + pd.Timedelta(days=1), notexp),
  "both_conservative": (com + pd.Timedelta(days=1), rg.fillna(rq + pd.Timedelta(days=14)) + pd.Timedelta(days=1), notexp),
  "r160_at_encerramento": (st + pd.Timedelta(days=7), enc.fillna(rq + pd.Timedelta(days=30)) + pd.Timedelta(days=1), notexp),
  "all_plus30d": (st + pd.Timedelta(days=37), rq + pd.Timedelta(days=31), notexp),
  "include_expired": (st + pd.Timedelta(days=7), rq + pd.Timedelta(days=1), np.ones(len(b), bool)),
  "add_icvm400_by_registro": (st.fillna(reg400) + pd.Timedelta(days=7), rq + pd.Timedelta(days=1), notexp),
}
for nm, (la, lb, kb) in variants.items():
    o = offers(la, lb, kb)
    Q["_s"] = S.issuer_event_counts(Q, o, {"x": 91})["x"].to_numpy()
    if nm == "orig":
        res["orig_flag_match"] = float(((Q["_s"] > 0) == (Q["sup_iss_90d"].fillna(0) > 0)).mean())
        print("flag match", res["orig_flag_match"])
    ev(rule("_s"), f"best|{nm}")
    ev(sup_only("_s"), f"supply_only|{nm}")

# 2 supply-flag composition: flagged P4Q rows whose bond is itself new (age<180d) vs seasoned
U = Q[Q["p4q"] & (Q["day"] >= H.START)]
fl = U["sup_iss_90d"].fillna(0) > 0
res["flag_share_p4q_rows"] = float(fl.mean())
if "age" in U:
    res["flag_rows_age_lt_180d_share"] = float((U.loc[fl, "age"] < 180).mean())
    Q["_new"] = (Q["age"] < 180).to_numpy()
    ev(lambda x: x["p4q"].to_numpy() & ~((x["sup_iss_90d"].fillna(0).to_numpy() > 0) & ~x["_new"].to_numpy()), "supply_only_seasoned_bonds")
    ev(lambda x: x["p4q"].to_numpy() & ~((x["sup_iss_90d"].fillna(0).to_numpy() > 0) & x["_new"].to_numpy()), "supply_only_new_bonds")
# by year of flag
res["flag_share_by_year"] = U.assign(f=fl).groupby(U["day"].dt.year)["f"].mean().round(3).to_dict()

# 3 nowcast lag: use nowcast value from 30 / 60 days earlier
nc = S.sector_nowcasts()
for lag in (30, 60):
    n2 = nc[["day", "sector", "nc_expo"]].copy(); n2["day"] = n2["day"] + pd.Timedelta(days=lag)
    Q = Q.drop(columns=["_nc"], errors="ignore").merge(n2.rename(columns={"nc_expo": "_nc"}), on=["day", "sector"], how="left")
    ev(rule("sup_iss_90d", "_nc"), f"best|nc_lag{lag}")
    ev(lambda x: x["p4q"].to_numpy() & ~(x["_nc"].to_numpy() < -1), f"nc_only|nc_lag{lag}")
# nowcast built only from daily/weekly series (drop World Bank latest-vintage monthly) -> check which sectors use WB
ex = pd.read_csv(S.ROOT / "research" / "data" / "sector_exposures.csv")
res["exposure_rows_WB_monthly_vars"] = sorted(set(ex["variable"]) & {"sugar","iron_ore","urea","dap","gas_eu","soy","corn","coal","copper","aluminum","potash","wheat","cotton"})
json.dump(res, open(OUT / "verify_results.json", "w"), indent=1, default=str)
print("done")
