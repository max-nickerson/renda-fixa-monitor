"""Event capture of the PD top-quintile flag inside P4 (pre-2026): of the P4 issuer-months whose issuer had a
distress event within 12 months (events < 2026), what share did each flag catch, vs worstQ and the P7 equity screen."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from research.nightly import harness as H
from research.nightly.combined import p7
from research.nightly.w2_expected_loss_carry.run import load

RD = Path("research/nightly/w2_expected_loss_carry")
OUT = H.HIST / "nightly" / "w2_expected_loss_carry"
P = load(False)
O = pd.read_pickle(OUT / "pd_oos.pkl")[["cnpj8", "day", "first_event"]]
P = P.merge(O, on=["cnpj8", "day"], how="left")
P["y"] = P["first_event"].notna() & (P["first_event"] < H.HOLDOUT)
E = pd.read_pickle(OUT / "events.pkl")
hard = E[E["type"] != "bond_loss"]
P["eqh"] = False
for d, idx in P[P["univ"]].groupby("day").groups.items():
    P.loc[idx, "eqh"] = p7.flag_eqh(P.loc[idx])
U = P[P["univ"] & (P["day"] >= H.START) & P["p4"]]
iss = U.groupby(["cnpj8", "day"]).agg(y=("y", "max"), pdl=("pdQ_logit", "max"), pdg=("pdQ_lgbm", "max"),
                                       wq=("worstQ", lambda s: bool(s.fillna(False).max())), eqh=("eqh", "max"),
                                       fwd=("fwd_126", "mean")).reset_index()
res = {"P4_issuer_months": int(len(iss)), "with_event_12m": int(iss["y"].sum()),
       "issuers_with_event": int(iss.loc[iss["y"], "cnpj8"].nunique())}
for f in ("pdl", "pdg", "wq", "eqh"):
    res[f] = {"flag_rate": float(iss[f].mean()), "recall_events": float(iss.loc[iss["y"], f].mean()),
              "precision": float(iss.loc[iss[f], "y"].mean()),
              "fwd126_flagged": float(iss.loc[iss[f], "fwd"].mean()), "fwd126_unflagged": float(iss.loc[~iss[f], "fwd"].mean())}
# events per year (onsets) overall and in the debenture universe
uni = set(P["cnpj8"])
on = E[E["onset"] & (E["date"] < H.HOLDOUT)]
res["onsets_per_year_all_types_debenture_issuers"] = on[on["cnpj8"].isin(uni)].groupby(on["date"].dt.year).size().to_dict()
res["hard_onsets_per_year_debenture_issuers"] = (hard[hard["cnpj8"].isin(uni) & (hard["date"] < H.HOLDOUT)]
                                                 .sort_values("date").drop_duplicates("cnpj8")
                                                 .groupby(lambda i: hard.loc[i, "date"].year).size().to_dict())
res["onsets_per_year_by_type_all_issuers"] = {str(k): v for k, v in on.groupby([on["date"].dt.year, "type"]).size().to_dict().items()}
json.dump(res, open(RD / "capture.json", "w"), indent=1, default=str)
print(json.dumps(res, indent=1, default=str))
