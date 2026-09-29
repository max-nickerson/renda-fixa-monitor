import json, pandas as pd
from research.nightly import harness as H
from research.nightly.factor_zoo import features as FZ
from research.nightly.factor_zoo.run import composite
spec = json.load(open("research/nightly/factor_zoo/results.json"))["composite_spec"]
P = composite(FZ.attach(H.load_panel("M")), spec, "zoo")
top = list(json.load(open("research/nightly/factor_zoo/verify_robustness/verify_results.json"))["top_avoided_issuers"])[:10]
x = P[P.p4q].copy()
x["thr"] = x.groupby("day")["zoo"].transform(lambda s: s.quantile(0.2)); x["drop"] = x.zoo <= x.thr
cols=[c for c in ["emissor","issuer","nome","sector"] if c in P.columns]
for c in top:
    z = x[x.cnpj8==c]
    print(c, z[cols].iloc[0].to_dict() if cols else "", "n_dates", z.day.nunique(), "drop_share", round(z["drop"].mean(),2),
          "listed", z.eq_r126.notna().mean().round(2), "fwd126_dropped", round(z.loc[z["drop"],"fwd_126"].mean(),3),
          "first_drop", z.loc[z["drop"],"day"].min(), "days", z.day.min().date(), z.day.max().date())
