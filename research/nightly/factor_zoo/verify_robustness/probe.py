import time, numpy as np, pandas as pd
from research.nightly import harness as H
from research.nightly.factor_zoo import features as FZ
from research.nightly.factor_zoo.run import composite, screen_fn
import json
spec = json.load(open("research/nightly/factor_zoo/results.json"))["composite_spec"]
P = FZ.attach(H.load_panel("M"))
P = composite(P, spec, "zoo")
U = P[P.univ]
for c in ["days_since_distress","age","bond_age_y","eq_r126","eq_vol63","ds_5","dres_21","eq_r21"]:
    print(c, U[c].notna().mean().round(3), U[c].describe()[["min","50%","max"]].round(2).to_dict())
q = P[P.p4q]
print("p4q rows", len(q), "zoo cov", q.zoo.notna().mean(), "listed", q.eq_r126.notna().mean())
t=time.time(); r = H.backtest(screen_fn("zoo",0.2), panel=P); print("bt s", time.time()-t)
b = H.baseline("P4Q")
print(H.stats(r["daily"], bench=b["daily"]))
