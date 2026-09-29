import time, pandas as pd, numpy as np
from research.nightly import harness as H
from research.nightly.structural_credit.run import worst_q
t=time.time()
PM=pd.read_pickle("data/history/nightly/structural_credit/panel_M_sc.pkl")
print(PM.shape, PM.day.max(), time.time()-t)
f=lambda x: x["p4q"].to_numpy() & ~worst_q(x,"sc_cg",low_bad=False)
r=H.backtest(f,panel=PM); b=H.baseline("P4Q")
print(time.time()-t); s=H.stats(r["daily"],bench=b["daily"]); print({k:s[k] for k in ["diff_ann_%","diff_t_nw","max_dd_%"]})
b2=H.backtest("p4q",panel=PM); print(H.stats(b2["daily"],bench=b["daily"])["diff_ann_%"], time.time()-t)
