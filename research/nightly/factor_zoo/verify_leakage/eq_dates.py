import sys; sys.path.insert(0,'.')
import numpy as np, pandas as pd
from research.nightly import harness as H
eq = pd.read_pickle(H.HIST/"equity_daily.pkl"); eq["date"]=pd.to_datetime(eq["date"])
print(eq["source"].value_counts().head()); print(eq["date"].dt.dayofweek.value_counts().sort_index().to_dict())
ix = H.index_levels()["IBOV"]; ib = np.log(ix).diff()
out=[]
for (t,src),g in eq.groupby(["ticker","source"]):
    s=np.log(g.set_index("date")["adj_close"].astype(float)).diff()
    if len(s)<200: continue
    j=pd.concat([s.rename("s"),ib.rename("i")],axis=1).dropna()
    out.append((src,t,j["s"].corr(j["i"]),j["s"].corr(j["i"].shift(1)),j["s"].corr(j["i"].shift(-1))))
o=pd.DataFrame(out,columns=["src","t","c0","c_lagIbov","c_leadIbov"])
print(o.groupby("src")[["c0","c_lagIbov","c_leadIbov"]].median()); print(len(o))
