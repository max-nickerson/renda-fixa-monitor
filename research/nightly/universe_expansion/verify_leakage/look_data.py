import pandas as pd, numpy as np
D="data/history/nightly/universe_expansion/"
F=pd.read_pickle(D+"fidc_panel.pkl")
print(F.shape, F.columns.tolist())
print(F["ret"].describe())
print("ret<=-1:",(F.ret<=-1).sum()," ret>=0.2:",(F.ret>=0.2).sum()," ret==0:",(F.ret==0).sum(), "ret in [0.05,0.2):",((F.ret>=0.05)&(F.ret<0.2)).sum(), "ret in [-1,-0.1]:",((F.ret>-1)&(F.ret<-0.1)).sum())
for c in ["29.492.605/0001-29|Senior 1","45.207.484/0001-82|Senior 1","28.152.199/0001-92|Senior 1"]:
    x=F[F.cls==c].sort_values("ref")[["ref","ret","pl","inad_ratio","CONDOM"]]
    print(c); print(x[(x.ref>="2021-10-01")&(x.ref<"2026-01-01")].to_string())
