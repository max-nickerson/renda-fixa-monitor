import pandas as pd
c=pd.read_pickle("data/history/commodities.pkl"); print(c.columns.tolist()); print(c.head())
c["available_date"]=pd.to_datetime(c["available_date"]); c["period_date"]=pd.to_datetime(c["period_date"])
c["lag"]=(c["available_date"]-c["period_date"]).dt.days
print(c.groupby("series").agg(n=("lag","size"),lagmin=("lag","min"),lagmed=("lag","median"),lagmax=("lag","max"),start=("period_date","min"),end=("period_date","max"), src=("source","first") if "source" in c else ("lag","size")).to_string())
