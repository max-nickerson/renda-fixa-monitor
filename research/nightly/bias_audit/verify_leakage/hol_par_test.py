"""Decisive test of the holiday-accrual claim: implied par = pu_avg / ratio (SND % PU da Curva).
For DI_SPREAD bonds traded on consecutive trade dates, count how many days of contract spread the par curve accrued
between them, net of CDI (BCB SGS 12 compounded over B3 business days)."""
import glob
from datetime import date
import numpy as np, pandas as pd
from research.nightly.bias_audit import build as B
from rfmonitor.history import bcb_series

lm = B.lab_min()
tr = lm.drop_duplicates(["codigo", "date"])[["codigo", "date", "ratio", "contract", "kind"]]
tr = tr[tr["kind"].isin(["DI_SPREAD", "PRE", "IPCA"])]
tr["date"] = pd.to_datetime(tr["date"])
fs = sorted(glob.glob(str(B.OUT / "snd_range_*.csv.gz")))
s = pd.concat([pd.read_csv(f, parse_dates=["date"]) for f in fs], ignore_index=True)
m = s.merge(tr, on=["codigo", "date"], how="inner")
m = m[(m["ratio"] > 0.5) & (m["pu_avg"] > 0)]
m["par"] = m["pu_avg"] / m["ratio"]
m = m.sort_values(["codigo", "date"])
m["par1"] = m.groupby("codigo")["par"].shift(-1)
m["d1"] = m.groupby("codigo")["date"].shift(-1)
m = m[m["d1"].notna()]
hol = B.b3_holidays()
d0 = m["date"].to_numpy().astype("datetime64[D]"); d1 = m["d1"].to_numpy().astype("datetime64[D]")
m["wk"] = np.busday_count(d0, d1); m["b3"] = np.busday_count(d0, d1, holidays=hol)
m = m[(m["wk"] >= 1) & (m["wk"] <= 3)]
cdi = bcb_series(12, date(2020, 12, 1)).astype(float) / 100
lc = np.log1p(cdi); cum = lc.cumsum()
cumd = pd.Series(cum.values, index=pd.DatetimeIndex(cdi.index))
# accrual from d0 to d1 over B3 days = sum of CDI of days d0 .. d1-1 (business days)
def cumbefore(d):
    idx = cumd.index.searchsorted(d) - 1
    return np.where(idx >= 0, cumd.values[np.clip(idx, 0, None)], 0.0)
m["lcdi"] = cumbefore(m["d1"].values) - cumbefore(m["date"].values)
m["lpar"] = np.log(m["par1"] / m["par"])
c = m["contract"] / 100
out = []
for k in ["DI_SPREAD", "PRE"]:
    x = m[m["kind"] == k].copy()
    xc = c[x.index]
    x["resid"] = x["lpar"] - (x["lcdi"] if k == "DI_SPREAD" else 0.0)
    x["ndays"] = x["resid"] / (np.log1p(xc) / 252)
    x = x[np.isfinite(x["ndays"]) & (xc > 0.005)]
    for (wk, b3), g in x.groupby(["wk", "b3"]):
        med = g["ndays"].median()
        out.append((k, wk, b3, len(g), round(med, 3), round((g["ndays"].sub(b3).abs() < 0.3).mean(), 3),
                    round((g["ndays"].sub(wk).abs() < 0.3).mean(), 3)))
print(pd.DataFrame(out, columns=["kind", "weekdays", "b3_days", "n", "median_implied_spread_days",
                                 "share_match_b3", "share_match_weekdays"]).to_string())
