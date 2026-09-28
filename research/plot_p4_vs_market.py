"""P4 over the whole backtest in total-return terms vs CDI, the debenture market (IDA-DI, IDA-IPCA) and Ibovespa.
P4 total return = CDI x (1 + rate-hedged credit excess) from the daily lab (weekly decisions, regime overlay,
next-trade execution, 25 bps per trade). Outputs research/out/p4_vs_market.png and p4_vs_market.json."""
import sys as _sys
from pathlib import Path as _P
_sys.path.insert(0, str(_P(__file__).resolve().parent.parent))  # run from anywhere

import json
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rfmonitor import history as h
from rfmonitor.config import DATA_DIR
from rfmonitor.ml import lab_daily as ld
from rfmonitor.ml.lab_daily import DRule
from rfmonitor.sources import brapi

OUT = Path(__file__).parent / "out"
curves = pd.read_pickle(DATA_DIR / "history" / "daily_curves.pkl")
p4 = curves["P4 decisões semanais"]
days = p4.index

u_path = DATA_DIR / "history" / "daily_u0.pkl"
if u_path.exists():
    u0 = pd.read_pickle(u_path)
else:
    u0 = ld.run(pd.read_pickle(DATA_DIR / "history" / "lab_daily.pkl"), DRule("U0", lambda r: True, lambda r: False),
                cost_bps=25)["daily"]
    u0.to_pickle(u_path)

cdi_d = (h.bcb_series(12, date(2021, 12, 1)) / 100).reindex(days).fillna(0)  # % a.d. → decimal per day
idx = pd.DataFrame(index=days)
idx["CDI"] = (1 + cdi_d).cumprod()
idx["P4 (crédito, juros hedgeados)"] = ((1 + cdi_d) * (1 + p4)).cumprod()
idx["Universo de debêntures (hedgeado)"] = ((1 + cdi_d) * (1 + u0.reindex(days).fillna(0))).cumprod()
for name, label in [("IDADI", "IDA-DI (mercado DI+)"), ("IDAIPCA", "IDA-IPCA (mercado IPCA+, sem hedge)")]:
    s = h.ida(name)["index"]
    idx[label] = s.reindex(days, method="ffill")
bv = pd.DataFrame(brapi.history("^BVSP", "5y"), columns=["d", "close", "vol"])
bv = bv.assign(d=pd.to_datetime(bv["d"])).set_index("d")["close"]
idx["Ibovespa"] = bv.reindex(days, method="ffill")
idx = idx.dropna()
idx = idx / idx.iloc[0]

yrs = (idx.index[-1] - idx.index[0]).days / 365.25
r = idx.pct_change().dropna()
cdi_r = r["CDI"]
stats = {}
for c in idx:
    cagr = idx[c].iloc[-1] ** (1 / yrs) - 1
    vol = r[c].std() * np.sqrt(252)
    ex = r[c] - cdi_r
    dd = (idx[c] / idx[c].cummax() - 1).min()
    stats[c] = {"total_%": (idx[c].iloc[-1] - 1) * 100, "cagr_%": cagr * 100, "vol_%": vol * 100,
                "pct_cdi": (idx[c].iloc[-1] - 1) / (idx["CDI"].iloc[-1] - 1) * 100,
                "sharpe_vs_cdi": ex.mean() * 252 / (ex.std() * np.sqrt(252)) if c != "CDI" else None,
                "max_dd_%": dd * 100}
cal = idx.resample("YE").last()
cal = pd.concat([idx.iloc[[0]], cal]).pct_change().dropna() * 100
cal.index = cal.index.year

colors = {"CDI": "#6b7280", "P4 (crédito, juros hedgeados)": "#0B6E63", "Universo de debêntures (hedgeado)": "#9ca3af",
          "IDA-DI (mercado DI+)": "#2563eb", "IDA-IPCA (mercado IPCA+, sem hedge)": "#a855f7", "Ibovespa": "#d97706"}
fig, ax = plt.subplots(2, 1, figsize=(12, 8), dpi=110, gridspec_kw={"height_ratios": [3, 1.3]}, sharex=True)
for c in idx:
    ax[0].plot(idx.index, (idx[c] - 1) * 100, label=f"{c}: {stats[c]['total_%']:+.0f}% ({stats[c]['cagr_%']:.1f}% a.a.)",
               color=colors[c], lw=2.4 if c.startswith("P4") else 1.4, ls="--" if c == "CDI" else "-")
    ax[1].plot(idx.index, (idx[c] / idx[c].cummax() - 1) * 100, color=colors[c], lw=1.8 if c.startswith("P4") else 1)
ax[0].set_title(f"P4 vs CDI, mercado de debêntures e Ibovespa · retorno total {idx.index[0]:%d/%m/%Y}–{idx.index[-1]:%d/%m/%Y}",
                fontsize=11)
ax[0].set_ylabel("retorno acumulado (%)")
ax[0].legend(fontsize=8.5, frameon=False, loc="upper left")
ax[0].grid(alpha=0.3)
ax[1].set_ylabel("drawdown (%)")
ax[1].grid(alpha=0.3)
fig.text(0.01, 0.005, "P4 = CDI + retorno de crédito com juros hedgeados (DI/DAP), decisão semanal, execução no negócio seguinte, "
         "25 bps por giro, vai para CDI em regime defensivo. Antes de taxas e IR. IDA-IPCA e Ibovespa sem hedge.", fontsize=7.5,
         color="#555")
fig.tight_layout(rect=(0, 0.02, 1, 1))
fig.savefig(OUT / "p4_vs_market.png")
json.dump({"period": [str(idx.index[0].date()), str(idx.index[-1].date())], "stats": stats,
           "calendar_%": {str(k): v for k, v in cal.round(2).T.to_dict().items()}},
          open(OUT / "p4_vs_market.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=float)
print(pd.DataFrame(stats).T.round(2).to_string())
print(cal.round(1).to_string())
