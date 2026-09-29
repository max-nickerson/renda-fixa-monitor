"""Same-sample comparison: within-issuer switch ranked on same-day vs 1-month-stale residual (and vs pair selection)."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from research.nightly import harness as H
from research.nightly.issuer_curve_rv import signals as S
OUT = Path(__file__).resolve().parent
PM = S.attach(H.load_panel("M"))
dts = np.sort(PM["day"].unique())
L = PM[["day", "codigo", "resid_bps", "cdi_bps"]].copy()
L["day"] = L["day"].map(pd.Series(dts[1:], index=dts[:-1]))
L = L.dropna(subset=["day"]).rename(columns={"resid_bps": "r_lag", "cdi_bps": "c_lag"})
PM = PM.merge(L, on=["day", "codigo"], how="left")
out = {}
for tgt in ("fwd_63", "fwd_126"):
    Hh = int(tgt.split("_")[1])
    X = PM[PM["univ"] & PM[f"dok_{Hh}"] & PM[tgt].notna() & PM["resid_bps"].notna() & PM["r_lag"].notna()]
    X = X[X.groupby(["day", "cnpj8"])["codigo"].transform("count") >= 2]
    res = {}
    for col in ("resid_bps", "r_lag", "cdi_bps", "c_lag"):
        g = X.groupby(["day", "cnpj8"])
        hi = X.loc[g[col].idxmax()].set_index(["day", "cnpj8"]); lo = X.loc[g[col].idxmin()].set_index(["day", "cnpj8"])
        ok = hi["codigo"].to_numpy() != lo["codigo"].to_numpy()
        d = (hi[tgt] - lo[tgt])[ok].reset_index(name="dret")
        s = d.groupby("day")["dret"].mean(); ann = 252 / Hh
        res[col] = {"ann_%": round(float(s.mean() * ann * 100), 3), "t": round(float(H.nw_t(s, Hh // 21)), 2), "n": int(ok.sum())}
    # agreement of pair picks
    out[tgt] = res
# how often do cheap/rich picks flip between lag and same day
print(json.dumps(out, indent=1))
json.dump(out, open(OUT / "verify_lag_results.json", "w"), indent=1)
