"""Leakage check: the harness uses TODAY's SND contract spread (reference table) for all history, both in
cdi_bps (selection) and in the par-accrual of daily returns. SND's pct_curve is relative to the PU-par in force
at the trade date, so PUpar_obs = pu_avg / (pct_curve/100) reveals the contract actually accruing at each date.
For DI_SPREAD bonds: step-implied spread s = (PUpar2/PUpar1 / CDIgrowth)^(252/nbd) - 1 over consecutive trades.
Output: per trade (codigo, date) the trailing (PIT) median implied spread over the last 120 calendar days,
and the full-sample-forward version (truth for returns)."""
import glob
import numpy as np, pandas as pd
from pathlib import Path
from datetime import date
from rfmonitor.ml.selection import _accruals, reference

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "data/history/nightly/portfolio_construction_verify_leakage"

def main():
    fs = sorted(glob.glob(str(ROOT / "data/history/snd_trades_*.csv.gz")))
    T = pd.concat([pd.read_csv(f, usecols=["date", "codigo", "qty", "pu_avg", "pct_curve"]) for f in fs])
    T["date"] = pd.to_datetime(T["date"])
    T = T[T["pct_curve"].between(40, 160) & (T["pu_avg"] > 0)]
    T = T.groupby(["codigo", "date"], as_index=False).agg(pu=("pu_avg", "mean"), pc=("pct_curve", "mean"))
    ref = reference()[["codigo", "kind", "contract"]]
    T = T.merge(ref, on="codigo", how="inner")
    T = T[T["kind"] == "DI_SPREAD"].sort_values(["codigo", "date"]).reset_index(drop=True)
    T["par"] = T["pu"] / (T["pc"] / 100)
    C, _ = _accruals(date(2020, 12, 1))
    C = C.dropna()
    Cv = C.reindex(pd.DatetimeIndex(T["date"]), method="ffill").to_numpy()
    T["C"] = Cv
    g = T.groupby("codigo")
    T["par0"], T["C0"], T["d0"] = g["par"].shift(1), g["C"].shift(1), g["date"].shift(1)
    nbd = np.busday_count(T["d0"].fillna(T["date"]).values.astype("datetime64[D]"), T["date"].values.astype("datetime64[D]"))
    T["nbd"] = nbd
    x = (T["par"] / T["par0"]) / (T["C"] / T["C0"])
    T["s_imp"] = (x ** (252 / np.maximum(nbd, 1)) - 1) * 100
    ok = T["par0"].notna() & (T["nbd"] >= 3) & (T["nbd"] <= 60) & T["s_imp"].between(-3, 40)
    S = T[ok][["codigo", "date", "s_imp", "contract", "nbd"]].copy()
    # PIT: rolling median of steps ending <= date, last 120 days, >= 3 steps
    out = []
    for c, s in S.groupby("codigo"):
        s = s.set_index("date").sort_index()
        m = s["s_imp"].rolling("120D", min_periods=3).median()
        out.append(pd.DataFrame({"codigo": c, "date": m.index, "s_pit": m.to_numpy(), "contract": s["contract"].to_numpy()}))
    O = pd.concat(out, ignore_index=True)
    O.to_pickle(OUT / "implied_contract_pit.pkl")
    S.to_pickle(OUT / "implied_steps.pkl")
    O["gap"] = O["contract"] - O["s_pit"]
    print(O["gap"].describe(percentiles=[.01, .05, .1, .25, .5, .75, .9, .95, .99]))
    bb = O.dropna().groupby("codigo")["gap"].agg(["median", "min", "max", "count"])
    print("bonds", len(bb), "with |median gap|>0.25pp:", int((bb["median"].abs() > 0.25).sum()),
          "max-min >0.5:", int(((bb["max"] - bb["min"]) > 0.5).sum()))
    print(bb.sort_values("median").head(15)); print(bb.sort_values("median").tail(15))

if __name__ == "__main__":
    main()
