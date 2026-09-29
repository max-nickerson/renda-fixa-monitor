"""Build the audit inputs (cached in data/history/nightly/bias_audit/):

  hol.pkl       holiday-accrual correction of the patched returns. build_returns() accrues the contract spread
                over np.busday_count (weekdays), but the SND par curve (ratio = % PU da Curva) accrues over B3
                business days, so every weekday holiday adds one day of spread (c/252) of fake return.
                B3 business days = the dates of BCB SGS 12 (CDI), known in advance (ANBIMA holiday calendar).
  exec.pkl      per grid cell (pos, b) execution penalties of the trade behind the mark:
                  spike_buy / spike_sell : raw print vs spike-cleaned print (the cleaning uses the NEXT trade)
                  rng_buy / rng_sell     : buy at the day's PU max / sell at the day's PU min (SND intraday range)
  names.pkl     stopped-bond / distress lists (pre-2026 information only)
"""
from __future__ import annotations

import glob
from datetime import date

import numpy as np
import pandas as pd

from research.nightly import harness as H

OUT = H.HIST / "nightly" / "bias_audit"


def lab_min() -> pd.DataFrame:
    p = OUT / "lab_min.pkl"
    if not p.exists():
        g = pd.read_pickle(H.HIST / "lab_daily.pkl")
        g = g[["codigo", "day", "date", "ratio", "cnpj8", "eligible", "cdi_bps", "kind", "fresh", "dur", "contract"]]
        g.to_pickle(p)
    return pd.read_pickle(p)


def b3_holidays() -> np.ndarray:
    from rfmonitor.history import bcb_series
    cdi = bcb_series(12, date(2020, 1, 1))
    bd = pd.DatetimeIndex(cdi.index)
    wk = pd.bdate_range(bd.min(), bd.max())
    hol = wk.difference(bd)
    return hol.to_numpy().astype("datetime64[D]")


def build_hol():
    p = OUT / "hol.pkl"
    if p.exists():
        return pd.read_pickle(p)
    C = H._core()
    G, codes = pd.read_pickle(H.HIST / "sellab_returns.pkl")
    dd = C["days"]
    g = lab_min()[["codigo", "day", "contract", "kind"]]
    G = G.sort_values(["b", "pos"]).reset_index(drop=True)
    G["day"] = dd[G["pos"].to_numpy()]
    G = G.merge(g, on=["codigo", "day"], how="left")
    nxt = G.groupby("b")["pos"].shift(-1)
    has = nxt.notna()
    d0 = G["day"].to_numpy().astype("datetime64[D]")
    d1 = dd[nxt.fillna(G["pos"]).astype(int).to_numpy()].to_numpy().astype("datetime64[D]")
    hol = b3_holidays()
    wk = np.busday_count(d0, d1)
    b3 = np.busday_count(d0, d1, holidays=hol)
    extra = np.where(has, wk - b3, 0)
    c = G["contract"].fillna(0).to_numpy() / 100
    f = (1 + c) ** (-extra / 252)
    rpos = np.where(has & (nxt - G["pos"] > 1), nxt.fillna(0).astype(int) - 1, G["pos"].to_numpy())
    out = pd.DataFrame({"pos": rpos, "b": G["b"].to_numpy(), "f": f, "extra": extra, "kind": G["kind"].to_numpy(),
                        "c": c})
    out = out[out["extra"] != 0]
    diag = {"n_rows_crossing_holiday": int(len(out)), "n_holidays_2021_2025": int(((hol >= np.datetime64("2021-01-01")) & (hol < np.datetime64("2026-01-01"))).sum()),
            "mean_extra_accrual_bps_per_row": float(((1 - out["f"]) * 1e4).mean())}
    pd.to_pickle((out, diag), p)
    return out, diag


def R_hol(R: np.ndarray) -> np.ndarray:
    out, _ = build_hol()
    R2 = R.copy()
    pos, b, f = out["pos"].to_numpy(), out["b"].to_numpy(), out["f"].to_numpy()
    R2[pos, b] = (1 + R2[pos, b]) * f - 1
    return R2


def snd_range() -> pd.DataFrame:
    fs = sorted(glob.glob(str(OUT / "snd_range_*.csv.gz")))
    s = pd.concat([pd.read_csv(f, parse_dates=["date"]) for f in fs], ignore_index=True)
    s = s[(s["pu_avg"] > 0) & (s["pu_min"] > 0) & (s["pu_max"] > 0)]
    s["hb"] = (s["pu_max"] / s["pu_avg"] - 1).clip(0, 0.2)
    s["hs"] = (1 - s["pu_min"] / s["pu_avg"]).clip(0, 0.2)
    return s


def build_exec():
    p = OUT / "exec.pkl"
    if p.exists():
        return pd.read_pickle(p)
    C = H._core()
    codes, TD, dd = C["codes"], C["TD"], C["days"]
    ND, NB = TD.shape
    # --- spike penalties (same rule as run_selection_lab.build_returns)
    g = lab_min()
    tr = g.drop_duplicates(["codigo", "date"])[["codigo", "date", "ratio"]].sort_values(["codigo", "date"]).copy()
    pv = tr.groupby("codigo")["ratio"].shift(1)
    nx = tr.groupby("codigo")["ratio"].shift(-1)
    spike = ((tr["ratio"] / pv - 1).abs() > 0.10) & ((nx / pv - 1).abs() < 0.03)
    tr["ratio_c"] = np.where(spike, pv, tr["ratio"])
    tr["sb"] = tr["ratio"] / tr["ratio_c"] - 1
    tr = tr[tr["sb"] != 0]
    # --- intraday range
    s = snd_range()
    s = s[s["codigo"].isin(set(codes))]
    # map both onto grid cells through (b, trade date) = TD
    cells_pos, cells_b = np.nonzero(TD >= 0)
    cell = pd.DataFrame({"pos": cells_pos.astype(np.int32), "b": cells_b.astype(np.int32),
                         "tdn": TD[cells_pos, cells_b]})
    def to_mat(df, col):
        df = df.copy()
        df["b"] = codes.get_indexer(df["codigo"]).astype(np.int32)
        df["tdn"] = df["date"].to_numpy().astype("datetime64[D]").astype(np.int64).astype(np.int32)
        m = cell.merge(df[["b", "tdn", col]], on=["b", "tdn"], how="inner")
        A = np.zeros((ND, NB), dtype=np.float32)
        A[m["pos"].to_numpy(), m["b"].to_numpy()] = m[col].to_numpy()
        return A
    SB = to_mat(tr, "sb")
    RB = to_mat(s, "hb")
    RS = to_mat(s, "hs")
    # coverage of range data on fresh cells before 2026
    hp = H._hpos()
    fresh = TD[:hp] == dd[:hp].to_numpy().astype("datetime64[D]").astype(np.int64)[:, None]
    cov = {"fresh_cells": int(fresh.sum()), "range_cells_on_fresh": int(((RB[:hp] > 0) | (RS[:hp] > 0))[fresh].sum()),
           "n_snd_range_rows": int(len(s)), "months": sorted({f[-13:-7] for f in glob.glob(str(OUT / 'snd_range_*.csv.gz'))})[:1]
           + sorted({f[-13:-7] for f in glob.glob(str(OUT / 'snd_range_*.csv.gz'))})[-1:],
           "n_spikes": int(len(tr))}
    obj = {"SB": SB, "RB": RB, "RS": RS, "cov": cov}
    pd.to_pickle(obj, p)
    return obj


if __name__ == "__main__":
    o, d = build_hol()
    print(d)
    e = build_exec()
    print(e["cov"])
