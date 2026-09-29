"""Bias-audit tranche engine: a line-by-line copy of harness v4 `_tranche_book` with hooks for the audit.

Hooks (all default to the harness behaviour, verified bit-for-bit in run.py):
  R            : return matrix (grid days x bonds) -> scenario returns (holiday accrual fix, recovery scenarios)
  cost_b       : per-bond round-trip cost in bps (array NB) instead of the scalar cost_bps
  entry_delay  : decisions execute at the first trade after day p + entry_delay (latency / stale-signal test)
  pen_buy/sell : callables (pos_array, b_array) -> fractional price penalty paid on the entry / exit trade
                 (e.g. buy at the day's max PU, sell at the day's min PU; raw vs spike-cleaned print)
  exit_fresh   : a tranche end whose mark is stale (> 0 days old) is extended to the bond's next real trade
                 (within 20 bdays; else the harness sell_pos rule)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from research.nightly import harness as H


def tranche_book(T: dict, hold: int = 126, cost_bps: float = 25.0, R: np.ndarray | None = None,
                 cost_b: np.ndarray | None = None, entry_delay: int = 0, pen_buy=None, pen_sell=None,
                 exit_fresh: bool = False, t_end: int | None = None, exit_flags: dict | None = None):
    C = H._core()
    dd = C["days"]
    ND = len(dd)
    if R is None:
        R = H._Rmat("base")
    if t_end is None:
        t_end = H._hpos()
    NR = H._next_row()
    TD = C["TD"]
    daynum = dd.to_numpy().astype("datetime64[D]").astype(np.int64)
    decs = sorted(T)
    step = int(np.median(np.diff(decs))) if len(decs) > 1 else 21
    M = max(int(round(hold / step)), 1)
    pnl = np.zeros(ND)
    expo = np.zeros(ND)
    xcost = np.zeros(ND)
    ncoh = np.zeros(ND)
    rows = []
    for p in decs:
        w = T[p]
        ncoh[p + 1:] += 1
        if not len(w):
            continue
        pe = min(p + entry_delay, ND - 1)
        ep = H.exec_pos(pe)
        ex_after = None
        if exit_flags is not None:
            ex_after = [(q, exit_flags[q]) for q in sorted(exit_flags) if q > p]
        for b, wi in w.items():
            k = int(ep[b])
            if k < 0 or k >= t_end:
                rows.append((p, b, wi, k, -1))
                continue
            e = k + hold
            if e < ND:
                x = int(NR[e, b])
                e = x if x < ND else e
                if exit_fresh and e < ND and TD[e, b] >= 0 and TD[e, b] < daynum[e]:
                    # the mark at the natural end is stale: sell at the next real trade instead
                    e2 = int(H.sell_pos(e - 1)[b]) if e >= 1 else e
                    e = max(e, e2)
            e = min(e, ND, t_end)
            early = False
            if ex_after:
                for q, fl in ex_after:
                    if q >= e:
                        break
                    if q > k and b in fl:
                        xq = int(H.sell_pos(q)[b])
                        if xq < e:
                            e, early = xq, True
                        break
            if e <= k:
                continue
            r = R[k:e, b]
            gv = np.r_[1.0, np.cumprod(1 + r[:-1])]
            pnl[k:e] += wi * gv * r
            expo[k:e] += wi * gv
            rows.append((p, b, wi, k, e))
            if pen_buy is not None:
                pb = float(pen_buy(np.array([k]), np.array([b]))[0])
                xcost[k] += wi * pb
            if pen_sell is not None and e < t_end:     # only exits that happen inside the window are charged
                ps = float(pen_sell(np.array([e]), np.array([b]))[0])
                xcost[min(e, ND - 1)] += wi * gv[-1] * (1 + r[-1]) * ps
            if early:
                cb = cost_bps if cost_b is None else cost_b[b]
                xcost[e] += wi * gv[-1] * (1 + r[-1]) * cb / 2 / 1e4
    div = np.maximum(np.minimum(M, ncoh), 1)
    pnl /= div
    expo /= div
    cost = xcost / div
    prev, turn = pd.Series(dtype=float), np.zeros(ND)
    for i, p in enumerate(decs):
        live = [T[x] for x in decs[max(0, i - M + 1): i + 1]]
        live = [s for s in live if len(s)]
        book = (pd.concat(live, axis=1).fillna(0).sum(axis=1) / min(M, i + 1)) if live else pd.Series(dtype=float)
        dlt = book.sub(prev, fill_value=0).abs()
        pc = min(p + 1, ND - 1)
        turn[pc] += dlt.sum()
        if cost_b is None:
            cost[pc] += dlt.sum() * cost_bps / 2 / 1e4
        else:
            cost[pc] += float((dlt * cost_b[dlt.index.to_numpy()]).sum()) / 2 / 1e4
        prev = book
    first = min(decs) + 1 if decs else 0
    idx = dd[first:t_end]
    net = pd.Series((pnl - cost)[first:t_end], index=idx)
    yrs = len(idx) / 252
    return {"daily": net, "gross": pd.Series(pnl[first:t_end], index=idx),
            "cost_ann_%": float(cost[first:t_end].sum() / yrs * 100),
            "turnover_ann": float(turn[first:t_end].sum() / yrs),
            "rows": pd.DataFrame(rows, columns=["dpos", "b", "w", "k", "e"])}


def targets(signal, panel, top_frac=0.2, issuer_cap=H.ISSUER_CAP, start=H.START, t_end=None):
    t_end = H._hpos() if t_end is None else t_end
    P = panel[(panel["day"] >= pd.Timestamp(start)) & (panel["dpos"] < t_end)]
    return H._targets(signal, P, top_frac, issuer_cap, False, H.MIN_NAMES, "univ")
