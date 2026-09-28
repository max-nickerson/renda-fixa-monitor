"""Strategy lab — weekly, point-in-time, realistic execution.

Grid: every Monday. Each bond's MARK at week k = its last SND trade in the 14 days before that Monday
(PU / par-curve ratio, implied CDI+ spread). Weekly bond excess return over CDI between marks:
    r(k→k+1) = G_par(m_k→m_{k+1}) × ratio_{k+1}/ratio_k − CDI growth
Decisions at week k use only data up to mark k (news: filings delivered before the Monday). Positions take
effect at mark k+LAG (default 1) — so the trade that generated a signal is never the execution price
(removes bid-ask bounce). Costs are charged per unit of one-way turnover.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from .. import history as h
from . import news as nw
from .selection import BUCKETS, _accruals, build_panel

STALE_WEEKS = 8


# ---------------------------------------------------------------- data
def weekly_grid(panel: pd.DataFrame) -> pd.DataFrame:
    p = panel.sort_values(["codigo", "date"]).reset_index(drop=True)
    # 30-day liquidity at each trade date
    p["trades_30d"] = p.groupby("codigo", group_keys=False)[["date", "trades"]].apply(
        lambda x: x.rolling("30D", on="date")["trades"].sum())
    mondays = pd.date_range(p["date"].min() + pd.Timedelta(days=21), p["date"].max(), freq="W-MON")
    codes = p.groupby("codigo").agg(first=("date", "min"), last=("date", "max"))
    grid = []
    for code, r in codes.iterrows():
        ms = mondays[(mondays > r["first"]) & (mondays <= r["last"] + pd.Timedelta(weeks=STALE_WEEKS))]
        grid.append(pd.DataFrame({"codigo": code, "week": ms}))
    g = pd.concat(grid, ignore_index=True)
    g["key"] = g["week"] - pd.Timedelta(days=1)
    g = g.sort_values("key")
    cols = ["codigo", "date", "ratio", "cdi_bps", "dur", "T", "kind", "contract", "cnpj", "incent", "trades_30d",
            "maturity", "pre_1y", "bench_rate"]
    m = pd.merge_asof(g, p[cols].sort_values("date"), left_on="key", right_on="date", by="codigo",
                      direction="backward")
    m = m.dropna(subset=["date"])
    m["age_days"] = (m["week"] - m["date"]).dt.days
    m = m[m["week"] < m["maturity"]]
    m["fresh"] = m["age_days"] <= 14
    return m.drop(columns="key").sort_values(["codigo", "week"]).reset_index(drop=True)


def add_returns(g: pd.DataFrame) -> pd.DataFrame:
    C, I = _accruals(g["week"].min().date() - pd.Timedelta(days=40).to_pytimedelta())
    Cw, Iw = C.reindex(g["week"].unique(), method="ffill"), I.reindex(g["week"].unique(), method="ffill")
    g = g.copy()
    g["next_week"] = g.groupby("codigo")["week"].shift(-1)
    g["next_ratio"] = g.groupby("codigo")["ratio"].shift(-1)
    ok = g["next_week"].notna() & ((g["next_week"] - g["week"]).dt.days == 7)
    tau = 5 / 252
    cg = Cw.reindex(g["next_week"]).to_numpy() / Cw.reindex(g["week"]).to_numpy()
    ig = Iw.reindex(g["next_week"]).to_numpy() / Iw.reindex(g["week"]).to_numpy()
    c = g["contract"].to_numpy() / 100
    gpar = np.select([g["kind"].eq("DI_SPREAD"), g["kind"].eq("IPCA")], [cg * (1 + c) ** tau, ig * (1 + c) ** tau],
                     (1 + c) ** tau)
    g["ret"] = np.where(ok, gpar * g["next_ratio"] / g["ratio"] - cg, np.nan)
    g.loc[g["ret"].abs() > 0.3, "ret"] = np.nan  # broken prints
    # Rate-hedged (pure credit) return: add back duration × change in the benchmark swap rate, as a PM
    # hedging IPCA+/Pré paper with DAP/DI futures would earn. DI+ floaters need no hedge.
    d_bench = (g.groupby("codigo")["bench_rate"].shift(-1) - g["bench_rate"]) / 100
    g["ret_hedged"] = g["ret"] + (g["dur"] * d_bench).where(g["kind"].isin(["IPCA", "PRE"]), 0.0).fillna(0.0)
    g.loc[g["ret_hedged"].abs() > 0.3, "ret_hedged"] = np.nan
    return g


def add_features(g: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    g = g.copy()
    g["peer"] = g["kind"] + "_" + g["incent"].astype(str)
    g["resid_bps"] = np.nan
    for (w, p), grp in g[g["fresh"]].groupby(["week", "peer"]):
        if len(grp) < 8:
            continue
        b = pd.cut(grp["dur"], BUCKETS)
        med = grp.groupby(b, observed=True).agg(d=("dur", "median"), s=("cdi_bps", "median")).dropna()
        fair = np.interp(grp["dur"], med["d"], med["s"]) if len(med) > 1 else med["s"].iloc[0]
        g.loc[grp.index, "resid_bps"] = grp["cdi_bps"] - fair
    wk = g.groupby("week")["resid_bps"]
    med, mad = wk.transform("median"), wk.transform(lambda s: (s - s.median()).abs().median() * 1.4826)
    g["resid_z"] = ((g["resid_bps"] - med) / mad.replace(0, np.nan)).clip(-5, 5)
    s = g.groupby("codigo")["cdi_bps"]
    mu = s.transform(lambda x: x.shift(1).rolling(26, min_periods=8).mean())
    sd = s.transform(lambda x: x.shift(1).rolling(26, min_periods=8).std())
    g["own_z"] = ((g["cdi_bps"] - mu) / sd.replace(0, np.nan)).clip(-5, 5)
    g["d_spread_4w"] = s.diff(4)
    g["d_spread_1w"] = s.diff(1)
    g["d_ratio_4w"] = g.groupby("codigo")["ratio"].pct_change(4)
    g["carry_per_dur"] = g["cdi_bps"] / g["dur"]
    g["cnpj8"] = g["cnpj"].astype(str).str.replace(r"\D", "", regex=True).str[:8]
    g["asof"] = g["week"]
    f = nw.features_at(events, g[["cnpj8", "asof"]])
    g = pd.concat([g, f], axis=1)
    g["news_any_30d"] = g[[c for c in f.columns if c.endswith("_30d")]].sum(axis=1)
    g["distress_90d"] = g["n_distress_90d"] > 0
    g["mkt_distress_z"] = nw.market_gauge(events, pd.DatetimeIndex(g["week"].unique())).reindex(g["week"]).to_numpy()
    ida = h.ida("IDADI")["index"].pct_change()
    cdi = h.bcb_series(12, date(2020, 1, 1)) / 100
    mom = (ida - cdi.reindex(ida.index)).rolling(21).sum()
    g["mkt_mom_21"] = mom.reindex(pd.DatetimeIndex(g["week"].unique()), method="ffill").reindex(g["week"]).to_numpy()
    g["eligible"] = g["fresh"] & g["ratio"].between(0.9, 1.1) & (g["dur"] >= 0.5)  # entry filter only
    return g


def build(start: date = date(2021, 1, 1)) -> pd.DataFrame:
    panel = build_panel(start)
    g = add_returns(weekly_grid(panel))
    return add_features(g, nw.cvm_events(start.year))


# ---------------------------------------------------------------- backtest engine
@dataclass
class Rule:
    """Stateful per-bond entry/exit rule evaluated weekly. enter/exit/stop are functions of the row."""
    name: str
    enter: callable
    exit: callable
    reentry_weeks: int = 0
    max_weeks: int | None = None
    note: str = ""
    params: dict = field(default_factory=dict)


def run_rule(g: pd.DataFrame, rule: Rule, lag: int = 1, cost_bps: float = 25.0,
             start: str = "2022-01-01", overlay: pd.Series | None = None, ret_col: str = "ret") -> dict:
    """Equal-weight portfolio of all bonds currently 'in' by the rule. Weekly excess returns vs CDI."""
    weeks = sorted(g["week"].unique())
    by_week = {w: d.set_index("codigo") for w, d in g.groupby("week")}
    held: dict[str, dict] = {}         # code -> {"since": week, "entry_spread": x}
    banned: dict[str, pd.Timestamp] = {}
    decided: list[set] = []
    for w in weeks:
        d = by_week[w]
        # exits
        for code in list(held):
            row = d.loc[code] if code in d.index else None
            st = held[code]
            weeks_held = (w - st["since"]).days // 7
            if row is None or (row["age_days"] > 7 * STALE_WEEKS):
                held.pop(code)
                continue
            if rule.exit(row, st) or (rule.max_weeks and weeks_held >= rule.max_weeks):
                held.pop(code)
                if rule.reentry_weeks:
                    banned[code] = w + pd.Timedelta(weeks=rule.reentry_weeks)
        # entries (only fresh, eligible bonds)
        cand = d[d["eligible"]]
        for code, row in cand.iterrows():
            if code in held or (code in banned and w < banned[code]):
                continue
            if rule.enter(row):
                held[code] = {"since": w, "entry_spread": row["cdi_bps"], "entry_ratio": row["ratio"]}
        decided.append(set(held))
    # returns: portfolio decided at week i earns ret at week i+lag (ret(k) is mark k → k+1)
    rets, turns, ns = [], [], []
    for i in range(len(weeks) - lag):
        w_earn = weeks[i + lag]
        names = decided[i]
        prev = decided[i - 1] if i > 0 else set()
        d = by_week[w_earn]
        r = d.loc[d.index.intersection(list(names)), ret_col].fillna(0.0)
        n = len(names)
        port = r.mean() if n else 0.0
        turn = (len(names ^ prev) / max(n, len(prev), 1)) if (names or prev) else 0.0
        if overlay is not None:
            on = overlay.asof(weeks[i]) if len(overlay) else 1.0
            port *= on
        rets.append(port - turn * cost_bps / 1e4 * (0.5 if turn else 0))
        turns.append(turn)
        ns.append(n)
    s = pd.Series(rets, index=weeks[lag:])
    s = s[s.index >= pd.Timestamp(start)]
    return {"name": rule.name, "weekly": s, "turnover": float(np.mean(turns[-len(s):])) if len(s) else np.nan,
            "avg_n": float(np.mean(ns[-len(s):])) if len(s) else 0.0, "note": rule.note}


def stats(s: pd.Series, bench: pd.Series | None = None) -> dict:
    eq = (1 + s).cumprod()
    yrs = len(s) / 52
    ann = eq.iloc[-1] ** (1 / yrs) - 1 if yrs > 0 else np.nan
    vol = s.std() * np.sqrt(52)
    out = {"excess_ann_%": ann * 100, "vol_%": vol * 100, "sharpe": ann / vol if vol else np.nan,
           "max_dd_%": ((eq / eq.cummax()) - 1).min() * 100, "worst_week_%": s.min() * 100}
    if bench is not None:
        b = bench.reindex(s.index).fillna(0)
        out["vs_bench_ann_%"] = (s - b).mean() * 5200
        te = (s - b).std() * np.sqrt(52)
        out["info_ratio"] = (s - b).mean() * 52 / te if te else np.nan
    return out
