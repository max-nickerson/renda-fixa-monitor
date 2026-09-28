"""Daily ('live') strategy lab — decisions at each business day's close, execution at the bond's NEXT REAL TRADE.

Differences from the weekly lab (lab.py):
- the grid is every business day; a bond's mark on day d is its last SND trade ≤ d (≤ 14 days old);
- information known at the close of d: stock closes ≤ d, news/filings/ratings dated ≤ d, ANBIMA-style marks ≤ d;
- an order decided at d executes on the first day k ≥ d + lag on which the bond has a trade dated AFTER d
  (a fresh print), so the signal's own print is never the execution price and illiquid bonds wait for a trade.
  Pending sells keep earning (losing) until they execute — realistic for debentures. Pending buys expire after
  20 business days without a trade.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from .. import history as h
from . import features_ext as fx
from . import news as nw
from . import press
from .lab import BUCKETS
from .selection import _accruals, build_panel

STALE_DAYS = 14


def build(start: date = date(2021, 1, 1)) -> pd.DataFrame:
    p = build_panel(start)
    p = p.sort_values(["codigo", "date"]).reset_index(drop=True)
    days = pd.bdate_range(p["date"].min() + pd.Timedelta(days=21), p["date"].max())
    span = p.groupby("codigo").agg(first=("date", "min"), last=("date", "max"), maturity=("maturity", "first"))
    grid = []
    for code, r in span.iterrows():
        ds = days[(days > r["first"]) & (days <= min(r["last"] + pd.Timedelta(days=STALE_DAYS), r["maturity"]))]
        if len(ds):
            grid.append(pd.DataFrame({"codigo": code, "day": ds}))
    g = pd.concat(grid, ignore_index=True).sort_values("day")
    cols = ["codigo", "date", "ratio", "cdi_bps", "dur", "kind", "contract", "cnpj", "incent", "bench_rate"]
    g = pd.merge_asof(g, p[cols].sort_values("date"), left_on="day", right_on="date", by="codigo", direction="backward")
    g = g.dropna(subset=["date"])
    g["age"] = (g["day"] - g["date"]).dt.days
    g = g[g["age"] <= STALE_DAYS].sort_values(["codigo", "day"]).reset_index(drop=True)

    # daily excess return (mark d → next business day), plus rate-hedged version
    C, I = _accruals(g["day"].min().date() - pd.Timedelta(days=40).to_pytimedelta())
    Cd, Id = C.reindex(days, method="ffill"), I.reindex(days, method="ffill")
    nxt = g.groupby("codigo")["day"].shift(-1)
    ok = nxt.notna() & (np.busday_count(g["day"].values.astype("datetime64[D]"),
                                        nxt.fillna(g["day"]).values.astype("datetime64[D]")) == 1)
    cg = Cd.reindex(nxt).to_numpy() / Cd.reindex(g["day"]).to_numpy()
    ig = Id.reindex(nxt).to_numpy() / Id.reindex(g["day"]).to_numpy()
    c, tau = g["contract"].to_numpy() / 100, 1 / 252
    gpar = np.select([g["kind"].eq("DI_SPREAD"), g["kind"].eq("IPCA")], [cg * (1 + c) ** tau, ig * (1 + c) ** tau],
                     (1 + c) ** tau)
    ratio_n = g.groupby("codigo")["ratio"].shift(-1)
    ret = np.where(ok, gpar * ratio_n / g["ratio"] - cg, np.nan)
    d_bench = (g.groupby("codigo")["bench_rate"].shift(-1) - g["bench_rate"]) / 100
    g["ret"] = ret + np.where(g["kind"].isin(["IPCA", "PRE"]), (g["dur"] * d_bench).fillna(0), 0)
    g.loc[g["ret"].abs() > 0.2, "ret"] = np.nan

    # peer residual (fresh marks only) and eligibility
    g["fresh"] = g["age"] <= 7
    g["peer"] = g["kind"] + "_" + g["incent"].astype(str)
    g["resid_bps"] = np.nan
    for (d, pe), grp in g[g["fresh"]].groupby(["day", "peer"]):
        if len(grp) < 8:
            continue
        b = pd.cut(grp["dur"], BUCKETS)
        med = grp.groupby(b, observed=True).agg(x=("dur", "median"), y=("cdi_bps", "median")).dropna()
        fair = np.interp(grp["dur"], med["x"], med["y"]) if len(med) > 1 else med["y"].iloc[0]
        g.loc[grp.index, "resid_bps"] = grp["cdi_bps"] - fair
    dd = g.groupby("day")["resid_bps"]
    mad = dd.transform(lambda s: (s - s.median()).abs().median() * 1.4826)
    g["resid_z"] = ((g["resid_bps"] - dd.transform("median")) / mad.replace(0, np.nan)).clip(-5, 5)
    g["eligible"] = g["fresh"] & g["ratio"].between(0.9, 1.1) & (g["dur"] >= 0.5)
    el = g["eligible"] & g["cdi_bps"].notna()
    g["cdi_pct"] = np.nan
    g.loc[el, "cdi_pct"] = g[el].groupby("day")["cdi_bps"].rank(pct=True, ascending=False)

    # information known at the CLOSE of day d → features_ext/news use (asof − 1 day), so pass asof = d + 1
    g["cnpj8"] = g["cnpj"].astype(str).str.replace(r"\D", "", regex=True).str[:8]
    keys = pd.DataFrame({"cnpj8": g["cnpj8"], "asof": g["day"] + pd.Timedelta(days=1)})
    cvm = nw.features_at(nw.cvm_events(start.year), keys, windows=(2, 30, 90))
    g["distress_2d"] = cvm["n_distress_2d"]
    g["fact_2d"] = cvm["n_fact_2d"]
    items = pd.read_pickle(h.HIST / "press_items.pkl") if (h.HIST / "press_items.pkl").exists() else pd.DataFrame()
    pr = press.weekly_features(items, keys)
    g["press_neg_7d"], g["press_neg_30d"] = pr["press_neg_7d"], pr["press_neg_30d"]
    rat = fx.rating_features(keys.rename(columns={"asof": "week"}))
    g["rat_days_since_down"] = rat["rat_days_since_down"]
    g.to_pickle(h.HIST / "lab_daily_partial.pkl")  # checkpoint: the steps above are the slow ones
    g = g.join(_equity_daily(g[["cnpj8", "day"]]))

    ida = h.ida("IDADI")["index"].pct_change()
    cdi = h.bcb_series(12, date(2020, 1, 1)) / 100
    mom = (ida - cdi.reindex(ida.index)).rolling(21).sum()
    g["mkt_mom_21"] = mom.reindex(days, method="ffill").reindex(g["day"]).to_numpy()
    return g


def _equity_daily(keys: pd.DataFrame) -> pd.DataFrame:
    """Stock return over the last 5 and 21 trading days as of the close of each day (issuer or parent)."""
    out = pd.DataFrame(index=keys.index, data={"eq_ret_1w": np.nan, "eq_ret_4w": np.nan})
    mp = fx._csv("equity_map.csv")
    px = fx._load("equity_daily.pkl")
    if mp is None or px is None:
        return out
    mp = mp[mp["ticker"].notna() & (mp["mapping_type"] != "none")].copy()
    mp["cnpj8"] = fx._cnpj8(mp["cnpj8"])
    mp["_c"] = mp["confidence"].map({"high": 0, "med": 1, "low": 2}).fillna(1)
    cands = mp.sort_values(["cnpj8", "_c"]).groupby("cnpj8")["ticker"].apply(list)
    px = px.copy()
    px["date"] = pd.to_datetime(px["date"])
    col = "adj_close" if "adj_close" in px else "close"
    series = {}
    for t, gg in px.groupby("ticker"):
        s = gg.sort_values("date").drop_duplicates("date").set_index("date")[col].astype(float)
        r = np.log(s.where(s > 0)).diff()
        r[r.abs() > 0.5] = 0.0
        lp = r.fillna(0).cumsum()
        series[t] = pd.DataFrame({"r1": lp - lp.shift(5), "r4": lp - lp.shift(21), "live": 1.0})
    for cnpj8, idx in keys.groupby("cnpj8").groups.items():
        days = keys.loc[idx, "day"]
        v1 = np.full(len(days), np.nan)
        v4 = np.full(len(days), np.nan)
        for t in cands.get(cnpj8, []):
            f = series.get(t)
            if f is None:
                continue
            ud = pd.DatetimeIndex(sorted(days.unique()))  # several bonds per issuer → dates repeat/interleave
            f = f.reindex(ud, method="ffill", limit=3).reindex(days)
            use = np.isnan(v4) & f["r4"].notna().to_numpy()
            v1[use], v4[use] = np.exp(f["r1"].to_numpy()[use]) - 1, np.exp(f["r4"].to_numpy()[use]) - 1
        out.loc[idx, "eq_ret_1w"], out.loc[idx, "eq_ret_4w"] = v1, v4
    return out


@dataclass
class DRule:
    name: str
    enter: callable              # row -> bool (evaluated only on decision days, eligible bonds)
    exit: callable               # row -> bool (evaluated every day for held bonds when event_exit, else on decision days)
    decide: str = "daily"        # 'daily' | 'weekly' (Mondays)
    event_exit: callable | None = None  # row -> bool, checked EVERY day regardless of `decide`
    reentry_days: int = 0


def run(g: pd.DataFrame, rule: DRule, lag: int = 1, cost_bps: float = 25.0, event_cost_bps: float | None = None,
        overlay: pd.Series | None = None, start: str = "2022-01-01") -> dict:
    days = sorted(g["day"].unique())
    by_day = {d: x.set_index("codigo") for d, x in g.groupby("day")}
    held: set[str] = set()
    pend_in: dict[str, pd.Timestamp] = {}
    pend_out: dict[str, tuple[pd.Timestamp, bool]] = {}
    banned: dict[str, pd.Timestamp] = {}
    rets, trades_log, ns = [], [], []
    bday = {d: i for i, d in enumerate(days)}
    ecost = cost_bps if event_cost_bps is None else event_cost_bps
    for d in days:
        rows = by_day[d].to_dict("index")
        i = bday[d]
        n_before = len(held)
        traded_bps = 0.0  # Σ one-way cost of executed trades; each bond is 1/N of the book
        # 1) execute pending orders on a fresh print dated after the decision day, at least `lag` bdays later
        for code, (dec, is_event) in list(pend_out.items()):
            r = rows.get(code)
            if r is None and code not in held:
                pend_out.pop(code)
                continue
            if r is not None and i - bday[dec] >= lag and r["date"] > dec:
                held.discard(code)
                pend_out.pop(code)
                traded_bps += (ecost if is_event else cost_bps) / 2
        for code, dec in list(pend_in.items()):
            r = rows.get(code)
            if i - bday[dec] > 20:
                pend_in.pop(code)
                continue
            if r is not None and i - bday[dec] >= lag and r["date"] > dec:
                held.add(code)
                pend_in.pop(code)
                traded_bps += cost_bps / 2
        # held bonds that disappeared (matured / no trades for STALE_DAYS) drop out
        for code in [c for c in held if c not in rows]:
            held.discard(code)
        cost = traded_bps / 1e4 / max(n_before, len(held), 1)
        # 2) earn today's return on the held book (ret = mark d → d+1)
        vals = [rows[c]["ret"] for c in held if c in rows]
        port = float(np.nanmean(vals)) if vals else 0.0
        port = 0.0 if port != port else port
        if overlay is not None:
            port *= float(overlay.asof(d)) if len(overlay) else 1.0
        rets.append(port - cost)
        ns.append(len(held))
        # 3) decide at the close of d
        decision_day = rule.decide == "daily" or pd.Timestamp(d).weekday() == 0
        for code in list(held):
            if code in pend_out:
                continue
            r = rows.get(code)
            if r is None:
                continue
            if rule.event_exit is not None and rule.event_exit(r):
                pend_out[code] = (d, True)
                if rule.reentry_days:
                    banned[code] = d + pd.Timedelta(days=rule.reentry_days)
            elif decision_day and rule.exit(r):
                pend_out[code] = (d, False)
        if decision_day:
            for code, r in rows.items():
                if (not r["eligible"] or code in held or code in pend_in
                        or (code in banned and d < banned[code])):
                    continue
                if rule.enter(r):
                    pend_in[code] = d
    s = pd.Series(rets, index=days)
    s = s[s.index >= pd.Timestamp(start)]
    return {"daily": s, "avg_n": float(np.mean(ns[-len(s):]))}


def stats(s: pd.Series, bench: pd.Series | None = None) -> dict:
    eq = (1 + s).cumprod()
    yrs = len(s) / 252
    ann = eq.iloc[-1] ** (1 / yrs) - 1
    vol = s.std() * np.sqrt(252)
    out = {"excess_ann_%": ann * 100, "vol_%": vol * 100, "sharpe": ann / vol if vol else np.nan,
           "max_dd_%": ((eq / eq.cummax()) - 1).min() * 100}
    if bench is not None:
        d = (s - bench.reindex(s.index).fillna(0))
        out["vs_bench_ann_%"] = d.mean() * 25200
        out["t_vs_bench"] = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
    return out
