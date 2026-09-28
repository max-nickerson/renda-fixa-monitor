"""Extra point-in-time feature blocks for the weekly lab, keyed by (cnpj8, week).

EQ   issuer (or parent) stock: returns, drawdown, volatility, abnormal volume     ← data/history/equity_daily.pkl
COM  production-chain margin shocks: Σ sector/issuer exposure × commodity z-move  ← commodities.pkl + exposure CSVs
RAT  rating actions (downgrades/upgrades)                                          ← rating_events.pkl
REG  regulator decisions for the issuer's sector                                   ← regulator_events.pkl
SEC  negative sector/commodity news volume                                         ← sector_news.pkl

Timing rule for a decision on Monday `week`: only data known by the end of Sunday (week − 1 day) is used —
daily closes up to Friday, commodity values whose `available_date` ≤ Sunday, news dated ≤ Sunday.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..config import DATA_DIR, ROOT

H = DATA_DIR / "history"
RD = ROOT / "research" / "data"

EQ_FEATURES = ["eq_ret_1w", "eq_ret_4w", "eq_ret_13w", "eq_dd_13w", "eq_vol_4w", "eq_vol_chg", "eq_volu_z", "eq_listed"]
COM_FEATURES = ["com_shock_4w", "com_shock_13w", "com_exposed"]
RAT_FEATURES = ["rat_down_90d", "rat_up_90d", "rat_days_since_down", "rat_net_365d"]
REG_FEATURES = ["reg_neg_30d", "reg_pos_30d", "reg_neg_90d"]
SEC_FEATURES = ["sec_news_z"]
MACRO_VARS = {"ipca", "selic", "ibc_br", "ibc_br_sa", "usdbrl"}
BLOCKS = {"EQ": EQ_FEATURES, "COM": COM_FEATURES, "RAT": RAT_FEATURES, "REG": REG_FEATURES, "SEC": SEC_FEATURES}

# topic → sector mapping for sector news (topic names as produced by research/data_news_extra.py)
TOPIC_SECTORS = {
    "fertilizantes": ["fertilizers"], "ureia": ["fertilizers"], "gas": ["fertilizers", "petrochemicals", "utilities_generation_thermal"],
    "etanol": ["sugar_ethanol", "fuel_distribution"], "acucar": ["sugar_ethanol"],
    "setor_eletrico": ["utilities_distribution", "utilities_generation_hydro", "utilities_generation_renewables",
                       "utilities_generation_thermal", "utilities_transmission"],
    "reservatorios": ["utilities_generation_hydro", "utilities_distribution"],
    "aereas": ["airlines"], "locadoras": ["car_rental_fleet", "trucking_equipment_rental"], "varejo": ["retail"],
    "construtoras": ["real_estate", "cement_construction_materials"], "agro": ["agribusiness_grains", "agribusiness_protein", "sugar_ethanol"],
    "frigorificos": ["agribusiness_protein"], "siderurgia": ["steel_metals", "mining"], "celulose": ["pulp_paper"],
    "saneamento": ["sanitation"],
}
REG_SECTORS = {
    "utilities": ["utilities_distribution", "utilities_generation_hydro", "utilities_generation_renewables",
                  "utilities_generation_thermal", "utilities_transmission"],
    "toll_roads": ["toll_roads"], "railways": ["railways"], "oil_gas": ["oil_gas", "fuel_distribution", "petrochemicals"],
    "sanitation": ["sanitation"], "airlines": ["airlines", "airports_ports_logistics"],
}


def _load(name: str):
    p = H / name
    return pd.read_pickle(p) if p.exists() else None


def _csv(name: str) -> pd.DataFrame | None:
    p = RD / name
    return pd.read_csv(p, dtype={"cnpj8": str}) if p.exists() else None


def _cnpj8(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace(r"\D", "", regex=True).str.zfill(8).str[:8]


# ---------------------------------------------------------------- EQ
def equity_features(keys: pd.DataFrame) -> pd.DataFrame:
    """keys: DataFrame [cnpj8, week]. One row per key with EQ features (NaN when not listed)."""
    out = pd.DataFrame(index=keys.index, columns=EQ_FEATURES, dtype=float)
    out["eq_listed"] = 0.0
    mp, px = _csv("equity_map.csv"), _load("equity_daily.pkl")
    if mp is None or px is None or px.empty:
        return out
    mp = mp[mp["ticker"].notna() & (mp.get("mapping_type", "direct") != "none")].copy()
    mp["cnpj8"] = _cnpj8(mp["cnpj8"])
    order = {"high": 0, "med": 1, "medium": 1, "low": 2}
    mp["_c"] = mp.get("confidence", pd.Series("med", index=mp.index)).map(order).fillna(1)
    mp["_d"] = (mp.get("mapping_type", pd.Series("direct", index=mp.index)) != "direct").astype(int)
    mp["_o"] = range(len(mp))
    # Several candidate tickers per issuer (own listing, parent, pre/post change of control): each week uses
    # the most-preferred ticker that actually traded that week — point-in-time, no reliance on free-text dates.
    cands = mp.sort_values(["cnpj8", "_c", "_d", "_o"]).groupby("cnpj8")["ticker"].apply(list)
    px = px.copy()
    px["date"] = pd.to_datetime(px["date"])
    price_col = "adj_close" if "adj_close" in px and px["adj_close"].notna().any() else "close"
    weeks = pd.DatetimeIndex(sorted(keys["week"].unique()))
    feats = {}
    for tkr, g in px.groupby("ticker"):
        g = g.sort_values("date").drop_duplicates("date", keep="last").set_index("date")
        raw = g[price_col].astype(float).where(lambda s: s > 0).ffill()
        v = g["volume"].astype(float)
        r = np.log(raw).diff()
        r[r.abs() > 0.5] = 0.0  # unadjusted splits/reverse splits in B3 COTAHIST fills
        p = np.exp(r.fillna(0).cumsum())
        # weekly snapshots as of the Friday before each Monday
        asof = pd.DataFrame(index=weeks)
        idx = p.index.searchsorted(weeks - pd.Timedelta(days=2), side="right") - 1  # last close ≤ Saturday
        ok = idx >= 0
        def at(series, lag_days=0):
            vals = np.full(len(weeks), np.nan)
            base = series.reindex(p.index)
            j = idx - lag_days
            m = ok & (j >= 0)
            vals[m] = base.to_numpy()[j[m]]
            return vals
        pv = at(p)
        asof["eq_ret_1w"] = np.log(pv / at(p, 5))
        asof["eq_ret_4w"] = np.log(pv / at(p, 21))
        asof["eq_ret_13w"] = np.log(pv / at(p, 63))
        hi13 = p.rolling(63, min_periods=20).max()
        asof["eq_dd_13w"] = pv / at(hi13) - 1
        vol4 = r.rolling(21, min_periods=10).std() * np.sqrt(252)
        vol26 = r.rolling(126, min_periods=40).std() * np.sqrt(252)
        asof["eq_vol_4w"] = at(vol4)
        asof["eq_vol_chg"] = at(vol4) / at(vol26) - 1
        lv = np.log1p(v)
        asof["eq_volu_z"] = (at(lv.rolling(20, min_periods=10).mean()) - at(lv.rolling(126, min_periods=40).mean())) / \
            at(lv.rolling(126, min_periods=40).std())
        # a ticker is "live" in a week only if it traded in the 10 days before it
        last_trade = pd.Series(p.index, index=p.index).reindex(weeks - pd.Timedelta(days=2), method="ffill")
        asof["_live"] = ((weeks - pd.Timedelta(days=2)) - pd.DatetimeIndex(last_trade.values)).days <= 10
        feats[tkr] = asof
    cols = [c for c in EQ_FEATURES if c != "eq_listed"]
    for cnpj8, idx_rows in keys.groupby("cnpj8").groups.items():
        tickers = [t for t in cands.get(cnpj8, []) if t in feats]
        if not tickers:
            continue
        wk = keys.loc[idx_rows, "week"]
        vals = np.full((len(wk), len(cols)), np.nan)
        filled = np.zeros(len(wk), dtype=bool)
        for t in tickers:
            f = feats[t].reindex(wk)
            use = (~filled) & f["_live"].fillna(False).to_numpy() & f["eq_ret_4w"].notna().to_numpy()
            vals[use] = f.loc[:, cols].to_numpy()[use]
            filled |= use
        out.loc[idx_rows, cols] = vals
        out.loc[idx_rows, "eq_listed"] = filled.astype(float)
    return out.astype(float)


# ---------------------------------------------------------------- COM
def _commodity_moves(weeks: pd.DatetimeIndex) -> dict[str, pd.DataFrame]:
    """Per series: z-scored 4w and 13w log changes as known at each week (point in time)."""
    c = _load("commodities.pkl")
    if c is None or c.empty:
        return {}
    c = c.copy()
    c["available_date"] = pd.to_datetime(c["available_date"])
    c["period_date"] = pd.to_datetime(c["period_date"])
    out = {}
    for s, g in c.groupby("series"):
        g = g.dropna(subset=["value"]).sort_values(["available_date", "period_date"])
        if len(g) < 10:
            continue
        vals = []
        for w in weeks:
            known = g[g["available_date"] <= w - pd.Timedelta(days=1)]
            if known.empty:
                vals.append((np.nan, np.nan))
                continue
            known = known.drop_duplicates("period_date", keep="last").set_index("period_date")["value"].sort_index()
            last_p = known.index[-1]
            def chg(days):
                past = known[known.index <= last_p - pd.Timedelta(days=days)]
                if past.empty or known.iloc[-1] <= 0 or past.iloc[-1] <= 0:
                    return np.nan
                return float(np.log(known.iloc[-1] / past.iloc[-1]))
            vals.append((chg(28), chg(91)))
        d = pd.DataFrame(vals, index=weeks, columns=["c4", "c13"])
        # z vs trailing 3y distribution of the same statistic (known at the time)
        for col in ("c4", "c13"):
            mu = d[col].rolling(156, min_periods=26).mean().shift(1)
            sd = d[col].rolling(156, min_periods=26).std().shift(1)
            d[col + "_z"] = ((d[col] - mu) / sd).clip(-4, 4)
        out[s] = d
    return out


def exposure_table() -> pd.DataFrame:
    """Long table cnpj8 → variable → weight (sector defaults + issuer overrides)."""
    sec, exp, ovr = _csv("issuer_sectors.csv"), _csv("sector_exposures.csv"), _csv("issuer_overrides.csv")
    if sec is None or exp is None:
        return pd.DataFrame(columns=["cnpj8", "variable", "weight"])
    sec["cnpj8"] = _cnpj8(sec["cnpj8"])
    e = sec[["cnpj8", "sector"]].merge(exp[["sector", "variable", "weight"]], on="sector", how="inner")
    if ovr is not None and not ovr.empty:
        ovr["cnpj8"] = _cnpj8(ovr["cnpj8"])
        e = e[~e["cnpj8"].isin(set(ovr["cnpj8"]))]
        e = pd.concat([e[["cnpj8", "variable", "weight"]], ovr[["cnpj8", "variable", "weight"]]])
    e = e[e["weight"].astype(float) != 0]
    return e[["cnpj8", "variable", "weight"]].astype({"weight": float})


def commodity_features(keys: pd.DataFrame, exposures: pd.DataFrame | None = None) -> pd.DataFrame:
    """Σ_v weight(issuer, v) × z-move(v). `exposures` can be a permuted table for the placebo test."""
    out = pd.DataFrame(index=keys.index, data={"com_shock_4w": np.nan, "com_shock_13w": np.nan, "com_exposed": 0.0})
    weeks = pd.DatetimeIndex(sorted(keys["week"].unique()))
    moves = _commodity_moves(weeks)
    e = exposure_table() if exposures is None else exposures
    if not moves or e.empty:
        return out
    # Production chain = traded commodities, energy and power only; macro links (inflation, rates, activity,
    # FX) are common to almost every issuer and would turn this into a macro factor.
    e = e[e["variable"].isin(moves) & ~e["variable"].isin(MACRO_VARS)]
    for cnpj8, idx_rows in keys.groupby("cnpj8").groups.items():
        ex = e[e["cnpj8"] == cnpj8]
        if ex.empty:
            continue
        rows = keys.loc[idx_rows, "week"]
        s4 = np.zeros(len(rows))
        s13 = np.zeros(len(rows))
        for var, w in zip(ex["variable"], ex["weight"]):
            m = moves[var].reindex(rows)
            s4 += w * m["c4_z"].fillna(0).to_numpy()
            s13 += w * m["c13_z"].fillna(0).to_numpy()
        out.loc[idx_rows, "com_shock_4w"] = s4
        out.loc[idx_rows, "com_shock_13w"] = s13
        out.loc[idx_rows, "com_exposed"] = 1.0
    return out


# ---------------------------------------------------------------- RAT / REG / SEC
def _count_window(dates: np.ndarray, asof: np.ndarray, days: int) -> np.ndarray:
    hi = np.searchsorted(dates, asof - np.timedelta64(1, "D"), side="right")
    lo = np.searchsorted(dates, asof - np.timedelta64(days, "D"), side="right")
    return hi - lo


def rating_features(keys: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=keys.index, data={"rat_down_90d": 0.0, "rat_up_90d": 0.0, "rat_days_since_down": 9999.0,
                                               "rat_net_365d": 0.0})
    ev = _load("rating_events.pkl")
    if ev is None or ev.empty:
        return out
    ev = ev.copy()
    ev["cnpj8"] = _cnpj8(ev["cnpj8"])
    ev["date"] = pd.to_datetime(ev["date"])
    for cnpj8, idx_rows in keys.groupby("cnpj8").groups.items():
        g = ev[ev["cnpj8"] == cnpj8]
        if g.empty:
            continue
        asof = keys.loc[idx_rows, "week"].to_numpy(dtype="datetime64[ns]")
        dn = np.sort(g.loc[g["direction"] < 0, "date"].to_numpy(dtype="datetime64[ns]"))
        up = np.sort(g.loc[g["direction"] > 0, "date"].to_numpy(dtype="datetime64[ns]"))
        out.loc[idx_rows, "rat_down_90d"] = _count_window(dn, asof, 90)
        out.loc[idx_rows, "rat_up_90d"] = _count_window(up, asof, 90)
        out.loc[idx_rows, "rat_net_365d"] = _count_window(up, asof, 365) - _count_window(dn, asof, 365)
        if len(dn):
            hi = np.searchsorted(dn, asof - np.timedelta64(1, "D"), side="right")
            last = np.where(hi > 0, dn[np.maximum(hi - 1, 0)], np.datetime64("NaT"))
            days = (asof - last) / np.timedelta64(1, "D")
            out.loc[idx_rows, "rat_days_since_down"] = np.where(np.isnan(days), 9999, days)
    return out


def issuer_sector_map() -> pd.Series:
    sec = _csv("issuer_sectors.csv")
    if sec is None:
        return pd.Series(dtype=str)
    sec["cnpj8"] = _cnpj8(sec["cnpj8"])
    return sec.drop_duplicates("cnpj8").set_index("cnpj8")["sector"]


def regulator_features(keys: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=keys.index, data={"reg_neg_30d": 0.0, "reg_pos_30d": 0.0, "reg_neg_90d": 0.0})
    ev = _load("regulator_events.pkl")
    sm = issuer_sector_map()
    if ev is None or ev.empty or sm.empty:
        return out
    ev = ev.copy()
    ev["date"] = pd.to_datetime(ev["date"])
    sector = keys["cnpj8"].map(sm)
    for grp, sectors in REG_SECTORS.items():
        g = ev[ev["sector_group"].astype(str).str.startswith(grp.split("_")[0])] if grp == "utilities" else \
            ev[ev["sector_group"].astype(str) == grp]
        if g.empty:
            continue
        rows = sector.isin(sectors)
        if not rows.any():
            continue
        asof = keys.loc[rows, "week"].to_numpy(dtype="datetime64[ns]")
        neg = np.sort(g.loc[g["direction"] < 0, "date"].to_numpy(dtype="datetime64[ns]"))
        pos = np.sort(g.loc[g["direction"] > 0, "date"].to_numpy(dtype="datetime64[ns]"))
        out.loc[rows, "reg_neg_30d"] = _count_window(neg, asof, 30)
        out.loc[rows, "reg_pos_30d"] = _count_window(pos, asof, 30)
        out.loc[rows, "reg_neg_90d"] = _count_window(neg, asof, 90)
    return out


def sector_news_features(keys: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=keys.index, data={"sec_news_z": np.nan})
    sn = _load("sector_news.pkl")
    sm = issuer_sector_map()
    if sn is None or sn.empty or sm.empty:
        return out
    sn = sn.copy()
    sn["date"] = pd.to_datetime(sn["date"])
    col = "n_negative" if "n_negative" in sn else "n_items"
    weeks = pd.DatetimeIndex(sorted(keys["week"].unique()))
    topic_z = {}
    for topic, g in sn.groupby("topic"):
        daily = g.groupby("date")[col].sum().reindex(pd.date_range(weeks.min() - pd.Timedelta(days=800), weeks.max()),
                                                     fill_value=0)
        roll = daily.rolling(30).sum().shift(1)
        z = (roll - roll.rolling(730, min_periods=180).mean()) / roll.rolling(730, min_periods=180).std()
        topic_z[str(topic)] = z.reindex(weeks, method="ffill")
    sector = keys["cnpj8"].map(sm)
    for sec_name in sector.dropna().unique():
        topics = [t for t, secs in TOPIC_SECTORS.items() if sec_name in secs]
        zs = [topic_z[t] for t in topic_z if any(t.startswith(x) or x in t for x in topics)]
        if not zs:
            continue
        zmax = pd.concat(zs, axis=1).max(axis=1)
        rows = sector == sec_name
        out.loc[rows, "sec_news_z"] = zmax.reindex(keys.loc[rows, "week"]).to_numpy()
    return out


def all_blocks(keys: pd.DataFrame) -> pd.DataFrame:
    parts = [equity_features(keys), commodity_features(keys), rating_features(keys), regulator_features(keys),
             sector_news_features(keys)]
    return pd.concat(parts, axis=1)


def coverage(g: pd.DataFrame) -> pd.DataFrame:
    """Share of eligible bond-weeks (2022+) where each block has information."""
    el = g[g["eligible"] & (g["week"] >= "2022-01-01")]
    rows = {
        "EQ (listed stock)": (el["eq_listed"] > 0).mean(),
        "COM (commodity-exposed)": (el["com_exposed"] > 0).mean(),
        "RAT (≥1 rating event ever)": el.groupby("cnpj8")["rat_net_365d"].transform(lambda s: (s != 0).any()).mean()
        if "rat_net_365d" in el else np.nan,
        "REG (sector has regulator)": (el["reg_neg_90d"].notna()).mean(),
        "SEC (sector news topic)": el["sec_news_z"].notna().mean(),
    }
    return pd.Series(rows, name="coverage").to_frame()
