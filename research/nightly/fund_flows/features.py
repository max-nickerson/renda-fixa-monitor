"""Point-in-time fund-flow / fund-holdings features for debentures.

Sources (built by build_data.py from CVM open data):
  cda_deb.pkl  CDA BLC_4 debenture positions per fund x reference month (SND code = CD_ATIVO)
  cda_pl.pkl   fund net assets per CDA reference month
  inf.pkl      daily fund reports (PL, subscriptions CAPTC_DIA, redemptions RESG_DIA, #investors)

PIT rules
  * CDA reference month m is used only from m_end + CDA_LAG_DAYS (60 calendar days; the legal deadline is ~10
    business days, but confidential positions / late filers / file rewrites -> conservative).
  * inf_diario for day t is used only INF_LAG (5) business-day rows later.
  * Credit fund (PIT) = debenture holdings / PL >= 25% in the latest available CDA vintage.

Public API
  agg_flows()          -> DataFrame on business days: aggregate credit-fund net-flow ratios (af21, af63, af126),
                          investor-count change (dcot63), #credit funds, credit PL; already lagged (PIT).
  bond_features(days)  -> DataFrame (day, codigo, nh, lnh, dnh3, dq3, nb3, hhi, cshare, press21, press63, stress63)
  signals(panel)       -> panel[['day','codigo']] merged with bond_features (cached per set of panel dates).
"""
import os
import hashlib
import numpy as np
import pandas as pd

D = "data/history/nightly/fund_flows"
CDA_LAG_DAYS = 60
INF_LAG = 5
CREDIT_SHARE = 0.25
_M = {}


def _load():
    if "deb" not in _M:
        deb = pd.read_pickle(f"{D}/cda_deb.pkl")
        deb = deb[deb["val"].fillna(0) > 0].copy()
        _M["deb"] = deb
        _M["pl"] = pd.read_pickle(f"{D}/cda_pl.pkl")[["fund", "ref", "pl"]]
    return _M["deb"], _M["pl"]


def fund_class():
    """fund x ref: debenture share of PL and PIT credit-fund flag."""
    if "cls" in _M:
        return _M["cls"]
    cache = f"{D}/fund_class.pkl"
    if os.path.exists(cache):
        _M["cls"] = pd.read_pickle(cache)
        return _M["cls"]
    deb, pl = _load()
    s = deb.groupby(["fund", "ref"])["val"].sum().rename("deb_val").reset_index()
    s = s.merge(pl, on=["fund", "ref"], how="left")
    s["deb_share"] = s["deb_val"] / s["pl"].where(s["pl"] > 1e6)
    s["credit"] = s["deb_share"] >= CREDIT_SHARE
    s.to_pickle(cache)
    _M["cls"] = s
    return s


def _flow_mats():
    """Business-day matrices (days x funds) of net flow, PL and #investors (float32)."""
    if "fm" in _M:
        return _M["fm"]
    cache = f"{D}/flow_mats.pkl"
    if os.path.exists(cache):
        _M["fm"] = pd.read_pickle(cache)
        return _M["fm"]
    inf = pd.read_pickle(f"{D}/inf.pkl")
    inf = inf[inf["day"].dt.dayofweek < 5].copy()
    inf["net"] = inf["captc"] - inf["resg"]
    net = inf.pivot_table(index="day", columns="fund", values="net", aggfunc="sum").astype("float32")
    pl = inf.pivot_table(index="day", columns="fund", values="pl", aggfunc="sum").astype("float32")
    ncot = inf.pivot_table(index="day", columns="fund", values="ncot", aggfunc="sum").astype("float32")
    cnt = pl.notna().sum(1)
    keep = cnt > 0.5 * cnt.rolling(21, min_periods=1).median()
    net, pl, ncot = net[keep], pl[keep], ncot[keep]
    pl = pl.where(pl > 0).ffill(limit=5)
    net = net.fillna(0.0)
    _M["fm"] = {"net": net, "pl": pl, "ncot": ncot.ffill(limit=5)}
    pd.to_pickle(_M["fm"], cache)
    return _M["fm"]


def fund_flow_ratios(W):
    """flow_W(t) = sum of net flows over (t-W, t] / PL(t-W), clipped to [-1, 1]; NaN if PL(t-W) < R$5m.
    Indexed by report day t (NOT yet lagged)."""
    key = f"fr{W}"
    if key in _M:
        return _M[key]
    fm = _flow_mats()
    cs = fm["net"].cumsum()
    num = cs - cs.shift(W)
    den = fm["pl"].shift(W)
    r = (num / den.where(den > 5e6)).clip(-1, 1).astype("float32")
    _M[key] = r
    return r


def _vintage_for(days):
    """latest CDA reference month available at each day (ref + CDA_LAG_DAYS <= day)."""
    deb, _ = _load()
    refs = np.array(sorted(deb["ref"].unique()), dtype="datetime64[ns]")
    avail = refs + np.timedelta64(CDA_LAG_DAYS, "D")
    idx = np.searchsorted(avail, np.asarray(pd.DatetimeIndex(days).values), side="right") - 1
    return pd.Series([pd.Timestamp(refs[i]) if i >= 0 else pd.NaT for i in idx], index=pd.DatetimeIndex(days))


def _lagged(df, day):
    """row of df (report-day index) usable at `day`: INF_LAG rows before the last report <= day."""
    pos = df.index.searchsorted(pd.Timestamp(day), side="right") - 1 - INF_LAG
    return df.iloc[pos] if pos >= 0 else None


def agg_flows():
    """Aggregate flows of PIT credit funds on report days, lagged INF_LAG rows (value at t usable at close t)."""
    cache = f"{D}/agg_flows.pkl"
    if os.path.exists(cache):
        return pd.read_pickle(cache)
    fm = _flow_mats()
    cls = fund_class()
    days = fm["net"].index
    vint = _vintage_for(days)
    cs = fm["net"].cumsum()
    out = []
    for v, dd in vint.groupby(vint):
        funds = cls.loc[(cls["ref"] == v) & cls["credit"], "fund"]
        cols = fm["net"].columns.intersection(funds)
        sub_cs = cs[cols].sum(1)
        sub_pl = fm["pl"][cols].sum(1, min_count=1)
        sub_n = fm["pl"][cols].notna().sum(1)
        sub_cot = fm["ncot"][cols].sum(1, min_count=1)
        rows = pd.DataFrame(index=dd.index)
        for W in (5, 21, 63, 126):
            rows[f"af{W}"] = (sub_cs - sub_cs.shift(W)).reindex(dd.index) / sub_pl.shift(W).reindex(dd.index)
        rows["dcot63"] = (sub_cot / sub_cot.shift(63) - 1).reindex(dd.index)
        rows["n_credit"] = sub_n.reindex(dd.index)
        rows["credit_pl"] = sub_pl.reindex(dd.index)
        out.append(rows)
    A = pd.concat(out).sort_index()
    A = A.shift(INF_LAG)
    A.to_pickle(cache)
    return A


def bond_features(days):
    days = pd.DatetimeIndex(sorted(set(pd.to_datetime(days))))
    deb, _ = _load()
    cls = fund_class()
    fr21, fr63 = fund_flow_ratios(21), fund_flow_ratios(63)
    vint = _vintage_for(days)
    refs = np.array(sorted(deb["ref"].unique()))
    by_ref = {r: g for r, g in deb.groupby("ref")}
    cls_by = {r: g.set_index("fund")["credit"] for r, g in cls.groupby("ref")}
    out = []
    for day in days:
        v = vint.loc[day]
        if pd.isna(v):
            continue
        h = by_ref[v][["fund", "codigo", "qty", "val"]].copy()
        i = int(np.searchsorted(refs, np.datetime64(v)))
        v3 = refs[i - 3] if i >= 3 else None
        h["credit"] = h["fund"].map(cls_by[v]).fillna(False).astype(float)
        f21, f63 = _lagged(fr21, day), _lagged(fr63, day)
        h["f21"] = h["fund"].map(f21).astype(float) if f21 is not None else np.nan
        h["f63"] = h["fund"].map(f63).astype(float) if f63 is not None else np.nan
        h["tot"] = h.groupby("codigo")["val"].transform("sum")
        h["w"] = h["val"] / h["tot"]
        h["wf21"] = h["w"] * h["f21"].fillna(0)
        h["wf63"] = h["w"] * h["f63"].fillna(0)
        h["wcov"] = h["w"] * h["f63"].notna()
        h["wst"] = h["w"] * (h["f63"] < -0.10)
        h["wc"] = h["w"] * h["credit"]
        h["w2"] = h["w"] ** 2
        g = h.groupby("codigo")
        F = pd.DataFrame({"nh": g.size(), "val": g["val"].sum(), "qty": g["qty"].sum(), "hhi": g["w2"].sum(),
                          "cshare": g["wc"].sum(), "cov": g["wcov"].sum(),
                          "press21": g["wf21"].sum(), "press63": g["wf63"].sum(), "stress63": g["wst"].sum()})
        for c in ["press21", "press63", "stress63"]:
            F[c] = (F[c] / F["cov"]).where(F["cov"] >= 0.5)
        if v3 is not None:
            h3 = by_ref[v3].groupby("codigo").agg(nh3=("fund", "size"), qty3=("qty", "sum"))
            F = F.join(h3, how="left")
            F["dnh3"] = np.log1p(F["nh"]) - np.log1p(F["nh3"].fillna(0))
            F["dq3"] = (F["qty"] / F["qty3"] - 1).clip(-1, 3)
            m3 = deb[(deb["ref"] > v3) & (deb["ref"] <= v)]
            nb = m3.groupby("codigo")[["buy", "sell"]].sum(min_count=1)
            F["nb3"] = ((nb["buy"].reindex(F.index).fillna(0) - nb["sell"].reindex(F.index).fillna(0)) / F["val"]).clip(-3, 3)
        else:
            F["dnh3"] = np.nan; F["dq3"] = np.nan; F["nb3"] = np.nan
        F["lnh"] = np.log1p(F["nh"])
        F["day"] = day
        F["vintage"] = v
        out.append(F.reset_index().rename(columns={"index": "codigo"}))
    B = pd.concat(out, ignore_index=True)
    return B[["day", "codigo", "vintage", "nh", "lnh", "dnh3", "dq3", "nb3", "hhi", "cshare", "press21", "press63",
              "stress63", "val"]]


def signals(panel):
    """Merge the PIT bond features onto a harness panel (rows day, codigo). Bonds with no fund holder: nh=lnh=0,
    other features NaN."""
    dd = sorted(pd.to_datetime(panel["day"].unique()))
    key = hashlib.md5(",".join(str(d.date()) for d in dd).encode()).hexdigest()[:10]
    cache = f"{D}/bond_features_{key}.pkl"
    if os.path.exists(cache):
        B = pd.read_pickle(cache)
    else:
        B = bond_features(dd)
        B.to_pickle(cache)
    X = panel[["day", "codigo"]].merge(B, on=["day", "codigo"], how="left")
    X["nh"] = X["nh"].fillna(0)
    X["lnh"] = X["lnh"].fillna(0)
    return X
