"""Study B — bond selection: which debentures to buy (cross-sectional), from SND trade history.

Panel = monthly snapshots (first business day of each month) of every DI+ / IPCA+ / Pré debenture that
traded in the previous 10 business days (SND trades, all assets, since 2021).

Implied CDI+ spread from a trade:  market rate ≈ contract rate + (1 − PU/PU_par_curve) / D, then converted
to CDI+ with the B3 DI x Pré / DI x IPCA curves of that day (bonds.cdi_equivalent_bps). D is estimated as
T × f(indexer), f calibrated on today's ANBIMA durations.

Label: excess return over CDI from the snapshot trade to the next snapshot trade
       = G_par × (ratio₂ / ratio₁) − CDI growth,   G_par = contract accrual (CDI·(1+x)^τ, IPCA·(1+c)^τ, (1+c)^τ)

Models (walk-forward, trained only on snapshots whose label ended before the rebalance date):
  heuristic score (the live screener), Ridge regression, gradient boosting.
Portfolio: equal-weight top quintile each month vs the equal-weight eligible universe, after bid-ask costs.
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .. import history as h
from ..bonds import br_float
from ..sources import anbima_public, snd

KINDS = ("DI_SPREAD", "IPCA", "PRE")
FEATURES = ["cdi_bps", "resid_bps", "resid_z", "own_z", "mom_1m", "mom_3m", "carry_per_dur", "ratio",
            "ratio_chg_1m", "log_trades_21d", "log_qty_21d", "days_since_trade", "dur", "T", "incent",
            "kind_ipca", "kind_pre", "issuer_n", "issuer_resid", "mkt_mom_21", "pre_1y", "pre_chg_3m"]
BUCKETS = [0, 1, 2, 3, 4, 5, 7, 10, 40]


# ---------------------------------------------------------------- reference data
def reference() -> pd.DataFrame:
    t = snd.table().drop_duplicates("Codigo do Ativo").copy()
    idx = t["indice"].astype(str).str.upper()
    pct = t["Percentual Multiplicador/Rentabilidade"].map(br_float)
    rate = t["Juros Criterio Novo - Taxa"].map(br_float)
    kind = np.select([idx.eq("DI") & (pct.fillna(100) <= 100), idx.eq("IPCA"), idx.str.startswith("PR")],
                     ["DI_SPREAD", "IPCA", "PRE"], "OTHER")
    return pd.DataFrame({
        "codigo": t["Codigo do Ativo"].astype(str), "kind": kind, "contract": rate,
        "maturity": pd.to_datetime(t["Data de Vencimento"], format="%d/%m/%Y", errors="coerce"),
        "cnpj": t["CNPJ"].astype(str), "incent": t["Deb. Incent. (Lei 12.431)"].eq("S").astype(int),
        "issuer": t["Empresa"].astype(str),
    }).query("kind in @KINDS").dropna(subset=["contract", "maturity"])


def duration_factors() -> dict[str, float]:
    """Median ANBIMA duration / time-to-maturity by indexer (amortisation & coupons shorten duration)."""
    d, deb = anbima_public.latest(anbima_public.debentures)
    f = {"DI_SPREAD": 0.8, "IPCA": 0.6, "PRE": 0.75}
    if deb is None:
        return f
    deb = deb.dropna(subset=["duration_du", "vencimento"]).copy()
    deb["T"] = (deb["vencimento"] - pd.Timestamp(d)).dt.days / 365.25
    deb["k"] = deb["indice"].map(lambda s: anbima_public.parse_indexer(s)[0])
    deb = deb[deb["T"] > 0.5]
    for k in KINDS:
        g = deb[deb["k"] == k]
        if len(g) > 10:
            f[k] = float((g["duration_du"] / 252 / g["T"]).median())
    return f


def _interp_rows(years: np.ndarray, grid: pd.DataFrame, curve: str) -> np.ndarray:
    tenors = np.array([126, 252, 504, 756, 1260, 1764, 2520]) / 252
    cols = [f"{curve}_{int(t * 252)}" for t in tenors]
    vals = grid[cols].to_numpy()
    out = np.full(len(years), np.nan)
    for i, (y, row) in enumerate(zip(years, vals)):
        ok = ~np.isnan(row)
        if ok.sum() >= 2:
            out[i] = np.interp(y, tenors[ok], row[ok])
    return out


# ---------------------------------------------------------------- panel
def build_panel(start: date = date(2021, 1, 1), codes: set[str] | None = None) -> pd.DataFrame:
    tr = h.snd_trades(start)
    if codes:
        tr = tr[tr["codigo"].isin(codes)]
    tr = tr[tr["pct_curve"].between(40, 160)].copy()
    tr = (tr.sort_values(["codigo", "date", "trades"]).groupby(["codigo", "date"], as_index=False)
          .agg(qty=("qty", "sum"), trades=("trades", "sum"), pu_avg=("pu_avg", "mean"), pct_curve=("pct_curve", "mean")))
    ref = reference()
    tr = tr.merge(ref, on="codigo", how="inner")
    tr["T"] = (tr["maturity"] - tr["date"]).dt.days / 365.25
    tr = tr[tr["T"] > 0.3]
    f = duration_factors()
    tr["dur"] = (tr["T"] * tr["kind"].map(f)).clip(lower=0.25)
    tr["ratio"] = tr["pct_curve"] / 100
    # Exact (compounded) form: PU/PU_par = ((1+c)/(1+y))^D  →  y = (1+c)·ratio^(−1/D) − 1
    tr["mkt_rate"] = ((1 + tr["contract"] / 100) * tr["ratio"] ** (-1 / tr["dur"]) - 1) * 100

    curves = h.b3_curve_panel(start - pd.Timedelta(days=10).to_pytimedelta()).reset_index()
    curves["date"] = curves["date"].astype("datetime64[ns]")
    tr["date"] = tr["date"].astype("datetime64[ns]")
    tr = tr.sort_values("date")
    tr = pd.merge_asof(tr, curves, on="date", direction="backward")
    pre = _interp_rows(tr["dur"].to_numpy(), tr, "PRE")
    dic = _interp_rows(tr["dur"].to_numpy(), tr, "DIC")
    r = tr["mkt_rate"].to_numpy() / 100
    tr["cdi_bps"] = np.select(
        [tr["kind"].eq("DI_SPREAD"), tr["kind"].eq("PRE"), tr["kind"].eq("IPCA")],
        [r * 1e4, ((1 + r) / (1 + pre / 100) - 1) * 1e4, ((1 + r) / (1 + dic / 100) - 1) * 1e4], np.nan)
    tr["pre_1y"] = tr["PRE_252"]
    tr = tr.dropna(subset=["cdi_bps"])
    tr = tr[tr["cdi_bps"].abs() < 2500]  # drop broken prints / distressed outliers
    return tr.drop(columns=[c for c in tr.columns if c[:4] in ("PRE_", "DIC_")])


def _accruals(start: date) -> tuple[pd.Series, pd.Series]:
    cdi = h.bcb_series(12, start) / 100
    C = (1 + cdi).cumprod()
    ipca_m = h.bcb_series(433, date(start.year - 1, 1, 1)) / 100
    days = pd.bdate_range(C.index.min(), pd.Timestamp.today())
    per_day = []
    for d in days:
        m = pd.Timestamp(d.year, d.month, 1)
        v = ipca_m.get(m, ipca_m.iloc[-1] if len(ipca_m) else 0.0)
        n = len(pd.bdate_range(m, m + pd.offsets.MonthEnd(0)))
        per_day.append((1 + v) ** (1 / n))
    I = pd.Series(np.cumprod(per_day), index=days)
    return C.reindex(days).ffill(), I


def snapshots(panel: pd.DataFrame, rebalances: list[pd.Timestamp] | None = None,
              with_labels: bool = True) -> pd.DataFrame:
    """One row per bond per month-start: its last trade in the prior 10 business days + features + label.
    `rebalances` overrides the month starts (e.g. [today] for live scoring)."""
    months = rebalances if rebalances is not None else \
        pd.date_range(panel["date"].min() + pd.offsets.MonthBegin(1), panel["date"].max(), freq="BMS")
    C, I = _accruals(panel["date"].min().date())
    ida = h.ida("IDADI")["index"].pct_change()
    cdi_d = C.pct_change()
    mkt_mom = (ida - cdi_d.reindex(ida.index)).rolling(21).sum()
    by_code = {k: g.set_index("date").sort_index() for k, g in panel.groupby("codigo")}
    rows = []
    for m in months:
        lo = m - pd.offsets.BDay(10)
        win = panel[(panel["date"] >= lo) & (panel["date"] < m)]
        if win.empty:
            continue
        last = win.sort_values("date").groupby("codigo").tail(1).copy()
        liq = panel[(panel["date"] >= m - pd.offsets.BDay(21)) & (panel["date"] < m)].groupby("codigo")
        last["log_trades_21d"] = np.log1p(last["codigo"].map(liq["trades"].sum()))
        last["log_qty_21d"] = np.log1p(last["codigo"].map(liq["qty"].sum()))
        last["days_since_trade"] = (m - last["date"]).dt.days
        last["rebalance"] = m
        rows.append(last)
    snap = pd.concat(rows, ignore_index=True)

    # Peer curve (indexer × tax status), residuals — computed per rebalance date.
    snap["peer"] = snap["kind"] + "_" + snap["incent"].astype(str)
    parts = []
    for (m, p), g in snap.groupby(["rebalance", "peer"]):
        g = g.copy()
        if len(g) >= 8:
            b = pd.cut(g["dur"], BUCKETS)
            med = g.groupby(b, observed=True).agg(d=("dur", "median"), s=("cdi_bps", "median")).dropna()
            fair = np.interp(g["dur"], med["d"], med["s"]) if len(med) > 1 else np.full(len(g), med["s"].iloc[0])
            g["resid_bps"] = g["cdi_bps"] - fair
            mad = (g["resid_bps"] - g["resid_bps"].median()).abs().median() * 1.4826 or 1
            g["resid_z"] = ((g["resid_bps"] - g["resid_bps"].median()) / mad).clip(-4, 4)
        else:
            g["resid_bps"] = g["resid_z"] = np.nan
        parts.append(g)
    snap = pd.concat(parts).sort_values(["codigo", "rebalance"])

    # Own history & momentum (per bond across snapshots).
    snap = snap.sort_values(["codigo", "rebalance"])
    grp = snap.groupby("codigo")["cdi_bps"]
    snap["mom_1m"] = grp.diff(1)
    snap["mom_3m"] = grp.diff(3)
    roll_mean = grp.transform(lambda s: s.shift(1).rolling(12, min_periods=4).mean())
    roll_std = grp.transform(lambda s: s.shift(1).rolling(12, min_periods=4).std())
    snap["own_z"] = ((snap["cdi_bps"] - roll_mean) / roll_std.replace(0, np.nan)).clip(-4, 4)
    snap["ratio_chg_1m"] = snap.groupby("codigo")["ratio"].diff(1)
    snap["carry_per_dur"] = snap["cdi_bps"] / snap["dur"]
    snap["kind_ipca"] = snap["kind"].eq("IPCA").astype(int)
    snap["kind_pre"] = snap["kind"].eq("PRE").astype(int)
    iss = snap.groupby(["rebalance", "cnpj"])
    snap["issuer_n"] = iss["codigo"].transform("count")
    snap["issuer_resid"] = iss["resid_bps"].transform("mean")
    snap["mkt_mom_21"] = mkt_mom.reindex(snap["rebalance"], method="ffill").to_numpy()
    snap["pre_chg_3m"] = snap.groupby("codigo")["pre_1y"].diff(3)

    # Labels. Realistic execution: ENTER at the first trade after the rebalance date and EXIT at the first
    # trade after the next rebalance. Features only use trades before the rebalance, so bid-ask bounce in
    # the last pre-rebalance print can't leak into the label. `y_naive` (from the feature trade) is kept
    # only to measure how much of a naive backtest is bounce.
    def excess(r, t1, ratio1, t2, ratio2):
        tau = np.busday_count(t1.date(), t2.date()) / 252
        cg = C.asof(t2) / C.asof(t1)
        if r["kind"] == "DI_SPREAD":
            gpar = cg * (1 + r["contract"] / 100) ** tau
        elif r["kind"] == "IPCA":
            gpar = I.asof(t2) / I.asof(t1) * (1 + r["contract"] / 100) ** tau
        else:
            gpar = (1 + r["contract"] / 100) ** tau
        return gpar * (ratio2 / ratio1) - cg

    if not with_labels:
        return snap.reset_index(drop=True)
    lab, naive, t2s = [], [], []
    for _, r in snap.iterrows():
        g = by_code[r["codigo"]]
        m, nxt_m = r["rebalance"], r["rebalance"] + pd.offsets.BMonthBegin(1)
        entry = g[(g.index >= m) & (g.index < m + pd.offsets.BDay(10))]
        exit_ = g[(g.index >= nxt_m) & (g.index < nxt_m + pd.offsets.BDay(10))]
        if exit_.empty:
            lab.append(np.nan)
            naive.append(np.nan)
            t2s.append(pd.NaT)
            continue
        t2, x = exit_.index[0], exit_.iloc[0]
        naive.append(excess(r, r["date"], r["ratio"], t2, x["ratio"]))
        lab.append(excess(r, entry.index[0], entry.iloc[0]["ratio"], t2, x["ratio"]) if not entry.empty else np.nan)
        t2s.append(t2)
    snap["y"] = lab
    snap["y_naive"] = naive
    snap["t2"] = t2s
    return snap.reset_index(drop=True)


# ---------------------------------------------------------------- live scoring
def live_scores(snap_hist: pd.DataFrame, panel: pd.DataFrame, asof: pd.Timestamp | None = None) -> pd.DataFrame:
    """Train Ridge on all labelled history and score today's bonds. Momentum/own-history features need the
    previous monthly snapshots, so today's row is scored together with them."""
    asof = asof or pd.Timestamp.today().normalize()
    train = eligible(snap_hist[snap_hist["y"].notna()])
    y = train["y"].clip(train["y"].quantile(0.01), train["y"].quantile(0.99))
    med = train[FEATURES].median()
    model = make_pipeline(StandardScaler(), Ridge(alpha=10.0)).fit(train[FEATURES].fillna(med), y)
    months = sorted(snap_hist["rebalance"].unique())[-13:]
    both = snapshots(panel, rebalances=[*months, asof], with_labels=False)
    cur = both[both["rebalance"] == asof].copy()
    cur["ml_pred"] = model.predict(cur[FEATURES].fillna(med))
    # Rank only the universe the backtest traded (0.5y+ duration, price 90–110% of the par curve): outside
    # it (distressed / broken prints) the model was never validated.
    el = eligible(cur).index
    cur["eligible"] = cur.index.isin(el)
    cur.loc[el, "heur"] = heuristic_score(cur.loc[el])
    cur.loc[el, "blend"] = cur.loc[el, "heur"].rank(pct=True) + cur.loc[el, "ml_pred"].rank(pct=True)
    cur["blend_pct"] = cur["blend"].rank(pct=True, ascending=False)  # 0 = best; NaN = not eligible
    coefs = dict(zip(FEATURES, model[-1].coef_ * 1e4))  # bps of 1m excess per 1σ of feature
    cur.attrs["coefs"] = coefs
    return cur


# ---------------------------------------------------------------- walk-forward
def heuristic_score(g: pd.DataFrame) -> pd.Series:
    """The live screener's rule: value 45%, carry 20%, momentum 15%, liquidity 20% (robust z within date)."""
    def rz(s):
        s = s.astype(float)
        mad = (s - s.median()).abs().median() * 1.4826
        return ((s - s.median()) / (mad if mad else (s.std() or 1))).clip(-3, 3)
    return (0.45 * g["resid_z"].fillna(0) + 0.20 * rz(g["carry_per_dur"]) - 0.15 * rz(g["mom_1m"].fillna(0))
            + 0.20 * rz(g["log_trades_21d"].fillna(0)))


def eligible(g: pd.DataFrame) -> pd.DataFrame:
    return g[(g["dur"] >= 0.5) & g["ratio"].between(0.9, 1.1) & g["resid_z"].notna()]


def walk_forward(snap: pd.DataFrame, test_start: str = "2023-01-01", q: float = 0.2,
                 cost_bps: float = 25.0, label: str = "y", exit_q: float | None = None,
                 blend: bool = False) -> dict:
    """exit_q: turnover buffer — buy the top `q`, keep a holding until it falls out of the top `exit_q`.
    blend: add a 50/50 rank blend of the heuristic and Ridge."""
    snap = snap.assign(y=snap[label])
    months = sorted(snap["rebalance"].unique())
    months = [m for m in months if m >= pd.Timestamp(test_start)]
    models = {
        "Ridge": lambda: make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
        "Gradient boosting": lambda: HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=300,
                                                                   min_samples_leaf=200, l2_regularization=1.0),
    }
    rec, ics, holdings_prev = [], {k: [] for k in ["Heurística (screener)", *models]}, {}
    for m in months:
        cur = eligible(snap[snap["rebalance"] == m]).copy()
        if len(cur) < 30:
            continue
        train = snap[(snap["t2"] < m) & snap["y"].notna()]
        train = eligible(train)
        y = train["y"].clip(train["y"].quantile(0.01), train["y"].quantile(0.99))
        preds = {"Heurística (screener)": heuristic_score(cur)}
        X = train[FEATURES].fillna(train[FEATURES].median())
        Xc = cur[FEATURES].fillna(train[FEATURES].median())
        for k, mk in models.items():
            if len(train) < 500:
                preds[k] = pd.Series(np.nan, index=cur.index)
                continue
            preds[k] = pd.Series(mk().fit(X, y).predict(Xc), index=cur.index)
        if blend and "Ridge" in preds and not preds["Ridge"].isna().all():
            preds["Blend (heurística+Ridge)"] = (preds["Heurística (screener)"].rank(pct=True)
                                                 + preds["Ridge"].rank(pct=True))
            ics.setdefault("Blend (heurística+Ridge)", [])
        realized = cur["y"].fillna(
            # no trade in the next window: assume price unchanged relative to par → carry only
            cur["y"].median() if cur["y"].notna().any() else 0.0)
        row = {"month": m, "Universo (EW)": realized.mean(), "n": len(cur)}
        for k, p in preds.items():
            if p.isna().all():
                row[k] = np.nan
                continue
            pct = p.rank(ascending=False, pct=True)
            prev = holdings_prev.get(k, set())
            top = pct <= q
            if exit_q:
                top = top | ((pct <= exit_q) & cur["codigo"].isin(prev))
            names = set(cur.loc[top, "codigo"])
            turnover = 1.0 if not prev else len(names - prev) / max(len(names), 1)
            row[k] = realized[top].mean() - turnover * cost_bps / 1e4
            row[f"{k} turnover"] = turnover
            holdings_prev[k] = names
            ok = cur["y"].notna()
            if ok.sum() > 20:
                ics[k].append(pd.Series(p[ok]).rank().corr(cur.loc[ok, "y"].rank()))
        rec.append(row)
    res = pd.DataFrame(rec).set_index("month")
    strat_cols = [c for c in ["Universo (EW)", "Heurística (screener)", *models, "Blend (heurística+Ridge)"]
                  if c in res]
    stats = {}
    for c in strat_cols:
        r = res[c].dropna()
        eq = (1 + r).cumprod()
        yrs = len(r) / 12
        ann = eq.iloc[-1] ** (1 / yrs) - 1 if yrs else np.nan
        vol = r.std() * np.sqrt(12)
        ic = ics.get(c, [])
        stats[c] = {"excess_ann_%": ann * 100, "vol_%": vol * 100, "sharpe": ann / vol if vol else np.nan,
                    "max_dd_%": ((eq / eq.cummax()) - 1).min() * 100,
                    "vs_universe_ann_%": (r - res.loc[r.index, "Universo (EW)"]).mean() * 1200,
                    "IC_mean": float(np.mean(ic)) if ic else np.nan,
                    "IC_tstat": float(np.mean(ic) / (np.std(ic) / np.sqrt(len(ic)))) if len(ic) > 2 and np.std(ic) else np.nan,
                    "turnover_%": res.get(f"{c} turnover", pd.Series(dtype=float)).mean() * 100}
    return {"monthly": res, "stats": pd.DataFrame(stats).T}
