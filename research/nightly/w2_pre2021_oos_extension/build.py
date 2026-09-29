"""Pre-2021 out-of-sample extension: point-in-time debenture grid 2014-01 .. 2021-06 rebuilt from raw SND prints,
in the SAME format as the nightly harness core + monthly panel, so the frozen P4 / P4+Q / P7 rules can be run on it.

Inputs (all cached in data/history/nightly/pre2021_oos_extension/ by fetch_snd.py / fetch_aux.py / fetch_eq_fin.py):
  snd_YYYYMM.csv.gz       SND secondary-market daily prints 2014-01..2020-12 (PU medio, % PU da curva, qty, #trades)
  data/history/snd_trades_2021{01..06}.csv.gz (read-only, same endpoint) to extend the grid into 2021-H1
  cvm_ipe_YYYY.zip        CVM IPE filings (distress / days_since_distress; dated by Data_Entrega)
  itr/dfp_cia_aberta_YYYY CVM statements 2011-2020 -> fundamentals, available_date_strict (latest receipt date)
  cotahist_YYYY.csv.gz    B3 COTAHIST closes 2013-2020 for equity_map tickers (+ predecessors)
  bcb_12.csv / bcb_433    CDI and IPCA from BCB SGS
  data/cache/tesouro_direto.csv (read-only): NTN-B real curve (IPCA kind, same source as the 2021+ lab) and
                                             LTN/NTN-F nominal curve (PRE kind; the 2021+ lab uses B3 DI x pre)
  SND registry (rfmonitor.sources.snd.table, a 2026 SNAPSHOT: contract rate, maturity, indexer, incentive flag)

Construction follows rfmonitor/ml/selection.build_panel + rfmonitor/ml/lab_daily.build + run_selection_lab.build_returns
+ harness v4 core (gap moves booked at realisation, rec40), with the bias_audit holiday fix: the contract and IPCA par
accrual run on B3 business days (days with a CDI print), not weekdays.
Outputs: data/history/nightly/pre2021_oos_extension/{core.pkl, panel_M.pkl, grid.pkl, fund_pre.pkl}
"""
from __future__ import annotations

import glob
import io
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from rfmonitor.ml.selection import BUCKETS, _interp_rows, duration_factors, reference  # noqa: E402

OUT = ROOT / "data" / "history" / "nightly" / "pre2021_oos_extension"
HIST = ROOT / "data" / "history"
GRID_END = pd.Timestamp("2021-06-30")
DEC_START, DEC_END = pd.Timestamp("2014-07-01"), pd.Timestamp("2021-03-31")
STALE = 14
REC = 0.40
T0 = time.time()
TENORS = [126, 252, 504, 756, 1260, 1764, 2520]


def log(*a):
    print(f"[build +{time.time() - T0:6.1f}s]", *a, flush=True)


# ------------------------------------------------------------------------------------------------ calendars / CDI
def bcb(code):
    s = pd.read_csv(OUT / f"bcb_{code}.csv", parse_dates=["date"]).set_index("date")["valor"] / 100
    return s[~s.index.duplicated()].sort_index()


def calendars(days):
    cdi = bcb(12)
    cdi = cdi[cdi.index >= "2013-01-01"]
    wk = pd.bdate_range(cdi.index.min(), GRID_END)
    hol = wk.difference(cdi.index)                     # weekday B3 holidays (no CDI print)
    C = (1 + cdi).cumprod()
    Cd = C.reindex(C.index.union(days)).ffill().reindex(days)
    ipca = bcb(433)
    b3d = pd.DatetimeIndex(cdi.index)
    fac = []
    for d in wk:
        if d in hol:
            fac.append(1.0)
            continue
        m = pd.Timestamp(d.year, d.month, 1)
        v = ipca.get(m, ipca.iloc[-1])
        n = ((b3d >= m) & (b3d <= m + pd.offsets.MonthEnd(0))).sum()
        fac.append((1 + v) ** (1 / max(n, 1)))
    I = pd.Series(np.cumprod(fac), index=wk)
    Id = I.reindex(I.index.union(days)).ffill().reindex(days)
    return cdi, hol, Cd, Id


# ------------------------------------------------------------------------------------------------ curves (Tesouro Direto)
def td_curves() -> pd.DataFrame:
    df = pd.read_csv(HIST.parent / "cache" / "tesouro_direto.csv", sep=";", decimal=",", encoding="latin1")
    df = df.rename(columns={"Tipo Titulo": "nome", "Data Vencimento": "venc", "Data Base": "date",
                            "Taxa Compra Manha": "taxa"})
    df["venc"] = pd.to_datetime(df["venc"], format="%d/%m/%Y")
    df["date"] = pd.to_datetime(df["date"], format="%d/%m/%Y")
    df = df[(df["date"] >= "2013-06-01") & (df["date"] <= GRID_END)]
    real = df["nome"].isin(["Tesouro IPCA+ com Juros Semestrais", "Tesouro IPCA+"])
    nom = df["nome"].isin(["Tesouro Prefixado", "Tesouro Prefixado com Juros Semestrais"])
    rows = {}
    for (d, g) in df[real | nom].groupby("date"):
        row = {}
        for pre, m in (("DIC", real.loc[g.index]), ("PRE", nom.loc[g.index])):
            gg = g[m.to_numpy()].drop_duplicates("venc")
            x = ((gg["venc"] - d).dt.days / 365.25 * 252).to_numpy()
            y = gg["taxa"].to_numpy(float)
            ok = (x > 20) & np.isfinite(y) & (y > 0)
            o = np.argsort(x[ok])
            if ok.sum() >= 2:
                for k in TENORS:
                    row[f"{pre}_{k}"] = float(np.interp(k, x[ok][o], y[ok][o]))
        rows[d] = row
    return pd.DataFrame.from_dict(rows, orient="index").sort_index()


# ------------------------------------------------------------------------------------------------ trades -> spreads
def trades() -> pd.DataFrame:
    fs = sorted(glob.glob(str(OUT / "snd_2*.csv.gz")))
    a = pd.concat([pd.read_csv(f, parse_dates=["date"]) for f in fs], ignore_index=True)
    b = pd.concat([pd.read_csv(HIST / f"snd_trades_2021{m:02d}.csv.gz", parse_dates=["date"]) for m in range(1, 7)],
                  ignore_index=True)
    cols = ["date", "codigo", "qty", "trades", "pu_avg", "pct_curve"]
    t = pd.concat([a[cols], b[cols]], ignore_index=True)
    t = t[(t["date"] >= "2014-01-01") & (t["date"] <= GRID_END)]
    t["codigo"] = t["codigo"].astype(str)
    return t


def spreads(raw: pd.DataFrame, curves: pd.DataFrame) -> pd.DataFrame:
    tr = raw[raw["pct_curve"].between(40, 160)].copy()
    tr = (tr.sort_values(["codigo", "date", "trades"]).groupby(["codigo", "date"], as_index=False)
          .agg(qty=("qty", "sum"), trades=("trades", "sum"), pu_avg=("pu_avg", "mean"), pct_curve=("pct_curve", "mean")))
    ref = reference()
    tr = tr.merge(ref, on="codigo", how="inner")
    tr["T"] = (tr["maturity"] - tr["date"]).dt.days / 365.25
    tr = tr[tr["T"] > 0.3]
    f = duration_factors()
    log("duration factors", f)
    tr["dur"] = (tr["T"] * tr["kind"].map(f)).clip(lower=0.25)
    tr["ratio"] = tr["pct_curve"] / 100
    tr["mkt_rate"] = ((1 + tr["contract"] / 100) * tr["ratio"] ** (-1 / tr["dur"]) - 1) * 100
    cv = curves.reset_index().rename(columns={"index": "date"})
    cv["date"] = cv["date"].astype("datetime64[ns]")
    tr["date"] = tr["date"].astype("datetime64[ns]")
    tr = pd.merge_asof(tr.sort_values("date"), cv, on="date", direction="backward")
    pre = _interp_rows(tr["dur"].to_numpy(), tr, "PRE")
    dic = _interp_rows(tr["dur"].to_numpy(), tr, "DIC")
    r = tr["mkt_rate"].to_numpy() / 100
    tr["cdi_bps"] = np.select(
        [tr["kind"].eq("DI_SPREAD"), tr["kind"].eq("PRE"), tr["kind"].eq("IPCA")],
        [r * 1e4, ((1 + r) / (1 + pre / 100) - 1) * 1e4, ((1 + r) / (1 + dic / 100) - 1) * 1e4], np.nan)
    tr["bench_rate"] = np.select([tr["kind"].eq("IPCA"), tr["kind"].eq("PRE")], [dic, pre], np.nan)
    tr = tr.dropna(subset=["cdi_bps"])
    tr = tr[tr["cdi_bps"].abs() < 2500]
    return tr.drop(columns=[c for c in tr.columns if c[:4] in ("PRE_", "DIC_")])


# ------------------------------------------------------------------------------------------------ grid
def grid(p: pd.DataFrame, days: pd.DatetimeIndex) -> pd.DataFrame:
    span = p.groupby("codigo").agg(first=("date", "min"), last=("date", "max"), maturity=("maturity", "first"))
    parts = []
    fa = span["first"].to_numpy().astype("datetime64[D]")
    la = (span["last"] + pd.Timedelta(days=STALE)).to_numpy().astype("datetime64[D]")
    ma = span["maturity"].clip(upper=pd.Timestamp("2200-01-01")).to_numpy().astype("datetime64[D]")
    dv = days.to_numpy().astype("datetime64[D]")
    for i, code in enumerate(span.index):
        hi = la[i] if np.isnat(ma[i]) else min(la[i], ma[i])
        ds = days[(dv > fa[i]) & (dv <= hi)]
        if len(ds):
            parts.append(pd.DataFrame({"codigo": code, "day": ds}))
    g = pd.concat(parts, ignore_index=True).sort_values("day")
    cols = ["codigo", "date", "ratio", "cdi_bps", "dur", "kind", "contract", "cnpj", "incent", "bench_rate", "maturity"]
    g = pd.merge_asof(g, p[cols].sort_values("date"), left_on="day", right_on="date", by="codigo", direction="backward")
    g = g.dropna(subset=["date"])
    g["age"] = (g["day"] - g["date"]).dt.days
    g = g[g["age"] <= STALE].sort_values(["codigo", "day"]).reset_index(drop=True)
    g["fresh"] = g["age"] <= 7
    g["peer"] = g["kind"] + "_" + g["incent"].astype(str)
    g["resid_bps"] = np.nan
    fr = g[g["fresh"]]
    res = np.full(len(g), np.nan)
    for (d, pe), grp in fr.groupby(["day", "peer"]):
        if len(grp) < 8:
            continue
        b = pd.cut(grp["dur"], BUCKETS)
        med = grp.groupby(b, observed=True).agg(x=("dur", "median"), y=("cdi_bps", "median")).dropna()
        fair = np.interp(grp["dur"], med["x"], med["y"]) if len(med) > 1 else med["y"].iloc[0]
        res[grp.index.to_numpy()] = grp["cdi_bps"].to_numpy() - fair
    g["resid_bps"] = res
    dd = g.groupby("day")["resid_bps"]
    mad = dd.transform(lambda s: (s - s.median()).abs().median() * 1.4826)
    g["resid_z"] = ((g["resid_bps"] - dd.transform("median")) / mad.replace(0, np.nan)).clip(-5, 5)
    g["eligible"] = g["fresh"] & g["ratio"].between(0.9, 1.1) & (g["dur"] >= 0.5)
    el = g["eligible"] & g["cdi_bps"].notna()
    g["cdi_pct"] = np.nan
    g.loc[el, "cdi_pct"] = g[el].groupby("day")["cdi_bps"].rank(pct=True, ascending=False)
    g["cnpj8"] = g["cnpj"].astype(str).str.replace(r"\D", "", regex=True).str[:8]
    return g


def patched_returns(g, days, hol, Cd, Id):
    """run_selection_lab.build_returns with B3-business-day accrual (bias_audit holiday fix)."""
    g = g.sort_values(["codigo", "day"]).reset_index(drop=True)
    tr = g.drop_duplicates(["codigo", "date"])[["codigo", "date", "ratio"]].sort_values(["codigo", "date"]).copy()
    pv = tr.groupby("codigo")["ratio"].shift(1)
    nx = tr.groupby("codigo")["ratio"].shift(-1)
    spike = ((tr["ratio"] / pv - 1).abs() > 0.10) & ((nx / pv - 1).abs() < 0.03)
    tr["ratio_c"] = np.where(spike, pv, tr["ratio"])
    g = g.merge(tr[["codigo", "date", "ratio_c"]], on=["codigo", "date"], how="left")
    nday = g.groupby("codigo")["day"].shift(-1)
    has = nday.notna()
    d0 = g["day"].values.astype("datetime64[D]")
    d1 = nday.fillna(g["day"]).values.astype("datetime64[D]")
    nbd = np.busday_count(d0, d1, holidays=hol.values.astype("datetime64[D]"))
    cg = Cd.reindex(nday.fillna(g["day"])).to_numpy() / Cd.reindex(g["day"]).to_numpy()
    ig = Id.reindex(nday.fillna(g["day"])).to_numpy() / Id.reindex(g["day"]).to_numpy()
    c, tau = g["contract"].to_numpy() / 100, nbd / 252
    gpar = np.select([g["kind"].eq("DI_SPREAD"), g["kind"].eq("IPCA")], [cg * (1 + c) ** tau, ig * (1 + c) ** tau],
                     (1 + c) ** tau)
    rn = g.groupby("codigo")["ratio_c"].shift(-1)
    r = gpar * rn / g["ratio_c"] - cg
    db = (g.groupby("codigo")["bench_rate"].shift(-1) - g["bench_rate"]) / 100
    r = r + np.where(g["kind"].isin(["IPCA", "PRE"]), (g["dur"] * db).fillna(0), 0)
    r = np.where(has, r, 0.0)
    g["r_patch"] = np.clip(np.nan_to_num(r, nan=0.0), -0.95, 1.0)
    g["pos"] = days.get_indexer(g["day"])
    return g, int(spike.sum())


def core(g, days, cdi):
    codes = pd.Index(sorted(g["codigo"].unique()))
    ND, NB = len(days), len(codes)
    g["b"] = codes.get_indexer(g["codigo"])
    G = g.sort_values(["b", "pos"])
    nxt = G.groupby("b")["pos"].shift(-1)
    rpos = np.where(nxt.notna() & (nxt - G["pos"] > 1), nxt.fillna(0).astype(int) - 1, G["pos"].to_numpy())
    R = np.zeros((ND, NB))
    R[rpos, G["b"].to_numpy()] = G["r_patch"].to_numpy()
    TD = np.full((ND, NB), -1, dtype=np.int32)
    TD[g["pos"].to_numpy(), g["b"].to_numpy()] = g["date"].to_numpy().astype("datetime64[D]").astype(np.int64)
    last = G.groupby("codigo").agg(last_day=("day", "max"), maturity=("maturity", "first"))
    lr = G.sort_values("day").groupby("codigo")["ratio"].last()
    last["last_ratio"] = lr
    stopped = last[(last["last_day"] < days[-1] - pd.Timedelta(days=30))
                   & (last["maturity"].isna() | (last["maturity"] > last["last_day"] + pd.Timedelta(days=60)))]
    dist = stopped[stopped["last_ratio"] < 0.90]
    lastpos = G.groupby("b")["pos"].max()
    R40 = R.copy()
    for c_, lrat in zip(dist.index, dist["last_ratio"].to_numpy(float)):
        bb = codes.get_loc(c_)
        p = int(lastpos[bb])
        R40[p, bb] = (1 + R40[p, bb]) * (REC / lrat) - 1
    cidx = (1 + cdi).cumprod()
    cg = cidx.reindex(cidx.index.union(days)).ffill().reindex(days)
    CDI = (cg / cg.shift(1) - 1).fillna(0).to_numpy()
    return {"version": 4, "days": days, "codes": codes, "R": R.astype(np.float32), "R40": R40.astype(np.float32),
            "TD": TD, "CDI": CDI, "cdi_daily_raw": cdi, "last": last[["last_day", "last_ratio", "maturity"]],
            "dist_codes": list(dist.index),
            "survivorship": {"stopped_before_maturity": int(len(stopped)), "stopped_below_0.90": int(len(dist))}}


# ------------------------------------------------------------------------------------------------ fundamentals
def fundamentals_pre(universe: set[str]) -> pd.DataFrame:
    import research.data_fundamentals as DF
    DF.CVM_DIR = OUT
    DF.YEARS = range(2011, 2021)
    idx, L = DF.load_cvm(universe, refresh=False)
    cv = DF.cvm_filings(idx, L)
    cv["source_grp"] = "cvm"
    pan = DF.finish(cv)
    pan = DF.pit_order(pan)
    for c in ("available_date", "available_date_strict", "period_end"):
        pan[c] = pd.to_datetime(pan[c]).astype("datetime64[ns]")
    return pan


def fund_features(fp: pd.DataFrame) -> pd.DataFrame:
    """= harness._fundamentals(strict=True) on a given raw table."""
    f = fp.copy()
    f["avail"] = f["available_date_strict"]
    f = f.sort_values(["cnpj8", "avail", "period_end"])
    f["_mx"] = f.groupby("cnpj8")["period_end"].cummax().groupby(f["cnpj8"]).shift(1)
    f = f[f["_mx"].isna() | (f["period_end"] > f["_mx"])].drop_duplicates(["cnpj8", "avail"], keep="last")
    e = f["ebitda_ltm"]
    lev = np.where(e > 0, f["net_debt"] / e, 15.0)
    lev = np.where(f["net_debt"].isna() | e.isna(), np.nan, lev)
    fe = f["fin_exp_ltm"]
    cov = np.where(fe > 0, e / fe, np.where(fe.notna() & e.notna(), 30.0, np.nan))
    return pd.DataFrame({
        "cnpj8": f["cnpj8"].astype(str).to_numpy(), "avail": f["avail"].to_numpy(),
        "period_end": f["period_end"].to_numpy(),
        "f_lev": np.clip(lev, -3, 15), "f_cov": np.clip(cov, -5, 30),
        "f_cash_st": np.log1p(f["cash_to_st_debt"].clip(0, 50)).to_numpy(),
        "f_eq_ratio": f["equity_ratio"].clip(-1, 1).to_numpy(),
        "f_d_lev": f["d_net_debt_ebitda_4q"].clip(-10, 10).to_numpy(),
        "f_source": f["source"].to_numpy(),
    }).sort_values("avail")


# ------------------------------------------------------------------------------------------------ equity
PRED = {"AXIA3": ["ELET3"], "AZZA3": ["ARZZ3"], "BHIA3": ["VIIA3", "VVAR3"], "BRAV3": ["RRRP3"], "BRST3": ["BRIT3"],
        "DXCO3": ["DTEX3"], "IGTI11": ["IGTA3"], "ISAE4": ["TRPL4"], "MBRF3": ["MRFG3"], "MOTV3": ["CCRO3"],
        "NATU3": ["NTCO3"], "RIAA3": ["GUAR3"], "SBFG3": ["CNTO3"], "TIMS3": ["TIMP3"], "VBBR3": ["BRDT3"],
        "WIZC3": ["WIZS3"], "AMER3": ["BTOW3"], "ALOS3": ["ALSO3"], "SRNA3": ["OMGE3"], "ZAMP3": ["BKBR3"]}


def equity_grid(dd):
    """Log-price index per ticker on the grid: COTAHIST (unadjusted, |daily log move| > 0.5 zeroed as in the harness
    for cotahist sources; predecessor tickers spliced before the ticker's own first print) before 2020-06-01,
    equity_daily.pkl (brapi adjusted) from then on."""
    ch = pd.concat([pd.read_csv(f, parse_dates=["date"]) for f in sorted(glob.glob(str(OUT / "cotahist_*.csv.gz")))])
    ch = ch.sort_values("volume").drop_duplicates(["ticker", "date"], keep="last")
    ed = pd.read_pickle(HIST / "equity_daily.pkl")
    ed["date"] = pd.to_datetime(ed["date"])
    mp = pd.read_csv(ROOT / "research" / "data" / "equity_map.csv", dtype=str)
    mp = mp[mp["ticker"].notna() & (mp["mapping_type"] != "none")].copy()
    tick = sorted(mp["ticker"].unique())
    TI = {t: i for i, t in enumerate(tick)}
    SPX = np.full((len(dd), len(tick)), np.nan)
    for t in tick:
        own = ch[ch["ticker"] == t].set_index("date").sort_index()
        pieces = [own]
        f0 = own.index.min() if len(own) else pd.Timestamp("2099-01-01")
        for p in PRED.get(t, []):
            q = ch[(ch["ticker"] == p) & (ch["date"] < f0)].set_index("date").sort_index()
            if len(q):
                pieces.insert(0, q)
                f0 = q.index.min()
        c = pd.concat(pieces)
        c = c[~c.index.duplicated(keep="last")].sort_index()
        c = c[c.index < "2020-06-01"]
        lr = np.log(c["close"].where(c["close"] > 0)).diff()
        lr[lr.abs() > 0.5] = 0.0
        e = ed[(ed["ticker"] == t)].drop_duplicates("date").set_index("date").sort_index()
        e = e[e.index >= "2020-06-01"]
        le = np.log(e["adj_close"].astype(float).where(e["adj_close"] > 0)).diff()
        if e["source"].astype(str).str.startswith("cotahist").all():
            le[le.abs() > 0.5] = 0.0
        if len(c) and len(e):
            le.iloc[0] = np.log(float(e["adj_close"].iloc[0]) / float(c["close"].iloc[-1])) \
                if abs(np.log(float(e["adj_close"].iloc[0]) / float(c["close"].iloc[-1]))) < 0.5 else 0.0
        s = pd.concat([lr, le]).fillna(0)
        if not len(s):
            continue
        s = s[~s.index.duplicated()].sort_index()
        px = np.exp(s.cumsum()) * 100
        SPX[:, TI[t]] = px.reindex(px.index.union(dd)).ffill(limit=5).reindex(dd).to_numpy()
    mp = mp[mp["ticker"].isin(TI)].copy()
    mp["_c"] = mp["confidence"].map({"high": 0, "med": 1, "low": 2}).fillna(1)
    cands = {k: list(zip(v["ticker"], v["mapping_type"], v["confidence"]))
             for k, v in mp.sort_values(["cnpj8", "_c"]).groupby("cnpj8")}
    return tick, TI, SPX, cands


def equity_features(pairs, tick, TI, SPX, cands):
    """harness._build_panel equity block: ticker chosen PIT (needs a price at d and >= 150 prices in the prior 252)."""
    lr = np.diff(np.log(SPX), axis=0, prepend=np.nan)
    rows = []
    for c8, p in zip(pairs["cnpj8"], pairs["dpos"]):
        j = -1
        for t, mtt, cff in cands.get(c8, []):
            jj = TI[t]
            if not np.isnan(SPX[p, jj]) and np.isfinite(SPX[max(0, p - 252):p, jj]).sum() >= 150:
                j, mt = jj, mtt
                break
        if j < 0:
            rows.append((c8, p, None) + (np.nan,) * 6)
            continue
        px = SPX[:, j]

        def ret(n):
            return px[p] / px[p - n] - 1 if p - n >= 0 and px[p - n] > 0 else np.nan
        win = lr[max(1, p - 62):p + 1, j]
        vol = np.nanstd(win) * np.sqrt(252) if np.isfinite(win).sum() >= 40 else np.nan
        hi = np.nanmax(px[max(0, p - 251):p + 1])
        rows.append((c8, p, tick[j], ret(21), ret(63), ret(126), ret(252), vol, px[p] / hi - 1))
    return pd.DataFrame(rows, columns=["cnpj8", "dpos", "eq_ticker", "eq_r21", "eq_r63", "eq_r126", "eq_r252",
                                       "eq_vol63", "eq_dd252"])


# ------------------------------------------------------------------------------------------------ CVM IPE
def ipe_events() -> pd.DataFrame:
    from rfmonitor.ml.news import DISTRESS
    fr = []
    for y in range(2013, 2021):
        with zipfile.ZipFile(OUT / f"cvm_ipe_{y}.zip") as z:
            m = next(n for n in z.namelist() if n.startswith(f"ipe_cia_aberta_{y}"))
            fr.append(pd.read_csv(z.open(m), sep=";", encoding="latin1", dtype=str))
    # 2021 (read-only shared cache) for the 2021-H1 part of the grid
    p21 = HIST.parent / "cache" / "cvm_ipe_2021.zip"
    if p21.exists():
        with zipfile.ZipFile(p21) as z:
            m = next(n for n in z.namelist() if n.startswith("ipe_cia_aberta_2021"))
            fr.append(pd.read_csv(z.open(m), sep=";", encoding="latin1", dtype=str))
    df = pd.concat(fr, ignore_index=True)
    cat = df["Categoria"].fillna("").str.lower()
    subj = (df["Assunto"].fillna("") + " " + df["Tipo"].fillna("")).str.lower()
    out = pd.DataFrame({"cnpj8": df["CNPJ_Companhia"].str.replace(r"\D", "", regex=True).str[:8],
                        "date": pd.to_datetime(df["Data_Entrega"], errors="coerce"),
                        "distress": subj.str.contains(DISTRESS, regex=True) | cat.str.contains("recupera")})
    return out[out["distress"]].dropna(subset=["date"])


def days_since(ev: pd.DataFrame, keys: pd.DataFrame) -> np.ndarray:
    """days since the last distress filing delivered <= day (asof = day + 1 as in lab_daily; 9999 if none)."""
    out = np.full(len(keys), 9999.0)
    E = {k: np.sort(g["date"].to_numpy(dtype="datetime64[ns]")) for k, g in ev.groupby("cnpj8")}
    for c8, idx in keys.groupby("cnpj8").groups.items():
        d = E.get(c8)
        if d is None:
            continue
        asof = (keys.loc[idx, "day"] + pd.Timedelta(days=1)).to_numpy(dtype="datetime64[ns]")
        hi = np.searchsorted(d, asof - np.timedelta64(1, "D"), side="right")
        last = np.where(hi > 0, d[np.maximum(hi - 1, 0)], np.datetime64("NaT"))
        dd = (asof - last) / np.timedelta64(1, "D")
        out[keys.index.get_indexer(idx)] = np.where(np.isnan(dd), 9999, dd)
    return out


# ------------------------------------------------------------------------------------------------ main
def main():
    raw = trades()
    log("raw prints", len(raw), raw["codigo"].nunique())
    curves = td_curves()
    log("curves", curves.shape, curves.index.min(), curves.index.max())
    p = spreads(raw, curves)
    log("priced prints", len(p), p["codigo"].nunique())
    days = pd.bdate_range(p["date"].min() + pd.Timedelta(days=21), GRID_END)
    cdi, hol, Cd, Id = calendars(days)
    g = grid(p, days)
    log("grid rows", len(g), "bonds", g["codigo"].nunique())
    g, nsp = patched_returns(g, days, hol, Cd, Id)
    C = core(g, days, cdi)
    C["n_spikes_removed"] = nsp
    pd.to_pickle(C, OUT / "core.pkl")
    log("core", C["R"].shape, C["survivorship"])

    # --- matrices for factor_zoo features (grid-position lags)
    ND, NB = len(days), len(C["codes"])
    CS = np.full((ND, NB), np.nan)
    RB = np.full((ND, NB), np.nan)
    CS[g["pos"], g["b"]] = g["cdi_bps"]
    RB[g["pos"], g["b"]] = g["resid_bps"]

    # --- monthly decision panel
    s = pd.Series(np.arange(ND), index=days)
    s = s[(days >= DEC_START) & (days <= DEC_END)]
    decs = sorted(int(v) for v in s.groupby([s.index.year, s.index.month]).min().to_numpy())
    P = g[g["pos"].isin(decs)].copy()
    P["dpos"] = P["pos"]
    P["univ"] = P["eligible"] & P["cdi_bps"].notna()
    P["ds_5"] = CS[P["dpos"], P["b"]] - CS[P["dpos"] - 5, P["b"]]
    P["dres_21"] = RB[P["dpos"], P["b"]] - RB[P["dpos"] - 21, P["b"]]
    # execution / forward labels (diagnostics)
    import research.nightly.harness as H
    H._MEM.clear()
    H._MEM["core"] = C
    ent = np.full(len(P), -1)
    P = P.reset_index(drop=True)
    for q, idx in P.groupby("dpos").groups.items():
        ep = H.exec_pos(int(q))
        ent[P.index.get_indexer(idx)] = ep[P.loc[idx, "b"].to_numpy()]
    P["entry_pos"] = ent
    P["executed"] = ent >= 0
    L = H._LC("base")
    for Hh in (63, 126):
        ok = (ent >= 0) & (ent + Hh <= ND)
        v = np.full(len(P), np.nan)
        v[ok] = np.expm1(L[ent[ok] + Hh, P["b"].to_numpy()[ok]] - L[ent[ok], P["b"].to_numpy()[ok]])
        P[f"fwd_{Hh}"] = v

    # fundamentals: CVM 2011-2020 built here + brapi rows of the shared fundamentals_pit (parents / gap-fill)
    uni = set(P["cnpj8"].astype(str))
    fpath = OUT / "fund_pre.pkl"
    if fpath.exists() and "--refund" not in sys.argv:
        fp = pd.read_pickle(fpath)
    else:
        fp = fundamentals_pre(uni | set(g["cnpj8"].astype(str)))
        fp.to_pickle(fpath)
    shared = pd.read_pickle(HIST / "fundamentals_pit.pkl")
    shared = shared[~shared["source"].astype(str).str.startswith("cvm")]
    allf = pd.concat([fp, shared[fp.columns.intersection(shared.columns)]], ignore_index=True)
    fu = fund_features(allf)
    m = pd.merge_asof(P[["day", "cnpj8"]].reset_index().sort_values("day"), fu, left_on="day", right_on="avail",
                      by="cnpj8", direction="backward").set_index("index").reindex(P.index)
    stale = (m["day"] - m["period_end"]).dt.days > 460
    for c in ["f_lev", "f_cov", "f_cash_st", "f_eq_ratio", "f_d_lev"]:
        P[c] = m[c].where(~stale).to_numpy()
    P["f_source"] = m["f_source"].where(~stale).to_numpy()
    P["covered"] = P["f_lev"].notna() & P["f_cov"].notna()
    QSIGN = {"f_lev": -1, "f_cov": 1, "f_cash_st": 1, "f_eq_ratio": 1, "f_d_lev": -1}
    for c in QSIGN:
        P.loc[~P["covered"], c] = np.nan
    U = P[P["univ"]]
    rk = pd.DataFrame({c: U.groupby("day")[c].rank(pct=True) * s_ for c, s_ in QSIGN.items()})
    P["f_quality"] = rk.mean(axis=1, skipna=True).reindex(P.index).where(P["covered"] & P["univ"])
    thr = P[P["univ"]].groupby("day")["f_quality"].quantile(0.2)
    P["worstQ"] = P["covered"] & P["univ"] & (P["f_quality"] <= P["day"].map(thr))
    log("fundamentals attached; covered share of univ", float(P.loc[P["univ"], "covered"].mean()))

    # equity
    tick, TI, SPX, cands = equity_grid(days)
    E = equity_features(P[["cnpj8", "dpos"]].drop_duplicates(), tick, TI, SPX, cands)
    P = P.merge(E, on=["cnpj8", "dpos"], how="left")
    P["listed"] = P["eq_ticker"].notna()
    log("equity attached; listed share of univ", float(P.loc[P["univ"], "listed"].mean()))

    # CVM distress + registry issue date + liquidity (vol91)
    ev = ipe_events()
    P["days_since_distress"] = days_since(ev, P[["cnpj8", "day"]])
    from rfmonitor.sources import snd
    t = snd.table().drop_duplicates("Codigo do Ativo").set_index("Codigo do Ativo")
    iss = pd.to_datetime(t["Data de Emissao"], format="%d/%m/%Y", errors="coerce")
    P["issue_date"] = P["codigo"].map(iss)
    P["bond_age_y"] = (P["day"] - P["issue_date"]).dt.days / 365.25
    vb = raw.assign(v=raw["qty"] * raw["pu_avg"]).groupby(["codigo", "date"])["v"].sum().reset_index()
    vols = []
    for c_, gg in vb.groupby("codigo"):
        cs = gg.set_index("date")["v"].sort_index().cumsum()
        dd_ = P.loc[P["codigo"] == c_, "day"]
        if not len(dd_):
            continue
        hi = cs.reindex(cs.index.union(dd_)).ffill().reindex(dd_).fillna(0).to_numpy()
        lo = cs.reindex(cs.index.union(dd_ - pd.Timedelta(days=91))).ffill().reindex(dd_ - pd.Timedelta(days=91)) \
            .fillna(0).to_numpy()
        vols.append(pd.DataFrame({"codigo": c_, "day": dd_.to_numpy(), "vol91_brl": hi - lo}))
    P = P.merge(pd.concat(vols), on=["codigo", "day"], how="left")
    P["press_neg_30d"] = np.nan          # Google-News press starts 2020-10: the P4 press filter is inactive pre-2021
    nz = lambda s_, v: s_.fillna(v)
    P["p4f"] = ~(nz(P["resid_z"], 0) <= -1.5) & (nz(P["press_neg_30d"], 0) < 1)
    P["p4"] = P["univ"] & (nz(P["cdi_pct"], 1) <= 0.3) & P["p4f"]
    P["p4q"] = P["p4"] & ~P["worstQ"]
    P["rich"] = nz(P["resid_z"], 0) <= -1.5
    for c in ("codigo", "cnpj8", "kind", "peer"):
        P[c] = P[c].astype(str)
    P = P.sort_values(["day", "codigo"]).reset_index(drop=True)
    P.to_pickle(OUT / "panel_M.pkl")
    g[["codigo", "day", "date", "cdi_bps", "resid_z", "ratio", "r_patch", "kind", "eligible"]].to_pickle(OUT / "grid.pkl")
    log("panel", P.shape, "decisions", P["dpos"].nunique())


if __name__ == "__main__":
    main()
