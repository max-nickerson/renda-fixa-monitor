"""Structural-credit dataset: point-in-time market cap, equity vol, debt and Merton / CreditGrades metrics
for every listed ticker used by the harness panel (issuer's own stock or listed parent).

Output: data/history/nightly/structural_credit/structural_daily.pkl
  one row per (ticker, date) on equity trading days, all quantities known at the close of `date`.

Point-in-time rules
  * shares outstanding: CVM ITR/DFP `composicao_capital` (total - treasury), available at DT_RECEB of the
    version whose values are published (strict: latest version's receipt date).
  * split handling: market cap = S_ref * q_ref * close_t, where close is brapi split-adjusted (in today's share
    units) and q_ref = unadjusted B3 COTAHIST close / brapi close on the filing's reference date. This converts the
    filing's share count into today's units without using any share count published after t. (The split factor
    itself is mechanical; the only "future" object is the brapi split adjustment, which is price-neutral.)
  * class/unit factor c: constant per ticker = brapi marketCap today / our raw market cap today (clipped), used only
    to convert units (e.g. TAEE11 = 1 ON + 2 PN) and class mixes; 1 when not available.
  * debt: fundamentals_pit (CVM strict availability; brapi rows +45/+90d), for the listed company's CNPJ
    (from CVM FCA valor_mobiliario ticker -> CNPJ), fallback rows tagged with this ticker. Stale > 460d dropped.
  * equity vol: 252d (min 120 obs) stdev of daily log adj_close returns, blended 50/50 with 63d; floor 10%.
  * risk-free: CDI annualised (log) from the harness.
"""
from __future__ import annotations

import glob
import json
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[3]
HIST = ROOT / "data" / "history"
CACHE = HIST / "nightly" / "structural_credit"
CACHE.mkdir(parents=True, exist_ok=True)
OUT = CACHE / "structural_daily.pkl"


UNIT_K = {"TAEE11": 3, "ALUP11": 3, "ENGI11": 5, "SAPR11": 5, "KLBN11": 5}  # shares per unit (CVM FCA)


def _c8(s):
    return re.sub(r"\D", "", str(s)).zfill(14)[:8]


# ---------------------------------------------------------------- ticker -> company CNPJ (CVM FCA)
def ticker_cnpj() -> pd.DataFrame:
    rows = []
    for f in sorted(CACHE.glob("fca_*.zip")):
        z = zipfile.ZipFile(f)
        n = [i for i in z.namelist() if "valor_mobiliario" in i][0]
        d = pd.read_csv(z.open(n), sep=";", encoding="latin1", dtype=str)
        d = d[d["Codigo_Negociacao"].notna()]
        rows.append(pd.DataFrame({"ticker": d["Codigo_Negociacao"].str.strip().str.upper(),
                                  "cnpj8": d["CNPJ_Companhia"].map(_c8),
                                  "unit": d["Composicao_BDR_Unit"]}))
    t = pd.concat(rows).drop_duplicates(["ticker", "cnpj8"])
    return t


# ---------------------------------------------------------------- shares outstanding, PIT
def shares_pit() -> pd.DataFrame:
    out = []
    for f in sorted((HIST / "cvm_fin").glob("*_cia_aberta_20*.zip")):
        kind = f.name.split("_")[0]
        y = f.stem.split("_")[-1]
        z = zipfile.ZipFile(f)
        idx = pd.read_csv(z.open(f"{kind}_cia_aberta_{y}.csv"), sep=";", encoding="latin1", dtype=str)
        idx = idx[["CNPJ_CIA", "DT_REFER", "VERSAO", "DT_RECEB"]]
        n = f"{kind}_cia_aberta_composicao_capital_{y}.csv"
        if n not in z.namelist():
            continue
        c = pd.read_csv(z.open(n), sep=";", encoding="latin1", dtype=str)
        c = c.merge(idx, on=["CNPJ_CIA", "DT_REFER", "VERSAO"], how="left")
        num = lambda s: pd.to_numeric(c[s], errors="coerce").fillna(0)
        c["shares"] = num("QT_ACAO_TOTAL_CAP_INTEGR") - num("QT_ACAO_TOTAL_TESOURO")
        out.append(pd.DataFrame({"cnpj8": c["CNPJ_CIA"].map(_c8), "ref": pd.to_datetime(c["DT_REFER"]),
                                 "avail": pd.to_datetime(c["DT_RECEB"]), "shares": c["shares"],
                                 "ver": pd.to_numeric(c["VERSAO"], errors="coerce")}))
    s = pd.concat(out); s["ref"] = s["ref"].astype("datetime64[ns]"); s["avail"] = s["avail"].astype("datetime64[ns]")
    s = s[(s["shares"] > 0) & s["avail"].notna()]
    s = s.sort_values(["cnpj8", "ref", "ver"]).drop_duplicates(["cnpj8", "ref"], keep="last")
    # never go backwards in reference date
    s = s.sort_values(["cnpj8", "avail", "ref"])
    s["_mx"] = s.groupby("cnpj8")["ref"].cummax().groupby(s["cnpj8"]).shift(1)
    s = s[s["_mx"].isna() | (s["ref"] > s["_mx"])].drop(columns=["_mx", "ver"])
    return s.drop_duplicates(["cnpj8", "avail"], keep="last")


# ---------------------------------------------------------------- unadjusted prices
def cotahist() -> pd.DataFrame:
    fs = sorted((HIST / "equity_raw").glob("cotahist_*.csv.gz"))
    d = pd.concat([pd.read_csv(f, usecols=["ticker", "date", "close", "volume"]) for f in fs])
    d["date"] = pd.to_datetime(d["date"])
    return d


def brapi_mcap() -> dict:
    m = {}
    for f in glob.glob(str(HIST / "brapi_fund" / "*.json")):
        try:
            d = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue
        if isinstance(d, dict) and d.get("marketCap") and d.get("regularMarketPrice"):
            m[Path(f).stem] = (float(d["marketCap"]), float(d["regularMarketPrice"]), d.get("regularMarketTime"))
    return m


# ---------------------------------------------------------------- debt, PIT
def debt_pit() -> pd.DataFrame:
    f = pd.read_pickle(HIST / "fundamentals_pit.pkl").copy()
    f["avail"] = f["available_date_strict"].astype("datetime64[ns]"); f["period_end"] = f["period_end"].astype("datetime64[ns]")
    f["cnpj8"] = f["cnpj8"].astype(str)
    f = f.sort_values(["cnpj8", "avail", "period_end"])
    f["_mx"] = f.groupby("cnpj8")["period_end"].cummax().groupby(f["cnpj8"]).shift(1)
    f = f[f["_mx"].isna() | (f["period_end"] > f["_mx"])].drop_duplicates(["cnpj8", "avail"], keep="last")
    return f[["cnpj8", "ticker", "is_parent", "avail", "period_end", "gross_debt", "st_debt", "cash", "total_assets",
              "equity", "ebitda_ltm"]]


# ---------------------------------------------------------------- models
def merton(E, sE, F, r, T=1.0, iters=60):
    """KMV-style Merton: solve V, sV from E = V N(d1) - F e^{-rT} N(d2), sE E = N(d1) sV V (vectorised)."""
    E, sE, F, r = map(lambda a: np.asarray(a, float), (E, sE, F, r))
    V = E + F * np.exp(-r * T)
    sV = sE * E / V
    sq = np.sqrt(T)
    for _ in range(iters):
        # Newton on V for given sV (a few steps)
        for _ in range(3):
            d1 = (np.log(V / F) + (r + 0.5 * sV ** 2) * T) / (sV * sq)
            d2 = d1 - sV * sq
            f = V * norm.cdf(d1) - F * np.exp(-r * T) * norm.cdf(d2) - E
            V = np.maximum(V - f / np.maximum(norm.cdf(d1), 1e-6), E * 1.0001)
        d1 = (np.log(V / F) + (r + 0.5 * sV ** 2) * T) / (sV * sq)
        sV_new = np.clip(sE * E / (V * np.maximum(norm.cdf(d1), 1e-6)), 0.01, 3.0)
        if np.nanmax(np.abs(sV_new - sV)) < 1e-6:
            sV = sV_new
            break
        sV = sV_new
    d1 = (np.log(V / F) + (r + 0.5 * sV ** 2) * T) / (sV * sq)
    d2 = d1 - sV * sq
    dd = (np.log(V / F) + (r - 0.5 * sV ** 2) * T) / (sV * sq)   # risk-neutral drift (= d2)
    B = np.maximum(V - E, 1e-9)                                     # market value of debt
    spread = -np.log(B / F) / T - r
    return V, sV, dd, np.maximum(spread, 0.0)


def naive_dd(E, sE, F, r, T=1.0):
    """Bharath-Shumway (2008) naive DD, drift = r."""
    sD = 0.05 + 0.25 * sE
    sV = E / (E + F) * sE + F / (E + F) * sD
    return (np.log((E + F) / F) + (r - 0.5 * sV ** 2) * T) / (sV * np.sqrt(T))


def creditgrades(S, D, sE, r, T=5.0, Lbar=0.5, lam=0.3, R=0.5):
    """CreditGrades (Finger et al. 2002) par CDS spread, totals instead of per-share quantities."""
    S, D, sE, r = map(lambda a: np.asarray(a, float), (S, D, sE, r))
    LD = Lbar * D
    sig = sE * S / (S + LD)
    d = (S + LD) / LD * np.exp(lam ** 2)
    ld = np.log(d)
    A = lambda t: np.sqrt(sig ** 2 * t + lam ** 2)
    P = lambda t: norm.cdf(-A(t) / 2 + ld / A(t)) - d * norm.cdf(-A(t) / 2 - ld / A(t))
    xi = lam ** 2 / sig ** 2
    z = np.sqrt(0.25 + 2 * r / sig ** 2)

    def G(u):
        su = sig * np.sqrt(u)
        return d ** (z + 0.5) * norm.cdf(-ld / su - z * su) + d ** (-z + 0.5) * norm.cdf(-ld / su + z * su)
    H = np.exp(r * xi) * (G(T + xi) - G(xi))
    P0 = P(0.0)
    PT = P(T)
    c = r * (1 - R) * (1 - P0 + H) / np.maximum(P0 - PT * np.exp(-r * T) - H, 1e-12)
    surv = PT
    return np.maximum(c, 0.0), surv


# ---------------------------------------------------------------- build
def build(tickers: list[str], cdi_ann: pd.Series) -> pd.DataFrame:
    eq = pd.read_pickle(HIST / "equity_daily.pkl")
    eq["date"] = pd.to_datetime(eq["date"])
    eq = eq[eq["ticker"].isin(tickers)].sort_values(["ticker", "date"]).drop_duplicates(["ticker", "date"])
    tc = ticker_cnpj()
    sh = shares_pit()
    ch = cotahist()
    mc = brapi_mcap()
    de = debt_pit()
    mp = pd.read_csv(ROOT / "research" / "data" / "equity_map.csv", dtype=str)
    rf = np.log1p(cdi_ann.astype(float)).rename("rf")
    out, diag = [], []
    for t, g in eq.groupby("ticker"):
        g = g.set_index("date")
        cn = tc.loc[tc["ticker"] == t, "cnpj8"].unique().tolist()
        if not cn:  # fall back to the map's direct row
            cn = mp.loc[(mp["ticker"] == t) & (mp["mapping_type"] == "direct"), "cnpj8"].str.zfill(8).unique().tolist()
        cn8 = cn[0] if cn else None
        # --- vol
        px = g["adj_close"].where(g["adj_close"] > 0)
        lr = np.log(px).diff()
        if g["source"].astype(str).str.startswith("cotahist").all():
            lr[lr.abs() > 0.5] = 0.0
        v252 = lr.rolling(252, min_periods=120).std() * np.sqrt(252)
        v63 = lr.rolling(63, min_periods=40).std() * np.sqrt(252)
        sE = (0.5 * v252 + 0.5 * v63).fillna(v63).clip(lower=0.10)
        # --- shares, split factor
        s = sh[sh["cnpj8"] == cn8].sort_values("avail") if cn8 else sh.iloc[:0]
        suf = re.sub(r"^[A-Z]+", "", t)
        sib = tc.loc[tc["cnpj8"] == cn8, "ticker"].tolist() if cn8 else []
        sib = [x for x in sib if re.sub(r"^[A-Z]+", "", x) == suf] or [t]
        u = ch[ch["ticker"].isin(sib + [t])].sort_values(["date", "volume"]).drop_duplicates("date", keep="last")
        u = u.set_index("date")["close"]
        close = g["close"].astype(float)
        if len(s):
            qs = []
            for ref in s["ref"]:
                cb = close[:ref].tail(1)
                ub = u[:ref].tail(1)
                if len(cb) and len(ub) and (cb.index[0] - ub.index[0]).days == 0 and cb.iloc[0] > 0:
                    q = ub.iloc[0] / cb.iloc[0]
                    q = q if 1e-4 < q < 1e4 else np.nan
                    # snap near-1 ratios (dividend noise / rounding) to 1
                    q = 1.0 if abs(np.log(q)) < 0.08 else q
                else:
                    q = np.nan
                qs.append(q)
            s = s.assign(q=pd.Series(qs, index=s.index).ffill().bfill().fillna(1.0).to_numpy())
            sq_raw = (s["shares"] * s["q"]).to_numpy(float)
            # issuance index: chain filing-to-filing ratios; ~1000x jumps are CVM unit changes (thousands of shares),
            # anything else outside [1/30, 30] is treated as a data error (ratio 1)
            idx = [1.0]
            qv = s["q"].to_numpy(float)
            sv = s["shares"].to_numpy(float)
            for i in range(1, len(sq_raw)):
                rr = sv[i] / sv[i - 1]
                qc = qv[i] / qv[i - 1]
                if 2.5 < abs(np.log10(rr)) < 3.5 and abs(np.log(qc)) < 0.1:
                    rho = 1.0            # CVM switched between shares and thousands of shares
                else:
                    rho = rr * qc
                if not (1 / 30 <= rho <= 30):
                    rho = 1.0
                idx.append(idx[-1] * rho)
            # single-filing spikes that revert are data errors: rolling median (3 filings) in log space
            li = pd.Series(np.log(idx))
            li = li.rolling(3, center=True, min_periods=1).median()
            li.iloc[-1] = np.log(idx[-1])
            idx = list(np.exp(li))
            s = s.assign(I=np.array(idx) / idx[-1])
            # today-unit share count at the last filing (for tickers without a brapi anchor)
            s_last = sq_raw[-1]
            if s_last * close.iloc[-1] < 50e6:   # reported in thousands
                s_last *= 1000.0
            m = pd.merge_asof(pd.DataFrame({"date": g.index.astype("datetime64[ns]")}), s[["avail", "I", "ref"]],
                              left_on="date", right_on="avail", direction="backward")
            Irel = m["I"].to_numpy().astype(float).copy()
            Irel[((m["date"] - m["ref"]).dt.days > 460).to_numpy()] = np.nan
        else:
            Irel = np.full(len(g), np.nan)
            s_last = np.nan
        if t in mc:
            shares_today = mc[t][0] / mc[t][1]
            c = "brapi"
        elif np.isfinite(s_last) and (not t.endswith("11") or t in UNIT_K):
            shares_today = s_last / UNIT_K.get(t, 1)
            c = "cvm"
        else:
            shares_today = np.nan
            c = "none"
        # without CVM share filings assume constant share count (today's units)
        Irel = np.where(np.isnan(Irel) & (len(s) == 0), 1.0, Irel)
        mcap = shares_today * close.to_numpy() * Irel / 1e6  # R$ mn
        # --- debt (listed company's own filing, fallback ticker-tagged brapi rows)
        dd_ = de[de["cnpj8"] == cn8] if cn8 else de.iloc[:0]
        if dd_["gross_debt"].notna().sum() == 0:
            dd_ = de[de["ticker"] == t]
            dsrc = "ticker"
        else:
            dsrc = "cnpj"
        dd_ = dd_.dropna(subset=["gross_debt"]).sort_values("avail")
        if len(dd_):
            m2 = pd.merge_asof(pd.DataFrame({"date": g.index.astype("datetime64[ns]")}), dd_.drop(columns=["cnpj8", "ticker"]),
                               left_on="date", right_on="avail", direction="backward")
            stale = ((m2["date"] - m2["period_end"]).dt.days > 460).to_numpy()
            gd = m2["gross_debt"].to_numpy().astype(float).copy()
            st = m2["st_debt"].fillna(0.3 * m2["gross_debt"]).to_numpy().copy()
            cash = m2["cash"].to_numpy().copy()
            eb = m2["ebitda_ltm"].to_numpy().copy()
            beq = m2["equity"].to_numpy().astype(float).copy()
            for a in (gd, st, cash, eb, beq):
                a[stale] = np.nan
        else:
            gd = st = cash = eb = beq = np.full(len(g), np.nan)
        # sanity guard on the market cap reconstruction: absurd price-to-book => share-count error
        pb = mcap / beq
        bad = (beq > 0) & ((pb > 40) | (pb < 0.02))
        mcap = np.where(bad, np.nan, mcap)
        r = rf.reindex(g.index, method="ffill").to_numpy()
        diag.append(dict(ticker=t, cnpj8=cn8, calib=c, n_share_filings=len(s), debt_src=dsrc,
                         debt_rows=len(dd_), mcap_last=float(pd.Series(mcap).dropna().iloc[-1]) if np.isfinite(mcap).any() else np.nan))
        out.append(pd.DataFrame({"ticker": t, "date": g.index, "close": close.to_numpy(), "mcap": mcap,
                                 "sigE": sE.to_numpy(), "gross_debt": gd, "st_debt": st, "cash": cash,
                                 "ebitda": eb, "book_eq": beq, "rf": r}))
    D = pd.concat(out, ignore_index=True)
    ok = (D["mcap"] > 0) & (D["gross_debt"] > 0) & D["sigE"].notna() & D["rf"].notna()
    X = D[ok]
    F = (X["st_debt"].clip(lower=0) + 0.5 * (X["gross_debt"] - X["st_debt"]).clip(lower=0)).clip(lower=1e-3)
    V, sV, dd, msp = merton(X["mcap"], X["sigE"], F, X["rf"])
    D.loc[ok, "dp"] = F.to_numpy()
    D.loc[ok, "V"] = V
    D.loc[ok, "sigV"] = sV
    D.loc[ok, "dd"] = dd
    D.loc[ok, "pd1y"] = norm.cdf(-dd)
    D.loc[ok, "merton_spread_bps"] = msp * 1e4
    D.loc[ok, "dd_naive"] = naive_dd(X["mcap"].to_numpy(), X["sigE"].to_numpy(), F.to_numpy(), X["rf"].to_numpy())
    cg, surv = creditgrades(X["mcap"], X["gross_debt"], X["sigE"], X["rf"])
    D.loc[ok, "cg_spread_bps"] = np.clip(cg * 1e4, 0, 1e4)
    D.loc[ok, "cg_pd5y"] = 1 - surv
    D.loc[ok, "mkt_lev"] = (X["gross_debt"] / (X["gross_debt"] + X["mcap"])).to_numpy()
    D.loc[ok, "net_mkt_lev"] = ((X["gross_debt"] - X["cash"].fillna(0)).clip(lower=0)
                                / ((X["gross_debt"] - X["cash"].fillna(0)).clip(lower=0) + X["mcap"])).to_numpy()
    pd.DataFrame(diag).to_csv(CACHE / "build_diag.csv", index=False)
    return D


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(ROOT))
    from research.nightly import harness as H
    P = H.load_panel("W", holdout=True)
    tick = sorted(P["eq_ticker"].dropna().unique())
    cdi = H.cdi_daily()
    cdi_ann = ((1 + cdi.where(cdi > 0).ffill()) ** 252 - 1)
    D = build(tick, cdi_ann)
    D.to_pickle(OUT)
    print(D.shape, D["dd"].notna().sum(), D["ticker"].nunique())
    print(D.groupby("ticker")[["mcap", "gross_debt", "sigE", "dd", "cg_spread_bps"]].median().round(2).to_string())
