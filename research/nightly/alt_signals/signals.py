"""Point-in-time alternative & supply signals for the debenture panel.

Every raw observation carries an AVAILABILITY date (when a market participant could have known it);
a value is used on decision day d only if avail <= d (decisions are at the close of d; all lags below
are >= 1 calendar day).

Signal families (all "higher = better credit / expected to outperform" unless noted):

1. Primary supply (CVM offers registry, oferta_distribuicao.zip)
   - legacy ICVM 400/476 debenture offers: avail = Data_Inicio_Oferta + 7d (comunicado de inicio is due
     within 5 bdays of the first placement effort)
   - RCVM 160 debenture requests (2023+): avail = Data_requerimento + 1d (automatic rite, published on filing)
   columns: sup_iss_90d / sup_iss_365d  (# offers by the same cnpj8, raw count; sign tested both ways)
            sup_sec_mom  (sector offer count 90d vs its own trailing-2y quarterly average, log ratio; sector-level)
            sup_mkt_mom  (market-wide 90d count vs trailing-2y quarterly average; time series only)
            sup_inc_mom  (Lei 12.431 incentivised offers 90d vs trailing 2y; time series only)
2. Rating actions (data/history/rating_events.pkl, dated by action date; avail = date + 1d)
   rat_neg_180d (negative actions incl. outlook/watch), rat_out_neg_180d (negative outlook/watch only,
   no downgrade), rat_pos_180d
3. Sector fundamental nowcasts from commodity / power data (data/history/commodities.pkl, which already
   carries available_date) + ANEEL tariff flags (bandeiras; avail = first day of competence month - the
   flag is announced on the last Friday of the previous month)
   nc_margin : hand-built sector margin nowcast, z-scored with expanding (past-only) std, sector-level
   nc_expo   : exposure-weighted (research/data/sector_exposures.csv, commodity variables only) 6m-change z
4. ANEEL distribution tariff resets (tarifas homologadas; avail = DatInicioVigencia, the REH is published
   a few days before vigencia), matched to issuers by distributor CNPJ root:
   tar_adj_last (log change of B1 residential TE+TUSD at the last reset, within 400d), tar_days_since
"""
from __future__ import annotations
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "data" / "history" / "nightly" / "alt_signals" / "raw"
CACHE = ROOT / "data" / "history" / "nightly" / "alt_signals"


def _cnpj8(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace(r"\D", "", regex=True).str.zfill(14).str[:8]


# ---------------------------------------------------------------- 1. supply
def load_offers() -> pd.DataFrame:
    out = CACHE / "offers_deb.pkl"
    if out.exists():
        return pd.read_pickle(out)
    z = zipfile.ZipFile(RAW / "oferta_distribuicao.zip")
    a = pd.read_csv(z.open("oferta_distribuicao.csv"), sep=";", encoding="latin-1", low_memory=False)
    a = a[a["Tipo_Ativo"].str.contains("DEB", na=False)].copy()
    a["start"] = pd.to_datetime(a["Data_Inicio_Oferta"], errors="coerce")
    la = pd.DataFrame({
        "cnpj8": _cnpj8(a["CNPJ_Emissor"]), "event": a["start"], "avail": a["start"] + pd.Timedelta(days=7),
        "value": pd.to_numeric(a["Valor_Total"], errors="coerce"),
        "incent": a["Oferta_Incentivo_Fiscal"].eq("S"), "src": "legacy"})
    b = pd.read_csv(z.open("oferta_resolucao_160.csv"), sep=";", encoding="latin-1", low_memory=False)
    b = b[b["Valor_Mobiliario"].str.startswith("Deb", na=False)].copy()
    b["req"] = pd.to_datetime(b["Data_requerimento"], errors="coerce")
    b = b[~b["Status_Requerimento"].isin(["Requerimento Expirado"])]
    lb = pd.DataFrame({
        "cnpj8": _cnpj8(b["CNPJ_Emissor"]), "event": b["req"], "avail": b["req"] + pd.Timedelta(days=1),
        "value": pd.to_numeric(b["Valor_Total_Registrado"], errors="coerce"),
        "incent": b["Titulo_incentivado"].eq("S"), "src": "r160"})
    o = pd.concat([la, lb], ignore_index=True).dropna(subset=["event"])
    # one row per issuer x availability day (multi-series offers are filed as several rows)
    o = (o.groupby(["cnpj8", "avail"], as_index=False)
           .agg(event=("event", "min"), value=("value", "sum"), incent=("incent", "max"), src=("src", "first"),
                n_rows=("event", "size")))
    o.to_pickle(out)
    return o


def _count_in_window(ev_days: np.ndarray, q_days: np.ndarray, w: int) -> np.ndarray:
    """# events with avail in (d-w, d] for each query day d (both sorted int64 day numbers)."""
    hi = np.searchsorted(ev_days, q_days, side="right")
    lo = np.searchsorted(ev_days, q_days - w, side="right")
    return hi - lo


def issuer_event_counts(panel: pd.DataFrame, ev: pd.DataFrame, windows: dict[str, int]) -> pd.DataFrame:
    """ev: cnpj8, avail. Returns DataFrame aligned to panel.index with one column per window."""
    res = {k: np.zeros(len(panel)) for k in windows}
    pday = panel["day"].values.astype("datetime64[D]").astype(np.int64)
    evg = {k: np.sort(g["avail"].values.astype("datetime64[D]").astype(np.int64)) for k, g in ev.groupby("cnpj8")}
    cn = panel["cnpj8"].astype(str).values
    for c8 in np.unique(cn):
        if c8 not in evg:
            continue
        m = np.where(cn == c8)[0]
        for k, w in windows.items():
            res[k][m] = _count_in_window(evg[c8], pday[m], w)
    return pd.DataFrame(res, index=panel.index)


def supply_timeseries(days: pd.DatetimeIndex, sectors: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Market / incentivised / sector supply momentum on the given days (sector-level long frame)."""
    o = load_offers()
    o = o.merge(sectors[["cnpj8", "sector"]], on="cnpj8", how="left")
    q = days.values.astype("datetime64[D]").astype(np.int64)

    def mom(sub):
        e = np.sort(sub["avail"].values.astype("datetime64[D]").astype(np.int64))
        c90 = _count_in_window(e, q, 91)
        c2y = _count_in_window(e, q, 730)
        return np.log((c90 + 1.0) / (c2y / 8.0 + 1.0))

    ts = pd.DataFrame({"day": days, "sup_mkt_mom": mom(o), "sup_inc_mom": mom(o[o["incent"]])})
    rows = []
    for s, g in o.dropna(subset=["sector"]).groupby("sector"):
        rows.append(pd.DataFrame({"day": days, "sector": s, "sup_sec_mom": mom(g)}))
    return ts, pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------- 2. ratings
NEG_OUT = re.compile(r"outlook.{0,25}negat|negative (outlook|watch)|watch negat|perspectiva negat|observa\w* negat|"
                     r"credit ?watch neg|revises .{0,40}to negative|places .{0,60}negative", re.I)
POS_OUT = re.compile(r"outlook.{0,25}positi|positive (outlook|watch)|perspectiva positi|observa\w* positi|"
                     r"revises .{0,40}to positive|places .{0,60}positive", re.I)
DOWN = re.compile(r"downgrad|rebaix|lowers|cuts|corta|reduz .{0,20}rating", re.I)


def rating_events() -> pd.DataFrame:
    r = pd.read_pickle(ROOT / "data" / "history" / "rating_events.pkl").copy()
    r["cnpj8"] = r["cnpj8"].astype(str).str.zfill(8)
    r["avail"] = pd.to_datetime(r["date"]) + pd.Timedelta(days=1)
    t = r["title"].fillna("")
    r["out_neg"] = t.str.contains(NEG_OUT) & ~t.str.contains(DOWN)
    r["out_pos"] = t.str.contains(POS_OUT)
    r["neg"] = (r["direction"] < 0) | r["out_neg"]
    r["pos"] = (r["direction"] > 0) | r["out_pos"]
    return r


# ---------------------------------------------------------------- 3. sector nowcasts
def _pit_daily(comm: pd.DataFrame, series: str, cal: pd.DatetimeIndex) -> pd.Series:
    """Latest value known at each calendar day (by available_date), latest vintage."""
    s = comm[comm["series"] == series].sort_values(["available_date", "period_date"])
    s = s.groupby("available_date")["value"].last()
    return s.reindex(cal.union(s.index)).ffill().reindex(cal)


def _z_past(x: pd.Series, min_obs: int = 250) -> pd.Series:
    """x / expanding std of x (past only, daily calendar), no demeaning (changes are ~0-mean)."""
    sd = x.expanding(min_periods=min_obs).std().shift(1)
    return (x / sd).clip(-4, 4)


def _dlog(s: pd.Series, days: int = 182) -> pd.Series:
    return np.log(s).diff(days)


def bandeiras(cal: pd.DatetimeIndex) -> pd.Series:
    b = pd.read_csv(RAW / "aneel_bandeiras.csv", sep=";", encoding="utf-8", dtype=str)
    b["month"] = pd.to_datetime(b["DatCompetencia"])
    b["add"] = pd.to_numeric(b["VlrAdicionalBandeira"].str.replace(",", "."), errors="coerce").fillna(0.0)
    b = b.groupby("month")["add"].max()
    # announced on the last Friday of the previous month -> usable from the 1st of the competence month
    return b.reindex(cal.union(b.index)).ffill().reindex(cal)


def sector_nowcasts() -> pd.DataFrame:
    """Long frame (day, sector, nc_margin, nc_expo) on a daily calendar 2019-01 .. 2026-09."""
    out = CACHE / "sector_nowcasts.pkl"
    if out.exists():
        return pd.read_pickle(out)
    comm = pd.read_pickle(ROOT / "data" / "history" / "commodities.pkl")
    comm["available_date"] = pd.to_datetime(comm["available_date"])
    cal = pd.date_range("2018-01-01", "2026-09-30", freq="D")
    S = {k: _pit_daily(comm, k, cal) for k in comm["series"].unique()}
    D = {k: _z_past(_dlog(v)) for k, v in S.items() if (v.dropna() > 0).all()}

    # reservoir anomaly vs same calendar month in PAST years only (expanding)
    res = S["reservoir"]
    anom = pd.Series(np.nan, index=cal)
    df = pd.DataFrame({"v": res, "m": cal.month, "y": cal.year})
    for y in range(2019, 2027):
        past = df[df["y"] < y].groupby("m")["v"].mean()
        idx = df["y"] == y
        anom[idx] = df.loc[idx, "v"].values - df.loc[idx, "m"].map(past).values
    res_z = (anom / anom.expanding(250).std().shift(1)).clip(-4, 4)
    pld_lvl = np.log1p(S["pld"].rolling(30, min_periods=10).mean())
    pld_z = ((pld_lvl - pld_lvl.expanding(250).mean().shift(1)) / pld_lvl.expanding(250).std().shift(1)).clip(-4, 4)
    flag = bandeiras(cal).fillna(0.0)
    flag_z = ((flag - flag.expanding(365).mean().shift(1)) / flag.expanding(365).std().shift(1)).clip(-4, 4)
    # fuel-distribution margin: Brazil retail diesel (R$/l) minus NY ULSD converted to R$/l
    imp = S["ulsd_ny"] * S["usdbrl"] / 3.785
    marg = S["diesel_br"] - imp
    marg_z = _z_past(marg.diff(182))
    # ethanol/sugar parity: ethanol price relative to sugar (mill mix option value); revenue proxy
    sugar_rev = 0.5 * D["sugar_br"] + 0.5 * D["ethanol_br"]

    M = {
        "sugar_ethanol": sugar_rev - 0.2 * D["diesel_br"],
        "oil_gas": D["brent"] + 0.3 * D["usdbrl"],
        "fuel_distribution": marg_z,
        "petrochemicals": -0.5 * D["brent"] + 0.5 * D["usdbrl"],
        "mining": D["iron_ore"],
        "steel_metals": 0.5 * D["iron_ore"] + 0.5 * D["usdbrl"],
        "pulp_paper": D["usdbrl"],
        "utilities_generation_hydro": res_z,
        "utilities_generation_thermal": -res_z + 0.5 * pld_z,
        "utilities_generation_renewables": 0.5 * pld_z,
        "utilities_distribution": -flag_z,
        "agribusiness_protein": -D["cattle_br"] + 0.3 * D["usdbrl"],
        "agribusiness_grains": 0.5 * D["soy"] + 0.5 * D["corn"] - 0.2 * D["diesel_br"],
        "fertilizers": D["urea"] - 0.5 * D["gas_eu"],
        "trucking_equipment_rental": -D["diesel_br"],
        "railways": 0.5 * D["soy"] + 0.5 * D["corn"],
    }
    # exposure-weighted (existing judgmental map), commodity/power variables only
    ex = pd.read_csv(ROOT / "research" / "data" / "sector_exposures.csv").dropna(subset=["variable"])
    ex = ex[~ex["variable"].isin(["ipca", "selic", "ibc_br", "ibc_br_sa"])]
    Dx = dict(D)
    Dx["reservoir"] = res_z
    Dx["pld"] = pld_z
    Dx["naphtha"] = D["brent"]
    E = {}
    for s, g in ex.groupby("sector"):
        parts = [w * Dx[v] for v, w in zip(g["variable"], g["weight"]) if v in Dx]
        if parts:
            E[s] = sum(parts) / g["weight"].abs().sum()
    rows = []
    for s in sorted(set(M) | set(E)):
        rows.append(pd.DataFrame({"day": cal, "sector": s,
                                  "nc_margin": M[s].values if s in M else np.nan,
                                  "nc_expo": E[s].values if s in E else np.nan}))
    L = pd.concat(rows, ignore_index=True)
    L = L[L["day"] >= "2019-01-01"]
    L.to_pickle(out)
    return L


# ---------------------------------------------------------------- 4. ANEEL tariff resets
def tariff_resets() -> pd.DataFrame:
    out = CACHE / "aneel_resets.pkl"
    if out.exists():
        return pd.read_pickle(out)
    t = pd.read_csv(RAW / "aneel_tarifas_homologadas.csv", sep=";", encoding="utf-8", dtype=str,
                    usecols=["DscREH", "SigAgente", "NumCNPJDistribuidora", "DatInicioVigencia", "DscBaseTarifaria",
                             "DscSubGrupo", "DscModalidadeTarifaria", "DscClasse", "DscSubClasse", "DscDetalhe",
                             "NomPostoTarifario", "VlrTUSD", "VlrTE"])
    s = t[(t.DscBaseTarifaria == "Tarifa de Aplicação") & (t.DscSubGrupo == "B1")
          & (t.DscModalidadeTarifaria == "Convencional") & (t.DscClasse == "Residencial")
          & (t.DscSubClasse == "Residencial") & (t.DscDetalhe == "Não se aplica")].copy()
    for c in ["VlrTUSD", "VlrTE"]:
        s[c] = pd.to_numeric(s[c].str.replace(".", "", regex=False).str.replace(",", "."), errors="coerce")
    s["tar"] = s["VlrTUSD"] + s["VlrTE"]
    s["avail"] = pd.to_datetime(s["DatInicioVigencia"])
    s["cnpj8"] = _cnpj8(s["NumCNPJDistribuidora"])
    g = (s.groupby(["cnpj8", "avail"], as_index=False).agg(tar=("tar", "mean"), agent=("SigAgente", "first"))
           .sort_values(["cnpj8", "avail"]))
    g["adj"] = np.log(g["tar"]).groupby(g["cnpj8"]).diff()
    g = g.dropna(subset=["adj"])
    g = g[g["adj"].abs() < 0.8]
    g.to_pickle(out)
    return g


def _asof_issuer(panel: pd.DataFrame, ev: pd.DataFrame, col: str, max_age: int) -> tuple[np.ndarray, np.ndarray]:
    val = np.full(len(panel), np.nan)
    age = np.full(len(panel), np.nan)
    pday = panel["day"].values.astype("datetime64[D]").astype(np.int64)
    cn = panel["cnpj8"].astype(str).values
    for c8, g in ev.groupby("cnpj8"):
        m = np.where(cn == c8)[0]
        if not len(m):
            continue
        e = g["avail"].values.astype("datetime64[D]").astype(np.int64)
        v = g[col].values
        i = np.searchsorted(e, pday[m], side="right") - 1
        ok = i >= 0
        a = np.where(ok, pday[m] - e[np.clip(i, 0, None)], np.nan)
        ok &= a <= max_age
        val[m[ok]] = v[i[ok]]
        age[m[ok]] = a[ok]
    return val, age


# ---------------------------------------------------------------- public
def signals(panel: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of `panel` with all alt-signal columns added (point-in-time at panel['day'] close)."""
    P = panel.copy()
    P["cnpj8"] = P["cnpj8"].astype(str).str.zfill(8)
    o = load_offers()
    cnt = issuer_event_counts(P, o, {"sup_iss_90d": 91, "sup_iss_365d": 365})
    P[cnt.columns] = cnt.values
    sec = pd.read_csv(ROOT / "research" / "data" / "issuer_sectors.csv", dtype={"cnpj8": str})
    sec["cnpj8"] = sec["cnpj8"].str.zfill(8)
    days = pd.DatetimeIndex(sorted(P["day"].unique()))
    ts, ss = supply_timeseries(days, sec)
    P = P.merge(ts, on="day", how="left").merge(ss, on=["day", "sector"], how="left")
    r = rating_events()
    for nm, flag in [("rat_neg_180d", "neg"), ("rat_out_neg_180d", "out_neg"), ("rat_pos_180d", "pos")]:
        P[nm] = issuer_event_counts(P, r[r[flag]], {nm: 180})[nm].values
    nc = sector_nowcasts()
    P = P.merge(nc, on=["day", "sector"], how="left")
    tr = tariff_resets()
    v, a = _asof_issuer(P, tr, "adj", 400)
    P["tar_adj_last"] = v
    P["tar_days_since"] = a
    return P


SIGNAL_COLS = ["sup_iss_90d", "sup_iss_365d", "sup_sec_mom", "sup_mkt_mom", "sup_inc_mom",
               "rat_neg_180d", "rat_out_neg_180d", "rat_pos_180d", "nc_margin", "nc_expo",
               "tar_adj_last", "tar_days_since"]


def export_long(P: pd.DataFrame, path: Path) -> None:
    """(codigo, cnpj8, day, signal columns) for a combiner."""
    P[["codigo", "cnpj8", "day", "sector"] + SIGNAL_COLS].to_pickle(path)
