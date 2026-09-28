"""Historical datasets for research / ML (all free, keyless, cached under data/history/).

- SND secondary-market trades (all debentures, daily since ~2010): PU médio, quantity, #trades, % PU da curva
- B3 TaxaSwap reference curves (DI x Pré, DI x IPCA) since 2015 — the market DI curves
- ANBIMA IDA indices (IDA-DI, IDA-IPCA, IMA-B …) daily since 2009 — market-level credit returns
- BCB SGS: CDI (12) and IPCA (433)
"""
from __future__ import annotations

import io
import logging
import time
import zipfile
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from .bonds import br_float
from .config import DATA_DIR
from .http import get

log = logging.getLogger(__name__)
HIST = DATA_DIR / "history"
UA_HDR = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/130"}


def _month_starts(start: date, end: date) -> list[date]:
    out, d = [], date(start.year, start.month, 1)
    while d <= end:
        out.append(d)
        d = date(d.year + (d.month == 12), d.month % 12 + 1, 1)
    return out


# ---------------------------------------------------------------- SND trades
def _snd_fetch(a: date, b: date, tries: int = 3) -> str | None:
    url = ("https://www.debentures.com.br/exploreosnd/consultaadados/mercadosecundario/precosdenegociacao_e.asp"
           f"?op_exc=Nada&emissor=&isin=&ativo=&dt_ini={a:%Y%m%d}&dt_fim={b:%Y%m%d}")
    for k in range(tries):
        try:
            r = get(url, headers=UA_HDR, timeout=180)
            if r.status_code == 200:
                return r.content.decode("latin1")
        except Exception as e:
            log.debug("SND %s-%s: %s", a, b, e)
        time.sleep(3 * (k + 1))
    return None


def snd_trades_month(m: date) -> pd.DataFrame:
    HIST.mkdir(parents=True, exist_ok=True)
    last = (date(m.year + (m.month == 12), m.month % 12 + 1, 1) - timedelta(days=1))
    path = HIST / f"snd_trades_{m:%Y%m}.csv.gz"
    current = last >= date.today() - timedelta(days=3)
    if path.exists() and not current:
        return pd.read_csv(path, parse_dates=["date"])
    end = min(last, date.today())
    mid = m + timedelta(days=14)
    # The SND server sporadically 500s on big ranges: try the month, then two halves.
    whole = _snd_fetch(m, end)
    if whole is not None:
        texts = [whole]
    else:
        halves = [(m, min(mid, end)), (mid + timedelta(days=1), end)]
        texts = [t for a, b in halves if a <= b and (t := _snd_fetch(a, b)) is not None]
    frames = []
    for text in texts:
        lines = text.splitlines()
        try:
            h = next(i for i, l in enumerate(lines) if l.startswith("Data\t"))
        except StopIteration:
            continue
        frames.append(pd.read_csv(io.StringIO("\n".join(lines[h:])), sep="\t", dtype=str))
    if not frames:
        raise RuntimeError(f"SND trades unavailable for {m:%Y-%m}")
    df = pd.concat(frames, ignore_index=True)
    df.columns = [c.strip() for c in df.columns]
    out = pd.DataFrame({
        "date": pd.to_datetime(df["Data"].str.strip(), format="%d/%m/%Y", errors="coerce"),
        "codigo": df["Código do Ativo"].str.strip(),
        "isin": df["ISIN"].str.strip(),
        "qty": df["Quantidade"].map(br_float),
        "trades": df["Número de Negócios"].map(br_float),
        "pu_avg": df["PU Médio"].map(br_float),
        "pct_curve": df["% PU da Curva"].map(br_float),
    }).dropna(subset=["date", "pu_avg"])
    out.to_csv(path, index=False, compression="gzip")
    return out


def snd_trades(start: date, end: date | None = None) -> pd.DataFrame:
    frames = []
    for m in _month_starts(start, end or date.today()):
        try:
            frames.append(snd_trades_month(m))
        except Exception as e:
            log.warning("SND trades %s failed: %s", m, e)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ---------------------------------------------------------------- B3 curves
def b3_curves_day(d: date) -> pd.DataFrame | None:
    """PRE (DI x pré) and DIC (DI x IPCA) vertices for one business day: (curve, du, rate %)."""
    HIST.mkdir(parents=True, exist_ok=True)
    cache = HIST / "b3" / f"{d:%Y%m%d}.csv"
    if cache.exists():
        df = pd.read_csv(cache)
        return df if not df.empty else None
    cache.parent.mkdir(exist_ok=True)
    r = None
    for k in range(5):
        try:
            r = get(f"https://www.b3.com.br/pesquisapregao/download?filelist=TS{d:%y%m%d}.ex_",
                    headers=UA_HDR, retries=0)
        except Exception as e:  # timeouts / resets
            log.debug("B3 %s: %s", d, e)
            r = None
            time.sleep(5 * (k + 1))
            continue
        if r.status_code != 429:
            break
        time.sleep(10 * (k + 1))  # B3 rate-limits bursts
    if r is None or r.status_code != 200:
        return None  # don't cache failures — retry next run
    rows = []
    if r.content[:2] == b"PK":
        try:
            outer = zipfile.ZipFile(io.BytesIO(r.content))
            inner_bytes = outer.read(outer.namelist()[0])
            inner = zipfile.ZipFile(io.BytesIO(inner_bytes))
            txt = inner.read(inner.namelist()[0]).decode("latin1")
            for line in txt.splitlines():
                code = line[21:26].strip()
                if code in ("PRE", "DIC"):
                    du = int(line[46:51])
                    rate = int(line[52:66]) / 1e7 * (-1 if line[51] == "-" else 1)
                    rows.append((code, du, rate))
        except (zipfile.BadZipFile, ValueError, IndexError) as e:
            log.debug("TaxaSwap %s parse failed: %s", d, e)
    df = pd.DataFrame(rows, columns=["curve", "du", "rate"])
    df.to_csv(cache, index=False)
    return df if not df.empty else None


def b3_curve_panel(start: date, end: date | None = None, step_days: int = 1) -> pd.DataFrame:
    """Rates at standard tenors (0.5,1,2,3,5,7,10y) per date for PRE and DIC — compact for modelling."""
    tenors = [126, 252, 504, 756, 1260, 1764, 2520]
    out = []
    d, end = start, end or date.today()
    while d <= end:
        if d.weekday() < 5:
            try:
                df = b3_curves_day(d)
            except Exception as e:
                log.warning("B3 %s: %s", d, e)
                df = None
            if df is not None:
                row = {"date": pd.Timestamp(d)}
                for c in ("PRE", "DIC"):
                    g = df[df["curve"] == c].sort_values("du")
                    if len(g) > 2:
                        for t in tenors:
                            row[f"{c}_{t}"] = float(np.interp(t, g["du"], g["rate"]))
                out.append(row)
        d += timedelta(days=step_days)
    p = pd.DataFrame(out).set_index("date").sort_index() if out else pd.DataFrame()
    # B3's 'DIC' vertices are NOT a usable DI x IPCA real curve (avg −1.6pp vs NTN-B, erratic, e.g. ~4.5% at
    # 2y in Feb-2026). Real rates come from the NTN-B curve (Tesouro Direto) instead; DI x Pré is kept
    # (matches LTN within ~2 bps).
    if not p.empty:
        p = p.drop(columns=[c for c in p.columns if c.startswith("DIC_")])
        real = ntnb_panel(p.index.min(), p.index.max())
        p = p.join(real, how="left").ffill()
    return p


def ntnb_panel(start, end) -> pd.DataFrame:
    """NTN-B real yields at standard tenors (as DIC_<du> columns for compatibility), from Tesouro Direto."""
    from .sources import tesouro_direto as td
    t = td.curve_like_tpf(pd.Timestamp(start) - pd.Timedelta(days=10))
    t = t[(t["titulo"] == "NTN-B") & (t["date"] <= pd.Timestamp(end))]
    tenors = [126, 252, 504, 756, 1260, 1764, 2520]
    rows = {}
    for d, g in t.groupby("date"):
        x = ((g["vencimento"] - d).dt.days / 365.25 * 252).to_numpy()
        y = g["taxa_indicativa"].to_numpy()
        o = np.argsort(x)
        if len(o) >= 3:
            rows[d] = {f"DIC_{k}": float(np.interp(k, x[o], y[o])) for k in tenors}
    return pd.DataFrame.from_dict(rows, orient="index").sort_index()


def curve_rate(panel: pd.DataFrame, d: pd.Timestamp, curve: str, years: float) -> float | None:
    if panel.empty:
        return None
    row = panel[panel.index <= d].tail(1)
    if row.empty:
        return None
    tenors = [126, 252, 504, 756, 1260, 1764, 2520]
    vals = [row.iloc[0].get(f"{curve}_{t}") for t in tenors]
    pts = [(t, v) for t, v in zip(tenors, vals) if v == v and v is not None]
    if not pts:
        return None
    return float(np.interp(years * 252, [p[0] for p in pts], [p[1] for p in pts]))


# ---------------------------------------------------------------- IDA indices
IDA_URL = "https://s3-data-prd-use1-precos.s3.us-east-1.amazonaws.com/arquivos/indices-historico/{}-HISTORICO.xls"


def ida(name: str) -> pd.DataFrame:
    HIST.mkdir(parents=True, exist_ok=True)
    path = HIST / f"{name}.xlsx"
    if not path.exists() or (pd.Timestamp.now() - pd.Timestamp(path.stat().st_mtime, unit="s")).days >= 1:
        r = get(IDA_URL.format(name), headers=UA_HDR)
        r.raise_for_status()
        path.write_bytes(r.content)
    df = pd.read_excel(path)
    df.columns = [str(c).strip() for c in df.columns]
    date_col = next(c for c in df.columns if "Data" in c)
    idx_col = next(c for c in df.columns if "mero" in c)  # 'Número Índice'
    dur_col = next((c for c in df.columns if "Duration" in c), None)
    out = pd.DataFrame({"date": pd.to_datetime(df[date_col], dayfirst=True, errors="coerce"),
                        "index": pd.to_numeric(df[idx_col], errors="coerce")})
    if dur_col:
        out["duration_du"] = pd.to_numeric(df[dur_col], errors="coerce")
    return out.dropna(subset=["date", "index"]).set_index("date").sort_index()


# ---------------------------------------------------------------- BCB
def bcb_series(code: int, start: date) -> pd.Series:
    """BCB SGS series (e.g. 12 = CDI % per day, 433 = IPCA % per month). Queries in ≤10y chunks, cached 12 h."""
    HIST.mkdir(parents=True, exist_ok=True)
    cache = HIST / f"bcb_{code}_{start:%Y%m%d}.csv"
    if cache.exists() and (time.time() - cache.stat().st_mtime) < 12 * 3600:
        s = pd.read_csv(cache, index_col=0, parse_dates=True).iloc[:, 0]
        s.index.name = None
        return s
    frames = []
    s = start
    try:
        while s <= date.today():
            e = min(date(s.year + 9, 12, 31), date.today())
            url = (f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{code}/dados?formato=json"
                   f"&dataInicial={s:%d/%m/%Y}&dataFinal={e:%d/%m/%Y}")
            for k in range(4):  # BCB sometimes answers 200 with an HTML error page
                r = get(url)
                try:
                    frames.append(pd.DataFrame(r.json()))
                    break
                except ValueError:
                    time.sleep(5 * (k + 1))
            else:
                raise RuntimeError(f"BCB SGS {code}: no valid JSON")
            s = date(e.year + 1, 1, 1)
    except Exception:
        # fall back to any cached copy of the same series that starts on/before `start`
        olds = sorted(HIST.glob(f"bcb_{code}_*.csv"))
        for p in olds:
            if p.stem.split("_")[-1] <= f"{start:%Y%m%d}":
                s_ = pd.read_csv(p, index_col=0, parse_dates=True).iloc[:, 0]
                s_.index.name = None
                log.warning("BCB %s unavailable, using cached %s", code, p.name)
                return s_[s_.index >= pd.Timestamp(start)]
        raise
    df = pd.concat(frames)
    ser = pd.Series(pd.to_numeric(df["valor"]).values, index=pd.to_datetime(df["data"], dayfirst=True))
    ser = ser[~ser.index.duplicated()].sort_index()
    ser.rename("v").to_csv(cache)
    return ser
