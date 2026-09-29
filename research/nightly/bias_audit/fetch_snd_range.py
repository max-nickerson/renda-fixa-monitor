"""Download SND secondary-market daily prices WITH intraday PU min / max (the repo's cached snd_trades_*.csv.gz keep
only PU medio). Cached per month in data/history/nightly/bias_audit/snd_range_YYYYMM.csv.gz.
PIT: SND publishes after the close of the trade date; values are used only from the trade date on."""
import io, time, sys
from datetime import date, timedelta
from pathlib import Path
import httpx, pandas as pd

OUT = Path("data/history/nightly/bias_audit")
OUT.mkdir(parents=True, exist_ok=True)
URL = ("https://www.debentures.com.br/exploreosnd/consultaadados/mercadosecundario/precosdenegociacao_e.asp"
       "?op_exc=Nada&emissor=&isin=&ativo=&dt_ini={a:%Y%m%d}&dt_fim={b:%Y%m%d}")


def brf(s):
    s = str(s).strip()
    if not s or s == "nan":
        return float("nan")
    return float(s.replace(".", "").replace(",", "."))


def fetch(a, b):
    for k in range(4):
        try:
            r = httpx.get(URL.format(a=a, b=b), timeout=240, headers={"User-Agent": "Mozilla/5.0"})
            if r.status_code == 200:
                return r.content.decode("latin1")
        except Exception as e:
            print("err", a, b, e, flush=True)
        time.sleep(5 * (k + 1))
    return None


def month(m):
    path = OUT / f"snd_range_{m:%Y%m}.csv.gz"
    if path.exists():
        return
    last = date(m.year + (m.month == 12), m.month % 12 + 1, 1) - timedelta(days=1)
    chunks = [(m, min(m + timedelta(days=9), last)), (m + timedelta(days=10), min(m + timedelta(days=19), last)),
              (m + timedelta(days=20), last)]
    frames = []
    for a, b in chunks:
        t = fetch(a, b)
        if t is None:
            print("FAILED", a, b, flush=True)
            return
        lines = t.splitlines()
        try:
            h = next(i for i, l in enumerate(lines) if l.startswith("Data\t"))
        except StopIteration:
            continue
        frames.append(pd.read_csv(io.StringIO("\n".join(lines[h:])), sep="\t", dtype=str))
    df = pd.concat(frames, ignore_index=True)
    df.columns = [c.strip() for c in df.columns]
    cols = list(df.columns)
    out = pd.DataFrame({
        "date": pd.to_datetime(df[cols[0]].str.strip(), format="%d/%m/%Y", errors="coerce"),
        "codigo": df[cols[2]].str.strip(),
        "qty": df[cols[4]].map(brf), "trades": df[cols[5]].map(brf),
        "pu_min": df[cols[6]].map(brf), "pu_avg": df[cols[7]].map(brf), "pu_max": df[cols[8]].map(brf),
    }).dropna(subset=["date", "pu_avg"])
    out.to_csv(path, index=False, compression="gzip")
    print(m, len(out), flush=True)


if __name__ == "__main__":
    y0, m0, y1, m1 = map(int, sys.argv[1:5])
    d = date(y0, m0, 1)
    while (d.year, d.month) <= (y1, m1):
        month(d)
        d = date(d.year + (d.month == 12), d.month % 12 + 1, 1)
