"""Download SND secondary-market daily prints (all columns incl. PU min/avg/max and % PU da Curva) month by month.
Same endpoint / parsing as rfmonitor.history.snd_trades_month and bias_audit/fetch_snd_range.py.
Cache: data/history/nightly/pre2021_oos_extension/snd_YYYYMM.csv.gz.
PIT: SND publishes the day's aggregates after the close of the trade date; values are used from that date's close on."""
import io, sys, time
from datetime import date, timedelta
from pathlib import Path
import httpx, pandas as pd

OUT = Path("data/history/nightly/pre2021_oos_extension")
OUT.mkdir(parents=True, exist_ok=True)
URL = ("https://www.debentures.com.br/exploreosnd/consultaadados/mercadosecundario/precosdenegociacao_e.asp"
       "?op_exc=Nada&emissor=&isin=&ativo=&dt_ini={a:%Y%m%d}&dt_fim={b:%Y%m%d}")


def brf(s):
    s = str(s).strip()
    if not s or s == "nan":
        return float("nan")
    try:
        return float(s.replace(".", "").replace(",", "."))
    except ValueError:
        return float("nan")


def fetch(a, b, tries=4):
    for k in range(tries):
        try:
            r = httpx.get(URL.format(a=a, b=b), timeout=240, headers={"User-Agent": "Mozilla/5.0"})
            if r.status_code == 200:
                return r.content.decode("latin1")
            print("status", r.status_code, a, b, flush=True)
        except Exception as e:
            print("err", a, b, type(e).__name__, flush=True)
        time.sleep(5 * (k + 1))
    return None


def parse(t):
    lines = t.splitlines()
    try:
        h = next(i for i, l in enumerate(lines) if l.startswith("Data\t"))
    except StopIteration:
        return None
    return pd.read_csv(io.StringIO("\n".join(lines[h:])), sep="\t", dtype=str)


def month(m):
    path = OUT / f"snd_{m:%Y%m}.csv.gz"
    if path.exists():
        return
    last = date(m.year + (m.month == 12), m.month % 12 + 1, 1) - timedelta(days=1)
    frames = []
    a = m
    while a <= last:  # 5-day chunks (the server 500s on large ranges); fall back to single days
        b = min(a + timedelta(days=4), last)
        t = fetch(a, b, tries=2)
        if t is not None:
            f = parse(t)
            if f is not None:
                frames.append(f)
        else:
            d = a
            while d <= b:
                t = fetch(d, d)
                if t is None:  # the server 500s persistently on a few single days (e.g. 2020-12-16): skip, logged
                    print("FAILED (skipped day)", d, flush=True)
                    with open(OUT / "missing_days.txt", "a") as fh:
                        fh.write(f"{d}\n")
                    d += timedelta(days=1)
                    continue
                f = parse(t)
                if f is not None:
                    frames.append(f)
                d += timedelta(days=1)
        a = b + timedelta(days=1)
        time.sleep(0.5)
    if not frames:
        print("EMPTY", m, flush=True)
        return
    df = pd.concat(frames, ignore_index=True)
    df.columns = [c.strip() for c in df.columns]
    c = list(df.columns)
    pick = lambda key: next(x for x in c if key in x)
    out = pd.DataFrame({
        "date": pd.to_datetime(df[c[0]].str.strip(), format="%d/%m/%Y", errors="coerce"),
        "emissor": df[c[1]].str.strip(),
        "codigo": df[c[2]].str.strip(),
        "isin": df[c[3]].str.strip(),
        "qty": df[pick("Quantidade")].map(brf),
        "trades": df[pick("mero de Neg")].map(brf),
        "pu_min": df[pick("nimo")].map(brf),
        "pu_avg": df[pick("dio")].map(brf),
        "pu_max": df[pick("ximo")].map(brf),
        "pct_curve": df[pick("% PU da Curva")].map(brf),
    }).dropna(subset=["date", "pu_avg"])
    out.to_csv(path, index=False, compression="gzip")
    print(m, len(out), out["codigo"].nunique(), flush=True)
    time.sleep(1.5)


if __name__ == "__main__":
    y0, m0, y1, m1 = map(int, sys.argv[1:5])
    d = date(y0, m0, 1)
    while (d.year, d.month) <= (y1, m1):
        month(d)
        d = date(d.year + (d.month == 12), d.month % 12 + 1, 1)
