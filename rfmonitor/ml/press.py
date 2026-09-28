"""Point-in-time PRESS news from Google News RSS (after:/before: operators), per issuer brand and market-wide.

Each item carries a publication DATE (historical items have a placeholder time), so an item is treated as
known at the END of its date: weekly features for Monday m only count items dated ≤ m−1 (Sunday), i.e.
everything was public before the decision. Cached per (brand, month) under data/history/gnews/.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
import unicodedata
import xml.etree.ElementTree as ET
from datetime import date
from email.utils import parsedate_to_datetime
from urllib.parse import quote_plus

import numpy as np
import pandas as pd

from ..config import DATA_DIR
from ..http import get

CACHE = DATA_DIR / "history" / "gnews"
# ≤ 8 OR-terms: longer queries make Google silently drop the after:/before: operators.
NEG_TERMS = ['recuperação', 'rebaixa', 'calote', 'default', 'fraude', 'reestruturação', 'waiver',
             '"vencimento antecipado"']
BRAND_FIX = {"Rede Dor": "Rede D'Or", "Alianca": "Taesa", "Estado": "", "Sao": ""}
MARKET_QUERIES = {"mkt_rj": '"recuperação judicial"', "mkt_calote": 'calote OR "vencimento antecipado" debêntures'}
GENERIC = {"CIA", "CIA.", "COMPANHIA", "CONCESSIONARIA", "CONCESSIONÁRIA", "SPE", "S/A", "SA", "S.A.", "DE", "DA", "DO",
           "DOS", "DAS", "E", "EMPRESA", "HOLDING", "PARTICIPACOES", "PARTICIPAÇÕES", "ENERGIA", "ELETRICA", "ELÉTRICA",
           "TRANSMISSORA", "DISTRIBUIDORA", "GERACAO", "SANEAMENTO", "BASICO", "EST.", "ESTADO", "RODOVIAS", "LTDA",
           "BRASIL", "BRASILEIRA", "INVESTIMENTOS", "SERVICOS", "SERVIÇOS", "SOCIEDADE", "GRUPO", "NOVA", "CENTRAL"}


def brand(issuer: str) -> str:
    """'CIA SANEAMENTO BASICO EST. SP - SABESP' → 'Sabesp'; 'LOCALIZA RENT A CAR S/A' → 'Localiza'."""
    s = re.sub(r"\(.*?\)", "", issuer or "").strip()
    if " - " in s:
        s = s.split(" - ")[-1]
    words = [w for w in re.split(r"\s+", s.upper()) if w and w not in GENERIC and not re.fullmatch(r"[A-Z]{1,2}\.?", w)]
    if not words:
        return ""
    name = (words[0] if len(words[0]) >= 5 else " ".join(words[:2])).title()
    return BRAND_FIX.get(name, name)


def _fetch(query: str, a: date, b: date) -> list[dict]:
    url = (f"https://news.google.com/rss/search?q={quote_plus(query)}+after:{a:%Y-%m-%d}+before:{b:%Y-%m-%d}"
           "&hl=pt-BR&gl=BR&ceid=BR:pt-419")
    for k in range(4):
        r = get(url, retries=0)
        if r.status_code == 200:
            break
        time.sleep(15 * (k + 1))
    else:
        raise RuntimeError(f"Google News {r.status_code}")
    items = []
    for it in ET.fromstring(r.content).iter("item"):
        try:
            d = parsedate_to_datetime(it.findtext("pubDate")).date()
        except Exception:
            continue
        items.append({"date": d.isoformat(), "title": (it.findtext("title") or "").strip()})
    return items


def _add_months(m: date, n: int) -> date:
    k = m.month - 1 + n
    return date(m.year + k // 12, k % 12 + 1, 1)


def range_items(query: str, a: date, b: date) -> list[dict]:
    """Items dated in [a, b) — cached; if Google's 100-item cap is hit, split the range in months."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{hashlib.sha1(query.encode()).hexdigest()[:12]}_{a:%Y%m%d}_{b:%Y%m%d}.json"
    if path.exists() and b <= date.today():
        return json.loads(path.read_text(encoding="utf-8"))
    items = _fetch(query, a - pd.Timedelta(days=1).to_pytimedelta(), b)  # after/before are exclusive
    items = [i for i in items if a.isoformat() <= i["date"] < b.isoformat()]
    if len(items) >= 95 and _add_months(a, 1) < b:  # saturated → finer slices
        items, m = [], a
        while m < b:
            items += range_items(query, m, min(_add_months(m, 1), b))
            m = _add_months(m, 1)
    path.write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")
    return items


def month_items(key: str, query: str, m: date) -> list[dict]:
    return range_items(query, m, _add_months(m, 1))


def neg_query(brand_name: str) -> str:
    return f'"{brand_name}" ({" OR ".join(NEG_TERMS)})'


def _norm(s: str) -> str:
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()


def brand_items(b: str, start: date, end: date, sleep: float = 0.7) -> list[dict]:
    """Quarterly queries (split to months when saturated); keeps items whose title names the brand."""
    q, out, m = neg_query(b), [], start
    bn = _norm(b)
    while m < end:
        nxt = min(_add_months(m, 3), end)
        try:
            out += [it for it in range_items(q, m, nxt) if bn in _norm(it["title"])]
        except Exception:
            pass
        time.sleep(sleep)
        m = nxt
    return out


def items_frame(brands: dict[str, str], start: date, end: date, workers: int = 3, log=print) -> pd.DataFrame:
    """brands: {cnpj8: brand}. Returns one row per (cnpj8, item) with date and title."""
    from concurrent.futures import ThreadPoolExecutor
    uniq = sorted({b for b in brands.values() if b})
    with ThreadPoolExecutor(workers) as ex:
        res = dict(zip(uniq, ex.map(lambda b: brand_items(b, start, end), uniq)))
    log(f"press: {len(uniq)} brands, {sum(len(v) for v in res.values())} items")
    rows = [{"cnpj8": c, "brand": b, "date": it["date"], "title": it["title"]}
            for c, b in brands.items() if b for it in res.get(b, [])]
    df = pd.DataFrame(rows)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
    return df


def market_frame(start: date, end: date, sleep: float = 0.7) -> pd.DataFrame:
    rows = []
    for key, q in MARKET_QUERIES.items():
        m = start
        while m < end:
            try:
                rows += [{"series": key, "date": it["date"]} for it in range_items(q, m, _add_months(m, 1))]
            except Exception:
                pass
            time.sleep(sleep)
            m = _add_months(m, 1)
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df


def weekly_features(items: pd.DataFrame, keys: pd.DataFrame) -> pd.DataFrame:
    """keys: (cnpj8, asof). Counts of negative press items dated in [asof−w, asof−1] and a spike ratio."""
    out = pd.DataFrame(index=keys.index, data={"press_neg_7d": 0.0, "press_neg_30d": 0.0, "press_neg_90d": 0.0})
    if items.empty:
        out["press_spike"] = 0.0
        return out
    ev = {k: np.sort(g["date"].to_numpy(dtype="datetime64[ns]")) for k, g in items.groupby("cnpj8")}
    for cnpj8, idx in keys.groupby("cnpj8").groups.items():
        d = ev.get(cnpj8)
        if d is None:
            continue
        asof = keys.loc[idx, "asof"].to_numpy(dtype="datetime64[ns]")
        hi = np.searchsorted(d, asof - np.timedelta64(1, "D"), side="right")
        for w in (7, 30, 90):
            lo = np.searchsorted(d, asof - np.timedelta64(w, "D"), side="right")
            out.loc[idx, f"press_neg_{w}d"] = hi - lo
    base = (out["press_neg_90d"] - out["press_neg_30d"]) / 2  # avg per 30d over the prior 60 days
    out["press_spike"] = (out["press_neg_30d"] - base).clip(lower=0) / (base + 1)
    return out


def market_weekly(mkt: pd.DataFrame, weeks: pd.DatetimeIndex) -> pd.DataFrame:
    out = pd.DataFrame(index=weeks)
    for key, g in mkt.groupby("series"):
        daily = g.groupby("date").size().reindex(pd.date_range(weeks.min() - pd.Timedelta(days=400), weeks.max()),
                                                  fill_value=0)
        roll = daily.rolling(30).sum().shift(1)
        out[f"{key}_z"] = ((roll - roll.rolling(365, min_periods=90).mean()) /
                           roll.rolling(365, min_periods=90).std()).reindex(weeks, method="ffill")
    return out
