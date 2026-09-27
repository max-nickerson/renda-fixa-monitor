"""News via Google News RSS (personal, non-commercial use) — pt-BR and English feeds."""
from __future__ import annotations

import hashlib
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime
from urllib.parse import quote_plus

from ..config import NEGATIVE_KEYWORDS
from ..http import get

FEEDS = [
    "https://news.google.com/rss/search?q={q}&hl=pt-BR&gl=BR&ceid=BR:pt-419",
    "https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en",
]


def severity(title: str) -> str:
    t = title.lower()
    return "high" if any(k in t for k in NEGATIVE_KEYWORDS) else "info"


def search(query: str, when: str = "7d") -> list[dict]:
    items, seen = [], set()
    for tpl in FEEDS:
        url = tpl.format(q=quote_plus(f'"{query}" when:{when}'))
        try:
            r = get(url)
            root = ET.fromstring(r.content)
        except Exception:
            continue
        for it in root.iter("item"):
            title = (it.findtext("title") or "").strip()
            link = it.findtext("link")
            key = title.lower()
            if not title or key in seen:
                continue
            seen.add(key)
            try:
                ts = parsedate_to_datetime(it.findtext("pubDate")).isoformat()
            except Exception:
                ts = ""
            src = it.find("source")
            items.append({
                "title": title, "url": link, "ts": ts,
                "source": src.text if src is not None else "Google News",
                "severity": severity(title),
                "uid": "news:" + hashlib.sha1(key.encode()).hexdigest()[:16],
            })
    return items
