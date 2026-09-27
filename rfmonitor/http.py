"""Shared HTTP helpers: a client with a browser-like User-Agent, retries and a file cache."""
from __future__ import annotations

import hashlib
import time
from pathlib import Path

import httpx

from .config import CACHE_DIR, USER_AGENT

_client: httpx.Client | None = None


def client() -> httpx.Client:
    global _client
    if _client is None:
        _client = httpx.Client(
            headers={"User-Agent": USER_AGENT}, timeout=httpx.Timeout(60, connect=15), follow_redirects=True
        )
    return _client


def get(url: str, retries: int = 2, **kw) -> httpx.Response:
    last: Exception | None = None
    for attempt in range(retries + 1):
        try:
            r = client().get(url, **kw)
            if r.status_code in (429, 500, 502, 503, 504) and attempt < retries:
                time.sleep(2 * (attempt + 1))
                continue
            return r
        except httpx.HTTPError as e:  # network hiccup
            last = e
            time.sleep(2 * (attempt + 1))
    raise last or RuntimeError(url)


def cached_bytes(url: str, max_age_hours: float, name: str | None = None) -> bytes | None:
    """Download with an on-disk cache. Returns None on 404 (e.g. file for a holiday)."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path: Path = CACHE_DIR / (name or hashlib.sha1(url.encode()).hexdigest())
    if path.exists() and (time.time() - path.stat().st_mtime) < max_age_hours * 3600:
        data = path.read_bytes()
        return data or None
    r = get(url)
    if r.status_code == 404:
        path.write_bytes(b"")  # remember misses too
        return None
    r.raise_for_status()
    path.write_bytes(r.content)
    return r.content
