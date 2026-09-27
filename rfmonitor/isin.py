"""ISIN validation, check-digit completion and asset-class guessing."""
from __future__ import annotations

import re

_BODY = re.compile(r"^[A-Z]{2}[A-Z0-9]{9}$")


def check_digit(body: str) -> str:
    """ISO 6166 check digit for the first 11 characters of an ISIN."""
    digits = "".join(str(int(c, 36)) for c in body.upper())
    total = 0
    # Double every other digit starting from the rightmost (the check digit is appended after it).
    for i, d in enumerate(reversed(digits)):
        n = int(d) * (2 if i % 2 == 0 else 1)
        total += n // 10 + n % 10
    return str((10 - total % 10) % 10)


def normalize(raw: str) -> str:
    """Return a valid 12-char ISIN. Accepts 11-char bodies (as TradingView shows them)."""
    s = re.sub(r"\s", "", raw or "").upper()
    if len(s) == 11 and _BODY.match(s):
        return s + check_digit(s)
    if len(s) == 12 and _BODY.match(s[:11]) and s[11].isdigit():
        if check_digit(s[:11]) != s[11]:
            raise ValueError(f"Invalid ISIN check digit: {s}")
        return s
    raise ValueError(f"Not an ISIN: {raw!r}")


def guess_kind(isin: str) -> str:
    """Best guess from the ISIN structure; the resolver refines it with registries."""
    if not isin.startswith("BR"):
        return "eurobond"
    if isin[2:5] == "STN":
        return "tesouro"
    tag = isin[6:9]
    if tag in {"CRI", "CRA"}:
        return "cri_cra"
    if tag == "DBS":
        return "debenture"
    return "other"


def b3_issuer_code(isin: str) -> str | None:
    """Brazilian ISINs embed the 4-letter B3 issuer code (BR[BRKM]DBS...)."""
    if isin.startswith("BR") and isin[2:5] != "STN":
        return isin[2:6]
    return None
