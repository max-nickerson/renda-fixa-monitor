"""Bond math: yield-to-maturity, curve interpolation, pt-BR number parsing."""
from __future__ import annotations

import re
from datetime import date

import numpy as np


def br_float(s) -> float | None:
    """'1.092,829524' -> 1092.829524; '--', 'N/D', '' -> None."""
    if s is None:
        return None
    if isinstance(s, (int, float)):
        return float(s)
    t = str(s).strip().replace("%", "")
    if t in {"", "--", "-", "N/D", "nan"}:
        return None
    t = t.replace(".", "").replace(",", ".") if "," in t else t
    try:
        return float(t)
    except ValueError:
        return None


def years_between(d0: date, d1: date) -> float:
    return (d1 - d0).days / 365.25


def interp(x: float, xs, ys) -> float | None:
    """Linear interpolation with flat extrapolation; ignores NaNs."""
    pts = sorted((a, b) for a, b in zip(xs, ys) if a is not None and b is not None and b == b)
    if not pts:
        return None
    ax, ay = zip(*pts)
    return float(np.interp(x, ax, ay))


def cdi_equivalent_bps(kind: str, rate: float | None, pre: float | None, real: float | None) -> float | None:
    """Express any Brazilian credit yield as a 'CDI + x' spread (bps, multiplicative like ANBIMA's DI+).

    kind: DI_SPREAD (rate = x in % a.a.), DI_PCT (rate = % of CDI), PRE (nominal % a.a.), IPCA (real % a.a.)
    pre:  pre-fixed (≈ expected CDI) rate at the bond's duration, % a.a.
    real: real (IPCA) rate at the bond's duration, % a.a. (NTN-B curve)
    IPCA → nominal via breakeven (1+pre)/(1+real): the inflation term cancels, so CDI+ = (1+y)/(1+real) − 1.
    """
    if rate is None or rate != rate:
        return None
    if kind == "DI_SPREAD":
        return rate * 100
    if kind == "DI_PCT" and pre is not None:
        return (rate / 100 - 1) * pre * 100  # extra share of the expected CDI path
    if kind == "PRE" and pre is not None:
        return ((1 + rate / 100) / (1 + pre / 100) - 1) * 10000
    if kind in ("IPCA", "IGPM") and real is not None:
        return ((1 + rate / 100) / (1 + real / 100) - 1) * 10000
    return None


def price_for_spread_change(price: float, duration: float, d_spread_bps: float) -> float:
    """First-order price after a spread change: P × (1 − D × Δs)."""
    return price * (1 - duration * d_spread_bps / 10000)


def _add_months(d: date, months: int) -> date:
    m = d.month - 1 + months
    y, m = d.year + m // 12, m % 12 + 1
    day = min(d.day, [31, 29 if y % 4 == 0 and (y % 100 or y % 400 == 0) else 28, 31, 30, 31, 30,
                      31, 31, 30, 31, 30, 31][m - 1])
    return date(y, m, day)


def coupon_schedule(settle: date, maturity: date, freq: int = 2) -> tuple[list[date], date]:
    """Remaining coupon dates after `settle`, plus the last coupon date on/before it."""
    step = 12 // freq
    dates, d, k = [], maturity, 0
    while d > settle:
        dates.append(d)
        k += 1
        d = _add_months(maturity, -step * k)
    return sorted(dates), d  # d = last coupon date on/before settle


def ytm(clean_price: float, coupon_pct: float, maturity: date, settle: date, freq: int = 2) -> float | None:
    """Annualised yield (%, compounded `freq` times/yr) from a clean price quoted per 100."""
    if not clean_price or clean_price <= 0 or maturity <= settle:
        return None
    flows, prev = coupon_schedule(settle, maturity, freq)
    c = coupon_pct / freq
    nxt = flows[0]
    accrued = c * (settle - prev).days / max((nxt - prev).days, 1)
    dirty = clean_price + accrued
    t0 = (nxt - settle).days / max((nxt - prev).days, 1)  # fraction of period to next coupon

    def pv(y: float) -> float:
        r = y / 100 / freq
        return sum((c + (100 if i == len(flows) - 1 else 0)) / (1 + r) ** (t0 + i) for i in range(len(flows)))

    lo, hi = -50.0, 500.0
    if pv(hi) > dirty:
        return None
    for _ in range(200):
        mid = (lo + hi) / 2
        if pv(mid) > dirty:
            lo = mid
        else:
            hi = mid
    return round((lo + hi) / 2, 4)


def mod_duration(clean_price: float, coupon_pct: float, maturity: date, settle: date, freq: int = 2) -> float | None:
    y = ytm(clean_price, coupon_pct, maturity, settle, freq)
    if y is None:
        return None
    flows, prev = coupon_schedule(settle, maturity, freq)
    nxt = flows[0]
    t0 = (nxt - settle).days / max((nxt - prev).days, 1)
    r, c = y / 100 / freq, coupon_pct / freq
    cfs = [(t0 + i, c + (100 if i == len(flows) - 1 else 0)) for i in range(len(flows))]
    pv = sum(cf / (1 + r) ** t for t, cf in cfs)
    mac = sum(t * cf / (1 + r) ** t for t, cf in cfs) / pv / freq
    return mac / (1 + r)


_FIGI_TICKER = re.compile(r"(?P<cpn>\d+(?:\s+\d+/\d+)?(?:\.\d+)?)\s+(?P<m>\d{2})/(?P<d>\d{2})/(?P<y>\d{2,4})")


def parse_figi_ticker(ticker: str) -> tuple[float | None, date | None]:
    """'BRASKM 4.5 01/10/28 REGS' -> (4.5, 2028-01-10)."""
    m = _FIGI_TICKER.search(ticker or "")
    if not m:
        return None, None
    cpn_txt = m["cpn"]
    if " " in cpn_txt:  # '4 1/2'
        whole, frac = cpn_txt.split()
        a, b = frac.split("/")
        cpn = float(whole) + float(a) / float(b)
    else:
        cpn = float(cpn_txt)
    y = int(m["y"])
    y = y + 2000 if y < 100 else y
    return cpn, date(y, int(m["m"]), int(m["d"]))
