"""Offline unit tests (no network)."""
from datetime import date

import numpy as np
import pandas as pd
import pytest

from rfmonitor import bonds, isin, strategy
from rfmonitor.sources.anbima_public import parse_indexer


def test_isin_check_digit_completion():
    assert isin.normalize("USN15516AB8") == "USN15516AB83"   # Braskem 2028 (TradingView shows 11 chars)
    assert isin.normalize("BRBRKMDBS0A") == "BRBRKMDBS0A1"   # Braskem debenture BRKMA6
    assert isin.normalize("brstncntb0o7") == "BRSTNCNTB0O7"  # NTN-B 2035


def test_isin_rejects_bad_check_digit():
    with pytest.raises(ValueError):
        isin.normalize("USN15516AB84")


def test_guess_kind():
    assert isin.guess_kind("BRBRKMDBS0A1") == "debenture"
    assert isin.guess_kind("BRSTNCNTB0O7") == "tesouro"
    assert isin.guess_kind("USN15516AB83") == "eurobond"
    assert isin.b3_issuer_code("BRBRKMDBS0A1") == "BRKM"


def test_br_float():
    assert bonds.br_float("1.092,829524") == pytest.approx(1092.829524)
    assert bonds.br_float("--") is None and bonds.br_float("N/D") is None
    assert bonds.br_float("0,7126") == pytest.approx(0.7126)


def test_parse_indexer():
    assert parse_indexer("DI + 1,6%") == ("DI_SPREAD", 1.6)
    assert parse_indexer("IPCA + 4,07%") == ("IPCA", 4.07)
    assert parse_indexer("108% do DI")[0] == "DI_PCT"


def test_figi_ticker():
    assert bonds.parse_figi_ticker("BRASKM 4.5 01/10/28 REGS") == (4.5, date(2028, 1, 10))
    assert bonds.parse_figi_ticker("BNTNB 6 05/15/35") == (6.0, date(2035, 5, 15))


def test_ytm_par_and_discount():
    settle, mat = date(2026, 1, 10), date(2031, 1, 10)
    assert bonds.ytm(100, 5.0, mat, settle) == pytest.approx(5.0, abs=1e-3)   # par bond yields its coupon
    assert bonds.ytm(90, 5.0, mat, settle) > 7.0
    d = bonds.mod_duration(100, 5.0, mat, settle)
    assert 4.0 < d < 4.6


def test_interp_flat_extrapolation():
    assert bonds.interp(0.5, [1, 2], [10, 20]) == 10
    assert bonds.interp(1.5, [1, 2], [10, 20]) == 15
    assert bonds.interp(5, [1, 2], [10, 20]) == 20


def test_signal_components_no_lookahead():
    idx = pd.bdate_range("2025-01-01", periods=300)
    rng = np.random.default_rng(0)
    spread = pd.Series(300 + rng.normal(0, 5, 300).cumsum(), index=idx)
    df = pd.DataFrame({"spread_bps": spread, "duration": 3.0,
                       "stock_close": 10 * np.exp(rng.normal(0, 0.02, 300).cumsum())}, index=idx)
    full, _ = strategy.components(df)
    part, _ = strategy.components(df.iloc[:200])
    # Values up to day 200 must not change when future data is appended.
    pd.testing.assert_frame_equal(full.iloc[:200], part, check_freq=False)


def test_screener_fair_curve_and_residuals():
    from rfmonitor import screener
    rng = np.random.default_rng(1)
    dur = rng.uniform(0.5, 8, 200)
    spread = 80 + 10 * dur + rng.normal(0, 5, 200)
    g = pd.DataFrame({"duration": dur, "spread_bps": spread})
    g.loc[0, "spread_bps"] += 150  # one obviously cheap bond
    fair = screener._fair_curve(g)
    resid_z = screener._robust_z(g["spread_bps"] - fair)
    assert resid_z.idxmax() == 0 and resid_z.iloc[0] > 5
    assert abs((g["spread_bps"] - fair).iloc[1:].median()) < 5  # curve tracks the peer group


def test_composite_labels():
    comp = pd.DataFrame({"value": [3.0, -3.0, 0.0], "momentum": [0, 0, 0]})
    labels = [strategy.label_for(s) for s in strategy.composite(comp)]
    assert labels == ["BUY", "SELL", "HOLD"]
