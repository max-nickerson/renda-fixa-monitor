# Structural / equity-implied credit (nightly slug `structural_credit`)

**Bottom line:** within P4+Q, equity-implied risk is a real avoid signal. Dropping the worst-quintile names does add about +0.35%/yr. The structural model itself adds almost nothing, though: plain equity drawdown or volatility does about as well. The bond-vs-equity divergence works only inside the book, and the lead-lag is statistically real but economically small. No variant survives Holm across the 12 variants tried (smallest Holm p = 0.14).

All numbers come from harness v4: monthly decisions 2022-01 to 2025-12, 126-bday tranches, 25 bps, issuer cap 10%, NW lag 6, 48 months. The sealed holdout was run once at the very end, only for the frozen best variant.

## Rerun

```
PYTHONPATH=. PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/Scripts/python.exe research/nightly/structural_credit/build_data.py
PYTHONPATH=. PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/Scripts/python.exe research/nightly/structural_credit/run.py            # pre-2026 only
PYTHONPATH=. PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/Scripts/python.exe research/nightly/structural_credit/run.py --holdout  # + sealed 2026 (already done once)
```

- `build_data.py` takes about 2 min.
- `run.py` takes about 4 min.
- The logs are in `data/history/nightly/structural_credit/run*.log`.

## Data built (new, point-in-time)

`data/history/nightly/structural_credit/structural_daily.pkl` has one row per (ticker, date) for 153 tickers (the panel's `eq_ticker` set), 2020-06 to 2026-09. 182k rows have a DD. Columns:
- `mcap`, `sigE`, `gross_debt`, `st_debt`, `cash`, `book_eq`, `rf`;
- `dp` (default point), `V`, `sigV`, `dd`, `pd1y`, `merton_spread_bps`, `dd_naive`;
- `cg_spread_bps`, `cg_pd5y`, `mkt_lev`, `net_mkt_lev`.

**Market cap:**
- **Share count:** shares outstanding (total minus treasury) from CVM ITR/DFP `composicao_capital`. This file ships inside the already-cached zips. A count is available at its filing's `DT_RECEB` (the latest version, so it is strict).
- **Splits:** splits are neutralised with `q = COTAHIST unadjusted close / brapi split-adjusted close` at each filing's reference date.
- **Anchor:** the level is anchored to brapi `marketCap / price` today, which also handles units and class mixes. Units without brapi data use the CVM share count divided by the unit composition (TAEE11 = 3, ENGI11 = 5, and so on).
- **Issuance index:** filing-to-filing share ratios are chained into an issuance index.
  - A ~1000x jump with no price-split signature is treated as a CVM "thousands of shares" unit change and ignored.
  - Other jumps outside 1/30 to 30 are treated as data errors.
  - The index is smoothed with a 3-filing rolling median, which kills one-filing spikes.
- **Guard:** market caps with a price-to-book above 40 or below 0.02 are set to NaN.

**Ticker to CNPJ:** the listed company's CNPJ comes from CVM FCA `valor_mobiliario` 2021–2026, downloaded to `fca_*.zip` in the cache folder.

**Debt:** `fundamentals_pit.pkl` at `available_date_strict`, for the listed company's CNPJ; brapi rows tagged with the ticker are the fallback. Rows more than 460 days stale are dropped.

**Equity vol:** 50/50 blend of 252d and 63d volatility of daily log `adj_close`, with a floor of 10%.

**Risk-free rate:** CDI.

**Models:**
- **Merton/KMV:** 1y horizon, default point = ST debt + 0.5 × LT debt. V and σV are solved jointly by vectorised Newton/fixed point, and DD uses the risk-neutral drift.
- **Bharath–Shumway naive DD.**
- **CreditGrades:** 5y spread with L̄ = 0.5, λ = 0.3, R = 0.5, and D = gross debt.

**Panel signal caches:**
- `signals_M.pkl` and `signals_W.pkl` hold (codigo, cnpj8, day, eq_ticker, `sc_*`).
- `panel_M_sc.pkl` is the full monthly panel with the `sc_*` columns added.
- To add the columns to any harness panel: `research/nightly/structural_credit/signals.py: signals(panel)`.

**Coverage:**
- 63% of universe bond-months have a DD, covering 223 issuers and 131 tickers.
- Only 40% of P4+Q bond-months have one: high-carry names skew unlisted.

## Results

### IC (listed subset, per-date Spearman; `pic` = partial IC after ranking out carry `cdi_bps`)

| feature | IC fwd_63 (t) | IC fwd_126 (t) | partial vs carry, fwd_126 (t) |
|---|---|---|---|
| Merton DD | −0.025 (−0.7) | −0.035 (−0.8) | **+0.054 (5.1)** |
| CreditGrades log-spread | +0.020 | +0.032 | −0.050 (−4.9) |
| ΔDD 21d | +0.039 (2.4) | +0.036 (1.9) | +0.042 (3.4) |
| divergence residual (bond wide vs equity-implied) | +0.128 (1.9) | +0.173 (2.1) | +0.102 (4.4) |
| eq_r63 (for reference) | +0.048 (2.5) | +0.070 (2.6) | +0.067 (4.2) |
| eq_dd252 (for reference) | +0.018 | +0.023 | +0.084 (6.0) |

**How to read it:** raw DD is useless because it mostly proxies carry. Holding carry fixed, safer equity-implied credit earns more, since carry does not fully pay for equity-signalled risk.

### Books vs P4+Q (12 variants plus P4 and U in the Holm family; 25 bps)

| variant | exCDI | vs U (t) | vs P4+Q (t) | Holm p | 22–23 / 24–25 vs P4+Q | vol | maxDD | names |
|---|---|---|---|---|---|---|---|---|
| **A3 P4+Q minus worst-quintile CreditGrades spread** (best) | 2.57 | +1.47 (3.0) | **+0.35 (1.65)** | 1.00 | +0.17 / +0.53 | 0.93 | −0.74 | 82 |
| A1 P4+Q minus worst-quintile DD | 2.56 | +1.45 (3.0) | +0.34 (1.62) | 1.00 | +0.17 / +0.51 | 0.94 | −0.74 | 82 |
| A6 minus worst DD or worst ΔDD | 2.57 | +1.47 | +0.35 (1.58) | 1.00 | +0.18 / +0.52 | 0.96 | −0.68 | 78 |
| A2 minus DD < 2 | 2.42 | +1.32 | +0.20 (1.65) | 1.00 | +0.15 / +0.26 | 0.93 | −1.03 | 97 |
| A8 minus worst divergence, measured inside the book | 2.48 | +1.37 | +0.26 (**2.58**, raw p 0.010) | 0.14 | +0.30 / +0.22 | 1.00 | −0.90 | 95 |
| C2 control: minus worst eq_dd252 (no model) | 2.56 | +1.46 | +0.34 (1.84) | 0.72 | +0.19 / +0.50 | 0.95 | −0.72 | 85 |
| C1 control: minus worst eq_vol63 (no model) | 2.51 | +1.41 | +0.29 (1.39) | 1.00 | +0.11 / +0.48 | 0.95 | −0.72 | 80 |
| A7 P4 minus worst DD (DD instead of fundamentals Q) | 2.34 | +1.24 | +0.12 (0.5) | 1.00 | −0.16 / +0.41 | 0.98 | −1.08 | 99 |
| A4 minus worst ΔDD 63d | 2.22 | | −0.00 | 1.00 | | | | |
| A5 minus worst divergence, universe quintile (drops ~0 names: degenerate) | 2.22 | | +0.00 | 1.00 | | | | |
| B1 P4+Q plus top divergence names (buy side) | 2.19 | | −0.03 | 1.00 | | | | |
| B2 standalone: listed, top-30% divergence | 2.16 | +1.06 | −0.06 (−0.2) | 1.00 | +0.40 / −0.52 | 1.17 | −1.06 | 50 |
| P4+Q (benchmark) | 2.22 | +1.12 | 0 | | | 1.00 | −1.32 | 103 |

**Robustness of the best rule (A3):**

| check | result |
|---|---|
| 50 bps | +0.34 vs P4+Q (t 1.58) |
| rec40 | +0.34 (t 1.62) |
| cohort 6m | +0.35%/yr (t 1.9) |
| cohort 12m | +0.43%/yr (t 5.4; overlapping, so the t is inflated) |
| removed names | earned 0.92% less over 6m than the kept names (t 2.0) |
| random-exclusion placebo (drop the same number of listed P4+Q names at random, 20 draws) | +0.14 mean, p95 +0.19; actual +0.35 beats all 20 |

Part of the placebo mean is itself positive, because dropping any listed name tilts the book toward unlisted names.

**Sealed holdout, 2026-01 to 2026-09, run once for the frozen A3:** +0.39%/yr vs P4+Q (NW t 3.1 over 9 months; +3.7%/yr vs U). The direction is consistent, but 9 months is too short to count on.

### Lead-lag (weekly; fresh marks both weeks; week fixed effects; SEs clustered by issuer)

**Equity leads bond spreads:** Δspread(t→t+1) on the issuer stock's past weekly returns gives coefficients of −14.7, −12.5, −10.8 and −8.0 bps per 100% return at lags 0 to 3 (t −3.0, −2.7, −2.9, −2.0). Summed, a −10% equity week predicts roughly +4.6 bps of spread widening over the next 4 weeks. The 4-week version: eq_r21 gives −19 bps per 100% (t −2.9).
- **Direct listings:** the effect is concentrated here, at −19 (t −3.4).
- **Parent listings:** no effect (+2.3, t 0.4).
- **HY names:** the coefficient is larger (−24) but noisy.

**Bond spreads do not lead equity:** t = 0.5.

**Model DD changes add nothing:** ΔDD21 does not predict 4-week spread changes (t −0.7).

**Mark behaviour:** strong own-spread mean reversion, −0.41 per week (t −42), consistent with noisy or stale marks.

### Big losses: did DD flag them ahead of time?

**Across the universe (listed rows with a DD, 2022–25):**

| predictor | AUC for fwd_126 < −10% | AUC for fwd_126 < −5% |
|---|---|---|
| −DD | 0.80 | 0.63 |
| CreditGrades spread | 0.79 | 0.61 |
| −eq_dd252 | 0.82 | 0.64 |
| eq_vol63 | 0.80 | 0.65 |
| carry (cdi_bps) | 0.74 | 0.52 |
| −f_quality | 0.65 | 0.57 |

The < −10% sample has 129 bond-month events across 30 issuers.

**Inside P4+Q, what we actually hold:**
- The DD AUC is **0.91** for < −10% (17 events) and **0.83** for < −5% (50 events).
- The comparable AUCs are carry 0.59 / 0.55, fundamentals quality 0.80 / 0.79, and eq_r63 0.83 / 0.71.
- The DD worst quintile caught 94% of both event sets.
- In a logit on carry, DD, eq_r63 and quality (clustered by issuer), DD is significant for < −5% (z −2.5). For < −10%, eq_r63 takes over (z −3.0 vs DD −1.7).

**Issuer level, first < −10% event:**
- 50% of the 30 issuers were in the DD worst quintile at decision time, and 63% in the worst 40%.
- Clear early flags, with DD percentile at decision (1 = safest):
  - AMER3, 2022-11: DD 1.57, percentile 0.9%, two months before the fraud disclosure;
  - BHIA3, 2023-02: percentile 2%;
  - ONCO3, 2024-07: percentile 2%;
  - KRSA3, 2022-11: percentile 2%;
  - SIMH3, 2024-09: percentile 3%.
- Misses: CPLE3 in 2022, CMIN3, CSAN3 and MOTV3 all had high DD. These look like duration or rate or mark-driven moves, not credit events.

**Blind spot:** only 52% of all < −10% events have a DD, and 72% are listed. The unlisted tail (Aegea, SPEs and similar) is invisible to any equity-implied signal.

Charts: `equity_total_return.png` (total return CDI × (1+excess) vs CDI, U, P4, P4+Q, IDA-DI and Ibovespa, plus cumulative excess vs U), `cum_excess.png` and `dd_big_losses.png`.

## Insights
1. **The value is avoidance, not selection.** Equity-implied risk is priced into carry only partially. Holding carry fixed, safer names earn more (partial IC +0.05, t 5). Avoiding the worst equity-implied quintile inside P4+Q adds about +0.35%/yr, lowers vol, and nearly halves maxDD (−1.32 to −0.74).
2. **The structural model does not beat simple equity measures.** A 52-week drawdown filter (C2: +0.34, t 1.84) does the same as Merton DD or CreditGrades. Their Spearman correlations with DD are 0.60 (drawdown) and −0.87 (volatility). What adds information is the equity market; the balance-sheet leverage layer adds little (market leverage alone: AUC 0.72, book IC ≈ 0).
3. **DD complements fundamentals quality rather than replacing it.** P4 minus worst DD (A7, +0.12 vs P4+Q) is worse than P4+Q plus a DD filter. The two stack (A1/A3 ≈ +0.35 on top of Q's +0.33).
4. **The divergence trade only works on the short side, inside the book.** Buying "bond wide vs equity-implied" names loses (B1 −0.03, B2 −0.06, with a sign flip across halves). Dropping in-book names whose bond is tight relative to equity-implied risk is the most stable variant (A8 +0.26, t 2.6, both halves positive), but it does not survive Holm (p 0.14).
5. **Equity leads bond marks by 1 to 4 weeks, but only for direct listings and only by a few bps.** The effect is far too small to trade after 25 bps; fast reaction does not pay, consistent with earlier labs. Bonds never lead equity.

## Caveats
- **No multiple-testing survivor:** 12 variants (plus P4 and U) are in the Holm family, and the best Holm p is 0.14 (A8); A3 has Holm p 1.0. The gain over P4+Q is about the size of the P4 to P4+Q step, and the t-stats come from smooth, stale marks (see the harness caveats).
- **Survivorship:** rec40 is barely changed, but bonds that stop trading earn 0.
- **Coverage:** only 40% of P4+Q exposure has an equity-implied signal. The rules only ever remove listed names, so the book tilts toward unlisted, rarely marked issuers, which is exactly where survivorship bias lives. The random-exclusion placebo (+0.14) partly captures this tilt.
- **Market cap reconstruction is approximate:**
  - brapi's split adjustment of `close` and today's brapi marketCap are used as anchors (mechanical, price-neutral, but not strictly as-of).
  - A few tickers with serial reverse splits and recapitalisations (SEQL3, AMER3 2024, AMBP3) remain noisy.
  - Banks and insurers have no debt field and are excluded.
- **Parent-listed issuers:** DD is the parent's (group) risk, not the SPV's.
- **Debt:** gross debt excludes IFRS-16 leases and other liabilities, so the default point is understated for lease-heavy retailers.
- **Sealed holdout:** 9 months (2026-01 to 2026-09), consulted once for A3 only. It does not decide anything.
