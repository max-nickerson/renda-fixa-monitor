# w2_p7_adversarial_verification: independent check of P7 (research/nightly/combined)

**Verdict: FRAGILE.** P7's equity-health (EQH) screen is **not** a placebo artefact. It survives every placebo, parameter, lag and cost check here, and is larger under honest assumptions. But:
- it fails the multiplicity bar for a search of about 280–440 variants;
- its NW t is inflated by negative autocorrelation (the iid t is 1.72);
- the gain comes from a handful of blow-ups: the top 5 months, or the top 10 issuers.

The carry/liquidity construction half of P7 is not new: it is portfolio_construction's result.

**Corrected best-estimate increment over P4+Q: about +0.25 to +0.35%/yr**, down from the published +0.50. This is the 5%-winsorised diff (+0.34) and the matched-control alpha shrunk for the search; it is not significant after multiplicity.

All numbers are pre-2026: 48 months, harness v4, monthly decisions, 126-day tranches, 25 bps. The only holdout number is the recomputed published one.

## 1. Reproduction
- P7 vs P4+Q is **+0.500%/yr (NW t 3.29)**: exact. vs U it is +1.62. At 50 bps it is +0.48.
- My generic re-implementation (`lib.mk`) matches the daily series exactly (max abs diff 0.0).
- 2026 holdout, recomputed only: **+0.68%/yr (t 2.25, 9 months)**, as published.

## 2. Matched placebos: the published placebo was contaminated
40 draws each, same count dropped, identical carry/liquidity construction.

| placebo | overlap with the truly flagged names | placebo mean vs P4+Q | P7 − placebo (NW t) |
|---|---|---|---|
| A: P7's own (random listed P4+Q) | ~40% | +0.18 (p95 +0.28) | +0.32 (2.1) |
| B/C: carry decile × sector strata, bond / issuer, drawn from all listed names | **94–97%** | +0.43 / +0.42 | +0.07 / +0.08 (invalid: these re-draw the flagged names) |
| G: coarse permutation (carry tercile) | 46% | +0.13 | +0.37 (2.7) |
| **D/E/F: zero-overlap controls (non-flagged comparable names; bond dec×sector, bond tercile, issuer q5×sector)** | 0% | −0.01 / −0.05 / +0.01 | **+0.51 / +0.55 / +0.49 (2.3–2.4)** |
| H: zero-overlap, EW construction, vs the screen-only book | 0% | −0.14 | +0.52 (2.9) |

**Insight.** The "mechanical +0.16 from dropping listed names" in combined's README is almost all contamination: 40% of that placebo's random drops are the flagged names themselves. Dropping comparable non-flagged listed names does nothing (about 0). So the placebo-adjusted alpha of the screen is about +0.5, not +0.3.

One caveat on the controls: flagged names carry more (280 vs about 245 bps), and the non-flagged pool cannot fully match that. Dropping lower-carry controls should, if anything, *help* the placebo, so this does not flatter P7.

## 3. Bootstraps and concentration
- **6-month block bootstrap** of the monthly diff: 90% CI [+0.25, +0.80], p(≤0) = 0.0001. With 3-month blocks, p = 0.005.
- **Issuer-clustered cohort bootstrap** (cohort diff +0.64): 90% CI [+0.18, +1.11], p = 0.010. Two-way (issuers × 6-month blocks): p = 0.037.
- **The monthly diff is fat-tailed:** skew 2.2, kurtosis 10.8, median month +0.02%. It also has **lag-2 autocorrelation of −0.47**. The Bartlett NW variance is therefore *smaller* than the iid variance: **iid t 1.72 vs NW t 3.29**. The 5%-winsorised diff is +0.34 (t 1.83).
- **Months:**

  | months excluded | diff vs P4+Q (t) |
  |---|---|
  | top 1 | +0.31 (2.2) |
  | top 2 | +0.18 |
  | top 3 | +0.09 (0.46) |
  | top 5 | −0.00 |

  Leave-one-month-out ranges from +0.31 to +0.57.
- **Issuers:**
  - The top 1 / 3 / 5 / 10 issuers are 19% / 43% / 62% / 94% of the cohort gain.
  - The largest are Oncoclínicas (0.12%/yr), Americanas (0.08), and then oil & gas, textiles and toll-road names.
  - Dropping the top 1 / 3 / 5 / 10 issuers from both books (backtest) gives +0.39 / +0.27 / +0.22 / +0.10 (t 3.3 / 2.3 / 2.0 / 0.9).
  - Leave-one-issuer-out, over the top 15 and bottom 5 issuers, ranges from +0.39 to +0.59.

## 4. Decomposition and portfolio_construction
| component | %/yr (t) |
|---|---|
| construction on P4+Q alone (carry weights, liquidity cap R$500m) | +0.19 (1.5) |
| EQH screen on EW P4+Q | +0.38 (2.1) |
| screen on the carry book | +0.32 (1.7) |
| interaction | −0.06 (−1.3) |
| total | +0.50 (3.3) |

- Regressing (P7 − P4+Q) on the two single-component books gives both betas ≈ 1.0, R² 0.93 and alpha −0.07. **P7 is exactly additive: screen plus construction, nothing emergent.**
- **Comparison with portfolio_construction:**

  | comparison | %/yr (t) |
  |---|---|
  | P7 vs PC top-25 carry @R$500m | +0.29 (1.2) |
  | P7 vs PC mv_emp_g20_fund500m (recommended) | **−0.07** |
  | P7 vs PC top-25@500m + EQH screen | −0.07 |

- Spanning alpha of P7 − P4+Q: +0.43 (t 2.6) on the PC top-25@500m book, and +0.28 (t 2.0) on the PC mv@500m book.

**Conclusion:** the construction leg was already known. P7's only new content is the EQH screen, and bolting it onto the PC fund book gives the same result.

## 5. Multiplicity
- **Variants tried**, from each `research/nightly/*/results.json`: 277 excluding bias_audit, and 439 including its 162 audit backtests.
- **Minimum |t| for Bonferroni / Holm (first step) at 5%:** 3.29 at N = 50, 3.74 at N = 277, 3.86 at N = 439.
- **P7:** NW t 3.29 gives Holm/Bonferroni p 0.28–0.44 at N = 277–439, and only 0.05 at N = 50. The conservative iid t is 1.72.
- **Deflated Sharpe** (monthly SR 0.25, null variance of t = 1, skew and kurtosis adjusted) is 0.07 at N = 277 and 0.05 at N = 439.
- **Result: not significant after the search.**

## 6. Pre-declared new variant: flag vote
The vote counts four flags: EQH bottom 20%, zoo bottom 20% within P4+Q, NegEvents, and supply.

| variant | vs P4+Q (t) | vs P7 (t) | 50 bps vs P4+Q |
|---|---|---|---|
| **FV2 carry (primary), drops ~24 of 103 names** | **+0.60 (4.1)** | +0.10 (1.3) | +0.58 |
| FV3 carry (drops ~7) | +0.43 (4.2) | −0.07 | +0.41 |
| FV2 EW | +0.48 (2.6) | −0.02 | +0.46 |
| FV3 EW | +0.27 (2.4) | −0.23 (−2.1) | +0.26 |

- FV2 minus its zero-overlap control placebo is +0.70 (t 2.6). The stage-6 placebo overlapped 86% with the flagged names and is invalid.
- FV2 is marginally better than P7 but not significantly so (+0.10, t 1.3). It inherits the same multiplicity problem, and the ≥3-flag finding that motivated it was seen in-sample.

## 7. Sensitivity (P7 vs P4+Q, same assumptions on both sides)
- **EQH quantile:** q = 0.10 / 0.20 / 0.30 gives +0.45 / +0.50 / +0.48.
- **Single components:** eq_dd252 alone +0.47, eq_r126 alone +0.42, eq_vol63 alone +0.35. All have t ≥ 2.
- **Fundamentals lag** for Q, in both books: 60 / 90 / 120 days gives +0.49 / +0.49 / +0.52. The lag hurts P4+Q itself (−0.04 / −0.07 / −0.14 vs lag-0 P4+Q), not the increment.
- **Realistic costs and survivorship:**

  | assumption | diff vs P4+Q |
  |---|---|
  | liquidity-bucket costs | +0.47 |
  | harsh survivorship | +0.76 |
  | both, plus the holiday fix | +0.72 |

- **Fully honest** (costs, harsh survivorship, holiday fix and lag-90 Q): **+0.68 (NW t 4.2, iid t 2.5)**. P7 is +1.51 over CDI against +0.83 for P4+Q and +0.18 for U. Minus the zero-overlap placebo it is +0.69 to +0.70 (t 3.2–3.3).

  The harsh scenario helps P7 by construction, because EQH avoids names that later go silent. That is real avoidance only if the 40% recovery is right.

## Why "fragile" and not "robust"
The sign is very stable. Every check above is positive:
- all placebos with low overlap;
- the q / component / lag grids;
- cost and survivorship scenarios;
- both halves;
- the 2026 holdout.

The magnitude and the significance are not:
- 94% of the gain is from 10 issuers;
- five months carry it all;
- the iid t is 1.7;
- after a search of about 300–440 variants the needed t is about 3.8.

Treat it as a sensible tail-avoidance rule worth about +0.25–0.35%/yr in expectation. It is not a proven alpha.

## Files and rerun
- `lib.py` holds the pre-declared rules, the generic signal, the placebos (including zero-overlap controls), the lagged Q and the cohort attribution.
- `verify.py` holds the stages, cached in `data/history/nightly/p7_adversarial_verification/stage_*.pkl`.
- `report.py` writes `results.json`, `summary_grid.csv` and the charts:

  | chart | contents |
  |---|---|
  | `equity_total_return.png` | total return and cumulative excess vs universe |
  | `placebo_distributions.png` | placebo draws vs the actual result |
  | `cum_diff_decomposition.png` | decomposition of the gain |
  | `concentration_bootstrap.png` | issuers, months and bootstraps |
  | `sensitivity_grid.png` | the sensitivity grid |
  | `equity_incl_holdout.png` | holdout shaded, recomputed only |

```
PYTHONPATH=. PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/Scripts/python.exe research/nightly/p7_adversarial_verification/verify.py   # ~25 min
PYTHONPATH=. PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe research/nightly/p7_adversarial_verification/report.py
```
