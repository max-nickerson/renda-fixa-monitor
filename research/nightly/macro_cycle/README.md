# macro_cycle: macro and credit-cycle timing and sizing overlays on P4+Q

**Bottom line: a clean negative result.** Across 20 overlay variants, none beats P4+Q (tranche 126, 25 bps) before 2026 after Holm adjustment. The best was IDA-DI 63-day momentum in/out. It came out at +0.02%/yr vs P4+Q (t 0.05) and roughly halved the drawdown. Nearly all of its edge came from one episode, the Americanas crisis in Jan–Feb 2023. It then lost 1.04%/yr vs P4+Q in the sealed 2026 holdout (t −2.3).

Macro ML models (logistic, ridge, LightGBM, HMM, ensemble) lose 0.4–2.7%/yr against P4+Q. Two things drive this:
- the macro-to-credit relationships flip sign between 2010–21 and 2022–25;
- switching costs eat whatever timing skill is left.

The only row that "beats" P4+Q is static leverage. That is carry scaling, not skill, and the result assumes funding at CDI + 50 bps.

Harness v4 was used throughout: monthly decisions, 126-bday tranches and the harness overlay convention. Under that convention a value known at close t trades at t+1. Switching costs |Δexposure| × exposure × cost/2. My overlay code reproduces `H.backtest(..., overlay="ida")` exactly (diff 0.0).

## Rerun
```
PYTHONPATH=. PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/Scripts/python.exe research/nightly/macro_cycle/data.py   # rebuild PIT macro panel (~6 min, network)
PYTHONPATH=. PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/Scripts/python.exe research/nightly/macro_cycle/run.py    # ~15 min cold (walk-forward fits), ~80 s warm
```
`run.py` calls `data.build()`, which reuses the cached panel if it is less than 24 h old. Walk-forward predictions are cached in `data/history/nightly/macro_cycle/preds.pkl` for 12 h.

## New data: PIT macro panel (`data/history/nightly/macro_cycle/macro_daily.pkl`)
- **Shape:** a daily business-day panel from 2008-06 to 2026-09 with 79 columns. Each row holds only values known at the close of that date.
- **Raw cache:** downloads are cached in `raw/`.

| block | series | source | PIT rule |
|---|---|---|---|
| Credit indices | IDA-DI, IDA-IPCA, IDA-Geral: excess over CDI (5/21/63/126/252d), 252d drawdown, 63d vol | ANBIMA xlsx (read-only from `data/history`) | Close of d. The harness adds a further 1-day lag. |
| Rates | Selic target and its 126/252d changes; CDI | BCB SGS 432, 12 | Effective date |
| Curve | Nominal 1/3/5y, real 5y, slope 5y−1y, 5y breakeven, pre1y − Selic (priced policy path), 21/63d changes | Tesouro Direto morning rates (Tesouro Transparente) | +1 bday |
| Inflation | IPCA surprise vs Focus median, 3m sum; Focus 12m-ahead IPCA and its 63d change; real policy rate (Selic − Focus 12m) | IBGE/BCB SGS 433; BCB Olinda Focus monthly | IPCA: exact IBGE release + 1d from 2018; before 2018, the 15th of M+1. Focus: survey Friday + 3 days. |
| FX / equity | USD/BRL PTAX (21/63d, vol); Ibovespa excess (21/63/126d, drawdown, vol) | SGS 1; Yahoo ^BVSP | Same day |
| Global risk | VIX, BAA−10y, UST 10y/2y, Brent, HYG/IEF (US HY proxy), EMB/IEF (EM sovereign proxy), EWZ; US HY OAS (from 2023-10 only) | Yahoo, FRED | +1 bday (the US closes after B3) |
| Brazil country risk | Nominal 5y − UST10 − (Focus IPCA − 2%). No free CDS series exists, so this is a proxy. | Derived | Same as inputs |
| Primary issuance | Debenture registrations in R$ over a rolling 91d window; yoy log change | CVM `oferta_distribuicao` + `oferta_resolucao_160` | Registration date |
| Market credit news | Count of 30d "recuperação judicial" / "calote" news items, z-score (from 2020-11) | `press_market.pkl` | Publication date |

## Method
- **Target.** IDA-DI excess over CDI over the 21 bdays from t+2, which matches the overlay convention. It is available from 2009, so the training data includes the 2015–16 recession and the 2020 Covid crash, not just the harness's 2022–25.
- **Walk-forward.**
  - Monthly refits from 2012 onwards.
  - Training rows are 2010→, and each row's label must end before the refit date.
  - Training rows are sampled weekly.
  - Features are standardised on the training rows only.
- **Models:**
  - rules: `mom63` (IDA-DI 63d excess > 0), and `vote` (≥ 2 of these stress flags → out: VIX > 25, USD/BRL 21d > +5%, IDA-DI 21d < 0, Ibovespa −15% off its high);
  - `logit` (P(fwd > 0), C = 0.1);
  - `ridge` (α = 50);
  - `gbm` (LightGBM, 200 trees, 7 leaves);
  - `hmm` (2-state Gaussian HMM on IDA-DI 5d excess, IDA vol, Ibovespa 21d and VIX; FILTERED forward-only probabilities, refit every 6 months);
  - `ens` (vote, or mean percentile, of logit, ridge, gbm, hmm and mom21).
- **Exposure rules:**
  - `inout` (0/1);
  - `size`: 0–1.5×, computed as clip(2.5 × pct − 0.25), where pct is the percentile of today's prediction among the model's own past out-of-sample predictions. It is rounded to 0.25 steps.
  - Every exposure is updated weekly at the Friday close.
  - Leverage is funded at CDI + 50 bps (100 bps as a sensitivity check).
- **Implementations:**
  - `ov_*`: a whole-book overlay, the harness semantics. It is optimistic because it trades the whole book instantly at the mark.
  - `entry_*`: implementable. The exposure only scales each **new** monthly P4+Q tranche, with no forced selling. This uses `H._targets` and `H._tranche_book`.
- **Spread sizing (2021+ only):** the universe median CDI+ spread, as an expanding z-score, sets exposure = 1 + 0.5z.

## Results: pre-2026, 48 months, 25 bps
There are 20 variants. Holm adjustment covers all of them. `timing_alpha` compares each row with the same book held at a constant exposure equal to that row's average exposure (an ex-post diagnostic).

| variant | exCDI | vs U | vs P4+Q | t | Holm p | vol | Sharpe | maxDD | 22–23 / 24–25 vs P4+Q | avg expo | timing alpha (t) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **P4+Q** | 2.22 | +1.12 | 0 | | | 1.04 | 2.13 | −1.39 | | 1.00 | |
| P4+Q + IDA21 regime (current) | 1.74 | +0.64 | −0.48 | −1.25 | 0.98 | 0.94 | 1.86 | −1.15 | +0.32 / −1.28 | 0.79 | −0.01 (−0.0) |
| ov_mom63_inout (best) | 2.24 | +1.13 | +0.02 | 0.05 | 1.00 | 0.82 | 2.72 | −0.50 | +0.39 / −0.35 | 0.78 | +0.51 (1.8) |
| ov_vote_inout | 1.83 | +0.73 | −0.39 | −2.11 | 0.33 | 1.11 | 1.64 | −1.35 | −0.41 / −0.37 | 0.90 | −0.16 |
| ov_logit_inout | 1.81 | +0.70 | −0.41 | −2.36 | 0.22 | 1.07 | 1.69 | −1.51 | −0.28 / −0.55 | 0.93 | −0.25 |
| ov_ridge_inout | −0.22 | −1.32 | −2.43 | −4.08 | 0.00 | 0.86 | −0.25 | −2.20 | −1.29 / −3.58 | 0.62 | −1.59 (−3.7) |
| ov_gbm_inout | 0.95 | −0.15 | −1.27 | −2.91 | 0.06 | 0.96 | 0.98 | −1.27 | −0.66 / −1.87 | 0.69 | −0.57 |
| ov_hmm_inout | 0.44 | −0.66 | −1.78 | −2.85 | 0.07 | 0.78 | 0.57 | −0.86 | −1.35 / −2.21 | 0.50 | −0.66 |
| ov_logit_size | 0.74 | −0.36 | −1.48 | −2.28 | 0.25 | 0.70 | 1.05 | −0.71 | −0.31 / −2.65 | 0.56 | −0.50 |
| ov_ridge_size | −0.45 | −1.55 | −2.67 | −4.10 | 0.00 | 1.21 | −0.37 | −3.07 | −1.47 / −3.87 | 0.80 | −2.22 (−3.9) |
| ov_gbm_size | 0.09 | −1.01 | −2.12 | −4.22 | 0.00 | 1.21 | 0.08 | −2.50 | −1.41 / −2.84 | 0.84 | −1.76 (−3.9) |
| ov_hmm_size | 1.25 | +0.14 | −0.97 | −1.81 | 0.42 | 1.10 | 1.13 | −1.31 | −0.38 / −1.56 | 0.85 | −0.64 |
| ov_ens_inout | 1.20 | +0.10 | −1.01 | −2.13 | 0.33 | 1.07 | 1.12 | −1.27 | −0.25 / −1.78 | 0.78 | −0.53 |
| **ov_ens_size (pre-declared primary)** | 1.25 | +0.15 | −0.97 | −2.69 | 0.11 | 0.96 | 1.30 | −1.18 | −0.27 / −1.67 | 0.91 | −0.78 (−2.4) |
| ov_spread_size | 1.13 | +0.03 | −1.09 | −2.07 | 0.33 | 1.12 | 1.01 | −2.16 | −0.06 / −2.11 | 0.73 | −0.48 |
| entry_spread_size | 1.59 | +0.48 | −0.63 | −1.29 | 0.98 | 1.13 | 1.40 | −2.09 | +0.11 / −1.38 | 0.71 | +0.02 |
| entry_ens_size | 2.12 | +1.01 | −0.10 | −0.56 | 1.00 | 1.07 | 1.97 | −1.69 | −0.02 / −0.19 | 0.89 | +0.13 (0.6) |
| entry_ens_inout | 1.83 | +0.73 | −0.39 | −2.03 | 0.33 | 0.91 | 2.01 | −1.36 | −0.41 / −0.36 | 0.76 | +0.15 (0.7) |
| entry_gbm_size | 2.11 | +1.01 | −0.11 | −0.35 | 1.00 | 1.13 | 1.87 | −2.05 | −0.24 / +0.03 | 0.86 | +0.19 (0.5) |
| static 1.25× (reference) | 2.66 | +1.55 | +0.44 | 2.48 | 0.18 | 1.30 | 2.04 | −1.78 | +0.26 / +0.62 | 1.25 | 0 |
| static 1.5× (reference) | 3.09 | +1.99 | +0.88 | 2.48 | 0.18 | 1.56 | 1.98 | −2.16 | +0.51 / +1.24 | 1.50 | 0 |

**Robustness (difference vs P4+Q, %/yr):**

| variant | 50 bps (t) | +5d extra lag | +10d extra lag | funding CDI + 100 |
|---|---|---|---|---|
| IDA21 regime | −1.19 (−2.1) | −0.72 | −0.56 | −0.48 |
| ens_size | −2.11 (−5.0) | −0.90 | −1.05 | −1.05 |
| ens_inout | −1.83 (−2.7) | −1.13 | −1.09 | −1.01 |
| gbm_size | −4.01 (−5.9) | −2.01 | −2.20 | −2.24 |
| logit_inout | −0.77 (−3.0) | −0.36 | −0.51 | −0.41 |
| hmm_inout | −2.42 (−3.1) | −1.62 | −1.58 | −1.78 |
| spread_size | −1.13 (−2.3) | −1.06 | −1.04 | −1.15 |
| static 1.25× | +0.34 (1.9) | +0.44 | +0.44 | +0.32 |

**Placebo.** 200 circular shifts of each exposure path, so the average exposure and the number of switches are kept. The reported figure is the actual difference vs the placebo mean (p = share of placebos ≥ actual):
- ens_size: −0.97 vs −1.32 (p 0.18);
- ens_inout: −1.01 vs −1.25 (p 0.14);
- gbm_size: −2.12 vs −2.37 (p 0.32);
- IDA21: −0.48 vs −1.15 (p 0.00).

IDA momentum does carry real timing information relative to random timing. It still does not pay for its lower exposure and its costs.

**Long-sample out-of-sample test on IDA-DI itself.** This is an overlay on the index, with a 25 bps switch cost:

| model | ann. 2012–21 | Sharpe | maxDD | ann. 2022–25 | Sharpe | maxDD | switches/yr |
|---|---|---|---|---|---|---|---|
| buy & hold | 0.70 | 0.39 | −5.90 | 0.70 | 0.49 | −3.24 | 0 |
| mom21 daily (P4 regime) | 0.59 | 0.69 | −2.67 | 0.45 | 0.42 | −2.28 | 7 |
| **mom63 in/out** | **0.99** | **1.61** | **−0.83** | **1.08** | **1.21** | **−1.03** | 1.2 |
| ens_inout | 0.68 | 1.25 | −1.60 | −0.23 | −0.16 | −3.24 | 7.5 |
| gbm_inout | 0.61 | 1.19 | −1.18 | −0.38 | −0.31 | −2.45 | 8.5 |
| hmm_inout | −0.56 | −0.76 | −7.01 | −0.58 | −0.64 | −2.58 | 6.8 |
| gbm_size / ens_size | −0.62 / 0.25 | | | −1.03 / −0.03 | | | 17–27 |

**Sealed holdout, 2026-01 to 2026-09, 9 months, reported once.**
- P4+Q: +0.08%/yr.
- ov_ens_size (pre-declared primary): −0.04 (−0.12 vs P4+Q, t −0.35).
- ov_mom63_inout (best pre-2026): −0.96 (**−1.04 vs P4+Q, t −2.28**). It sat out in the recovery.
- IDA21 regime: +0.35 (+0.27 vs P4+Q, t 0.85).

## Insights
1. **Macro signals change sign across regimes.** Non-overlapping monthly Spearman correlation with IDA-DI's forward 21d excess, 2010–21 vs 2022–25:

   | feature | 2010–21 | 2022–25 |
   |---|---|---|
   | VIX level | +0.18 | −0.34 |
   | Brazil country-risk proxy | +0.10 | −0.37 |
   | USD/BRL 63d | −0.23 | +0.20 |
   | EMB/IEF | +0.24 | −0.10 |
   | Ibovespa 21d | +0.22 | −0.04 |
   | IDA-DI vol63 | +0.43 | −0.12 |

   Only a few features keep their sign: the index's own momentum (x5 +0.43/+0.44, x21 +0.40/+0.36, x63 +0.37/+0.30), Brent 63d (+0.24/+0.32) and US HY (HYG/IEF, +0.32/+0.18). A model fitted on 2010–21 therefore learns relationships that break in 2022–25. Ridge and GBM are the worst rows in the table: their timing alpha is −1.6 to −2.2%/yr, with t −3.7 to −3.9.
2. **The one robust predictor is the index's own momentum, and it is mostly an artifact of smooth marks.** IDA-DI returns are autocorrelated because marks are stale. The momentum signal predicts the index well, but a debenture book cannot be switched at those marks. Adding a 5–10 day extra lag hurts IDA21 by a further 0.08–0.24%/yr.
3. **Slow momentum (63d, 1.2 switches/yr) is the only timing rule that works on the long sample.**
   - 2012–21: Sharpe 1.61 vs 0.39 buy & hold, maxDD −0.8% vs −5.9%.
   - It also works on P4+Q before 2026: vol 0.82 vs 1.04, maxDD −0.50 vs −1.39, exposure-matched timing alpha +0.51 (t 1.8).
   - However, the cumulative difference vs P4+Q comes almost entirely from Jan–Feb 2023, the Americanas crisis (see `cum_excess.png`).
   - It lost 1.04%/yr in the 2026 holdout.
   - Read it as a drawdown-control tool with one good episode, not as alpha.
4. **Continuous sizing is worse than in/out.** Weekly changes in size mean 13–27 switches a year, which at 25–50 bps per round trip costs 0.5–2%/yr. Every `size` variant is negative, and at 50 bps ens_size falls to −2.11 (t −5.0).
5. **Implementable entry-sizing (scale only new tranches) removes most of the cost damage.** Timing alpha is about +0.13 to +0.19 (t < 0.8) with no skill. entry_ens_size and entry_gbm_size end at −0.10 vs P4+Q. This is the only honest way to express a macro view in an illiquid book, and there is no view worth expressing.
6. **Leaning into wide spreads did not pay.**
   - Buying more when the universe's median CDI+ spread is high (spread_size) lost 1.1%/yr as an overlay and 0.6%/yr at entry.
   - 2024–25 spreads compressed while carry fell, and the z-score kept the book under-invested.
   - Carry-level timing alpha ≈ 0.
7. **Leverage is the only lever that "beats" P4+Q, and it is carry scaling, not skill.**
   - Static 1.25× gives +0.44 (t 2.5) and 1.5× gives +0.88, with funding at CDI + 50.
   - MaxDD scales roughly with leverage: −1.78 and −2.16 vs −1.39.
   - It depends entirely on the ability to repo debentures at around CDI + 50 bps, which is not realistic for most HY names.
8. **Issuance and IPCA surprises carry no timing signal.** Their correlations are all below 0.04 except issuance yoy in 2022–25 (+0.15).

## Caveats
- **Few independent episodes.** 48 pre-2026 months hold about 2 credit episodes: Q1-2023 Americanas and late-2025. The long IDA sample adds 2015–16 and 2020, but it is the index, not the book.
- **Optimistic overlay mechanics.** Overlay variants switch the whole book instantly at the mark. Real exits in debentures are slow and costly, so every `ov_*` row is optimistic, including the harness IDA21 regime.
- **Smooth, stale marks** inflate Sharpe and t (see the harness caveats). Timing rules based on momentum partly exploit this autocorrelation.
- **Assumed funding cost.** The leverage funding rate is an assumption. No DI-hedged repo market for debentures was modelled.
- **Data coverage.** US HY OAS on FRED covers only about 3 years, so HYG/IEF was used as the proxy. There is no free Brazil CDS series. The pre-2018 IPCA availability date is a conservative approximation. Tesouro Direto morning rates are retail-desk rates.
- **Variant counting.** 20 variants were tried; the Holm adjustment covers all of them. The model hyperparameters (C, α, GBM size, the 2.5 × pct − 0.25 mapping, weekly updates) were fixed a priori and not tuned.

## Reusable outputs
- `data/history/nightly/macro_cycle/macro_daily.pkl`: the PIT daily macro panel. The index is the date, and each value is known at that date's close. Build it with `research/nightly/macro_cycle/data.py:build()`.
- `data/history/nightly/macro_cycle/macro_overlay_signals.pkl`: a wide table with a date index and these columns: `ens_size, ens_inout, gbm_size, hmm_inout, logit_inout, spread_size, ens_pct`. Each is an exposure known at the close of that date, walk-forward from 2012. Pass it to `harness.backtest(overlay=series)`, which applies the 1-day lag.
- `run.py:apply_overlay(base, ov, cost_bps, fund_bps, extra_lag)`: the harness overlay plus funding on the levered part.
- `run.py:entry_sized(signal, ov)`: sizes only new tranches.

## Files
- `run.py`, `data.py`
- `results.json`: all tables, the IDA long sample, univariate correlations, GBM importances, the holdout and data coverage
- `equity_total_return.png`, `cum_excess.png`, `ida_long_sample.png`, `exposures.png`
- `run.log`
