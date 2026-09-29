# ml_ranking: modern tabular models as rankers and as overlays on P4+Q

**Short answer:** no model beats P4+Q significantly once you correct for the 51 variants tried.

- **Best variant:** `lgb_peer:tilt`. It weights P4+Q names by 0.5 + the percentile of a LightGBM score trained on each bond's return in excess of its peer group.
  - Pre-2026 it earns **+0.17 %/yr vs P4+Q** (NW t 2.18, raw p 0.03, **Holm p = 1.0** across 51 variants). The result is the same at 50 bps (+0.17) and under rec40 (+0.17).
  - All of the gain comes in 2024–25: +0.01 in 2022–23 and +0.33 in 2024–25.
  - Sealed holdout (2026-01 to 2026-09, looked at once, variant frozen beforehand): **+0.11 %/yr vs P4+Q** (t 1.2, 7 of 9 months positive).
- **The main result:**
  - As standalone top-20% rankers, every ML model loses to P4+Q by −0.5 to −1.5 %/yr, mostly in 2022–23.
  - As **overlays inside P4+Q**, several are weakly positive: +0.1 to +0.17 %/yr.
  - A trivial non-ML control does comparably well per unit of risk. Tilting P4+Q weights by carry percentile (`ctrl_carry:tilt`) gives +0.06 %/yr with t 5.2 and Holm p < 0.001, because its tracking error is tiny.

All numbers use harness v4: monthly decisions, 126-bday overlapping tranches, 25 bps round trip, 10% issuer cap, pre-2026 unless stated.

## Method

- **Features:** `features.py` builds 92 point-in-time features.
  - **Harness panel columns (72):** carry, residual and richness, duration, press and CVM filing counts, weekly trading extras, CVM fundamentals and quality, issuer equity returns / vol / drawdown / ADTV, market state, and kind and sector codes.
  - **20 new features:**
    - bond path features from the harness return matrix (only moves realised by the decision close): past 21/63/126d return, 63d vol, 126d max drawdown, worst day, share of stale days;
    - carry vs the same-date sector and kind medians, carry × duration, per-date ranks;
    - issuer aggregates: number of bonds, mean resid_z, mean and worst past return, worst drawdown;
    - sector momentum and sector size.
- **Walk-forward:** `models.py`.
  - Predictions start 2022-01 (48 monthly dates) and models are refit every 6 months.
  - A model used at date p trains only on universe rows whose `lab_end_126 <= refit date`, so the embargo equals the horizon.
- **Hyperparameters:** tuned by nested time-series CV with optuna, once, at the first 2024 refit.
  - Inner validation is the last 6 labelled dates, with inner training embargoed before them. The objective is mean per-date Spearman IC (AUC for the classifier).
  - The tuned parameters are reused for 2024–25. 2022–23 use defaults, since there are too few labels for nested CV.
  - Budget: 6 trials, CatBoost excluded (see caveats). Holdout rows were never used.
- **Models:**

| name | learner | target |
|---|---|---|
| lgb_reg | LightGBM regression | fwd_126 winsorised, demeaned per date |
| lgb_mono | same, monotone constraints (carry +, cdi_pct −, resid +) | same |
| lgb_rank | LightGBM LambdaRank (quintile relevance) | per-date quintile |
| xgb_pair | XGBoost rank:pairwise | per-date quintile |
| cat_yeti | CatBoost YetiRank | per-date rank |
| enet_int | elastic net on per-date rank-Gaussian features + 66 pairwise interactions of 12 core drivers | raw |
| hier | ridge + empirical-Bayes issuer and sector random effects on training residuals (hierarchical shrinkage) | raw |
| lgb_peer | LightGBM regression | **excess over the same-date peer-group mean** |
| lgb_loss | LightGBM classifier | P(fwd_126 < −3%), ~7% base rate; used as a veto |
| ens | equal rank average of the 8 return models | |
| stack | non-negative weights re-fit each date on *past* OOS predictions with realised labels | |
| ctrl_carry / ctrl_resid | **non-ML controls**: raw `cdi_bps` / `resid_z` | |

- **Portfolio forms**, 4 per model (3 for the classifier):
  - `veto10` / `veto20`: P4+Q minus names in the bottom 10% / 20% of the universe by score;
  - `tilt`: P4+Q with weight ∝ 0.5 + score percentile;
  - `top20`: a standalone top 20% by score.
- **Total:** 51 variants, all in one `H.compare` call for Holm.

## Results

### Information coefficient (per-date Spearman vs fwd_126, 42 dates with complete labels)

| model | IC universe | NW t | IC inside P4+Q | t |
|---|---|---|---|---|
| lgb_reg | −0.01 | −0.1 | 0.21 | 5.9 |
| lgb_mono | −0.01 | −0.1 | 0.21 | 5.9 |
| lgb_rank | 0.06 | 0.6 | 0.21 | 5.5 |
| xgb_pair | 0.07 | 0.7 | 0.22 | 6.9 |
| cat_yeti | −0.01 | −0.1 | 0.15 | 2.7 |
| enet_int | −0.01 | −0.2 | 0.15 | 3.1 |
| hier | −0.05 | −0.6 | 0.09 | 4.1 |
| **lgb_peer** | **0.16** | **4.8** | **0.25** | 4.8 |
| lgb_loss (sign flipped) | −0.00 | −0.0 | −0.06 | −1.1 |
| ens | 0.04 | 0.4 | 0.23 | 7.5 |
| stack | 0.05 | 0.6 | 0.25 | 4.9 |
| **ctrl: cdi_bps** | **0.25** | 3.6 | **0.33** | 11.2 |
| **ctrl: resid_z** | 0.23 | **14.6** | 0.30 | 9.2 |

### Books vs P4+Q (25 bps, pre-2026, 51 variants; top 12 and a selection)

| variant | exCDI | exU | t vs U | vs P4+Q | t | Holm p | 22–23 | 24–25 | Sharpe | maxDD |
|---|---|---|---|---|---|---|---|---|---|---|
| **lgb_peer:tilt** | 2.39 | 1.29 | 3.1 | **+0.17** | 2.18 | 1.00 | +0.01 | +0.33 | 2.21 | −1.38 |
| stack:tilt | 2.38 | 1.28 | 3.3 | +0.16 | 1.57 | 1.00 | −0.00 | +0.32 | 2.12 | −1.38 |
| stack:veto20 | 2.37 | 1.26 | 3.0 | +0.15 | 2.89 | 0.18 | +0.10 | +0.19 | 2.25 | −1.38 |
| lgb_peer:veto20 | 2.34 | 1.24 | 2.8 | +0.12 | 2.61 | 0.41 | +0.05 | +0.19 | 2.27 | −1.38 |
| lgb_reg:tilt | 2.32 | 1.22 | 2.9 | +0.10 | 1.23 | 1.00 | +0.07 | +0.13 | 2.06 | −1.35 |
| lgb_mono:tilt | 2.31 | 1.21 | 2.9 | +0.09 | 1.12 | 1.00 | +0.08 | +0.10 | 2.05 | −1.35 |
| lgb_loss:veto10 | 2.30 | 1.20 | 2.4 | +0.08 | 1.39 | 1.00 | +0.16 | −0.01 | 2.38 | −1.32 |
| ctrl_resid:tilt | 2.29 | 1.19 | 2.6 | +0.08 | 3.23 | 0.06 | +0.10 | +0.06 | 2.16 | −1.32 |
| ctrl_carry:tilt | 2.28 | 1.18 | 2.5 | +0.06 | **5.22** | **<0.001** | +0.06 | +0.06 | 2.18 | −1.34 |
| xgb_pair:tilt | 2.26 | 1.16 | | +0.04 | 0.71 | 1.00 | | | | |
| cat_yeti:tilt | 2.25 | 1.15 | | +0.03 | 0.33 | 1.00 | | | | |
| lgb_peer:top20 | 1.72 | 0.62 | | −0.50 | −0.68 | 1.00 | −1.79 | +0.79 | | |
| stack:top20 | 1.36 | 0.25 | | −0.86 | −0.79 | 1.00 | −2.50 | +0.78 | | |
| lgb_rank:top20 | 0.88 | −0.23 | | −1.34 | −1.36 | 1.00 | −2.24 | −0.45 | | |
| **P4+Q** | 2.22 | 1.12 | 2.4 | 0 | | | | | 2.13 | −1.39 |
| P4 | 1.89 | 0.79 | | −0.33 | | | | | 1.81 | −1.56 |

The full table is in `results.json` → `table_25bps`.

**Robustness of the top 5** (difference vs P4+Q in %/yr, each against P4+Q run on the same settings):

| variant | 50 bps | rec40 | hold 63 | cohort 126d (t) |
|---|---|---|---|---|
| lgb_peer:tilt | +0.17 (t 2.1) | +0.17 | +0.23 | +0.14 (1.9) |
| stack:tilt | +0.16 (1.5) | +0.16 | +0.20 | +0.11 (1.2) |
| stack:veto20 | +0.14 (2.8) | +0.15 | +0.14 | +0.13 (2.5) |
| lgb_peer:veto20 | +0.12 (2.5) | +0.12 | +0.11 | +0.12 (2.4) |

**Placebo for the best.** A random tilt, with uniform weights 0.5–1.5 over the same P4+Q names, gives a mean of −0.01 and a p95 of +0.06 over 30 draws. The actual +0.17 is above every draw.

**Sealed holdout, once, frozen `lgb_peer:tilt`** (2026-01 to 2026-09, 9 months):

| | value |
|---|---|
| exCDI | +0.19 %/yr (P4+Q: +0.08) |
| vs P4+Q | **+0.11 %/yr**, t 1.2, hit 7 of 9 months |
| vs U | +3.45 %/yr (2026 was a bad year for the universe) |

**SHAP** (`shap_best.png`). TreeSHAP of the final pre-2026 `lgb_peer` fit (2025-10), measured on 2025 out-of-sample rows. The top drivers are:
- carry × duration, carry vs kind median, equity drawdown vs 252d high (lower drawdown helps), years to maturity, raw carry, duration;
- IDA-DI 63d momentum (interaction), carry vs sector, fundamental quality (+), equity 252d return (+), equity vol (−).

In other words, it is mostly **carry measured within peers, with duration, plus the issuer's equity health.**

Charts:
- `equity_total_return.png`: total return CDI × (1 + excess), with CDI, U, P4, P4+Q, IDA-DI and Ibov;
- `cum_excess.png`: cumulative excess vs U and cumulative paired difference vs P4+Q;
- `shap_best.png`.

## Insights

1. **The universe-level IC of the "raw" ML models is about 0 because they learn the wrong regime early.**
   - Trained on 2021 labels, every raw-target model learned "low carry is good". In 2022-03 to 2022-10 they posted ICs of −0.4 to −0.65, while carry alone had +0.6 to +0.7. They recovered from 2023.
   - The cross-section *across* peer groups (kind and duration buckets) is regime-driven and flips sign.
   - Demeaning the target within the peer group (`lgb_peer`) removes this. It is the only model with a positive IC on every sub-period: 0.16 overall, t 4.8.
   - This is the most useful modelling lesson: **predict within-peer excess, not raw return.**
2. **Nothing beats carry on raw ranking power.**
   - `cdi_bps` alone has IC 0.25 (0.33 inside P4+Q), and `resid_z` alone has IC 0.23 with NW t 14.6, the most stable signal of all. No model matches either.
   - Inside P4+Q, which is already the top 30% by carry, carry still ranks the future at 0.33.
3. **Overlays can only nudge P4+Q.** The best ML tilt adds +0.17 %/yr, and all of it comes in 2024–25. Removing the bottom 10% of the universe almost never touches P4+Q names, so veto10 is a near no-op for any carry-correlated model.
4. **The large-loss classifier is useless as a veto.**
   - Its universe AUC and IC are about 0. Its IC sign flips between years: it is negative for 2023 decisions and positive for 2024.
   - As a veto it adds +0.08 (t 1.4), all in 2022–23. This agrees with the prior finding that reacting to news does not pay.
5. **Learning-to-rank objectives do not help.** LambdaRank, pairwise and YetiRank are no better than plain L2 regression here: overlays within ±0.05. They optimise the top of the list, while an overlay needs the whole ordering inside P4+Q.
6. **Hierarchical issuer shrinkage hurts.** Past issuer alpha on realised labels does not persist: IC inside P4+Q falls to 0.09. Monotone constraints change nothing (lgb_mono ≈ lgb_reg).
7. **Stacking helps only a little.** NNLS stacking on past OOS predictions ends up weighting 92% `lgb_peer` and 8% `lgb_rank` by 2025-12 and lifts the ensemble's IC inside P4+Q to 0.25. Books are not better than `lgb_peer` alone.
8. **The one significant result is a trivial non-ML tilt.** `ctrl_carry:tilt` (weight ∝ 0.5 + carry percentile within P4+Q) gives +0.06 %/yr with t 5.2, Holm p < 0.001. The economic size is tiny, and the t-stat is inflated by very low tracking error and smooth marks. It is still a free, robust refinement for a combiner.

## Reusable signals

- `research/nightly/ml_ranking/run.py::signals(panel=None, name="ml_stack_pct")` returns `(codigo, cnpj8, date, value)`: per-date percentiles, walk-forward, known at the close of `date`, monthly dates 2022-01 to 2025-12.
- Available names: `ml_lgb_peer_pct` (the best model), `ml_stack_pct`, `ml_ens_pct`, `ml_lgb_rank_pct`, `ml_hier_pct`, and `ml_lgb_loss_pct` (higher = safer).
- Cache: `data/history/nightly/ml_ranking/signals_ml_ranking.pkl`.
- Raw walk-forward predictions per model are in `data/history/nightly/ml_ranking/pred_<model>.pkl`, including `lgb_peer`, with keys `pred` and `params`.
- The feature matrix, with the 20 new PIT features, is in `features_M.pkl`. Build it with `features.build("M")`.

## Caveats

- **Smooth, stale marks** inflate every t-stat, and paired t-stats at tracking errors of ~0.1 %/yr most of all (the carry-tilt t of 5.2). The halves show the ML gains are **2024–25 only**.
- **Early training sample is small.** The first models (2022) train on 5 to 10 labelled dates, which explains the 2022 inversion. A longer pre-2022 history, which does not exist here, might change the standalone result.
- **Reduced tuning budget.** On this shared machine LightGBM with 2 threads was 3× slower than with 1 (thread contention), so everything ran single-threaded:
  - one nested tuning point (2024), 6 trials;
  - CatBoost YetiRank at defaults (~60 s per fit);
  - refits every 6 months.
  More tuning is unlikely to close a 0.08 IC gap to carry.
- **Selection bias.** The best variant was chosen as the maximum of 51 pre-2026 results, hence Holm p = 1. The holdout (+0.11, t 1.2) is consistent in sign but short: 9 months.
- **Veto design.** The veto overlays cut universe-level percentiles, so for carry-correlated scores they rarely bind. A veto on within-P4+Q percentiles was not tested, to avoid growing the Holm family late.
- **Execution assumptions.** All results use the harness tranche engine: 126d tranches, execution at the next trade, 0 return after the last print. rec40 is reported for the top 5.

## Rerun

```
cd renda-fixa-monitor
PYTHONPATH=. PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/Scripts/python.exe research/nightly/ml_ranking/run.py all
# one-time sealed-holdout report of the frozen variant (already done; rewrites results.json['holdout']):
PYTHONPATH=. PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=1 .venv/Scripts/python.exe research/nightly/ml_ranking/run.py holdout lgb_peer:tilt
```

The `preds` stage takes ~20 min (CatBoost ~8 min of it) and `eval` takes ~6 min. Caches are in `data/history/nightly/ml_ranking/`. Delete `pred_*.pkl` to refit.
