# Deep lab: does deep learning beat P4 / P4 + quality at picking debentures?

Script: `research/run_deep_lab.py` (about 41 min on CPU, PyTorch 2.14 CPU). Rebuild:

```
PYTHONPATH=. PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe research/run_deep_lab.py [--quick]
```

`--quick` uses fewer seeds (about 22 min). The log is in `data/deep.log`. All numbers are in `deep_results.json`. The script reads the selection lab's cached panels (`data/history/sellab_panel.pkl`, `sellab_returns.pkl`), so run `run_selection_lab.py --rebuild` first if those caches change.

## Short answer

**No.** No deep-learning variant beats P4 at any horizon, and every one of them loses to **P4 + excl. worst-quintile quality**. Several of those losses are significant after Holm correction.

- The models do learn something real. Out-of-sample IC is +0.11 to +0.20 at 3m and 6m, and the placebo gives about 0.
- That signal, however, is weaker than raw carry alone. The IC of `cdi_bps` on the same dates is +0.20 / +0.24 / +0.24.
- Turning the model score into a top-20% portfolio does worse than the simple rules.

**Keep P4 + quality.**

## Excess vs universe (%/yr, gross; all variants on the same model cohorts)

| H (cohorts) | P4 | P4+Q | GBM sellab | MLP top20 | ListNet top20 | GRU top20 | P4∩MLP top50 | Placebo top20 |
|---|---|---|---|---|---|---|---|---|
| 3m (46, 2022-09→) | +1.12 | **+1.58** | +1.22 | +0.77 | +1.16 | +0.76 | +1.20 | +0.23 |
| 6m (40, 2022-12→) | +0.65 | **+1.20** | +0.74 | +0.74 | +0.64 | +0.19 | +0.55 | +0.07 |
| 12m (28, 2023-06→) | +0.53 | **+0.79** | +0.16 | +0.36 | +0.34 | +0.16 | +0.52 | −0.04 |

- Net of costs, the ranking is the same. The models turn over no more than P4: book cost is about 0.55 / 0.30 / 0.16 %/yr at 25 bps. Net 25 / 50 figures are in the JSON.
- On the full P4 sample (2022-01→), P4 earns +1.23 / +1.10 / +1.07. The models cannot trade their first 8 to 17 months because they need at least 6 months of fully realised labels.

**IC** (Spearman per date, NW t):

| Model | 3m | 6m | 12m |
|---|---|---|---|
| MLP | +0.11 (2.4) | +0.14 (1.6) | +0.06 (0.4) |
| ListNet | +0.14 (3.4) | +0.20 (5.9) | +0.11 (1.5) |
| GRU | +0.11 (2.8) | +0.12 (2.0) | +0.04 (0.3) |
| GBM on the same features | +0.20 | +0.15 | −0.01 |
| placebo | +0.01 | +0.01 | −0.01 |
| cdi_bps alone | +0.20 | +0.24 | +0.24 |

## Paired vs P4 / P4+quality

The Holm family covers 36 tests: 3 DL models × 2 portfolio forms × 2 bases × 3 horizons.

- **vs P4**: the differences run from −0.45 to +0.21 %/yr, and none is positive and significant. The best result is GRU P4∩top50 at 3m: +0.21, t 1.2, Holm p = 1. That gain appears only in 2022–23; it is −0.02 in 2024–26.
- **vs P4+quality**: all 18 DL comparisons are negative, from −0.25 to −1.01 %/yr. The significant ones after Holm are:

| Comparison | Diff (%/yr) | t | Holm p |
|---|---|---|---|
| MLP P4∩top50, 6m | −0.65 | −4.4 | 0.0003 |
| GRU P4∩top50, 6m | −0.62 | −4.3 | 0.0005 |
| GRU top20, 6m | −1.01 | −4.0 | 0.002 |
| ListNet P4∩top50, 6m | −0.57 | −3.5 | 0.016 |
| GRU top20, 12m | −0.63 | −3.4 | 0.022 |

- **Halves (2022–23 / 2024–26)**: the DL models look best in the first half. For example, MLP top20 at 3m earns +1.67 in 2022–23 and +0.29 in 2024–26, and GRU top20 earns +2.10 and +0.05. They then decay, while P4+Q holds up in both halves: +1.97 / +1.37 at 3m.
- **GBM on the same features as the MLP** (control) is the best ML model at 3m: +1.67 vs U, and +0.09 vs P4+Q (t 0.2). It is negative at 12m. The extra features help trees more than nets, but the model still does not beat P4+Q.

## Placebo and stability

- **Placebo** (target shuffled across bonds within each date before training, 3 seeds): IC is about 0 (+0.012 / +0.008 / −0.014), and top20 excess vs U is about 0 (+0.23 / +0.07 / −0.04). The placebo P4∩top50 matches P4 (+1.10 / +0.74 / +0.58), so the P4∩model variants get essentially all their return from P4.
- **Seed dispersion** is small next to the effects measured:

| Model | Seed std of IC | Seed std of top20 excess (%/yr) | Pairwise rank corr of predictions |
|---|---|---|---|
| MLP | 0.006–0.012 | 0.04–0.12 | 0.80–0.92 |
| ListNet | 0.009–0.016 | 0.09–0.15 | — |
| GRU | 0.013–0.021 | 0.04–0.12 | — |

  The ensembles are stable. They are simply not better.
- **Overfitting signature**: the early-stopping validation IC (0.15–0.38) is far above the out-of-sample IC (0.04–0.20), and the gap grows with the horizon. Validation dates sit next to the training dates, and at 12m their label windows overlap almost completely. 20–38% of fits stop at epoch 0 or 1.

## Equity (`deep_equity.png`)

The chart shows a daily tranche book at 6m with 25 bps cost, on the same 40 cohorts.

- ListNet ends highest (+2.7% cumulative), ahead of P4+Q (+1.8) and P4 (+1.0).
- MLP (+0.2), GRU (−0.7) and P4∩MLP (−0.3) trail GBM (+0.6).
- ListNet's lead comes almost entirely from avoiding the Dec-2024 drawdown. The cohort-average statistic, which is the tested quantity, puts ListNet at +0.64 vs P4+Q's +1.20. The path and the cohort mean differ because the book compounds daily and weights tranches differently. **Do not read the chart as evidence for ListNet.**

## Method

The following are identical to the selection lab: panel, patched hedged returns, entry rule, targets, `dok` completeness filter, portfolio construction (equal weight, 10% issuer cap), book-level cost, NW t (lag = H/21), strict fundamentals availability. The helpers are copied, because `run_selection_lab.py` runs everything at import.

**Training**
- Walk-forward with an expanding window. Training uses only samples with `lab_end{H} <= decision position`, meaning entry + H is already realised.
- Retraining happens every 3 months (GRU every 6), once at least 6 label dates exist.
- Early stopping uses the last 20% of training dates as validation, with a 1-date purge, and scores on the mean per-date IC.
- The target is the per-date rank of y{H}.

**Features** (52 tabular)
- All continuous features are per-date rank-gauss standardised, with missing-value indicators: cdi_bps, resid_bps, resid_z, dur, age, contract, press 7d/30d, distress, fact, eq_ret 1w/4w, log days since downgrade, 10 fundamentals and composite quality.
- Binary flags: incent, press, distress, downgrade in the last year, covered, is_parent.
- mkt_mom_21, plus kind and peer one-hots.

**Models**
- **MLP**: 64-32 hidden units, dropout 0.25, AdamW (wd 1e-3), 5 seeds, ensemble by rank-average.
- **ListNet**: the same network, trained with a within-date softmax cross-entropy loss (target × 8).
- **GRU**: hidden size 16, run over 26 weekly steps, concatenated with a static MLP branch. It uses 3 seeds and subsamples 20k rows per epoch to fit the time budget. The per-step inputs are:
  - present flag;
  - Δcdi_bps vs now;
  - resid_bps and resid_z;
  - weekly patched return, assigned to the end of each segment so nothing after the decision leaks through a trading gap;
  - weekly trade count.

## Data and training issues

- **Short sample**: there are 28 to 46 evaluable monthly cohorts, and first predictions come only in 2022-09, 2022-12 and 2023-06 (3m / 6m / 12m). 12m has just 28 overlapping cohorts, which is about 2.3 independent observations per year, so any 12m model claim is fragile.
- **Heavy cross-sectional correlation** (about 700 bonds a date, few issuers). It means the effective training sample is closer to "dates × issuers" than to 30k rows. The nets hit their best validation score within a few epochs.
- The known selection-lab caveats carry over: survivorship after the last trade, and the uncovered-bond edge.
- The sequence model adds nothing over the static features. The recent-path channels (spread moves, trades, weekly returns) carry little incremental information at these horizons.
