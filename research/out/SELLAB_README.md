# Selection lab: do issuer fundamentals help pick bonds beyond carry + relative value?

Script: `research/run_selection_lab.py` (about 10 min). Rebuild:

```
PYTHONPATH=. PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe research/run_selection_lab.py [--rebuild] [--quick]
```

`--rebuild` rebuilds the cached panels (`data/history/sellab_panel.pkl`, `sellab_returns.pkl`). `--quick` skips the ML models and runs 10 placebo permutations instead of 100. The log is in `data/sellab.log`. All tables are in `sellab_results.json`.

## Resposta curta

**Yes, but the gain is modest, and it only works as a screen on top of carry.** Fundamentals have almost no predictive power on their own. Their raw IC is about 0, because weak issuers pay more carry. **Once carry, RV, duration and peer group are held fixed**, better quality predicts better forward returns: the composite-quality partial IC is +0.04 / +0.05 / +0.08 at 3 / 6 / 12 months, with Newey-West t of 4.2 / 3.7 / 6.5.

The practical form is **P4 minus the worst quintile of composite quality** (bonds without fundamentals coverage stay in). Versus P4 it adds:

| horizon | P4 − universe | P4+quality − universe | increment | t NW | Holm p | placebo null mean | increment net of placebo | 2022–23 / 2024–26 |
|---|---|---|---|---|---|---|---|---|
| 3m | +1.23%/yr | +1.61 | **+0.38** | 3.6 | 0.007 | +0.12 | +0.27 (p<0.01) | +0.30 / +0.45 |
| 6m | +1.10 | +1.54 | **+0.44** | 3.2 | 0.024 | +0.11 | +0.34 (p<0.01) | +0.31 / +0.57 |
| 12m | +1.07 | +1.30 | **+0.24** | 3.2 | 0.021 | +0.08 | +0.15 (p<0.01) | +0.24 / +0.23 |

- The single-metric screens give similar results: worst-quintile interest coverage adds +0.49%/yr at 3m and 6m, ND/EBITDA adds +0.19 to +0.34, and rising leverage adds +0.19 to +0.33. Most of these survive Holm correction across the 24 tests.
- **Placebo** (fundamentals shuffled across issuers within each date, 100 runs): the real result beat every run. However, about a third of the raw gain appears under the placebo too, because any screen that drops covered bonds shifts weight toward uncovered bonds, and those did better (see below). The "net of placebo" column is the honest estimate: **roughly +0.15 to +0.35%/yr**.
- **Cost**: the screen barely changes turnover. Book-level cost is 0.33%/yr at 25 bps, versus 0.30 for P4, at 6m. Excess vs universe net of costs is +1.40 at 25 bps and +1.26 at 50 bps, versus P4's +0.98 and +0.86.
- **Strict vs non-strict availability dates** (`available_date_strict` vs `available_date`): the results are practically identical (JSON `check_nonstrict`).
- **The cleanest test**, restricted to covered bonds only (P4 covered-only, with vs without the screen), gives +0.31 / +0.44 / +0.17%/yr (t 1.8 / 2.5 / 1.9). The placebo is significant (p<0.01, null about 0), but it **does not survive Holm** (p 0.17 to 0.36).

**What did NOT work:**
- **Expected-return score** (carry − walk-forward logistic PD × LGD + convergence × duration): the EL term changes almost nothing. It adds +0.01 to +0.05%/yr versus the same score without EL. The score also beats P4 on neither horizon measured per cohort.
- **Walk-forward Ridge/GBM**: adding fundamentals helps each model by +0.2 to +0.4%/yr (t 1.6 to 2.4), but **none survives Holm**. The gain is concentrated in 2024–26 and is negative or zero in 2022–23 at 6m/12m. Worse, the models themselves lose to the simple P4 rule: GBM+fund at 6m is +0.74 vs universe, against P4's +1.10. The ML route is not worth it.
- **Within-peer IC** of fundamentals is about 0 (quality at 6m: +0.008, t 0.5). The information shows up only relative to carry: at equal carry, the better issuer wins.

## Existing signals (full universe, 6m)

IC: cdi_bps +0.28 (t 4.7), resid_bps +0.22 (t 13.6), dur +0.01 (t 0.1). Carry and RV remain the main drivers.

## Uncovered bonds (~17% of the universe, 31% of P4)

These are mostly unlisted SPVs and subsidiaries. They beat covered bonds by about +1.5%/yr (t 2.8 to 3.8), **even after controlling for carry, RV and peer group** (rank coefficient t 2.0 to 6.3). They have higher carry (median 116 vs 101 bps). **Dropping them hurts**: "P4 covered only" earns only +0.5%/yr over the universe. Screens should therefore keep uncovered bonds, as the main variant does. Part of this edge may reflect survivorship or illiquidity, since these bonds are rarely marked (see below).

## Indexer (hedged)

Averaged over cohorts: DI+ +0.4 to +0.9%/yr vs the universe, IPCA+ −0.5 to −0.8, Pré +1.4 to +2.1. **None is significant** (|t| ≤ 1.4). Ranking carry within each indexer instead of across all of them changes P4 by −0.2 to +0.2 (t < 0.5). P4 is already 78% DI+ because it ranks raw CDI+ carry, and incentivized IPCA bonds have low CDI+-equivalent carry. The indexer does not matter for selection once returns are hedged.

## Method

- **Decisions**: first grid day of each month, from 2022-01 to the latest date whose label is complete. That gives 54 / 51 / 45 cohorts at 3 / 6 / 12m.
- **Universe**: `eligible` and a valid `cdi_bps`. Signals are taken at the close of the decision day.
- **Execution**: as in `lab_daily`, entry is the first trade dated after the decision, at least 1 business day later and within 20 business days. If no trade happens, that weight stays in cash (0 excess).
- **Target**: compounded rate-hedged excess return over CDI for 63 / 126 / 252 business days from entry.
- **Portfolios**: equal weight with a 10% issuer cap, held for the full horizon as overlapping monthly tranches. Per-cohort excess vs the universe tranche is averaged, with Newey-West t (lag = horizon in months) and non-overlapping t in the JSON. Cost is **book-level**: |Δ weights| of the tranche book × cost/2 per side, at 25 and 50 bps.
- **Fundamentals**: `merge_asof` on issuer (cnpj8) using `available_date_strict`. After re-sorting by the strict date, the as-of period can never go backwards. Rows with period_end more than 460 days old are discarded. Coverage requires both ND/EBITDA and interest coverage, which excludes banks and insurers: 83% of decision rows are covered.
- **Clipping**: EBITDA ≤ 0 is set to leverage 15, equity ≤ 0 is set to gross debt/equity 10, and the other ratios are clipped.
- **Composite quality**: the mean of cross-sectional ranks of −ND/EBITDA, coverage, cash/ST debt, equity ratio and −ΔND/EBITDA over 4 quarters.
- **IC**: Spearman per date. The partial IC comes from rank residuals on carry rank, RV rank, duration rank and peer dummies.
- **ER score**: carry × H − PD × (H/126) × LGD + 0.25 × (H/126) × resid_bps × dur.
  - PD comes from a walk-forward logistic on the event "6m hedged return < −10%", fitted on labels already known at the decision date. The fit starts once there are at least 30 events, which is why this variant has fewer cohorts. LGD is the walk-forward mean loss of those events.
  - Bonds without coverage get the historical event rate of uncovered bonds.
- **ML**: Ridge and HistGBM on daily cross-sectional ranks, predicting the rank of the forward return. Training uses an expanding window and only labels complete by the decision date. The top quintile is selected.
- **Holm correction** runs over all 24 incremental tests (variant vs its no-fundamentals base × 3 horizons).

## Data problems (important)

1. **`ret` in `lab_daily` loses large moves.** It is NaN across any gap longer than 14 days without trades (18,450 segments, 125 of them with losses over 10%, down to −63%), and it is NaN when |ret| > 20% (654 contiguous rows). This lab rebuilds a **patched return**:
   - the same formula, generalized to multi-day gaps;
   - 810 isolated spike prints removed (a move over 10% that reverts to within 3% on the next trade);
   - gap carry included.

   Carry across gaps is the main reason total log-return nearly doubles (Σ 95.8 vs 50.4). With the unpatched `ret`, the conclusion is unchanged: P4+quality minus P4 is +0.22 / +0.31 / +0.23%/yr.
2. **Survivorship**: 1,403 bonds stop trading more than 30 days before the data end and before maturity (or have no known maturity). 55 of them had a last mark below 0.90 and 28 below 0.80. Losses up to the last trade are kept, but anything after that (defaults with no further prints) is not observed. This favors illiquid, distressed bonds, and probably inflates the uncovered-bond edge.
3. **Fundamentals**:
   - brapi parent numbers (`is_parent`) describe the group, not the issuing SPV;
   - 3% of rows use the net financial result as the interest expense, which biases coverage upward;
   - no financial issuers are included.

## Files

- `sellab_results.json`:
  - `main_strict` holds `ic` (raw / partial / within_peer / signals_full_universe), `portfolios`, `incremental` (with Holm p), `covered_vs_uncovered` and `indexer`;
  - also `check_nonstrict`, `check_unpatched_ret_excess_vs_U_ann_%`, `placebo` and `survivorship`.
- `sellab_ic.png`: raw and partial IC by horizon (* = |t| ≥ 2).
- `sellab_equity.png`: cumulative excess vs the universe at 6m for the main variants. The ER and ML variants only start trading once their models can be fitted, which is later than P4.
