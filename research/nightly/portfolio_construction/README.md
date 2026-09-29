# Portfolio construction on the P4+Q alpha (nightly, harness v4)

**Question.** Take P4+Q as the alpha source. How should the book be built: weights, risk model, constraints and rebalancing? And what would a real R$500m fund run?

**Short answer.**
- **Concentration drives everything.** Nearly all the value from construction comes from concentrating *within* P4+Q in its highest-carry names. Risk models add almost nothing.
  - Simple version: equal weight over the 25 highest-carry P4+Q names, issuer cap 5%. It earns **+1.76%/yr over P4+Q** (NW t 4.7, Holm p < 0.001 across 34 variants), in both halves, at 50 bps and in rec40.
  - A low-risk-aversion mean-variance optimizer ends up holding the same book.
- **Liquidity takes most of it away at fund size.** Cap each position at 25% of the bond's 91-day SND volume, and the edge falls:

  | fund size | edge over P4+Q |
  |---|---|
  | R$50m | +0.99 |
  | R$250m | +0.46 |
  | R$500m | +0.21 |
  | R$1bn | −0.09 |

  The carry-concentration premium sits in bonds a R$500m fund cannot hold at size.
- **Risk-based constructions lose to plain equal weight** by 0.2–0.4%/yr, because they tilt away from carry. That covers min-variance, inverse-vol (DTS), max-diversification, issuer risk parity and CVaR with default scenarios. They cut realised drawdown only modestly.
- **Rebalancing is a cost question.** Replacing the whole book every week costs 1.4%/yr vs P4+Q. Quarterly replacement, or longer tranches, is the best of the engine choices.
- **Recommended for a real R$500m fund:** `mv_emp_g20_fund500m`, a liquidity-constrained mean-variance book (defined below).
  - Pre-2026: +0.57%/yr vs P4+Q (t 2.5, raw p 0.013). **Holm p is 0.30, so it is not significant after the multiple-testing correction.**
  - Sealed 2026 holdout, 9 months: +0.60%/yr vs P4+Q (t 1.5).

## Method

- **Harness and engine.** Everything runs through `research/nightly/harness.py` (v4): monthly decisions, 126-bday overlapping tranches, 25 bps round trip, patched rate-hedged excess returns. The sealed holdout is every decision dated 2026-01-01 or later.
- **Candidate set.** P4+Q names on each date, except `wide_mv`, which uses: carry in the top 50%, not rich, no negative press, not worst-quintile quality.
- **Weights.** Weights are computed point-in-time at the decision close and passed to `H.backtest(weights_df, issuer_cap=1.0)`. The constraints are enforced inside the construction.
- **Risk models** (`risk.py`). Each uses only the R rows realised by the decision close.
  - **`emp`, de-smoothed empirical.**
    - Inputs: weekly returns over the trailing 104 weeks, turned into overlapping 4-week sums. This is variance-ratio de-smoothing: it captures stale-mark autocorrelation up to lag 3.
    - Vol: the bond's own vol when it has 26 or more observations, shrunk 50/50 towards κ·DTS. Otherwise κ·DTS alone.
    - Correlation: a 50/50 shrink of the pairwise sample correlation towards a block structure (issuer / sector / market), clipped to PSD.
  - **`dts`, structural.** σ = κ·duration·spread, the Duration-Times-Spread measure.
    - The block correlations are floored at economic priors: issuer 0.8, sector 0.35, market 0.2.
    - The floor is needed because stale marks make measured correlations tiny. Same-issuer pairs average only **0.18**, sector 0.12 and market 0.09, even on 4-week sums. This is an Epps-like effect of smooth marks.
- **Default scenarios** (CVaR). Over a 6-month horizon:
  - Default is simulated per issuer with a Gaussian copula: market 0.15, sector 0.30, and all bonds of an issuer default together.
  - PD = 1 − exp(−ψ·s/(1−R)·h), with R = 40% and ψ = 0.5 (the ratio of real-world to risk-neutral PD).
  - Loss on default = 1 − 0.40/price. A DTS spread-MTM term is added.
  - The optimisation is Rockafellar–Uryasev CVaR97.5 in an LP solved with HiGHS.
  - Every variant also gets an **ex-ante 6-month CVaR99** from a common scenario set.
- **Constructions.**
  - Equal weight with issuer cap 10% (= P4+Q) or 5%, with and without a sector cap of 25%.
  - Score-weighted: a composite tilt (0.5 carry + 0.25 cheapness vs the peer curve + 0.25 quality), carry-proportional weights, and the top-50 by composite.
  - Inverse-vol (DTS), min-variance (emp), max-diversification (DTS), equal risk contribution by issuer (emp).
  - Mean-variance with μ = ½·spread: γ = 20 or 100, emp or dts covariance, an L1 turnover penalty, and a duration band.
  - CVaR with λ = 1 or 5.
  - Top-25 by carry.
  - **Fund-constrained books:**
    - issuer ≤ 5% and sector ≤ 25%;
    - duration within ±1 year of the universe average;
    - **liquidity:** a position may not exceed 25% of the bond's R$ volume over the prior 91 days (from SND trades, point-in-time), at R$500m or R$2bn;
    - minimum lot 0.25%;
    - if the constraints are infeasible, the carry ranking is extended before the liquidity cap is relaxed.
- **Rebalancing and engine variants of P4+Q.**
  - "Live" replace-the-book at weekly, monthly or quarterly frequency.
  - A ±50% band instead of ±25%.
  - A hysteresis buffer: hold a name while its carry stays in the top 45%.
  - A sticky book that sells only when a name turns rich.
  - Tranche lengths of 63 and 252 days, weekly tranches, and quarterly tranches.

## Results (pre-2026, 48 months, net 25 bps; paired vs P4+Q tranche; Holm across all 34 variants)

| variant | exCDI | vs U | vs P4+Q | t | Holm p | maxDD | 22–23 / 24–25 vs P4+Q | rec40 vs P4+Q | 50 bps vs P4+Q | turnover | ex-ante CVaR99 6m | eff. N issuers | liq. breach @500m |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mv_emp_g20_durband | 4.06 | 2.96 | +1.84 | 4.7 | <0.001 | −0.18 | +1.72 / +1.95 | +1.76 | +1.82 | 3.1 | 9.7 | 22 | 47% |
| **top25_carry_cap5** | 3.98 | 2.88 | **+1.76** | 4.7 | <0.001 | −0.34 | +1.42 / +2.11 | +1.68 | +1.73 | 3.2 | | | |
| mv_emp_g20 | 3.83 | 2.73 | +1.62 | 4.1 | <0.001 | −0.15 | +1.89 / +1.34 | +1.54 | +1.60 | 3.1 | 9.8 | 21 | 50% |
| mv_emp_g100 | 3.26 | 2.16 | +1.04 | 2.6 | 0.25 | −0.28 | +1.47 / +0.60 | +1.02 | +1.02 | 3.1 | 9.1 | 22 | 46% |
| top50_comp | 3.05 | 1.95 | +0.84 | 5.6 | <0.001 | −0.64 | +0.67 / +1.00 | +0.82 | +0.84 | 2.9 | 7.7 | 34 | 31% |
| carry_prop | 2.90 | 1.80 | +0.69 | 5.6 | <0.001 | −0.94 | +0.55 / +0.83 | +0.68 | +0.69 | 2.9 | 7.4 | 45 | 23% |
| **mv_emp_g20_fund500m (recommended)** | 2.79 | 1.68 | **+0.57** | 2.5 | 0.30 | −0.98 | +0.75 / +0.38 | +0.51 | +0.55 | 3.1 | 8.3 | 25 | 0% |
| score_tilt | 2.51 | 1.41 | +0.29 | 4.5 | <0.001 | −1.05 | +0.28 / +0.30 | +0.29 | +0.30 | 2.9 | 6.4 | 53 | 19% |
| top25_carry_fund500m | 2.43 | 1.33 | +0.21 | 1.1 | 1 | −1.91 | +0.12 / +0.31 | +0.16 | +0.19 | 3.1 | | | 0% |
| EW_cap10 (= P4+Q) | 2.22 | 1.12 | 0 | | | −1.39 | | | | 2.9 | 6.1 | 52 | 17% |
| EW_cap5_sec25 | 2.23 | 1.13 | +0.01 | 0.8 | 1 | −1.38 | 0.00 / +0.03 | +0.01 | +0.02 | 2.9 | 5.9 | 56 | 17% |
| EW_fund500m | 2.12 | 1.02 | −0.10 | −0.8 | 1 | −1.84 | −0.13 / −0.08 | −0.11 | −0.11 | 3.0 | 5.9 | 48 | 0% |
| EW_fund2bn | 1.99 | 0.89 | −0.23 | −0.7 | 1 | −2.61 | −0.23 / −0.22 | −0.24 | −0.25 | 3.1 | 6.5 | 36 | 0% |
| cvar_l1 / cvar_l5 | 2.03 / 1.89 | 0.93 / 0.78 | −0.19 / −0.33 | −1.1 / −1.7 | 1 | −1.27 / −1.35 | | −0.16 / −0.29 | | 3.6 | 6.1 / 5.9 | 40 | 28% |
| erc_issuer_emp | 2.02 | 0.92 | −0.20 | −0.9 | 1 | −1.06 | +0.11 / −0.51 | −0.19 | −0.22 | 3.1 | 6.0 | 48 | 24% |
| minvar_emp | 1.89 | 0.79 | −0.33 | −0.8 | 1 | −0.84 | +0.20 / −0.87 | −0.30 | −0.38 | 3.3 | 7.3 | 23 | 39% |
| maxdiv_dts | 1.81 | 0.71 | −0.41 | −1.6 | 1 | −1.04 | +0.05 / −0.87 | −0.39 | −0.43 | 3.1 | 6.0 | 45 | 24% |
| invvol_dts | 1.79 | 0.69 | −0.43 | −1.7 | 1 | −1.06 | +0.07 / −0.92 | −0.40 | −0.45 | 3.1 | 6.4 | 39 | 20% |
| mv_dts_g100 | 2.77 | 1.67 | +0.55 | 1.1 | 1 | −0.27 | +1.19 / −0.08 | +0.59 | +0.53 | 3.2 | 8.6 | 22 | 48% |
| carry_over_var_tilt | 2.23 | 1.13 | +0.02 | 0.05 | 1 | −0.62 | +0.50 / −0.47 | | | 3.1 | 6.8 | 34 | 27% |

Other variants are in `results_table.csv`: mv_emp_g20_tc, wide_mv and EW_cap5.

**Engines and rebalancing (P4+Q names; "vs live-M" = same engine, monthly replace):**

| variant | exCDI | vs P4+Q | t | vs live-M | turnover/yr | cost %/yr | maxDD |
|---|---|---|---|---|---|---|---|
| live_W_replace | 0.80 | −1.42 | −3.6 | −1.12 | 14.5 | 1.81 | −2.88 |
| live_M_replace | 1.92 | −0.30 | −1.6 | 0 | 9.1 | 1.14 | −1.69 |
| live_M_band50 | 1.90 | −0.32 | −1.7 | −0.02 | 8.8 | 1.10 | −1.69 |
| live_M_buffer45 | 1.94 | −0.28 | −1.4 | +0.02 | 8.1 | 1.02 | −1.67 |
| live_M_sticky_rich | 2.10 | −0.12 | −0.4 | +0.18 | 2.3 | 0.29 | −1.71 |
| live_Q_replace | 2.51 | +0.29 | 1.2 | +0.59 | 3.8 | 0.48 | −1.51 |
| tranche_h63 / h252 | 2.26 / 2.37 | +0.05 / +0.15 | 0.3 / 1.2 | | 4.8 / 2.1 | 0.60 / 0.26 | −1.82 / −1.22 |
| tranche_W126 / Q126 | 2.28 / 2.44 | +0.06 / +0.22 | 0.7 / 2.0 | | 3.3 / 2.7 | 0.42 / 0.34 | −1.39 / −1.24 |
| tranche_buffer45 | 2.07 | −0.15 | −2.6 | | 2.8 | 0.35 | −1.45 |

**Placebo for the recommended book.** The same weights on random universe names score +0.81 on average (p95 +1.04); the book scores +2.79.

**Capacity curve** (`capacity.json`). Top-25-carry, made liquidity-feasible at each fund size, vs P4+Q:

| fund size | vs P4+Q | t | avg names |
|---|---|---|---|
| R$50m | +0.99 | 2.8 | 34 |
| R$100m | +0.83 | 2.3 | 36 |
| R$250m | +0.46 | 2.3 | 42 |
| R$500m | +0.21 | 1.1 | 47 |
| R$1bn | −0.09 | −0.3 | 59 |
| R$2bn | −0.18 | −0.5 | 71 |

**Diagnostics** (`diag.json`; P4+Q cross-sections, fwd_126):
- **Carry within P4+Q.** Carry (cdi_bps) has a Spearman IC of **+0.33 (t 11)** within P4+Q, so the 30% cut leaves a lot of carry ranking unused.
- **DTS.** The IC of DTS is **−0.23 (t −3.2)**: high duration × spread predicts worse rate-hedged returns.
- **Vol.** The empirical vol and carry/variance ICs are insignificant or negative.
- **What the optimizer buys.** The MV book holds 91% DI floaters, with duration 2.1 vs 3.4 and a spread of 461 vs 289 bps. It is less covered (46% vs 65%) and less listed (27% vs 44%). Only 86% of its weight actually executes within 20 bdays, against 92% for equal weight.

**Sealed holdout (2026-01 to 2026-09, run once after freezing the recommendation):**

| book | exCDI %/yr | vs P4+Q | t |
|---|---|---|---|
| U | −3.26 | | |
| P4+Q | +0.08 | | |
| **mv_emp_g20_fund500m** | **+0.68** | **+0.60** | 1.5 |
| top25_carry_fund500m | +0.60 | +0.52 | 1.3 |
| EW_fund500m | −0.08 | −0.16 | −2.5 |
| mv_emp_g20 (unconstrained) | +1.69 | +1.61 | 2.4 |
| top25_carry_cap5 (unconstrained) | +2.65 | +2.57 | 3.4 |
| cvar_l5 | −0.45 | −0.53 | −1.4 |

Charts:
- `equity_total_return.png`: total return CDI × (1 + excess), plus cumulative excess vs the universe, with CDI / U / P4 / P4+Q / IDA-DI / Ibov as references.
- `cum_excess.png`: cumulative excess vs the universe and vs P4+Q.
- `frontier.png`: excess vs ex-ante CVaR99, realised max drawdown and turnover. In this chart, green marks engine variants and the two top-25-carry books, which have no ex-ante risk metrics. It is not only engine variants.
- `equity_total_return_incl_holdout.png`.

## Recommended construction for a R$500m fund (frozen before the holdout)

`mv_emp_g20_fund500m`:
- **Decisions.** Monthly, as overlapping 126-bday tranches: each month buys a sixth of the book.
- **Candidates.** P4+Q names.
- **Objective.** Maximise ½·spread·w − 10·w′Σw, with Σ the de-smoothed empirical covariance (`risk.cov(kind="emp")`).
- **Constraints:**
  - issuer ≤ 5% and sector ≤ 25%;
  - duration within the universe average ± 1 year;
  - each position ≤ 25% of the bond's SND volume over the previous 91 days (at R$500m);
  - minimum lot 0.25%.

**Why this one:**
- It is the best construction that is **liquidity-feasible**, with 0% liquidity breach vs 50% for the unconstrained optimizer.
- It is positive in both halves (+0.75 / +0.38) and robust at 50 bps (+0.55) and in rec40 (+0.51).
- It has a lower max drawdown than P4+Q (−0.98 vs −1.39).

**The cost.** It carries a higher ex-ante default tail than P4+Q: 6-month CVaR99 of 8.3% vs 6.1%.

**What it is not.** It is not statistically established: Holm p is 0.30 with 34 variants.

A simpler, nearly equivalent rule for operations: rank P4+Q by carry, buy down the ranking until the liquidity, issuer, sector and duration constraints are met. That is `top25_carry_fund500m`, which is weaker pre-2026 (+0.21).

## Insights

1. **The construction edge is the carry slope inside the selection, not risk modelling.** Within P4+Q, carry still ranks forward returns strongly: IC +0.33, t 11. Equal weight throws that away.
   - Top-25 by carry, or carry-proportional weights, beats every risk model.
   - Mean-variance helps only because, with near-diagonal empirical covariance, it collapses into "max carry subject to caps".
2. **Smooth marks break correlation-based optimisers.** Measured same-issuer correlation is 0.18 (market 0.09), so empirical covariances treat two bonds of one issuer as almost independent.
   - Without an issuer cap, a naive MV would stack one issuer.
   - The structural DTS model with economic correlation floors, together with the default copula, gives the tail view that the marks hide.
3. **The two tail measures disagree.** In the ex-ante tail model, the concentrated books are the riskiest (CVaR99 9.7% vs 6.1%). Realised drawdowns show the reverse: the concentrated books have the smallest maxDD (−0.15 to −0.34).
   - Realised default losses in 2022–25 were far below even half the risk-neutral PDs.
   - So the concentration premium is at least partly an unpriced-in-sample short on default-cluster risk.
4. **Risk-based constructions lose 0.2–0.4%/yr.** This covers min-variance, max-diversification, inverse-vol, issuer risk parity and CVaR, because they underweight high-DTS carry.
   - Their only benefit is a smaller drawdown for min-var and max-div: −0.84 to −1.04 vs −1.39.
   - DTS itself is a negative predictor (IC −0.23). So a DTS-neutral carry tilt, meaning high spread at short duration, is the efficient direction; the duration-band MV was the best in-sample book.
5. **Capacity is the binding constraint.** The unconstrained concentrated book needs a median of 174 days (at 20% participation) to build a position at R$500m.
   - The carry-concentration premium decays roughly linearly in log fund size and is gone by R$1bn.
   - A R$500m fund should expect about +0.2 to +0.6%/yr from construction over P4+Q, not +1.8.
6. **Rebalancing:**
   - Every replace-the-book engine loses to tranches or quarterly replacement, purely through costs (weekly: 14.5×/yr turnover, 1.8%/yr cost).
   - Bands and hysteresis buffers barely help (−0.02 / +0.02).
   - Quarterly replace (+0.29) and 252-day tranches (+0.15) are the cheapest ways to hold P4+Q.
   - Sector caps at 25% and issuer caps at 5% vs 10% cost nothing (+0.01).

## Caveats

- **Smooth, stale, survivorship-biased marks.** The t-stats are inflated. The concentrated high-carry books load on exactly the illiquid, less-covered names where this bias lives. rec40 only stresses the 55 known stop-trading names.
- **Only 48 months, with few defaults in the chosen names.** Realised default losses were small. The default-scenario model says the concentrated books carry about 1.6× the tail.
- **The liquidity proxy is rough.** It uses SND reported volume, which includes interdealer back-to-back trades. Real capacity is probably lower. The 25% participation rate is an assumption.
- **Execution.** Execution is at the mark, with no market impact beyond cost_bps. Market impact would hurt the concentrated books most.
- **Holm across 34 variants.** Only the unconstrained carry-concentration books survive it. The recommended fund book does not (0.30). Several variants were added after the first results were seen (mv_emp_g20_durband, top25_carry_*, mv_emp_g20_fund500m, carry_over_var_tilt). All are counted in the Holm correction.
- **The holdout is thin.** It covers 9 months in a stressed tape (universe −3.3%/yr).

## Reusable outputs

- **Liquidity dataset.** `data/history/nightly/portfolio_construction/liquidity.pkl`, with columns `codigo, day, vol91_brl, tdays91, adv_brl`.
  - It covers every harness M/W decision day from 2021-03 to 2026-09.
  - PIT: only SND trades dated on or before the decision day.
- **Vol / carry signals.** `signals.signals(panel)`, cached at `data/history/nightly/portfolio_construction/signals_M.pkl`, with columns `codigo, day, emp_vol, dts, vol_hat, carry_over_var, n_obs4w`.
- **Risk models.** `risk.cov(p, x, kind)`, `risk.default_scenarios(x, ...)` and `risk.block_params(...)`.
- **Constructors.** `construct.project / minvar / meanvar / maxdiv / erc_issuer / cvar_opt`, with issuer, sector, duration, liquidity and min-lot constraints.
- **Recommended weights.** `data/history/nightly/portfolio_construction/recommended_weights_pre2026.pkl`, with columns `day, codigo, weight`.
- **All variant weights.** `weights_pre.pkl` (pre-2026) and `weights_all.pkl` (including 2026).

## Rerun

```
PYTHONPATH=. PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/Scripts/python.exe research/nightly/portfolio_construction/data.py
PYTHONPATH=. PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/Scripts/python.exe research/nightly/portfolio_construction/build_weights.py
PYTHONPATH=. PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/Scripts/python.exe research/nightly/portfolio_construction/signals.py
PYTHONPATH=. PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/Scripts/python.exe research/nightly/portfolio_construction/run.py
PYTHONPATH=. PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/Scripts/python.exe research/nightly/portfolio_construction/diag.py
PYTHONPATH=. PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/Scripts/python.exe research/nightly/portfolio_construction/capacity.py
# once, after freezing RECOMMENDED:
PYTHONPATH=. PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/Scripts/python.exe research/nightly/portfolio_construction/run.py --holdout
```

The weights caches (`weights_pre.pkl`, `weights_all.pkl`) are reused if present. Delete them from `data/history/nightly/portfolio_construction/` to rebuild, which takes about 5 minutes each. The total runtime is about 25 minutes.
