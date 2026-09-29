# combined: P7, the combined wave-1 strategy (harness v4)

**Bottom line.** P7 is P4+Q, minus the worst equity-health quintile, carry-weighted and capped by liquidity for an R$500m fund.
- **Before 2026:** +2.72%/yr over CDI and **+0.50%/yr vs P4+Q** (NW t 3.29, Holm p 0.010 across the 11 variants plus P4).
- **Halves:** +0.37 in 2022–23 and +0.63 in 2024–25.
- **Costs:** +0.48 at 50 bps, and +0.47 under realistic liquidity-bucket costs.
- **Sealed 2026 holdout** (9 months, read once): **+0.68%/yr vs P4+Q** (t 2.2). That is +0.76%/yr over CDI, against +0.08 for P4+Q and −3.26 for the universe.

**The main caveat.** The edge is concentrated. Excluding the 3 best months leaves +0.09 (t 0.46). Dropping the top-10 contributing issuers from both books leaves +0.10 (t 0.86). P7 is better at avoiding the same few blow-ups (Oncoclínicas, Americanas and others), plus a harvested carry slope. It is not a broad new alpha.

## Pre-registered spec (`p7.py` docstring; written before any P7 run, wave-1 default parameters, nothing tuned here)
1. **Selection:** P4+Q.
2. **Avoid screen (EQH):** an equity-health composite, the mean percentile rank of eq_dd252 (+), −eq_vol63 and eq_r126 (+) within the universe's listed rows.
   - P4+Q names in the bottom 20% are dropped. Unlisted names are kept.
   - This is the common core that survived in structural_credit (C2 ≈ Merton DD), factor_zoo (the equity block) and ml_ranking (SHAP).
3. **Construction:** weights ∝ carry (cdi_bps), issuer cap 5%, position ≤ 25% of the 91-day SND volume at R$500m AUM.
   - If the caps are infeasible they are relaxed ×1.5.
   - This follows portfolio_construction: the carry slope inside P4+Q, capacity-limited.
4. **Timing / hedging:** none. No wave-1 overlay or hedge survived, and both are tested below as additions.
5. **Engine:** harness monthly decisions, 126-bday tranches, 25 bps.

## Results before 2026 (48 months; Holm across all 12 rows)
| variant | exCDI | vs U (t) | vs P4+Q (t) | Holm p | 22–23 / 24–25 | maxDD | 50 bps vs P4+Q |
|---|---|---|---|---|---|---|---|
| **P7** | **2.72** | +1.62 (3.5) | **+0.50 (3.29)** | **0.010** | +0.37 / +0.63 | −1.00 | +0.48 |
| V1 screen only (EQH, EW cap 10%) | 2.60 | +1.50 | +0.38 (2.14) | 0.16 | +0.28 / +0.48 | −0.70 | +0.36 |
| V2 construction only | 2.40 | +1.30 | +0.19 (1.48) | 0.41 | | −1.66 | +0.18 |
| V4 P7 without liquidity cap (not investable) | 3.15 | +2.05 | +0.93 (3.36) | 0.009 | | −0.45 | +0.92 |
| V5 P7 with the zoo composite as the screen | 2.76 | +1.65 | +0.54 (4.32) | <0.001 | +0.58 / +0.50 | −1.04 | +0.50 |
| V6 P7 + event and supply flags | 2.57 | | +0.36 (1.38) | 0.41 | −0.13 / +0.84 | | +0.30 |
| V7 P7, top-25 carry construction | 2.79 | | +0.57 (2.76) | 0.05 | | | +0.54 |
| V8 P7 + IDA-DI mom63 entry gate | 2.33 | | +0.11 (0.45) | 0.65 | | −0.74 | |
| V9 P7 + Ibov-slump entry gate | 2.48 | | +0.27 (1.86) | 0.25 | | −0.97 | |
| V10 P7 with EW instead of the carry tilt (liquidity cap kept) | 2.50 | | +0.28 (2.42) | 0.11 | | | |
| V11 P7 on P4 (no Q) | 2.62 | | +0.40 (2.54) | 0.09 | | −1.39 | |
| P4 / P4+Q / U | 1.89 / 2.22 / 1.10 | | −0.33 / 0 / | | | −1.56 / −1.39 / −3.51 | |

**References, pre-2026, excess over CDI:** IDA-DI +1.15%/yr, Ibovespa +0.23%/yr.

**Ablation**, each change measured vs P7:

| change | vs P7 | t |
|---|---|---|
| no screen | −0.32 | −1.7 |
| no carry tilt (EW, liquidity cap kept) | −0.22 | −3.1 |
| no carry/liquidity construction at all | −0.12 | −0.9 |
| no Q filter | −0.10 | −0.7 |
| adding the mom63 gate | −0.39 | |
| adding the Ibov gate | −0.23 | −2.5 |
| adding event/supply flags | −0.14 | |

Removing the liquidity cap adds +0.43, but that version cannot be bought.

## Robustness of P7 (vs P4+Q, paired, same assumptions on both sides)
| check | diff | t |
|---|---|---|
| rec40 | +0.46 | 2.8 |
| hold 63 / 252 | +0.73 / +0.41 | 3.1 / 3.1 |
| realistic costs by liquidity bucket (bias_audit: DI 45–65, IPCA/PRE 136–169 bps round trip) | +0.47 (P7 +1.89 over CDI vs P4+Q +1.42) | 3.2 |
| harsh survivorship (every stop < 0.98 goes to 40%) | +0.76 | 3.6 |
| both of the above ("honest") | **+0.73** (P7 +1.77 over CDI, P4+Q +1.04, U +0.32) | 3.4 |
| capacity at R$100m / 250m / 500m / 1bn / 2bn | +0.56 / +0.55 / +0.50 / +0.35 / +0.34 | |
| placebo: drop the same number of listed P4+Q names at random, same construction (20 draws) | mean +0.16, p95 +0.25 | actual +0.50 |
| by year | 2022 +0.09, 2023 +0.66, 2024 +1.19, 2025 +0.07 | |
| **excluding the 3 best months** (2024-12, 2025-04, 2023-01) | **+0.09** | 0.46 |
| dropping the top-5 / top-10 contributing issuers from both books | +0.24 / +0.10 | 2.3 / 0.9 |

The placebo's positive mean shows that dropping any listed name helps a bit, because it tilts the book toward unlisted names.

## Sealed holdout, 2026-01..09 (run once, after everything above was frozen)
| book | exCDI %/yr | vs P4+Q | vs U | maxDD |
|---|---|---|---|---|
| **P7** | **+0.76** | **+0.68 (t 2.2)** | +4.02 | −0.67 |
| P4+Q | +0.08 | 0 | +3.34 | |
| P4 | −0.73 | −0.81 | +2.53 | −0.94 |
| U | −3.26 | | | −3.06 |
| IDA-DI / Ibov | +0.53 / +5.68 | | | |

In the holdout, P7 vs P4+Q is +0.65 at 50 bps and +0.60 with liquidity-bucket costs. It was positive in 6 of 9 months.

## Insights
- **The wave-1 avoid signals are mostly one signal.** Across P4+Q rows, the Jaccard overlaps of the dropped sets are:

  | pair | Jaccard |
  |---|---|
  | equity-health / zoo | 0.29 |
  | equity-health / events | 0.26 |
  | events / supply | 0.19 |

  Names flagged by only one signal do not underperform: their 6-month return relative to P4+Q is −0.02 to +0.33%. Names flagged by 3 or more signals underperform by −1.36% over 6 months. Stacking every flag (V6) is worse than the equity screen alone (−0.14 vs P7) because it removes 33% of the book.
- **Screen and construction are complementary.** The screen removes tail risk: maxDD falls from −1.39 to −0.70 and vol from 1.04 to 0.93. The carry tilt adds return, with carry 301 vs 289 bps. Each alone is +0.19 to +0.38; together they are +0.50.
- **The liquidity cap costs about 0.43%/yr but makes the book real.** Liquidity breach at R$500m falls from 17% of weight for P4+Q (28% for uncapped carry) to 0.3%. Capacity fades only slowly, to +0.34 at R$2bn, because the carry tilt is mild.
- **Honest levels are much lower, but the relative edge grows.** Under liquidity-bucket costs plus harsh survivorship, P7 is +1.77%/yr over CDI (P4+Q +1.04, U +0.32). The gap vs P4+Q grows to +0.73, because P7 avoids more of the bonds that later go silent.
- **Timing and hedging still subtract.** The mom63 and Ibov entry gates cost 0.23–0.39%/yr, even though they cut drawdown.

## Caveats
- **Concentration in months and issuers.** Most of the gain comes from 2023–24 blow-up avoidance: 2022 and 2025 are about +0.08 each. Excluding the 3 best months, or the top-10 issuers, the gain is not significant.
- **Choices informed by wave-1.** The components were chosen after looking at wave-1 results, on the same 2022–25 sample. So the pre-2026 Holm p (0.010) understates the true search, which spans more than 300 wave-1 variants. The holdout is the only clean test, and it is only 9 months long.
- **Inflated statistics and marks.** Smooth, stale marks inflate t and Sharpe. The harness also has a holiday-accrual bug (bias_audit), worth about −0.13 on every book's level.
- **Liquidity proxy.** SND volume includes interdealer back-to-back trades, so real capacity is probably lower. The cost bucket uses lifetime median ADV.
- **Coverage of the screen.** EQH only sees listed issuers (about 41% of P4+Q). The unlisted tail is unscreened.

## Files and rerun
- `p7.py`: the spec, `make_signal()`, and `signals()`, which exports P7 target weights (day, codigo, weight).
- `run.py`: stage 1 (pre-2026) and stage 2 (`--holdout`).
- `results.json` and `compare_pre2026.csv`.
- Charts: `equity_total_return.png` (the holdout is shaded), `equity_pre2026.png` and `cum_excess.png`.
- Weight caches: `data/history/nightly/combined/p7_weights_pre2026.pkl` and `p7_weights_all.pkl`.

To rerun:

```
PYTHONPATH=. PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/Scripts/python.exe research/nightly/combined/run.py            # ~3 min
PYTHONPATH=. PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/Scripts/python.exe research/nightly/combined/run.py --holdout  # already run once
```
