# w2_primary_market_concession: the debenture new-issue concession and a primary-market sleeve

**Bottom line.** On average, Brazilian local debentures in 2021–25 were **not** issued with a new-issue concession.

**Measuring the concession**
- **First print vs issue spread.** The median first SND print sits exactly at the issue spread. On average the first print is 5.7 bps *wider*.
- **Against the issuer's own curve.** New series priced 16 bps *inside* the issuer's outstanding bonds (t −4.0).
- **Against the peer curve.** The apparent "concession" vs the peer curve (+36 bps) is a carry and credit proxy. Once the issue spread is controlled for, it predicts neither first-print tightening nor later relative return.

**The primary-sleeve result depends on the marks.**
- **Best book.** A primary sleeve buying new issues at the bookbuilding rate beats P4+Q before 2026: A2 (concession > 50 bps plus the P4 press, P4+Q quality and P7 EQH screens) returns **+1.31%/yr vs P4+Q** (NW t 2.98).
- **Not significant after multiple testing.** Across the 30 variants tried, the global Holm p is **0.067**.
- **The result depends on how bonds that never trade are marked.** 69% of A2's series never print within the 126-bday hold.
  - Marked to the peer-curve model: +1.31.
  - Marked at cash: −0.29.
  - Marked at par plus carry: +1.79.
- **Buying at the issue rate adds no premium** over buying the same bonds at their first print: −0.37%/yr (t −0.5).
- **Half the edge is carry.** Simply restricting P4+Q to bonds with a spread of at least 260 bps (A2's median issue spread) earns +0.74 (t 3.5).

**Sealed 2026 holdout** (9 months, read once):
- A2 is +2.49%/yr vs P4+Q (t 4.1), and +1.48 even with never-printed bonds at cash.
- In the redemption wave, young bonds held up better than the seasoned book.

Harness v4, 25 bps, pre-2026 unless stated. Everything is point-in-time: decisions use information at the close of the grid day before settlement.

## Data built (`build_data.py`, cached in `data/history/nightly/w2_primary_market_concession/`)

| file | content | PIT rule |
|---|---|---|
| `new_issues.pkl` | 4,215 new series settling 2021-04 → 2026-09 (DI+ 3,008, IPCA 1,077, Pré 130), from the SND registry. Columns: contract (bookbuilding) rate, settlement (= start of accrual), CVM regime (ICVM 476 / RCVM 160 automatic / ICVM 400), size (qty × VNE), lead manager, guarantee, Lei 12.431 flag, first-time vs repeat issuer, first SND print (lab_daily), CDI+-equivalent issue spread, same-day peer fair value, issuer effect, press/rating screens, the linked CVM offer (90% linked), IDA-DI momentum, credit-fund flows (af21, lagged 5 bdays) | Fair values use only fresh lab_daily marks of `fair_day`, the grid day before settlement. The curve is the last B3 PRE / NTN-B curve on or before `fair_day`. |
| `curves.pkl` | B3 PRE and NTN-B real curve panel, 2021 → 2026-09 (`rfmonitor.history.b3_curve_panel`) | publication day |
| `events.pkl` | per-series outcomes: first-print jump `J`, post-print paths (21/63/126 bdays) vs the universe and vs the peer cohort, spread path | outcome only |
| `univ_bh.pkl` | a buy-and-hold universe cohort benchmark per grid day and horizon | |
| `peer_bucket_spread.pkl` | daily median spread per (kind × incentive, duration bucket), used to mark never-printed bonds to model | same-day marks |

**Definitions.**
- **Issue spread** is the CDI+ equivalent at par, using the same formulas as `rfmonitor.ml.selection`:
  - DI+: the contract rate;
  - IPCA+ and Pré: (1+c)/(1+curve(dur)) − 1.
- **Concession.** `conc_peer` = issue spread − the same-day peer-curve fair value (kind × incent, duration-bucket medians: the resid framework). `conc_iss` = `conc_peer` − the issuer effect, i.e. the shrunk mean residual of the issuer's other fresh bonds. This is the `issuer_curve_rv` e_iss logic with k = 1. `conc` uses `conc_iss` where the issuer has fresh bonds (38%), and `conc_peer` otherwise.

## 1. Event study (`analysis.py`, `analysis_results.json`, `event_study.png`), 3,566 series settling 2021-06 → 2025

**Coverage.**
- Only **56%** of new series ever print on the SND grid.
- The median gap from settlement to first print is 23 bdays. Only 27% print within 21 bdays.

### What the concession measures
| concession measure | mean (bps) | median | t |
|---|---|---|---|
| vs peer curve | +36 | +22 | 10.7 |
| **vs issuer curve** (n = 1,331) | **−16** | −9 | −4.0 |
| issuer's own bonds vs peers | +19 | +1 | 4.5 |

New issues price *inside* the issuer's secondary curve. The peer "concession" is the issuer's credit premium.

### First print (n = 1,868)
- **Spread move:** the median spread change is 0.0 bps (the first print is usually at the issue price) and the mean is +5.7 bps (t 2.5).
- **Primary buyer's gain:** the rate-hedged excess from par at settlement to the first print is +0.26% (t 1.7), essentially the carry accrued over the median 23 bdays.
- **Controls:** with month-clustered regressions controlling for the issue spread, `conc` has no effect on the first-print spread move (−0.11 bps per bp, t −0.9).

### After the first print (126 bdays, month-clustered t)
- **Bought at the first print:** **+0.42% vs the peer cohort** (t 5.0) and +0.46% vs the universe cohort (t 4.0). This is a modest seasoning drift.
- **The primary buyer at par, from settlement:** +0.11% vs peers (t 0.3), and −0.03% vs the universe.

**By concession quintile** (primary buyer vs peers over 126 bdays, %):

| quintile | Q1 | Q2 | Q3 | Q4 | Q5 |
|---|---|---|---|---|---|
| concession | −90 bps | | | | +267 bps |
| return | −2.18 | −0.28 | −0.14 | +1.37 | +2.00 |

The mean issue spread rises from 65 to 384 bps across the quintiles. After controlling for the issue spread, the concession has t = 1.0.

**Splits** (post-print vs peers over 126 bdays, then primary vs peers):

| split | post-print vs peers | primary vs peers |
|---|---|---|
| Lei 12.431 | +1.16 | +0.82 |
| taxable | +0.07 | −0.27 |
| first-time issuer (average concession 152 bps) | +0.97 | +1.28 |
| repeat issuer | +0.28 | −0.14 |
| RCVM 160 | +0.52 | +0.37 |
| ICVM 476 | +0.32 | −0.23 |
| ICVM 400 (only 26 series) | −1.39 | −6.39 |
| IPCA | +1.13 | |
| DI+ | +0.02 | |

- **Market regime:** the IDA-DI momentum and credit-fund flow splits are flat.
- **2024:** primary buyers lost (−1.12 vs peers). New issues were priced tight: the average concession was 8 bps and the first print was +24 bps wider.

## 2. Primary sleeve vs P4+Q and P7 (`portfolio.py`)

**Engine.**
- Each selected series is bought at **par (the bookbuilding rate) at settlement**. Each deal gets weight 1, spread across its series.
- Until the first print the position accrues the issue spread. The first print books the hedged move −dur × (s_fp − s_iss).
- After that it earns the harness's patched returns. It is held 126 bdays, using the harness gap rule.
- Never-printed series are **marked to model**: carry − dur × Δ(peer-bucket median spread).
- The primary entry is free. The exit costs cost/2.
- Blends combine P4+Q with the sleeve on the same capital.

| variant | exCDI | vs U | vs P4+Q (t) | 22–23 / 24–25 | 50 bps | rec40 | vs P7 | maxDD |
|---|---|---|---|---|---|---|---|---|
| A0 every new issue | 0.82 | −0.28 | −1.39 (−2.5) | −1.22 / −1.57 | −1.25 | −1.36 | −1.90 | −3.15 |
| A1 conc > 0 + screens | 2.92 | +1.82 | +0.70 (1.6) | +0.08 / +1.33 | +0.84 | +0.73 | +0.20 | −2.19 |
| **A2 conc > 50 + screens** | **3.53** | **+2.43** | **+1.31 (2.98)** | **+0.96 / +1.67** | **+1.45** | **+1.35** | +0.81 (1.6) | −2.10 |
| A3 top-30% carry + screens | 2.95 | +1.85 | +0.73 (1.7) | +0.43 / +1.04 | +0.87 | +0.77 | +0.23 | −2.36 |
| A4 Lei 12.431 only | 3.87 | +2.77 | +1.65 (1.8) | −0.71 / +4.02 | | | | −2.75 |
| A5 first-time issuers | 2.84 | +1.74 | +0.62 (0.9) | | | | | −2.95 |
| A6 = A1 with never-printed at cash | 1.70 | +0.59 | −0.52 (−1.1) | | | | | |
| A7 = A1 with never-printed at par + carry | 3.41 | +2.31 | +1.19 (2.6) | | | | | 0.00 |
| B1 = A1 bought at the first print (≤ 21 bdays) | 3.29 | +2.18 | +1.07 (1.6) | −0.03 / +2.17 | | | | −1.50 |
| P4+Q + 20% A2 (100% fill) | 2.48 | +1.38 | +0.27 (2.98) | | +0.29 | | | −1.46 |
| P4+Q + 20% A2 (**30% fill**) | 2.30 | +1.20 | +0.08 | | +0.09 | | | −1.40 |
| P7 | 2.72 | +1.62 | +0.50 (3.3) | +0.37 / +0.63 | | | | −1.00 |

Placebo, random new issues (40 draws, the same count per month as A1): −1.47 vs P4+Q (p95 −1.07).

### Robustness of A2 (`robustness.py`)
- **Marking and selection:**
  - never-printed at cash: −0.29;
  - never-printed at par + carry: +1.79;
  - split by look-ahead: printed series +2.35 (t 4.5) vs never-printed +0.39 (t 0.6);
  - bought at the first print (B2, 164 positions): +1.81 (t 2.5).
- **Parameters:**
  - no screens: +1.48;
  - concession > 25: +1.10;
  - concession > 100: +1.57;
  - hold 63: +0.94;
  - hold 252: +0.91.
- **Concentration:** excluding the 10 most frequent issuers leaves +1.33.
- **By year:** 2022 +2.0, 2023 −0.1, 2024 +2.3, 2025 +1.1.
- **Carry controls:**
  - spread-matched random new issues (same month × spread tercile, 30 draws): +0.59, p95 +0.91;
  - P4+Q restricted to spread ≥ 260: +0.74 (t 3.5);
  - the universe restricted to spread ≥ 260: +0.16.

So about half of A2 is carry. The rest (≈ +0.6) sits mostly in the never-printed, model-marked series, where credit risk is invisible.

### Harness-native (secondary, market-marked) test of the seasoning drift (`seasoning.py`)
| variant | vs P4+Q | t |
|---|---|---|
| C1: universe names first printed ≤ 31 days ago | −1.10 | −2.8 |
| C2: C1 ∩ P4 | −1.81 | |
| C3: P4+Q ∪ new P4-like names | −0.01 | |
| C4: P4+Q minus new issues | +0.07 | |

- **C1:** its cohort excess vs the universe is +0.45%/yr (t 2.2), and it is +0.02 vs U. The placebo mean is +0.66 (p95 +0.92) against +1.12 actual. C2 is thin: often fewer than 5 names.
- **Conclusion:** there is a small seasoning drift, but it cannot be harvested in the secondary book.

**Multiple testing.** 30 variants were tried (`results.json` → `global_holm`):
- A2's global Holm p is **0.067**.
- Only two rows pass: the par + carry marking (0.009) and the "P4+Q ∩ spread ≥ 260" carry control (0.014).
- The look-ahead printed-only split also shows p 0.000, but it is a diagnostic, not a strategy.

## 3. Sealed 2026 holdout (read once, 9 months; `holdout.log`)
| | exCDI | vs P4+Q (t) |
|---|---|---|
| P4+Q | +0.08 | |
| **A2** | **+2.57** | **+2.49 (4.1)** |
| A2, never-printed at cash | +1.56 | +1.48 (2.3) |
| P4+Q + 20% A2 | +0.58 | +0.50 (4.1) |
| A3 | +2.72 | +2.64 |
| B3 (secondary top-30% carry new issues) | +2.91 | +2.83 (4.8) |
| A4 (Lei 12.431) | −0.89 | −0.97 |
| P7 | +0.76 | +0.68 (2.2) |
| C1 (harness: all new issues) | −2.33 | −2.41 |

- **Why new issues held up.** In the 2026 redemption wave, bonds issued in the last 6 months did not blow up (low early default hazard), while the seasoned book did.
- **Limits.** This is one episode, and the never-printed part is still model-marked.

## Insights
1. **There is no average new-issue concession in the local debenture market.**
   - New series come at or inside the issuer's curve (−16 bps).
   - The first print is at the issue price.
   - Unlike US IG (where a new-issue premium is well documented), the 2021–25 fund-inflow boom let issuers price flat.
2. **The "concession vs peers" is carry.** It is monotone in outcomes, but after controlling for the issue spread it has no incremental power (t ≈ 0 to 1).
3. **An allocation at the bookbuilding rate is worth nothing extra.** Buying the same selection at the first print does as well (A1 − B1 = −0.37, t −0.5), so the primary-fill assumption is not where value lies.
4. **Where the value is:**
   - carry: wide new issues;
   - Lei 12.431 and IPCA new issues, which beat their own peer cohort by about +1.1% over 6 months;
   - first-time issuers, who pay more and outperform (+1.3% vs peers primary, but +0.6 in books and not significant).
5. **Primary-sleeve backtests are dominated by the marking of bonds that never trade** (44% of all new series, 69% of the wide ones). Any claim of primary alpha must state its marking convention. The honest range for A2 is −0.3 to +1.8%/yr vs P4+Q.
6. **Realistic allocations shrink the edge.** At 30% fill of a 20% sleeve, the blended book adds only +0.08%/yr.

## Caveats
- **The bookbuilding-rate fill is optimistic.** It assumes you get allocated at the clearing rate in every selected deal. In hot deals (the ones you want) allocations are scaled back; in cold deals you get filled in full. The 30% fill blend is a crude haircut.
- **The contract rate is today's registry rate.** Repactuated bonds could show a later rate. Few new series 2021–25 are affected.
- **The settlement date is proxied.** It is the registry "Início da Rentabilidade". The true bookbuilding date is a few days earlier, and the curve used for IPCA conversion is from the day before settlement. 12 series that traded before that date were dropped.
- **Never-printed bonds are model-marked.** They have no market price, so defaults and credit events among them are invisible until they print. This is the stale-mark and survivorship zone, now at 44–69% of the sleeve.
- **Screens cover only issuers already on the grid.** EQH, Q and press use issuers with grid bonds. First-time issuers are kept unscreened ("uncovered kept"), as in P4+Q.
- **No letter ratings are available.** The "rating" split uses the issue spread vs the universe median + 100 bps, and negative rating actions in the last 365 days.
- **The sleeve's own construction.** It is value-weighted over live positions and fully invested whenever any position is live. Its vol (1.7) is higher than P4+Q's (1.0), and Sharpe and t are inflated by smooth marks, as for every book.

## Reusable signals (all PIT at `fair_day`, the grid day before settlement)
- `data/history/nightly/w2_primary_market_concession/new_issues.pkl`:
  - identifiers and terms: `codigo, cnpj8, settle, fair_day, s_iss`;
  - concession: `conc_peer, conc_iss, conc` (fair values);
  - issuer and deal: `first_time, n_prior_series, regime, deal_size, incent, guarantee, lead`;
  - screens: `press_neg_30d, rat_neg_365d`;
  - linked offer: `off_*`.
- `research/nightly/w2_primary_market_concession/portfolio.py`:
  - `sleeve(sel, mode='primary'|'secondary', noprint='model'|'carry'|'cash')`: the primary-sleeve engine;
  - `blend(base_daily, sleeve, a)`: same-capital blending;
  - `load()`: new issues plus issuer EQH/Q flags.
- `seasoning.py:add_new(panel, first_print())`: `new_d`, the days since a bond's first SND print, for any harness panel.

Rerun:
```
PYTHONPATH=. PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=2 .venv/Scripts/python.exe research/nightly/w2_primary_market_concession/build_data.py
... analysis.py ; portfolio.py pre ; robustness.py ; seasoning.py ; summarize.py pre ; summarize.py holdout
```
