# w2_pre2021_oos_extension: a genuine pre-2021 out-of-sample test of the frozen rules

**Question.** Every result so far rests on 48 pre-2026 months (2022–25), and the 2026 holdout is partly snooped. Do the frozen P4, P4+Q and P7 rules, the factor_zoo screen and the macro_cycle mom63 gate keep their edges on 2015–2020? That window includes the 2015–16 recession default era and the 2020 COVID credit-fund crash. It was never used for any design choice.

**Answer.** Mostly yes, but only in sign:
- P4 and P4+Q beat the universe out of sample by a wide margin.
- P7 keeps a small positive edge over P4+Q (+0.12%/yr, t 0.86, Holm 1.0).
- The factor_zoo screen flips sign (−0.35%/yr).
- The mom63 gate behaves as drawdown insurance: it pays in 2020 and costs in 2015–19.

No variant beats P4+Q significantly. The sealed 2026 holdout was not touched.

## What was built (all point-in-time; caches in `data/history/nightly/pre2021_oos_extension/`)

| step | file | notes |
|---|---|---|
| SND prints 2014-01..2020-12 | `fetch_snd.py` → `snd_YYYYMM.csv.gz` | Same endpoint as `bias_audit/fetch_snd_range.py`, in 5-day chunks with polite sleeps. One day (2020-12-16) returns 500 persistently and was skipped (`missing_days.txt`). 2021-01..06 comes from the shared `snd_trades_2021*` files. |
| CVM IPE 2013–20, BCB CDI/IPCA | `fetch_aux.py` | IPE is dated by `Data_Entrega`. |
| B3 COTAHIST 2013–20, CVM ITR/DFP 2011–20 | `fetch_eq_fin.py` | COTAHIST closes are unadjusted, with \|daily log move\| > 0.5 zeroed as in the harness. Predecessor tickers are spliced in (e.g. BTOW3→AMER3, CCRO3→MOTV3). brapi-adjusted `equity_daily.pkl` is used from 2020-06. |
| grid / core / panel | `build.py` → `core.pkl`, `panel_M.pkl`, `fund_pre.pkl`, `grid.pkl` | See the construction notes below the table. |
| backtests | `run.py`, `decomp.py` → `results.json`, charts | The harness engine is used unchanged. The rebuilt core is injected into `harness._MEM`, and `backtest(panel=...)` is called with monthly decisions, 126-bday tranches, a 10% issuer cap, and 25 / 50 bps plus rec40. |

**`build.py` construction:**
- Spreads, returns and flags follow `selection.build_panel` + `lab_daily.build` + `run_selection_lab.build_returns` + harness v4.
- The IPCA real curve is the NTN-B curve from Tesouro Direto, the same source as the 2021+ lab. The PRE curve is LTN/NTN-F; the 2021+ lab uses B3 DI×pré.
- Gap moves are booked at realisation. The rec40 scenario is included.
- **Holiday fix:** the contract and IPCA par accrue on B3 business days, not weekdays.
- Fundamentals are CVM-only, with `available_date_strict`, plus the brapi rows of `fundamentals_pit`.
- `p4`, `p4q`, `worstQ` and the equity features are exact copies of the harness code.

**Overlap validation, 2021-03, rebuilt vs harness** (`results.json → overlap_validation_2021`):

| check | value |
|---|---|
| `cdi_bps` correlation | 1.0000 |
| median \|Δ cdi_bps\| | 0.08 bps |
| `resid_z` correlation | 1.0000 |
| universe agreement | 100% |
| P4 flag agreement | 92% |
| worstQ agreement | 92% |
| daily return correlation | 0.96 |

- **P4 flag.** The 8% gap comes from the press filter. Press data starts in 2020-10, so the filter is inactive in the rebuild.
- **worstQ.** The gap comes from differences in fundamentals sources.
- **Daily returns.** The large per-bond differences are all bonds whose harness grid starts on 2021-01-25, whereas the rebuilt grid carries their 2020 gap moves.

## Coverage (thin before 2018)

Mean per monthly decision:

| year | grid bonds | universe | P4 | P4+Q | issuers | fundamentals covered | listed |
|---|---|---|---|---|---|---|---|
| 2015 | 117 | 78 | 22 | 17 | 51 | 89% | 44% |
| 2016 | 127 | 82 | 24 | 17 | 55 | 80% | 44% |
| 2017 | 133 | 88 | 26 | 16 | 54 | 81% | 51% |
| 2018 | 190 | 135 | 39 | 29 | 81 | 79% | 54% |
| 2019 | 252 | 150 | 43 | 33 | 85 | 83% | 57% |
| 2020 | 340 | 231 | 69 | 55 | 126 | 82% | 60% |

The 2021–25 universe had about 700 bonds. Monthly figures are in `coverage_by_month.csv`.

## Results: out of sample, 2015-01..2020-12 (72 months), 25 bps

Stats are on calendar months with NW lag 6. "vs P4+Q" is the paired difference.

| book | exCDI %/yr | vs U (t) | vs P4+Q (t) | Holm (7 variants) | vs P4+Q 2015–16 / 2017–19 / 2020 | Sharpe | maxDD (monthly) |
|---|---|---|---|---|---|---|---|
| U (universe) | 0.92 | – | −1.83 (−3.7) | | −2.85 / −1.90 / +0.40 | 0.49 | −5.3 |
| P4 | 2.60 | +1.68 (3.5) | −0.15 (−0.4) | 1.0 | −0.35 / +0.09 / −0.51 | 1.31 | −3.1 |
| **P4+Q** | **2.75** | **+1.83 (3.7)** | 0 | | | 1.39 | −3.4 |
| **P7** (frozen) | **2.87** | +1.95 (4.0) | **+0.12 (0.86)** | 1.0 | +0.20 / −0.07 / +0.55 | 1.42 | −3.1 |
| P7 without the liquidity cap (AUM R$500m is large for 2015) | 2.96 | +2.05 (4.4) | +0.21 (1.43) | 1.0 | +0.21 / +0.00 / +0.84 | 1.56 | −2.7 |
| factor_zoo screen20 | 2.40 | +1.49 (2.6) | **−0.35 (−1.42)** | 1.0 | −0.67 / −0.18 / −0.20 | 1.36 | −3.3 |
| P4+Q + mom63 gate | 2.89 | +1.97 (3.5) | +0.14 (0.28) | 1.0 | −0.28 / −0.54 / **+3.00** | 1.97 | **−1.1** |
| P7 decomposition: eqh screen alone (EW) | 2.85 | | +0.10 (0.68) | 1.0 | +0.10 / −0.10 / +0.67 | | |
| P7 decomposition: carry weights alone | 2.77 | | +0.02 (0.26) | 1.0 | +0.02 / −0.02 / +0.15 | | |
| IDA-DI (index) | 0.65 | | | | 2015–16 +1.15, 2017–19 +0.68, 2020 −0.44 | 0.28 | −5.8 |

Sensitivities:

| run | P7 vs P4+Q | P7 without liquidity cap vs P4+Q | P4+Q vs U | ZOO vs P4+Q |
|---|---|---|---|---|
| 50 bps | +0.10 (t 0.72) | +0.19 | +1.68 | −0.36 |
| rec40 | +0.12 | +0.21 | +1.87 | −0.35 |

rec40 barely moves: only 17 bonds stop trading below 0.90 in the rebuilt grid.

**Placebos:**
- P4+Q +2.92 vs random books of the same size, which average +1.03 (p95 +1.26), so selection is real. The placebo window is 2015-01..2021-06, including the tail of the grid.
- P7's random-screen placebo (dropping as many listed P4+Q names at random) averages +0.04 vs P4+Q (p95 +0.105). P7's +0.12 sits just above that p95.

**Drawdowns.** Worst daily peak-to-trough of cumulative excess over CDI:

| book | 2015–16 | 2020 | Mar–May 2020 cumulative |
|---|---|---|---|
| U | −3.4% | −4.2% | −2.7% |
| P4 | −2.2% | −4.7% | −2.7% |
| P4+Q | −1.9% | −5.5% | −3.4% |
| P7 | −2.1% | −5.2% | −3.1% |
| mom63-gated P4+Q | −1.9% | −0.6% | +0.5% |
| IDA-DI | −0.7% | −6.9% | −4.5% |

**In-sample reference** (same code, harness core, stats 2022–25): P7 vs P4+Q +0.44 (t 2.6); P7 without the liquidity cap +1.00 (t 3.9); ZOO +0.34 (t 2.3). The out-of-sample edges are roughly one quarter of the in-sample ones, or reversed.

## Insights

1. **The carry-selection core generalises; the add-ons mostly do not.**
   - P4+Q beats the universe by +1.83%/yr (t 3.7) on 72 never-seen months, which is larger than in-sample (+1.12).
   - P7's extra +0.12 is a quarter of its in-sample +0.44 and insignificant.
   - The factor_zoo screen loses 0.35%/yr.
2. **The universe itself was a bad place in 2015–16.**
   - U earned +0.33%/yr over CDI, against +1.15 for IDA-DI.
   - P4+Q beat U by +2.85%/yr (t 4.9) in 2015–16 and by +1.90 in 2017–19.
   - In 2020, U beat P4+Q (+0.40): high-carry names fell hardest in the COVID redemption crash (Mar–May: P4+Q −3.4% vs U −2.7%).
3. **The Q screen's sign holds, its size does not.**
   - P4+Q − P4 is +0.15 (t 0.44), against +0.3 in 2022–25. It is +0.35 in 2015–16, −0.09 in 2017–19 and +0.51 in 2020.
   - Inside P4 the quality composite's 6m IC is actually **negative**: −0.11 (t −2.6). Carry-neutral in the universe it is −0.00.
   - So "drop the worst quintile" helps as a tail screen, not as a ranking.
4. **The equity-health score is the one component with robust out-of-sample information.**
   - Its IC inside P4 is +0.135 (t 3.4, 33 dates with at least 10 listed P4 names).
   - Of P7's +0.12 edge, the eqh screen provides +0.10; carry weighting alone adds only +0.02.
   - In-sample the carry weighting was the bigger piece. Carry's IC inside P4 is +0.18 (t 5.9) out of sample, but equal-weight P4+Q already harvests most of it.
5. **The mom63 gate is insurance, not alpha, and 2020 confirms it.**
   - It avoided almost all of COVID (+3.0%/yr vs P4+Q in 2020; 2020 maxDD −0.6% vs −5.5%).
   - It lost 0.28 in 2015–16 and 0.54 in 2017–19 through whipsaws and switch costs, so the net is +0.14 (t 0.28).
   - Across 2015–2025 it has saved two episodes (COVID, Americanas) and paid for them in between.
6. **The factor_zoo composite is a 2022–25 artefact.**
   - It is negative in every sub-period out of sample, worst in 2015 (−1.4%/yr).
   - This supports the wave-1 verdict that it mostly dodged a handful of blow-ups.

## Caveats

- **Registry snapshot.** Bond terms (contract rate, maturity, indexer, Lei 12.431 flag) come from the 2026 SND registry snapshot. Repactuated bonds carry today's terms back to 2015. Duration factors use today's ANBIMA ratios.
- **Thin cross-section.** P4+Q holds only 16–17 names from 2015–17, and the universe is 78–88 bonds. With the 10% issuer cap the books are concentrated, and the t-stats rest on few names.
- **Press filter inactive.** Google-News press starts in 2020-10, so P4's no-negative-press filter is off before 2021. The rebuilt P4 is therefore "P4 without press".
- **Equity data.** COTAHIST closes are not dividend-adjusted, so eq_r126 and eq_dd252 are slightly biased down for high-yield payers. Moves larger than 50% (splits) are zeroed. Listed coverage is 44–60%.
- **Fundamentals.** They are CVM-only before 2021; there is no brapi parent history before about 2020.
- **PRE curve.** The nominal curve is Tesouro Direto LTN/NTN-F "taxa compra", not B3 DI×pré. This affects only PRE bonds, a small share.
- **P7 liquidity cap.** The R$500m AUM makes the cap bind hard in 2015, and the caps relax ×1.5 until feasible. The uncapped variant is reported as well.
- **Marks and survivorship.** Smooth, stale SND marks inflate t and Sharpe, as in the harness. Survivorship is limited: the registry keeps excluded bonds, and rec40 affects 17 bonds.
- **Splice.** The equity curves use the rebuilt series up to 2021-02-28 and the harness in-sample series from 2021-03-01 (shaded). 2026 is not shown and was not used.
- **Multiple testing.** 7 variants were compared with P4+Q: P4, P7, P7 without the liquidity cap, ZOO, mom63, and the two P7 components. Holm p = 1.0 for all of them.

## Files

- `fetch_snd.py`, `fetch_aux.py`, `fetch_eq_fin.py`: downloads.
- `build.py`: grid, core, panel.
- `run.py`: books, stats, charts.
- `decomp.py`: P7 decomposition and the 7-variant Holm correction.
- `results.json`: all numbers.
- `coverage_by_month.csv`: coverage per monthly decision.
- `equity_curves_2015_2025.png`:
  - top panel: total return CDI × (1 + excess);
  - bottom panel: cumulative excess vs U;
  - shading: the 2021+ in-sample period, 2015–16 and COVID.
- `oos_vs_p4q.png`: cumulative excess vs P4+Q over 2015–2020.
- Reusable data:
  - `data/history/nightly/pre2021_oos_extension/core.pkl`: harness-format core for 2014-02..2021-06;
  - `panel_M.pkl`: monthly panel with `univ`, `p4`, `p4q`, `worstQ`, eq_*, `zoo` inputs and `vol91_brl`;
  - `oos_series.pkl`: daily excess series.

Rerun (about 12 min once downloads are cached):

```
PYTHONPATH=. PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=2 .venv/Scripts/python.exe research/nightly/w2_pre2021_oos_extension/fetch_snd.py 2014 1 2020 12
PYTHONPATH=. .venv/Scripts/python.exe research/nightly/w2_pre2021_oos_extension/fetch_aux.py
PYTHONPATH=. .venv/Scripts/python.exe research/nightly/w2_pre2021_oos_extension/fetch_eq_fin.py
PYTHONPATH=. PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=2 .venv/Scripts/python.exe research/nightly/w2_pre2021_oos_extension/build.py
PYTHONPATH=. PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=2 .venv/Scripts/python.exe research/nightly/w2_pre2021_oos_extension/run.py
PYTHONPATH=. PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=2 .venv/Scripts/python.exe research/nightly/w2_pre2021_oos_extension/decomp.py
```
