# Fund flows & holdings (CVM inf_diario + CDA) vs debentures — nightly `fund_flows`

**Bottom line: clean negative.** Brazilian credit-fund flows are very visible in the data: the 2023 post-Americanas wave took out about 5.5% of credit-fund net assets per month, and there was a smaller late-2024 wave. But flows **follow** returns rather than lead them. None of the 15 flow or holdings variants beats P4+Q out of sample: the best is +0.08%/yr (t 0.86, Holm p = 1). The sealed 2026 holdout confirms this: the best variant lands at −0.00%/yr vs P4+Q.

## Data built (new, all public CVM open data, cached under `data/history/nightly/fund_flows/`)

| file | content | coverage | PIT rule |
|---|---|---|---|
| `raw/inf_diario_fi_*.zip` → `inf.pkl` | daily report for each fund: PL, subscriptions (CAPTC_DIA), redemptions (RESG_DIA), number of investors | 2020-01 → 2026-09; 12,067 funds (every fund that ever held a debenture, plus credit/incentivized/infra-named funds) | the value for day t is used only **5 business days later** |
| `raw/cda_fi_*.zip` → `cda_deb.pkl` | CDA BLC_4 debenture positions: fund × month × SND code (CD_ATIVO), qty, market value, month buys/sells | 2020-01 → 2026-08; 7.2M rows; about 4,000 funds and about 2,000 bonds per month by 2025 | reference month m is used only from **m-end + 60 calendar days**. The legal deadline is about 10 business days; the extra margin covers confidential positions and late filers. |
| `cda_pl.pkl` | fund PL at each CDA month | same | same |
| `fund_class.pkl` | debenture share of PL; **credit fund = debentures ≥ 25% of PL** | 2,005 credit funds holding R$1.1 tn at end-2025 | same as CDA |
| `agg_flows.pkl` / `agg_flows_export.pkl` | aggregate PIT credit-fund net flow over 5/21/63/126 days as % of PL (`af*`), change in investor count (`dcot63`), `n_credit`, `credit_pl` | business days 2020-06 → 2026-09 | already shifted by 5 report days |
| `bond_signals_M.pkl` | per (day, codigo) holdings features for the monthly panel | 94% of universe rows have ≥ 1 fund holder; median 59 holders | as above |

**Bond features** (`features.py`, per decision date, from the latest available CDA vintage and flows lagged by 5 days):
- `nh`, `lnh`: number of fund holders.
- `dnh3`, `dq3`: 3-month change in holder count and in the quantity held by funds.
- `nb3`: funds' net buying (buys − sells) over 3 months / value held.
- `hhi`: concentration of holders.
- `cshare`: share of the value held by credit funds.
- `press21`, `press63`: holding-weighted 21/63-day net flow of the holders (fire-sale pressure, Coval–Stafford style).
- `stress63`: share of holders suffering outflows above 10% over 63 days.

## Method

- Everything runs on the common harness v4: monthly decisions, 126-bday tranches, 25 bps, 10% issuer cap, pre-2026 only.
- Paired Newey–West t vs P4+Q.
- Holm correction across **all 15 variants tried**: 11 in `run.py` (6 cross-sectional, 4 overlays, plus the IDA overlay reference) and 4 follow-ups.
- Cross-sectional variants drop the worst quintile of a feature from P4+Q. Quintiles are set among covered universe names; uncovered names are kept.
- Timing variants are P4+Q with a 0/1 overlay built from aggregate credit-fund flows. The overlay is lagged one more day by the harness, and switching costs are charged.

## Results (pre-2026, 48 months, 25 bps)

Baselines: U +1.10 exCDI; P4 +1.89 (−0.33 vs P4+Q, t −2.3); **P4+Q +2.22** (vs U +1.12, t 2.4).

| variant | exCDI % | vs U % | vs P4+Q % | t | Holm p (15) | 22–23 / 24–25 vs P4+Q | 50 bps vs P4+Q | rec40 vs P4+Q |
|---|---|---|---|---|---|---|---|---|
| X1 ex top-quintile stress63 (holders in >10% outflow) | 2.13 | 1.03 | −0.09 | −1.0 | 1 | +0.05 / −0.23 | −0.12 | −0.10 |
| X2 ex bottom-quintile press21 (holders' 21d outflow) | 2.13 | 1.03 | −0.09 | −1.1 | 1 | +0.07 / −0.25 | −0.12 | −0.08 |
| X3 ex bottom-quintile nb3 (funds net selling) | 2.13 | 1.03 | −0.09 | −1.9 | 0.83 | −0.13 / −0.05 | −0.11 | −0.09 |
| **X4 ex bottom-quintile cshare (few credit-fund holders)** | **2.30** | **1.20** | **+0.08** | **0.86** | 1 | −0.08 / +0.24 | +0.08 | +0.08 |
| X5 ex orphans (no fund holder) | 2.06 | 0.96 | −0.16 | −2.4 | 0.27 | −0.13 / −0.19 | −0.17 | −0.16 |
| X6 ex bottom-quintile combo (nb3 + cshare − stress63) | 2.15 | 1.05 | −0.07 | −1.0 | 1 | −0.15 / +0.01 | −0.11 | −0.08 |
| T1 overlay: in if af21 > 0 | 1.75 | 0.64 | −0.47 | −1.2 | 1 | +0.08 / −1.02 | −0.84 | −0.47 |
| T2 overlay: in if af63 > 0 | 1.68 | 0.57 | −0.54 | −1.7 | 1 | −0.38 / −0.71 | −0.60 | −0.54 |
| T3 overlay: out if af21 < −2% | 2.16 | 1.06 | −0.06 | −0.4 | 1 | −0.12 / 0.00 | −0.15 | −0.06 |
| T4 overlay: out if IDA regime off AND af21 < 0 | 2.10 | 1.00 | −0.12 | −0.5 | 1 | +0.27 / −0.50 | −0.51 | −0.12 |
| (ref) P4+Q + IDA overlay | 1.74 | 0.64 | −0.48 | −1.3 | 1 | +0.32 / −1.28 | | |
| F1 weekly, 21d tranches, ex press21 (vs P4+Q on the same engine, +1.87) | 1.75 | | −0.11 | −1.0 | 1 | +0.12 / −0.35 | −0.26 | |
| F2 weekly, 21d tranches, ex stress63 | 1.72 | | −0.15 | −1.5 | 1 | −0.14 / −0.15 | −0.30 | |
| F3 ex low cshare within kind | 2.25 | | +0.03 | 0.45 | 1 | +0.12 / −0.06 | | |
| X7 ex low cshare among DI_SPREAD only | 2.22 | 1.12 | +0.01 | 0.12 | 1 | −0.06 / +0.07 | +0.00 | |

**Best variant (X4):**
- Sharpe 2.17 vs 2.13 for P4+Q; maxDD −1.43% vs −1.39%.
- Placebo (drop the same number of P4+Q names at random, 20 draws): mean −0.03, p95 +0.06. X4's +0.08 sits just above the p95, but Holm across the 15 variants gives p = 1.

**Cohort level vs P4+Q** (per-date tranche returns, %/yr, NW t):

| variant | 1 month | 3 months | 6 months |
|---|---|---|---|
| X2 ex press21 | **+0.32 (t 2.3)** | −0.03 | −0.11 |
| X6 combo | +0.32 (t 1.8) | +0.02 | −0.08 |
| X1 ex stress63 | +0.25 (t 1.3) | −0.12 | −0.11 |
| X5 ex orphans | −0.20 (t −2.8) | −0.08 | −0.13 (t −3.5) |
| X4 ex low cshare | +0.05 | +0.11 | +0.06 |

**Sealed holdout (2026-01 → 2026-09, reported once, nothing chosen from it):**
- X4 exCDI +0.08, vs P4+Q **−0.00** (t −0.04). P4+Q itself is at +0.08 exCDI and +3.34 vs U.
- T3 (second-best, out if af21 < −2%) comes in at **−1.89 vs P4+Q (t −3.1)**. The 2026 redemption wave triggered the exit, and the book missed the recovery.

## Insights

1. **Flows chase returns; they do not lead them.**
   - The 21-day credit-fund flow correlates +0.41 with P4+Q's *previous*-month return (IDA-DI: +0.60).
   - In r(m+1) = a + b·flow(m) + c·r(m), flow is never significant: t = 0.6 for af21, 1.4 for af5, 0.07 for af63.
   - The 2023 outflow wave started *after* the Americanas event (Feb 2023), and it ended (flows positive again from about Aug 2023) while spreads were still near their 200 bps peak.
   - Flow overlays therefore behave like the IDA momentum overlay: they pay switching costs and exit near the bottom. T1 −0.47, T2 −0.54, and the 2026 holdout T3 −1.9.
2. **Fire-sale pressure is real but lasts about a month.**
   - Dropping bonds whose holders face outflows adds +0.32%/yr at a 1-month horizon (t 2.3).
   - The effect is gone by 3 months and reverses slightly at 6 (−0.11), consistent with Coval–Stafford price pressure followed by reversal.
   - Harvesting it needs weekly decisions with 21-day tranches, i.e. about 14×/yr turnover. Costs eat it: −0.11 at 25 bps, −0.26 at 50 bps (t −2.1).
3. **"Orphan" debentures, held by no investment fund, outperform.**
   - They are 6% of the universe and 9% of P4+Q.
   - Removing them costs −0.16%/yr (t −2.4, cohort 6m t −3.5), and the result survives rec40.
   - Raw 6-month forward return: 2.0% for orphans vs 1.15% for held bonds, with lower carry (110 vs 118 bps) and fewer trades.
   - This looks like a neglect/illiquidity premium, but it sits in the stale-mark / survivorship zone that the harness caveats warn about. Treat it as a hypothesis, not a tilt.
4. **Credit-fund ownership share is a carry proxy.**
   - Within DI-spread bonds, `cshare` has an IC of +0.13 at 3 months (t 5.4). Carry-neutral, it is +0.01 (t 0.8).
   - Credit funds own the higher-carry names, and P4+Q already selects on carry, so X4, F3 and X7 are all about 0.
5. **Holding changes carry no information.** Fund net buying (`nb3`), holder-count changes (`dnh3`), quantity changes (`dq3`) and holder concentration (`hhi`) all have |IC| ≤ 0.03 and are insignificant, raw or carry-neutral.
6. **The structural boom.**
   - Credit funds (debentures ≥ 25% of PL) grew from 597 to 2,005, and their PL from about R$0.25 tn to R$1.1 tn (2022-01 → 2025-12).
   - Positions per CDA month grew from 44k to more than 200k.
   - The investor count of credit funds (`dcot63`) fell −14% in Q1 2026. That is the 2026 redemption wave now in the holdout.

## Caveats

- **CDA files are rewritten by CVM**; 2023 files were re-generated in 2024. They may therefore contain positions that were confidential or filed late at the time. The 60-day lag mitigates but does not remove this. The last months are visibly incomplete: the 2026-07 vintage has only 1,272 credit funds, vs about 2,300.
- **Flows are measured where the debentures are held**, i.e. at master funds. Feeder (FIC) flows reach them only through master subscriptions and redemptions. Mergers and incorporations create spikes: fund ratios are clipped to ±100% and funds with PL below R$5m are ignored.
- **The harness caveats apply**: smooth and stale marks inflate t and Sharpe; there is survivorship after the last print; and execution is at SND marks. Overlays switch instantly at the mark, which is optimistic, and they still lose.
- **Directions were set before the book tests**, from the fire-sale hypothesis (X1–X3, X6, T1–T4). X4 and X5 were symmetric guesses. F1–F3 and X7 were chosen *after* seeing ICs and cohorts, and all are counted in the Holm correction (15).

## Reusable signals

- `research/nightly/fund_flows/features.py`:
  - `signals(panel)`: merge onto any harness panel; returns `day, codigo, nh, lnh, dnh3, dq3, nb3, hhi, cshare, press21, press63, stress63`.
  - `agg_flows()`: business-day aggregate credit-fund flows, already lagged.
  - `fund_class()`: PIT credit-fund flags.
  - `fund_flow_ratios(W)`: fund-level flow ratios, not lagged. Apply `_lagged` or a 5-day shift before using them.
- Cached: `data/history/nightly/fund_flows/bond_signals_M.pkl` (monthly panel, keyed by day and codigo, with cnpj8) and `agg_flows_export.pkl`.
- Best candidates for a combiner:
  - `press21` as a short-horizon (≤ 1 month) avoid signal;
  - `nh == 0` (orphan) as a possible positive tilt, after a survivorship check;
  - `af21` as a regime *descriptor* (e.g. for conditioning), not as a timing rule.

## Files and rerun

- Code:
  - `download.py`: CVM downloads, about 1.5 GB, a few minutes.
  - `build_data.py`: parse, about 15 min.
  - `features.py`
  - `run.py`: main analysis; `--holdout` adds the sealed block.
  - `followup.py`, `followup2.py`
- Outputs:
  - `results.json`
  - `equity_total_return.png`
  - `cum_excess.png`
  - `flows_vs_spreads.png`

```
cd renda-fixa-monitor
export PYTHONPATH=. PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
.venv/Scripts/python.exe research/nightly/fund_flows/download.py all
.venv/Scripts/python.exe research/nightly/fund_flows/build_data.py all
.venv/Scripts/python.exe research/nightly/fund_flows/run.py --holdout
.venv/Scripts/python.exe research/nightly/fund_flows/followup.py
.venv/Scripts/python.exe research/nightly/fund_flows/followup2.py
```
