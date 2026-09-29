# bias_audit: independent bias audit of P4 / P4+Q (harness v4)

**Bottom line.** The v4 P4+Q level over CDI is overstated by about 1 pp/yr: +2.22%/yr falls to **+1.20%/yr** under conservative assumptions. The universe falls further, from +1.10 to **+0.25**. So the *relative* edge survives:

- **P4+Q − U** goes from +1.12 (t 2.4) to **+0.95%/yr (t 2.05)**. That is under conservative assumptions, before 2026, with the same assumptions applied to both sides.
- **P4 − U** goes from +0.79 to **+0.75** (t 1.4).
- **P4+Q − P4** goes from +0.33 (t 2.3) to **+0.20 (t 1.4)**. The Q filter is the fragile part. A 90-day lag on fundamentals alone removes 0.07 of it.

Two things drive the level haircut:
- **Realistic, liquidity-dependent costs** take −0.8 to −1.0 pp.
- **A new bug in the patched returns.** They accrue the contract spread on *weekdays*, not B3 business days, which adds −0.13 pp on every book.

**Nothing here "beats P4+Q".** This is an audit. Its product is the haircut table and the honest baselines.

All numbers are pre-2026: monthly decisions, 126-bday tranches, 48 months, Newey-West lag 6. The engine is `engine.py`, a copy of harness `_tranche_book` with hooks. With the hooks off, it reproduces `H.baseline('U'/'P4Q')` to 0.0 max abs difference.

## Haircut table

### Individual: each bias alone on top of v4

All figures are ann. %/yr. Every row is a separate backtest in which **U, P4 and P4+Q all get the same assumption**.

| bias / check | U | P4 | P4+Q | P4+Q−U (t) | P4+Q−P4 (t) | Δ P4+Q level | Δ(P4+Q−U) |
|---|---|---|---|---|---|---|---|
| **v4 baseline** | 1.10 | 1.89 | 2.22 | 1.12 (2.4) | 0.33 (2.3) | | |
| Holiday accrual fix (new bug) | 0.97 | 1.76 | 2.09 | 1.12 (2.4) | 0.33 | **−0.13** | +0.01 |
| Survivorship rec40 (harness: stops < 0.90) | 1.07 | 1.80 | 2.19 | 1.11 | 0.39 | −0.03 | −0.01 |
| Survivorship: PIT distress issuers, silent bonds → 40% | 1.05 | 1.82 | 2.15 | 1.10 | 0.33 | −0.07 | −0.02 |
| Survivorship harsh: every stop < 0.98 → 40% | 0.95 | 1.39 | 1.83 | 0.89 (1.8) | 0.45 | −0.38 | −0.23 |
| Raw prints at execution (no spike cleaning) | 1.10 | 1.90 | 2.23 | 1.13 | 0.33 | +0.01 | +0.01 |
| Exit only at a real trade (no stale-mark exits) | 1.10 | 1.95 | 2.27 | 1.17 | 0.32 | +0.05 | +0.06 |
| Entry latency +1 bday | 1.08 | 1.90 | 2.23 | 1.15 | 0.33 | +0.01 | +0.04 |
| Entry latency +5 bdays | 1.00 | 1.84 | 2.14 | 1.13 (2.5) | 0.30 | −0.08 | +0.02 |
| Cost flat 50 bps | 0.89 | 1.54 | 1.84 | 0.95 | 0.30 | −0.38 | −0.17 |
| **Cost by liquidity bucket (institutional Roll: DI 45–65, IPCA 136–169 bps round trip)** | 0.47 | 1.15 | 1.41 | 0.94 (2.0) | 0.26 | **−0.81** | −0.18 |
| Cost from ANBIMA bid/ask (DI 64–77, IPCA 82–185 bps) | 0.52 | 1.10 | 1.37 | 0.85 (1.8) | 0.27 | −0.85 | −0.26 |
| Cost from SND intraday range (flawed, see caveats) | 0.67 | 1.78 | 2.11 | 1.43 | 0.33 | −0.11 | +0.32 |
| Worst-of-day execution: buy at PU max, sell at PU min (stress) | −0.87 | 0.86 | 1.31 | 2.18 | 0.45 | −0.91 | (+1.07, artefact) |
| Fundamentals +90d lag (Q filter) | 1.10 | 1.89 | 2.15 | 1.05 | 0.26 (2.0) | −0.07 | −0.07 |
| Fundamentals without brapi parent rows | 1.10 | 1.89 | 2.21 | 1.11 | 0.33 | −0.00 | −0.00 |
| Fundamentals non-strict dates (less conservative) | 1.10 | 1.89 | 2.21 | 1.11 | 0.32 | −0.01 | −0.01 |
| Parameter snooping (72 neighbours of P4+Q) | | | | median 1.05 (p10 0.78, p90 1.41) | | | **−0.07** |

### Cumulative: building the 'honest' baseline

| step | U | P4 | P4+Q | P4+Q−U (t) | P4−U (t) | P4+Q−P4 (t) | P4+Q−U 22–23 / 24–25 |
|---|---|---|---|---|---|---|---|
| v4 baseline | 1.10 | 1.89 | 2.22 | 1.12 (2.41) | 0.79 (1.59) | 0.33 (2.28) | 1.63 / 0.60 |
| + holiday accrual fix | 0.97 | 1.76 | 2.09 | 1.12 (2.43) | 0.80 | 0.33 | 1.64 / 0.61 |
| + distress-issuer recovery 40% | 0.92 | 1.69 | 2.02 | 1.10 (2.30) | 0.78 | 0.33 | 1.68 / 0.53 |
| + raw prints at execution | 0.91 | 1.71 | 2.03 | 1.12 (2.37) | 0.79 | 0.33 | 1.68 / 0.56 |
| + exit only at a real trade | 0.88 | 1.74 | 2.05 | 1.17 (2.38) | 0.85 | 0.31 | 1.76 / 0.57 |
| + liquidity-bucket costs | 0.25 | 1.00 | 1.24 | 0.99 (2.02) | 0.75 (1.41) | 0.24 (1.54) | 1.65 / 0.33 |
| **+ fundamentals lag 90d = HONEST** | **0.25** | **1.00** | **1.20** | **0.95 (2.05)** | **0.75 (1.41)** | **0.20 (1.41)** | **1.47 / 0.44** |
| Stress (honest + worst-of-day prices + harsh survivorship) | −1.80 | −0.46 | −0.09 | 1.72 (artefact of the IPCA range) | | | |

**Total haircut, v4 to honest:**

| | v4 | honest | haircut |
|---|---|---|---|
| U | +1.10 | +0.25 | −0.85 |
| P4 | +1.89 | +1.00 | −0.89 |
| P4+Q | +2.22 | +1.20 | −1.02 |
| P4+Q−U | +1.12 | +0.95 | −0.16 |
| P4+Q−P4 | +0.33 | +0.20 | −0.13 |
| snooping (applied to the relative edge) | | | ≈ −0.07 more |

**Honest P4+Q vs v4 P4+Q:** −1.02%/yr, NW t −7.6, Holm p ≈ 0. The comparison covers 21 rows, and 162 backtests were run in total, counting the parameter neighbourhood.

## Other checks

- **Stale-mark smoothing (Getmansky-Lo-Makarov MA(2) on monthly excess):**
  - honest P4+Q: θ0 0.67. Sharpe 1.23 observed falls to **0.87** de-smoothed, and vol rises from 0.98 to 1.37%.
  - v4 P4+Q: Sharpe falls from 2.13 to 1.57.
  - The spread P4+Q − U is barely smoothed (θ0 0.92). Its Sharpe is 0.96 observed and 0.92 de-smoothed. NW t is 2.04 at lag 6 and 2.24 at lag 12.
- **Placebo on the honest engine:** random books with the same size and churn score −1.23%/yr vs honest U (p95 −1.05). Honest P4+Q scores +0.95. Random books lose to U because they churn like P4+Q (about 3×/yr turnover), while U's tranche book turns over about 1.7×/yr and pays about 0.65%/yr in honest costs.
- **Universe look-ahead:** every universe row has a trade date ≤ the decision day, a mark ≤ 7 days old and a ratio in [0.9, 1.1]. All sampled entries are strictly after the decision. The +1 day and +5 day latency tests cost ≤ 0.08%/yr, so the edge does not depend on same-day information.
- **Off-grid bonds:**
  - 527 SND codes, 7.3% of pre-2026 volume, never reach the grid. 226 of them have no `% PU da Curva`, mostly %CDI, IGP-M or TR bonds.
  - They include Americanas' LAME29 / LAMEA1-3 / LAMEA6 / LAMEA8 / BTOW15, CVC, Oi's OIBRA2 and AZUL11.
  - Off-grid bonds crash below 70% of par slightly more often: 6.6% vs 4.1%. This is a mild universe-definition bias that flatters U. It cannot flatter P4, because P4 can only buy on-grid bonds.
- **Defaults:** `default_cases.csv` lists 33 PIT distress issuers. `distress_events_pit.csv` has their event dates.
  - P4+Q had a live position at the event in 12 of them, and P4 in 20.
  - Those positions were in Dasa, Giga+, Oncoclínicas, Aeris, Ambipar and Americanas.
  - Most kept trading, so the losses are in the marks: Oncoclínicas −28% and Ambipar −17% mean bond return around the event.
  - The real hole is bonds that print near par and then go silent, as Americanas LAMEA4/5 did. Adding a 40% recovery for these costs only 0.07%/yr.

## Insights

1. **New bug in `run_selection_lab.build_returns`, which is used by the harness.** `np.busday_count` counts weekdays. The SND par curve (`% PU da Curva`) accrues on B3 business days. Every weekday holiday therefore books one extra day of contract accrual:
   - about 0.9 bp for DI+ bonds;
   - 2.7 bp for IPCA bonds;
   - 5.4 bp for PRE bonds.

   That is 48 holidays over 2021–25 and 54,610 rows. It inflates **all** book levels by 0.13%/yr and the IPCA-heavy universe the most. It cancels in P4+Q − U. The fix is in `build.py: R_hol()`.
2. **Costs dominate the level; the relative edge is robust to them.** With institutional costs by liquidity bucket, U drops 0.64 and P4+Q drops 0.81. P4+Q − U falls only 0.18, because the universe book pays too. Two independent estimates agree within 0.04 pp: ANBIMA indicative bid/ask (Sep 2026 cross-section, 10k quotes) and marks_liquidity's Roll model.
3. **The Q filter is the fragile part of P4+Q.**
   - A 90-day lag on fundamentals cuts P4+Q − P4 from 0.33 to 0.26. In the honest stack the figure is 0.20 (t 1.4).
   - Dropping brapi rows or using non-strict dates changes nothing, so restatement leakage is not the issue. Freshness is.
   - In 2022–23 the honest P4+Q − U is +1.47. In 2024–25 it is +0.44.
4. **P4 was not cherry-picked.** It sits at the 54th percentile of 72 neighbouring rules (carry cut 20/30/40%, rich threshold none/−1/−1.5/−2, press on/off, worst-Q 10/20/30%), and P4+Q at the 57th. The snooping haircut is only about 0.07 pp. Every neighbour has P4+Q − U > +0.7.
5. **Stale exits were not flattering the books.** Forcing exits at real trades *raises* P4+Q by 0.05. Execution at the raw print instead of the spike-cleaned one is neutral. Removing both engine optimisms costs nothing. The optimism lies in the price level (costs), not the timing.
6. **Worst-of-day execution is not a valid cost model for IPCA bonds.** The SND intraday range grows with liquidity: the median round trip is 525 bps in the most liquid IPCA quintile, because retail and institutional prints mix. DI floaters show a median range of about 1 bp. Using it as a cost punishes the IPCA-heavy universe and *inflates* P4+Q − U to +2.2, so treat it only as a level stress. Under it, P4+Q earns about CDI (−0.09).

## Sealed holdout (2026-01 to 2026-09, reported once, spec frozen before)

| | v4 | honest |
|---|---|---|
| U | −3.26 | −3.96 |
| P4 | −0.73 | −1.68 |
| P4+Q | +0.08 | |
| P4+Q (lag-90 Q) | −0.35 | −1.44 |

- Honest P4+Q − honest U is +2.52%/yr, t 2.8, over 9 months.
- Honest P4+Q vs v4 P4+Q is −1.52%/yr.
- The relative edge held. The level did not.

## Files

- `fetch_snd_range.py` downloads the SND daily prices **with PU min / max** for 2021-06 to 2025-12. They go to `data/history/nightly/bias_audit/snd_range_YYYYMM.csv.gz`: 527k rows, dated by trade date and published after the close.
- `build.py` builds the holiday correction (`hol.pkl`, `R_hol(R)`), the execution-penalty matrices (`exec.pkl`) and `lab_min.pkl`.
- `engine.py` is the hookable tranche engine: per-bond costs, entry delay, buy / sell penalties, real-trade exits and scenario returns.
- `run.py` runs all scenarios, the cumulative build-up, the snooping grid, GLM, the placebo, the default table, the charts and the holdout (with `--holdout`).
- Outputs:
  - `results.json`;
  - `haircut_individual.csv` and `haircut_cumulative.csv`;
  - `compare_vs_base_P4Q.csv`;
  - `param_neighbourhood.csv`;
  - `default_cases.csv` and `distress_events_pit.csv`;
  - `equity_total_return.png` and `cum_excess.png`.
- **Reusable pieces:**
  - `data/history/nightly/bias_audit/honest_daily.pkl` holds the daily net excess of honest and v4 U / P4 / P4+Q.
  - `build.R_hol(R)` applies the holiday fix to any return matrix.
  - `run.cost_vectors(...)` gives per-bond round-trip costs.
  - `run.worstq_variant(panel, lag_days, strict, drop_brapi, q)` gives a lagged worst-quality flag.
  - `distress_events_pit.csv` gives (cnpj8, event_date, name) and is PIT: first DI+ mark below 0.85, or the public filing date.

## Rerun

```
PYTHONPATH=. PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/Scripts/python.exe research/nightly/bias_audit/fetch_snd_range.py 2021 6 2025 12
PYTHONPATH=. PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/Scripts/python.exe research/nightly/bias_audit/build.py
PYTHONPATH=. PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/Scripts/python.exe research/nightly/bias_audit/run.py --holdout
```

Runtime is about 10 minutes and peak memory about 1.5 GB.

## Caveats

- **The ANBIMA bid/ask comes from a single cross-section (Sep 2026, 10 days).** Historical ANBIMA files return 404. It is applied back to 2022 as a liquidity-bucket table.
- **Liquidity quintiles use each bond's lifetime ADV**, which contains mild look-ahead. This is fine for a cost haircut but must not be used as a signal.
- **The distress recovery rule is a judgement call.**
  - Rule: a bond went silent within 180 days before, or 365 days after, its issuer's credit event, or it stopped below 0.97 at a distressed issuer. Such bonds go to 40% of par.
  - Some of the 25 bonds hit were probably redeemed at par, which makes the rule conservative.
  - A first, over-inclusive list used `cdi_bps > 1000`, which is IPCA-conversion noise. It flagged 83 issuers, including Sabesp and Petrobras, and 199 bonds redeemed at par. It was rejected, not used.
- **Not fixed:**
  - static contract terms (repactuations);
  - the static duration factor behind `cdi_bps` and the rate hedge;
  - IPCA "rate-hedged" returns keep the NTN-B-vs-CDI carry, so they are not a true swap hedge;
  - the IDA overlay is not audited (it is not in the tranche baselines).
- **Capacity is not included.** marks_liquidity shows that the edge shrinks further at R$500m+.
- **Holdout:** 9 months, a bad universe year, and not evidence on its own.
