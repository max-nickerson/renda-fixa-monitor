# marks_liquidity: better prices, liquidity and execution for P4+Q

**Bottom line.** Better marks don't improve selection. Relative value (RV) is not a print-noise artifact: it works just as well on filtered fair values. What changes the picture is **execution**. Realistic, liquidity-dependent costs cut P4+Q from +2.2%/yr to about +1.2%/yr over CDI with no size constraint. They cut it to about +0.4%/yr at R$500m and about 0 at R$2bn. Sizing positions by liquidity recovers only about +0.1%/yr. None of the 7 variants beats P4+Q after Holm correction.

All results are pre-2026 (sealed holdout rule) and use harness v4 unless stated otherwise: monthly decisions, 126-bday tranches, next-trade execution.

## What was built (point-in-time)

| piece | file | PIT rule |
|---|---|---|
| Trade table: 653k bond-days, with R$ volume, number of trades, ticket and CDI+ spread per print | `build_marks.py` -> `data/history/nightly/marks_liquidity/trades.pkl` | SND daily aggregates, dated by trade date (published after the close) |
| Peer factor: median spread change of bonds printed on consecutive days, per kind × incentive | `peer_factor.pkl` | only prints dated ≤ t |
| **(a) Kalman fair spread** (`kf_s`, `kf_sd`), one filter per issuer | `marks.pkl` | filtered, never smoothed: uses prints ≤ day |
| **(b) Clean / volume-weighted marks** (`cl_s`, `vw_s`) | `marks.pkl` | the outlier test only looks backward |
| **(c) Liquidity features**: `adv63`, `tdays63`, `ticket63`, `ntr63`, and a bond-level Roll half-spread (`roll_hs`) | `marks.pkl` | past 63 bdays, or 252 for Roll. Each Roll product is dated at the later print |
| Cost table: Roll half-spread by indexer group × ADV quintile | `cost_table.json`, `results.json["costs"]` | estimated only on prints before 2024 |
| **Reusable signals** | `signals.py: signals(panel)` | merges onto any harness panel. Adds `kf_s`, `kf_gap(_z)`, `kf_resid_bps/z`, `vw_resid_*`, `adv63`, `liq_q`, `cost_rt_bps` |

### How each piece works

**(a) Kalman fair spread.**
- The state has an issuer-common random walk plus one random walk per bond, around the peer factor.
- Observation noise depends on the indexer group, the duration and the number of trades.
- Prints with |z| > 2.5 are gated with bounded influence.
- Parameters come from robust (MAD) variance-vs-horizon fits on prints before 2024 (`params.json`).

**(b) Clean and volume-weighted marks.**
- Odd lots are dropped: a single trade under R$200k, or total volume under R$50k.
- Outliers are dropped: more than max(40 bps, 4 × MAD) away from the median of the last 5 clean prints. The history resets after 3 same-side outliers.
- `vw_s` is the volume-weighted spread of the last 5 or fewer clean prints within 10 bdays.

## Results

### 1. Print noise and cost of trading

**Noise in the marks.**
- **DI+ floaters:** typical noise is small, about 4 bps of price, but heavy-tailed. The Roll half-spread is about 40 bps on all prints and about 23 bps on institutional prints (≥ R$200k).
- **IPCA/PRE:** noise is huge. The robust typical print noise is about 75 bps of price, and day-to-day spread changes have a robust sd of about 27 bps with no horizon slope. The Roll half-spread is about 124 bps on all prints and 77 bps on institutional prints.

**Print-size tiers in IPCA bonds** (deviation from the Kalman prediction, median):
- Small tickets (under R$200k) print close to fair, about −7 to −9 bps.
- Tickets of R$1–5m print about 38 bps cheaper.
- Tickets over R$5m print about 60 bps cheaper.

**DI bonds show no such tiering** (under 1 bp).

**Estimated round trip by ADV quintile, institutional tier (the main cost model):**

| ADV quintile | DI | IPCA |
|---|---|---|
| 1 (least liquid) | 65 bps | 169 bps |
| 5 (most liquid) | 45 bps | 136 bps |

Taking all prints, the round trip is about 1.75 times higher. The harness assumes 25 bps.

### 2. Does RV still work on better marks? Yes, and it is not bounce.

Monthly Spearman IC against `fwd_126`:

| signal | IC | t |
|---|---|---|
| `resid_bps` on raw marks | 0.209 | 11.5 |
| `kf_resid_bps` | 0.208 | 15.1 |
| `vw_resid_bps` | 0.207 | 14.5 |
| noise part (raw − KF resid) | 0.006 | 0.3 |
| `kf_gap` (last print − fair) | 0.049 | 3.2 |
| carry, raw vs KF | 0.142 vs 0.136 | |

- The noise component carries no information.
- RV is as strong in illiquid bonds (IC 0.19) as in liquid ones (0.21).
- It is much stronger in DI (0.38) than in IPCA (0.16), which is consistent with IPCA marks being noisy.

### 3. Selection variants

Harness engine, 25 bps, compared with P4+Q (+2.22 exCDI, +1.12 vs universe). **7 variants tried**; Holm p = 1.0 for all.

| variant | exCDI | vs U | vs P4+Q | t | 22–23 / 24–25 | at 50 bps vs P4+Q | liquidity-bucket cost vs P4+Q |
|---|---|---|---|---|---|---|---|
| P4Q_kf (KF carry + KF rich) | 2.22 | 1.12 | +0.00 | 0.1 | −0.01 / +0.02 | +0.01 | +0.03 |
| P4Q_kf_rich_only | 2.21 | 1.10 | −0.01 | −1.4 | 0.00 / −0.02 | −0.01 | −0.02 |
| P4Q_vw | 2.21 | 1.11 | −0.01 | −0.3 | −0.04 / +0.02 | −0.01 | −0.01 |
| **P4Q_gapfilter** (skip bonds whose last print is > 2 sd rich vs own fair) | **2.27** | **1.16** | **+0.05** | 1.1 | +0.04 / +0.05 | +0.04 | +0.02 |
| P4Q_liq_q2plus (drop the least-liquid ADV quintile) | 2.08 | 0.97 | −0.14 | −1.1 | −0.04 / −0.25 | −0.15 | −0.05 |
| P4Q_netcost (carry − estimated round-trip cost) | 2.15 | 1.05 | −0.06 | −0.3 | +0.31 / −0.44 | −0.05 | +0.21 (t 0.8) |
| P4Q_LA @R$500m (own cost model, vs P4+Q @R$500m) | | | +0.10 | 1.1 | | | |

**Placebo for the best variant (gapfilter):** +2.27, against random books of the same size at +0.83 (p95 +1.00).

### 4. Capacity

**Setup.**
- Costs are per bond: the half-spread of the bond's ADV quintile (institutional tier) plus square-root impact, 1.0 × σ_daily × √(trade / ADV63).
- Fills are limited to 20% of the bond's actual SND volume in the 20-bday entry window. The unfilled part stays in cash.
- Exits are assumed filled, which is optimistic.

| AUM | U exCDI | P4+Q exCDI | P4+Q fill | P4+Q cost %/yr (spread + impact) | P4Q-LA exCDI | LA fill |
|---|---|---|---|---|---|---|
| no constraint (bucket cost only) | 0.32 | 1.16 | 1.00 | 1.39 | | |
| all-prints Roll (upper-bound cost) | −0.32 | 0.20 | 1.00 | 2.32 | | |
| R$100m | 0.29 | 0.82 | 0.87 | 1.11 + 0.37 | 0.86 | 0.90 |
| R$500m | 0.17 | 0.41 | 0.77 | 0.95 + 0.58 | 0.51 | 0.83 |
| R$2bn | −0.03 | 0.04 | 0.59 | 0.71 + 0.70 | 0.12 | 0.73 |

- **P4+Q − U** is +0.53 (t 1.1) at R$100m, +0.23 at R$500m and +0.06 at R$2bn.
- **Scale of the market:** total SND volume is about R$2.1–2.4bn per bday in 2024–25, against a median bond ADV of about R$0.9m.
- **P4+Q tilts toward illiquid bonds:**
  - 27% of P4+Q is in ADV quintile 1, against 20% of the universe;
  - the median P4+Q bond prints on 29% of days, against 44% for the universe.

### 5. De-smoothed risk (Getmansky-Lo-Makarov MA(2) on monthly excess)

| book | Sharpe observed | Sharpe GLM | Lo-adjusted | θ0 | vol observed → true |
|---|---|---|---|---|---|
| U | 0.74 | 0.51 | 0.55 | 0.65 | 1.50 → 2.16 |
| P4 | 1.81 | 1.31 | 1.38 | 0.68 | 1.05 → 1.44 |
| P4+Q | 2.13 | 1.57 | 1.61 | 0.70 | 1.04 → 1.41 |
| IDA-DI (the index itself) | 0.81 | 0.52 | 0.58 | 0.57 | 1.42 → 2.19 |
| P4+Q − U (the spread) | 1.06 | 0.94 | 1.06 | 0.82 | about 1.05 → 1.19 |

- About 30–35% of the monthly return of debenture books shows up with a lag of 1–2 months.
- True vol is about 1.4–1.5 times the observed vol, and Sharpe is about 25–30% lower.
- The relative bet (P4+Q vs U) is much less smoothed, because it largely cancels out.

### 6. Sealed holdout (2026, reported once, all choices frozen before)

| | exCDI | vs P4+Q |
|---|---|---|
| P4Q_gapfilter | +0.49%/yr | +0.41 (t 1.5) |
| P4+Q | +0.08%/yr | |

The universe did badly in 2026, so P4+Q is +3.3 vs U. There are only about 8 months, so this is not evidence.

## Insights

1. **Stale and noisy marks are not what drives RV.** Filtering the marks leaves the resid IC unchanged (0.21) and raises its t from 11.5 to 15. The noise component predicts nothing. The +0.1%/yr the rich filter earns in books is not a bounce artifact, but it is also not something better marks improve.
2. **IPCA debenture prints are two-tiered.**
   - Small retail tickets trade 40–60 bps of price richer than institutional blocks.
   - The IPCA "noise" (about 75 bps typical per print) is largely this tier-mixing.
   - Marking IPCA books at the last print therefore overstates what an institution can sell at.
   - DI floaters do not show the effect.
3. **Costs, not signals, are the binding constraint.** At realistic institutional spreads the harness's 25 bps round trip is about 2–5 times too low. P4+Q turns over 2.9 times a year, which costs about 1.4%/yr and more than halves its excess over CDI. Its edge over the universe roughly halves (+0.84 → +0.53 at R$100m).
4. **Capacity is small.** At R$500m only 77% of the target fills at 20% participation, and the edge over U drops to +0.23%/yr. At R$2bn P4+Q is about CDI. Liquidity-aware sizing plus a spill-over into the next-best carry names adds +0.03 to +0.10%/yr (not significant).
5. **Filtering liquidity out does not help.** Dropping the least-liquid quintile costs 0.14%/yr at 25 bps and is neutral under bucket costs. Ranking carry net of estimated cost helps only when costs are realistic (+0.21, t 0.8), and flips sign between the two halves.

## Caveats

- **The cost estimates come from daily-average prints.** SND gives no intraday high/low or bid/ask, so Corwin-Schultz is impossible. Roll on daily averages mixes retail/institutional tiering with true bid-ask, and the all-prints Roll is dominated by fat tails. The institutional-tier scale (0.57 for DI, 0.62 for IPCA) is a single ratio.
- **The impact coefficient (Y = 1) is assumed, not estimated.** Exits are assumed filled, and fills use actual future window volume, as a simulation of what would have executed.
- **The IPCA spread series is itself noisy.** The CDI-equivalent conversion uses a static duration factor and static contract terms. Part of the "noise" may be a conversion artifact.
- **The Kalman parameters are pooled by indexer group.** The issuer-common share is fixed at 0.8 for DI (a cap) and 0.15 for IPCA. The filter is not re-estimated by bond.
- **The harness caveats still apply:** smooth marks, survivorship (use `rec40`), and the 2026 holdout being only about 8 months.
- **The engines differ.** The capacity and bucket-cost books use `tranche_book` in `run.py`, which mirrors the harness's tranche engine exactly (maximum difference 2e-19 at 25 bps) but adds per-bond costs and partial fills. Compare them only with each other.
- **The chart is inconsistent on costs.** In `cum_excess.png`, the AUM curves are shown against the harness universe at 25 bps, so they look worse relative to U than they are on a like-for-like basis. Use the `capacity_paired` numbers for the like-for-like comparison.

## Files

- **Code:** `build_marks.py`, `run.py`, `signals.py`
- **Results:** `results.json`, `params.json`, `cost_table.json`
- **Charts:** `equity_total_return.png`, `cum_excess.png`, `capacity.png`, `equity_harness_style.png`
- **Caches:** `data/history/nightly/marks_liquidity/` (`trades.pkl`, `marks.pkl`, `peer_factor.pkl`, `trade_flags.pkl`, logs)

## Rerun

```
PYTHONPATH=. PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/Scripts/python.exe research/nightly/marks_liquidity/build_marks.py
PYTHONPATH=. PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/Scripts/python.exe research/nightly/marks_liquidity/run.py [--holdout]
```

`build_marks.py` takes about 9 minutes and `run.py` about 4.
