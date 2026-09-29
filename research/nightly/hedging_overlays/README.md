# Hedging & overlays for the P4+Q debenture book (harness v4)

**Short answer.** No overlay or hedge beats P4+Q net of costs. The two things worth taking away are:

1. **The harness's IPCA+ rate hedge over-hedges.** This inflates the "credit" excess of IPCA+ bonds in 2024–25.
2. **The only cheap drawdown reducer is realistic, and small:** do not open new tranches when the Ibovespa is in a 63-day slump.

Everything else fails:
- equity, bank and CDS beta hedges;
- Ibovespa puts;
- volatility targeting, drawdown stops, momentum overlays and cash buffers;
- bond-level stop-losses.

Each of these either does nothing, because the book's equity beta is about 0.003, or costs more return than the drawdown it saves.

## Files

| file | what |
|---|---|
| `data.py` | Caches in `data/history/nightly/hedging_overlays/`: DI x Pre curve (B3 files), NTN-B real curve, EMBI+ Brazil (Ipeadata; the series ends 2024-07-30), Ibov / SMAL11 / bank closes |
| `rates.py` | Alternative rate and inflation hedges, as additive adjustments to the harness return matrix R, booked at the harness realisation position (v4 gap rule); `use_R` context manager |
| `run.py` | Main study, sections A–G, 24 candidates. Holm correction, pre-registered selection, robustness, placebo, charts and the **single** holdout evaluation → `results.json` |
| `hedge_ratio.py`, `run_helpers.py` | Follow-up, pre-2026 only: point-in-time yield beta of IPCA+ debentures to NTN-B, and books with an empirical hedge ratio → `hedge_ratio_results.json` |
| `equity_total_return.png` | Total return CDI x (1+excess) vs CDI / Universe / P4 / P4+Q / IDA-DI / Ibov |
| `cum_excess.png` | Cumulative excess vs the universe, plus drawdowns |
| `rate_hedges_ipca.png` | The IPCA+ part of P4+Q under each hedge choice |

**Rerun** (about 5 min, then about 2 min):
```
PYTHONPATH=. PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/Scripts/python.exe research/nightly/hedging_overlays/data.py
PYTHONPATH=. PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/Scripts/python.exe research/nightly/hedging_overlays/run.py
PYTHONPATH=. PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/Scripts/python.exe research/nightly/hedging_overlays/hedge_ratio.py
```

## Method

**Setup.**
- Base book: harness P4+Q. Monthly decisions, 126-bday tranches, 10% issuer cap, 25 bps (50 bps as the check).
- Statistics run on 48 months, 2022-01 → 2025-12.
- Every candidate is paired with P4+Q using Newey-West t (lag 6). Holm runs across all 24 `run.py` candidates.
- `hedge_ratio.py` adds 4 more variants, so **28 were tried in total**. No candidate reaches Holm p < 0.05 with a positive sign, so the combined count does not change any conclusion.

**Point-in-time handling.**
- Hedge betas are estimated on data up to week w and applied from week w+1.
- Overlays built from the book's own returns, or from the universe's, are lagged one extra day. This is because `R[t]` is the t→t+1 move, known only at close t+1. The harness then lags every overlay one more day.
- Curves use the trade date of each mark, the same timing as `bench_rate`.

**Pre-registered selection rule:**
- Among candidates losing no more than 0.15%/yr vs P4+Q at 25 bps, pick the largest improvement in max drawdown (tie-break: worst month).
- The long-short section G is excluded.
- The **sealed holdout was evaluated once**, for the winner only (`E_gate_ibov`). The later `hedge_ratio.py` follow-up is pre-2026 only.

## Results (pre-2026, 25 bps; vs P4+Q = paired annual difference)

| variant | what | exCDI | vs P4+Q (t) | vol | maxDD | worst mo | crisis Dec22–Jun23 | 50 bps vs P4+Q | 22–23 / 24–25 vs P4+Q |
|---|---|---|---|---|---|---|---|---|---|
| **P4+Q** | base | **+2.22** | — | 1.04 | −1.39 | −0.62 | −0.92 | — | — |
| P4 / U | refs | +1.89 / +1.10 | −0.33 / −1.12 | 1.05 / 1.50 | −1.56 / −3.51 | | −1.24 / −1.69 | | |
| A_raw | no rate hedge | +1.45 | −0.76 (−1.8) | 1.15 | −1.10 | −1.10 | −0.24 | −0.76 | +0.04 / −1.57 |
| A_vertex | IPCA+ hedged at one 5y NTN-B vertex, Pre at one 3y DI vertex | +2.21 | −0.01 (−0.2) | 1.07 | −1.56 | −0.68 | −1.03 | −0.01 | |
| A_di1_ipca | IPCA+ hedged with nominal DI1 (no DAP) | +2.24 | +0.03 (0.0) | **1.54** | **−2.50** | −0.74 | −2.05 | +0.03 | −0.61 / +0.66 |
| A_swap | exact hedge + IPCA x DI / Pre x DI carry swap | +2.36 | +0.14 (0.9) | 1.02 | −1.49 | −0.68 | −0.82 | +0.14 | +0.29 / 0.00 |
| B_IBOV_beta | short Ibov futures, rolling Dimson β | +2.20 | −0.02 (−0.7) | 1.06 | −1.39 | −0.62 | −0.92 | −0.02 | |
| B_SMAL11_beta | short SMAL11, β | +2.20 | −0.02 (−0.7) | 1.05 | −1.39 | | | −0.02 | |
| B_BANKS_beta | short bank basket (ITUB4, BBDC4, BBAS3, SANB11, BPAC11), β | +2.18 | −0.04 (−1.6) | 1.05 | −1.39 | | | −0.04 | |
| B_CDS_EMBI_beta | buy 5y Brazil CDS protection (EMBI+ proxy, to 2024-07), β | +2.18 | −0.04 (−1.6) | 1.04 | −1.40 | | | −0.04 | |
| B_IBOV_fixed10 | constant 10% Ibov futures short | +2.15 | −0.07 (−0.1) | **2.20** | −1.91 | −0.83 | −1.06 | −0.07 | |
| C_put25 | 1m 90% Ibov puts on 25% notional | +2.11 | −0.11 (−0.5) | 1.15 | −1.63 | −0.70 | −1.37 | −0.11 | |
| C_put100 | same, 100% notional | +1.79 | −0.43 (−0.5) | 1.83 | −3.53 | −0.92 | −2.72 | −0.43 | |
| D_voltarget | exposure = median vol / 63d vol, floor 0.25 | +1.40 | −0.81 (−3.3) | 0.78 | −0.89 | −0.51 | −0.71 | −0.87 | |
| D_ddstop | 50% when book DD < −0.5% | +1.11 | −1.11 (−1.5) | 1.28 | −3.70 | −1.19 | −2.34 | −2.19 | |
| D_umom | 50% when universe 21d excess < 0 | +0.44 | −1.78 (−6.2) | 1.08 | −2.42 | | | −3.56 | |
| D_ibovstress | 50% when Ibov 63d excess < −10% | +1.82 | −0.39 (−3.2) | 1.07 | −1.80 | | | −0.88 | |
| D_cash10 | constant 10% cash | +2.00 | −0.22 (−3.1) | 0.94 | −1.25 | −0.56 | −0.83 | −0.18 | |
| D_distress | 50% when CVM distress z > 2 | +2.08 | −0.14 (−1.9) | 1.05 | −1.47 | | | −0.27 | |
| E_gate_dd | skip new tranche when book DD < −0.5% | +1.61 | −0.60 (−1.6) | 0.92 | −1.23 | −0.56 | −0.82 | −0.57 | |
| **E_gate_ibov** | **skip new tranche when Ibov 63d excess < −10%** | **+2.12** | **−0.10 (−0.9)** | **0.94** | **−1.10** | **−0.53** | **−0.80** | **−0.08** | **+0.04 / −0.24** |
| E_gate_widen | skip when universe median CDI+ +15 bps m/m | +2.19 | −0.03 (−1.0) | 1.03 | −1.36 | | | −0.02 | |
| E_gate_tighten | skip when median CDI+ −15 bps m/m | +2.10 | −0.11 (−1.5) | 1.01 | −1.39 | | | −0.11 | |
| F_stop_4w3pct | sell a bond after a −3% 4-week price drop (weekly flags) | +1.97 | −0.25 (−0.9) | 0.80 | −0.95 | −0.68 | −0.56 | −0.30 | +0.08 / −0.58 |
| F_stop_below95 | sell when the mark is < 95% of par | +1.78 | −0.44 (−1.9) | 0.77 | −1.34 | −0.48 | −0.97 | −0.49 | |
| F_exit_rich | sell when rich (weekly) | +2.28 | +0.06 (0.8) | 0.99 | −1.30 | −0.54 | −0.69 | +0.05 | |

**Holm across the 24 candidates.**
- The only Holm-significant differences are **negative**: D_voltarget, D_umom, D_ibovstress and D_cash10.
- Every candidate with a positive difference has Holm p = 1.
- The F rows are compared with P4+Q on the monthly engine. Against the same weekly-flag engine, the differences are −0.26, −0.45 and +0.05 (`results.json` → `info.F_vs_same_engine`).

**Best under the pre-registered rule: `E_gate_ibov`.**
- Four of 48 decisions were gated: 2022-07, 2023-01, 2024-12 and 2025-01.
- maxDD improves from −1.39 to −1.10, and the worst month from −0.62 to −0.53, for −0.10%/yr (−0.08 at 50 bps).
- Placebo: gating 4 random decisions (40 draws) matches or beats that maxDD only 7.5% of the time. That is suggestive, not significant.
- rec40: −0.10 vs P4+Q rec40, with the same maxDD.
- **Sealed holdout (2026-01..09, used once):** +0.20 exCDI vs P4+Q +0.08. The difference is +0.12 (t 0.75), and maxDD is −0.62 vs −0.78. The direction holds, but it is 9 months and not significant.

### Rate / inflation hedges on the IPCA+ part of P4+Q (sub-book, pre-2026)

| hedge | exCDI | vol | maxDD | β to 5y real yield (% per +1pp) | 22–23 / 24–25 |
|---|---|---|---|---|---|
| raw (no hedge) | −1.61 | 3.38 | −8.04 | −2.37 | |
| harness exact (h = 1 × duration vs NTN-B at the bond's duration) | +1.87 | 2.71 | −5.05 | **+1.23** (over-hedged) | −0.72 / +4.47 |
| single vertex (5y NTN-B) | +1.82 | 2.83 | −5.50 | +1.64 | |
| nominal DI1 at matching duration | +2.17 | 6.92 | −14.3 | +5.13 | |
| exact + carry swap | +2.63 | 2.25 | **−2.17** | +1.12 | +0.64 / +4.63 |
| h = PIT yield beta (≈0.45) | −0.20 | 2.58 | −4.94 | −0.87 | −0.74 / +0.33 |
| h = PIT beta + swap | +0.56 | 2.18 | −2.34 | −0.99 | +0.62 / +0.51 |

## Insights

1. **IPCA+ debenture yields move about 0.4–0.55 for every 1.0 move in NTN-B yields.**
   - This is a point-in-time pooled trade-to-trade regression, trailing 12 months. It fell from 0.52 in 2021 to about 0.40 in 2023–24.
   - The harness hedges IPCA+ bonds 1:1 with model duration, so hedged IPCA+ returns carry a **positive** real-rate beta (+1.2% per +1pp on the sub-book).
   - Real yields rose sharply in 2024. The IPCA+ sleeve's "credit" excess in 2024–25 (+4.5%/yr) therefore collapses to +0.3%/yr at the empirical hedge ratio.
   - At book level, P4+Q drops from +2.22 to +1.76 (−0.45%/yr, t −1.8).
   - **Every nightly result that favours IPCA+ bonds, or that looks strong in 2024–25, should be re-checked against this.**
   - Monthly regressions put the zero-beta hedge ratio at about 0.65, so the true ratio lies between 0.45 and 0.65 and the artifact is 0.3–0.45%/yr of the book.
2. **The IPCA+ sleeve's drawdown comes from inflation carry, not duration.**
   - An IPCA x DI carry swap on top of the duration hedge cuts the IPCA+ sub-book maxDD from −5.1% to −2.2% and its vol from 2.7% to 2.2%.
   - It hedges realised IPCA plus the real yield against CDI.
   - At book level the difference is +0.14 (t 0.9). Cost: DAP/swap half-spread of about 2 bp on a book DV01 of about 1.3 years gives roughly 0.10%/yr. The DI1-only equivalent is about 0.03%/yr.
3. **Hedging IPCA+ bonds with DI1 instead of DAP or NTN-B is disastrous.**
   - The IPCA+ sub-book's vol goes from 2.7% to 6.9% and its maxDD from −5% to −14%.
   - The real-vs-nominal basis is larger than the credit risk.
   - Duration-only at one vertex vs bond-by-bond key-rate costs little: maxDD −1.56 vs −1.39.
   - Our B3 / Tesouro key-rate curve reproduces `bench_rate` with correlation 0.999.
4. **Proxy credit hedges are useless because the book has no equity or sovereign beta in normal times.**
   - The rolling β averages 0.003 (Ibov), 0.002 (SMAL11), 0.001 (banks) and 0.011 (CDS proxy).
   - Monthly correlations are slightly negative, about −0.1 to −0.2.
   - **Asymmetry:** in the book's 6 worst months, the correlation with Ibov and banks is +0.72 to +0.76. Stress is shared, but the average beta is zero, so a linear hedge cannot capture it. That is why a conditional gate works a little and a hedge does not.
   - A meaningful fixed 10% Ibov short doubles vol (2.2%) and deepens maxDD.
5. **Ibov puts are pure cost in this sample.**
   - Puts 10% OTM cost about 0.11% of notional per month, and paid off in only 2 of 48 months.
   - At 25% notional they cost 0.10%/yr and worsen maxDD to −1.63.
   - The 2023 debenture stress (Americanas, Light) was idiosyncratic, while the Ibovespa rallied.
6. **Instant overlays at the mark lose through switching costs and whipsaw.**
   - D_ddstop switched 76 times, costing about 1%/yr, and its maxDD **worsened** to −3.7%.
   - Vol targeting halves vol, but Sharpe falls from 2.13 to 1.81.
   - A 10% cash buffer scales everything proportionally, costing 0.22%/yr for 0.14 of maxDD. It only makes sense as a liquidity reserve.
   - All of these assume the whole book can be sold at the mark, which debentures cannot do, so real-world results would be worse.
7. **Entry gating is the only realistic de-risking tool, since it needs no forced selling.**
   - Skipping new tranches after an equity slump cuts maxDD by about 0.3pp for about 0.1%/yr.
   - Gating on the book's own drawdown is too slow: 9 of 48 decisions gated, −0.60%/yr.
8. **Bond-level stops reduce vol but cut return.**
   - Selling after a −3% 4-week drop reduces vol to 0.80 and maxDD to −0.95, but costs 0.25%/yr, all of it in 2024–25.
   - Stops realise losses on bonds that mean-revert.
   - The "sell when rich" exit remains the only exit rule that adds anything: +0.05%/yr, not significant.
9. **Long-short within peers (informational).**
   - **Borrowing:** B3's BTB lending platform covers equities and ETFs, and B3 has since added lending of federal bonds. We found no organised borrow market for debentures; B3 does accept debentures as collateral. This was a quick web check, not verified in depth. The best case is a bilateral loan or a short via a reverse repo with a dealer, so treat this leg as theoretical.
   - The long-short shorts the lowest-carry 30% of the same peers: excess ≈ 0 (−0.01%/yr at 2% borrow, −2.9% at 5%). Its correlation with the universe is −0.58.
   - The carry premium is harvested on the long side only. Shorting rich debentures does not pay even before realistic borrow costs.

## Reusable point-in-time signals

- **`data/history/nightly/hedging_overlays/ipca_yield_beta.pkl`:** a Series of month-end → PIT yield beta of IPCA+ debentures to NTN-B. Use it as a hedge ratio, applying it from the day after the month end.
- **`data/history/nightly/hedging_overlays/overlay_signals.pkl`:** a DataFrame by grid day with the D overlays (`D_voltarget`, `D_ddstop`, `D_umom`, `D_ibovstress`, `D_cash10`, `D_distress`). Each value is known at close t, and it can be passed straight to `H.backtest(..., overlay=col)`.
- **The Ibov 63-day entry gate:** `ex_daily(IBOV).shift(1).rolling(63).sum() < -0.10` on the decision day (`run.py`, `GATES["E_gate_ibov"]`).
- **`rates.build()["mats"]`:** additive R adjustments for the hedge variants (`raw`, `vertex`, `di1_ipca`, `swap`, `keyrate`), for use with `rates.use_R(adj)`.

## Caveats

- **Smooth, stale marks.** Vols and Sharpes are inflated, and drawdowns understated. Overlays are executed at the mark, which is optimistic.
- **Only 48 months, with one real credit stress (2023) and no market-wide credit crash.** Tail hedges are designed for a 2020-type event, which is not in the sample.
- **EMBI+ ends 2024-07.** The CDS proxy has no hedge after that, and it ignores the USD quanto effect.
- **Put pricing is Black-Scholes.** The implied vol is max(realised 63d, 15%) × 1.25, a guess at skew and vol premium, with no bid-ask.
- **The carry swap is approximated with a daily-reset fixed leg at the NTN-B real yield.** A real IPCA x DI swap locks the coupon at inception.
- **The yield-beta finding relies on the duration model:** the SND ratio and the `duration_factors` median. Part of the "beta < 1" could be duration mis-measurement. Either way, the hedge ratio the harness applies is too high relative to how marks actually move.
- **`hedge_ratio.py` was run after the holdout was spent.** Its numbers are pre-2026 only and were not used to choose the best variant.
