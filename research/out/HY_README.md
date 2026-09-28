# HY lab: can a high-yield debenture book beat P4, and do tail hedges pay for themselves?

Script: `research/run_hy_lab.py` (about 5 min). Rebuild:

```
PYTHONPATH=. PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe research/run_hy_lab.py
```

- **Inputs:** the selection-lab caches, read only (`data/history/sellab_panel.pkl`, `sellab_returns.pkl`). If they are missing, run `run_selection_lab.py --rebuild` first.
- **Other outputs:** the log goes to `data/hy.log`. The brapi ^BVSP / SMAL11 prices are cached in `research/out/hy_indices.pkl`.
- **Results and chart:** every table is in `hy_results.json`, with 3 horizons × base / 50 bps / rec40 scenarios. The chart is `hy_equity.png`.

## Short answer

**1. HY carry alone barely beats P4.**
- **Top quintile by CDI+ carry (Q5):** equal to P4 (−0.02%/yr, t −0.1). This bucket is the one that actually held Americanas.
- **Top decile (D10):** +0.53%/yr (t 1.4).
- **CDI+ ≥ 300 bps (A300):** +0.92%/yr (t 2.3). Its 2023 result is partly luck: Americanas' LAMEA7 traded at 215–271 bps, below both cuts.

**2. HY plus the P4 filters plus the quality screen does beat P4** at every horizon and in both halves of the sample:
- Q5: +0.68%/yr (t 4.0).
- D10: +1.6%/yr (t 4.6).
- A300: +2.3%/yr (t 7.2).
- It also has smaller drawdowns than P4.
- **Caveat:** 60–70% of these books are issuers with no listed stock, the rarely-marked segment that the selection lab suspects of survivorship bias. Restricted to bonds with fundamentals coverage, the edge shrinks:
  - Q5: −0.06 vs P4.
  - D10: +0.97 (19 bonds).
  - A300: +1.51 (17 bonds).
  - Against a covered-only P4 (+0.96), these increments are +0.5 / +1.5 / +2.1.

**3. None of the hedges pays for itself.** The filters are the hedge.
- **Equity short with a regression ratio:** the trailing bond-vs-stock beta has a median of about 0.002, so the hedge is roughly zero and does nothing.
- **Equity short with a structural ratio** (duration × spread, about 3% notional): +0.3%/yr, but only because HY issuers' stocks lagged CDI. That is an equity bet, not a hedge. It raises vol from 1.3% to 1.8%, deepens the max drawdown from −1.7% to −2.3%, and makes the 2023 window worse (−0.9% → −1.8%).
- **Short sized for a jump to default** (0.5 × notional, top-risk names only): recoups 20–35% of big-drop losses. It also triples vol to 3.5%, takes max DD to −4.2%, and loses 3.7% in the 2023 window.
- **Ibovespa / SMAL11 beta hedges:** the book's beta is about 0.01, so the hedge is immaterial (−0.02 to −0.13%/yr).
- **Ibovespa puts, 25% of notional:** cost 0.28%/yr at 22% implied vol and 0.8%/yr at 27%, and pay nothing. Ibovespa rose 6.5% in the stress window. HY stress in Brazil was idiosyncratic, not a market crash.

## Main table: H = 6m, book-level monthly series 2022-02 to 2026-09, 25 bps, marked to last price

"xCDI" and "xU" are the annualized excess over CDI and over the universe. "vs P4" is the annualized difference with a Newey-West t (lag 6). "Crisis" is the cumulative excess from Dec-22 to Jun-23. "Big" is the share of position losses coming from positions that lost more than 10%.

| variant | xCDI | xU | vs P4 (t) | vol | Sharpe | maxDD | worst mo | skew | hit | vs P4 22–23 / 24–26 | crisis | big |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CDI | 0 | | | | | | | | | | | |
| Universe | +0.45 | 0 | −1.08 (−2.1) | 1.69 | 0.27 | −3.60 | −1.18 | −0.50 | 63% | −1.46 / −0.80 | −1.67 | 28% |
| P4 | +1.53 | +1.08 | — | 1.09 | 1.41 | −1.52 | −0.64 | −0.63 | 72% | — | −1.27 | 50% |
| P4 + quality | +1.94 | +1.49 | +0.41 (2.9) | 1.05 | 1.85 | −1.32 | −0.59 | −0.60 | 74% | +0.32 / +0.47 | −0.91 | 40% |
| Q5 HY | +1.51 | +1.06 | −0.02 (−0.1) | 1.33 | 1.14 | −1.70 | −0.81 | −0.32 | 68% | +0.28 / −0.25 | −0.91 | 51% |
| Q5 HY + P4 filters | +1.75 | +1.30 | +0.22 (2.2) | 1.19 | 1.47 | −1.33 | −0.80 | −0.65 | 74% | +0.42 / +0.08 | −1.01 | 54% |
| Q5 HY + quality | +1.92 | +1.47 | +0.39 (2.7) | 1.20 | 1.60 | −1.32 | −0.70 | −0.49 | 72% | +0.55 / +0.27 | −0.66 | 49% |
| **Q5 HY + P4f + Q** | **+2.21** | +1.76 | **+0.68 (4.0)** | 1.15 | 1.92 | −1.15 | −0.59 | −0.61 | 77% | +0.67 / +0.68 | −0.69 | 47% |
| D10 HY | +2.06 | +1.61 | +0.53 (1.4) | 1.49 | 1.38 | −1.46 | −1.19 | −0.55 | 75% | +1.00 / +0.19 | +0.03 | 56% |
| **D10 HY + P4f + Q** | **+3.14** | +2.69 | **+1.61 (4.6)** | 1.25 | 2.51 | −1.05 | −0.80 | −0.44 | 79% | +1.67 / +1.57 | +0.88 | 47% |
| A300 HY | +2.45 | +2.00 | +0.92 (2.3) | 1.46 | 1.68 | −1.23 | −1.23 | −1.18 | 79% | +1.24 / +0.69 | +0.11 | 62% |
| **A300 HY + P4f + Q** | **+3.82** | +3.37 | **+2.29 (7.2)** | 1.13 | 3.38 | −0.46 | −0.35 | +0.40 | 86% | +2.07 / +2.45 | +0.94 | 54% |

**Hedges on Q5 HY** (unhedged: +1.51, vol 1.33, maxDD −1.70, crisis −0.91). Hedge P&L and cost are annualized, in % of the book.

| hedge | xCDI | hedge P&L | cost | vol | maxDD | crisis | % of big-drop loss recouped | avg notional |
|---|---|---|---|---|---|---|---|---|
| Equity short, regression ratio, 2% borrow | +1.52 | +0.02 | 0.01 | 1.33 | −1.71 | −1.00 | 0.5% | 0.2% |
| same, 5% borrow | +1.51 | +0.02 | 0.01 | 1.33 | −1.72 | −1.01 | | |
| Equity short, structural ratio (dur × spread) | +1.79 | +0.35 | 0.08 | 1.80 | −2.30 | −1.79 | 6.7% | 3.2% |
| Equity short, top-risk names only (regression ratio) | +1.52 | +0.02 | 0.00 | 1.32 | −1.70 | −0.94 | 0.6% | 0.1% |
| Equity short, triggered when stock ≤ −15% over 4w | +1.52 | +0.02 | 0.00 | 1.33 | −1.72 | −0.98 | 0.3% | 0.2% |
| Equity short, jump-to-default sized (0.5, top-risk) | +2.16 | +0.84 | 0.20 | 3.46 | −4.21 | −3.69 | 21.8% | 8.3% |
| Ibovespa futures, beta | +1.45 | −0.05 | 0.00 | 1.34 | −1.89 | −0.93 | | 0.5% |
| SMAL11 short, beta | +1.48 | −0.02 | 0.01 | 1.33 | −1.75 | −0.93 | | 0.3% |
| Ibovespa put proxy, beta notional, IV 22% | +1.49 | −0.01 (net) | | 1.33 | −1.71 | −0.91 | | 0.5% |
| Ibovespa put proxy, 25% notional, IV 22% | +1.22 | −0.28 (net) | | 1.58 | −1.95 | −1.20 | | 25% |
| Ibovespa put proxy, 25% notional, IV 27% | +0.70 | −0.79 (net) | | 1.59 | −2.29 | −1.49 | | 25% |

The same pattern holds on D10, A300 and the filtered bases (JSON): no hedge improves Sharpe or maxDD.
- On the filtered books, the jump-to-default short loses money: −0.6 to −0.7%/yr of hedge P&L on D10 / A300 + P4f + Q.
- A single-stock put on an HY issuer would behave like that jump-to-default short, and in practice no liquid market for one exists.

**Robustness**, excess vs P4 at 3 / 6 / 12m:

| variant | 3m | 6m | 12m |
|---|---|---|---|
| Q5 HY | −0.14 | −0.02 | −0.03 |
| D10 HY | +0.45 | +0.53 | +0.51 |
| A300 HY | +0.73 | +0.92 | +0.93 |
| Q5 HY + P4f + Q | +0.68 | +0.68 | +0.53 |
| D10 HY + P4f + Q | +1.80 | +1.61 | +1.32 |
| A300 HY + P4f + Q | +2.26 | +2.29 | +1.77 |

- **Cost at 50 bps:** P4 +1.21, Q5 HY +1.24, D10 HY +1.74, A300 HY +2.12, Q5 HY + P4f + Q +1.85, A300 HY + P4f + Q +3.44.
- **Recovery at 40% (rec40):** the 55 bonds that stopped trading below 0.90 jump to 40% of par. Results:
  - P4 +1.40, Q5 HY +1.42, D10 HY +1.91, A300 HY +2.16 (maxDD −2.4).
  - Q5 HY + P4f + Q +2.17, A300 HY + P4f + Q +3.74.
  - Ranking unchanged.

## Where the losses come from, and the 2023 window

- **Big drops** (positions losing more than 10%) account for about 50–60% of all position losses in the HY books, versus 28% in the universe. Bonds that stopped trading below 0.90 account for 27–38%.
- **Americanas (LAMEA7):** it was in Q5 and in P4 (215–271 bps, flagged `p4f` = clean, quality not in the worst quintile). It lost 50% in each of 3 cohorts and is the largest single loss in Q5 and P4. No screen caught it, and neither did the stock (−8% over 4w on 2023-01-02).
- **Other large losers:** ELFA12, AERI12, VVAR26, IRBR12, KRSA, TBCR18 and AGRU. See `crisis_anatomy_H126`.
- **Hedge P&L on those names:** about 0 with the regression ratio, and about 0.1–0.4 bp of book with the structural ratio.

## Implementability in Brazil

- **Single-name CDS:** there is no liquid market, so none was modeled.
- **Coverage.** Share of HY weight (full-horizon positions) by listed-equity status:

  | bucket | listed stock (direct / parent) | ADTV ≥ R$5m (shortable proxy) | ADTV ≥ R$200m (single-stock options proxy) |
  |---|---|---|---|
  | Q5 | 53% (46 / 7) | 42% | 5% |
  | D10 | 49% | 36% | 3% |
  | A300 | 48% | 35% | 2% |
  | filtered books (P4f + Q) | only 30–40% | 19–28% | ~2% |

  The riskiest names are mostly unlisted SPVs and subsidiaries, which cannot be hedged at all.
- **Stock borrow (BTC):** the 2% (5% check) fee is optimistic for distressed names. Their borrow typically becomes scarce and expensive exactly when the hedge is needed. The borrow cost is immaterial here only because the useful hedge ratios are tiny.
- **Ibovespa futures (WIN/IND):** fully implementable at near-zero cost. They do not help, because HY debenture stress was not market-wide.
- **SMAL11:** there is no liquid small-cap future. Shorting SMAL11 via borrow is feasible at small size.
- **Ibovespa puts:** liquidity is concentrated in short-dated, near-the-money strikes. A 3m 10%-OTM roll is less liquid, so it is **priced here as a Black-Scholes approximation at a fixed IV** (22%, and 27% for the skew). Premium is about 0.54% of notional per 3m roll at 22% and 1.04% at 27%, plus 10% of premium as spread. This is not a market-priced backtest.

## Method

- **Cohorts:** monthly decision dates from 2022-01 to 2026-09, from the selection-lab panel. The universe is `eligible` bonds (marked 0.9–1.1, duration ≥ 0.5).
- **Entry:** the first trade after the decision (within 20 bdays), otherwise cash.
- **Returns:** PATCHED rate-hedged excess returns, which include moves over 20% and gap carry. Weights are equal with a 10% issuer cap.
- **Book:** overlapping tranches held 63 / 126 / 252 bdays, averaged over the live tranches. Positions are **buy-and-hold**: daily contribution = w × cumulative growth × r, so the book sum equals the compounded tranche return. Cost is 25 bps round trip on book turnover (50 bps check).
- **Stats:** from the monthly series of the daily book. The Newey-West lag equals the horizon in months. `cohort_level` in the JSON gives the selection-lab-comparable per-cohort figures; P4 vs universe is +1.10%/yr there, matching SELLAB.
- **Buckets:** defined point-in-time on each decision cross-section.
  - **P4 filters (P4f):** `resid_z > −1.5` and no `press_neg_30d`.
  - **Quality screen (Q):** drops the worst quintile of the selection lab's composite quality (strict availability dates). Uncovered bonds are kept.
  - **Top-risk flag:** top 5% carry, OR worst-quintile quality, OR stock ≤ −15% over 4w.
- **Equity hedge ratios** use the issuer's stock or its listed parent (from `equity_map`) with ADTV ≥ R$5m.
  - **Regression:** a weekly Dimson beta (lag 0 + 1) of the bond's hedged return on the stock's excess return over the trailing 252 bdays. It is set to 50% own beta + 50% pooled HY median, then clipped to [0, 1].
  - **Structural:** duration × spread (elasticity −1), capped at 0.5.
  - **Jump-to-default:** 0.5 on top-risk names.
  - **Trigger:** turns on once the stock's 21d return is ≤ −15%.
  - **Sizing and costs:** the short is sized on the position's current value and pays borrow of 2% (5% check) plus 10 bps per side.
- **Macro hedge:** −β × (index − CDI). β is the trailing Dimson beta of the cohort basket's weekly returns.
- **Put proxy:** notional = book β × exposure, or 25% of exposure. The premium is financed at CDI.

## Caveats

1. **Smooth marks.** Debenture marks are stale and smooth, so vol, Sharpe and t-stats are overstated for every book, and more so for rarely-traded HY. A t of 7 means persistent carry, not that the result is safe. Losses often arrive as gaps that the patched returns do capture, but only once a trade prints.
2. **Survivorship.** 55 bonds stopped trading below 0.90, and anything after their last print is unobserved. The rec40 run bounds this. It does not cover bonds that stopped above 0.90 and later defaulted.
3. **Uncovered bonds.** Much of the HY edge comes from unlisted, uncovered issuers (see the covered-only rows in the JSON). These are the least verifiable and least hedgeable part of the book.
4. **Cut choice and a single stress event.** The cut choice drives the 2023 result: Americanas sat in Q5 but not in D10 or A300. With 3 buckets × 4 filters tested and one real stress episode, treat D10 and A300 as optimistic and Q5 as the honest "HY" read.
5. **Sample regime.** The sample (2022–2026) had high CDI and no broad credit crash. A market-wide sell-off like 2020 is exactly the scenario the index hedges and puts target, and it is not in the data.
6. **Book-engine fix.** An earlier fixed-weight daily sum (the approach also used in `sellab_equity.png`) overstated volatile bonds' returns: a bad-print dip and rebound sums to a gain. This lab uses buy-and-hold drift instead. Versus the naive sum, it lowers every book by about 1.0–1.5%/yr (P4 −1.05, Q5 HY −1.35, universe −1.46).
