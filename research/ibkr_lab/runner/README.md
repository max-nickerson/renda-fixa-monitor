# Forward validation runner (IBKR paper, shadow fills)

`forward.py` runs once per market day after the close:

```
PYTHONPATH=. PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -c "import sys; sys.path.insert(0,'.'); from research.ibkr_lab.runner import forward; print(forward.run_day())"
# intraday / extra marks only (no trading):  forward.run_day(mark_only=True)
```

For each plugin in `strategies/`, the runner does three things:

1. On rebalance days it calls `targets(ib, asof)`.
2. It trades the difference to the target at the quoted touch (buy at ask, sell at bid). The paper Gateway API is **Read-Only**, so every trade is a **shadow fill** (`mode='shadow'`). If the API ever accepts orders, the runner switches to real paper orders through `safe.place_paper_order`.
3. It marks every position and writes one NAV row per strategy.

Everything goes to `data/ibkr_lab/forward.db`, in the tables `strategies`, `positions`, `fills`, `nav` and `runs`. Each strategy starts with USD 250k. The runner connects only through `safe.connect_paper`: DU accounts only, ports 4002/7497 only, and a 50k cap per order.

## Changes to `forward.py` (kept small; the safety behaviour is unchanged)

| change | why |
|---|---|
| Market-data type 4 instead of 1 (live if subscribed, else delayed, and frozen after the close) | With type 1, CME/B3 futures, US options and SPX returned nothing on this login. |
| Valuation price is the mid. If the touch is wider than 1.5%, it uses the last/close when that lies inside the touch. | A stale last trade mis-marked options and bonds. One stale side of a thin quote mis-marked the mid. |
| One quote per contract per run (cache) | Speed. |
| A target may give an explicit `qty` (number of units) instead of `target_notional_usd` | Needed for option legs, DI1 contracts and bond face value. |
| **Per-position cap:** a target above 50k (`MAX_POS_USD`) is floored to fit. A plugin may lower the cap (`cap_usd`) but never raise it. If `safe.place_paper_order` raises its size error, the trade becomes a shadow fill; the cap is never bypassed. | Safety. |
| Bonds use multiplier 0.01 | IBKR bond size is face value in USD and the price is % of par. |
| Optional plugin hook `unit_value(contract, px, asof)` returns the value of one unit | Used by DI1, which is quoted as a rate. `positions.avg_px/last_px` then hold that value. `fills.px/bid/ask` always keep the raw quote, for slippage analysis. |
| A position that can no longer be qualified is skipped with a note instead of crashing the run | Robustness. |

## Plugins

Capital is 250k per strategy, with each position ≤ 48k (under the 50k cap).

| plugin | status | rebalance | what it holds | EXPECTED (scaled to this book) |
|---|---|---|---|---|
| `brl_tsmom` | candidate | monthly | CME 6L (BRE) front and next contract: long BRL if the 1/3/6/12m TS-momentum ensemble is positive. Notional = 250k × ens × min(10%/EWMA vol, 2), capped at 2×48k, integer contracts. **Roll:** hold the first two contracts whose last trade date is more than 35 days after the rebalance, so nothing can expire before the next rebalance. | Backtest OOS Sharpe 0.53, 4.3%/yr on 8.1% vol. Capped book: about 1.8%/yr on 3.5% vol. |
| `di1_mom_ts` | candidate | monthly | DI1 MOM_TS with the **look-ahead fix** (`sig_mom(bret.shift(1))`). Buckets are 1/2/3/5y; the contract is the one closest in du. DV01-sized: weight = s/4/modified duration, with one scale factor so the largest leg is ≤ 48k. Receiving the rate = SELL DI1. Valuation = -PU × CDI accrual, which reproduces the B3 daily settlement. | Sharpe 0.435 (OOS 0.28). About 0.5%/yr on 1.1% vol. |
| `spx_box_lend` | measurement | daily (log only) | **Nothing is held.** Every run day it appends the implied lending rate (mid and touch) of a ~1y SPX box with width 400 (USD 40k) versus the maturity-matched UST to `data/ibkr_lab/forward_box_rates.jsonl`. **Why log-only:** every 1y SPX box has a deep-ITM leg worth more than 50k, so the per-leg cap would clip it, and the runner has no combo contracts. | Study: mid +52 to +63 bp over UST; touch about -40 bp. |
| `spx_condor` | control | monthly | SPXW iron condor with legs chosen by delta (short 16, long 5), about 35 DTE, n condors with max loss ≤ 48k. It is held while expiry is more than 7 days away, then closed at the touch and re-opened at the next rebalance. | Sharpe 0.48 (2016-26: 0.22). About 0.6%/yr on 0.8% vol. |
| `credit_equity_residz` | control | weekly | B3 quintile long/short on debenture resid_z (`credit_equity/paper_check.basket()`), 125k per side, about 10k per name. **Shorts assume borrow.** IBKR shows shortable level 2.0 (a locate is needed), and the runner charges no borrow: deduct about 3%/yr on the short side. | 8.5%/yr, 9.3% vol, Sharpe 0.92 (2026 YTD 0.47). |
| `b3_factor_combo` | control | monthly | Walk-forward factor combo, long-only, top third of the B3 top-60, equal weight, about 12.5k each. It uses the latest `wf_weights` in `b3_factors/results.json`. | 10.5%/yr, 16.6% vol, Sharpe over CDI -0.05. vs BOVA11: IR -0.30. |
| `eurobond_carry` | control | monthly | Top 5 of the S2 carry long list (premium vs the Brazil sovereign curve, healthy bonds) with a quote. Face value is in 1,000 steps, ≤ 48k each. Long only, no hedge. | Uses the EW healthy-corporate benchmark: 1.9%/yr, 4.0% vol, Sharpe 0.48. The S2 L/S backtest (Sharpe 2.26) is not what is held. |

**Data freshness.** The runner does not rebuild research data except the BRL inputs, which re-download when older than 2 days. Before each rebalance, refresh:

- the IBKR store (`data/build.py`) for B3 prices;
- `data/history/b3` and the CDI cache for DI1;
- `lab_daily.pkl` for resid_z;
- `eurobonds/collect.py` + `rv.py` for the carry picks;
- `b3_factors/factors.py` monthly for the walk-forward weights.

Each fill note records the data date the signal used.

## First live day: 2026-09-29

The run happened after the B3 close. US quotes are 15-min delayed or frozen, and all fills are shadow fills.

| plugin | positions | gross USD | day-1 NAV | notes |
|---|---|---|---|---|
| brl_tsmom | 2 (2×6LZ6, 2×6LF7 long) | 75.6k | 249.37k | ens +1, lev 1.19, capped. The -630 is mostly the wide delayed touch on the thin serial month 6LF7. |
| di1_mom_ts | 3 (receive 1× each of V27, V28, V29) | 44.7k | 250.00k | Signal +1/3 on every bucket. The 5y leg rounds to 0 contracts, because 250k is too small for DV01 granularity. |
| spx_box_lend | 0 | 0 | 250.00k | Log line written: Sep-27 7400/7800 box, mid -> 5.05% (+61 bp over UST), touch 2.90% (-153 bp). |
| spx_condor | 4 legs, 1 condor, Nov-03 6850/7300P and 8000/8200C | 7.0k premium (max loss 41.2k) | 249.80k | |
| credit_equity_residz | 26 (13 long / 13 short) | 250k | 249.73k | B3 was closed, so fills are at the closing price (bid = ask). |
| b3_factor_combo | 20 long | 250k | 250.34k | Same: closing-price fills. |
| eurobond_carry | 5 long (Aegea 31, Oceanica 31, Movida 29/31, CSN 28) | 233k | 245.8k | Day-1 loss of -4.2k is the bid/ask (1.2-5% of price on these names). |

**Caveat for the next run.** The monthly plugins rebalance again on the first run in October (2026-10-01), because the month changes. Small trades are expected if the signals moved.

## How validation will be judged

Use `strategies.expected` against realised `nav` and `fills`. These are pre-registered rules:

- **Checkpoints.** The first read is at 13 weeks. The verdict comes at 26 weeks for the monthly and weekly books, and at 52 weeks for `brl_tsmom`/`di1_mom_ts` if they are still ambiguous. Over 6 months a Sharpe estimate has a standard error of about 1.4, so return and Sharpe are judged as *consistency*, not proof.
- **Return and risk.** Compute the daily NAV return, then realised annual vol, return and Sharpe.
  - **Pass:** realised vol is within 0.5-1.5× of the expected (scaled) vol, and the realised Sharpe is above EXPECTED Sharpe minus 1.64 × √(252/n_days).
  - **Fail:** the realised Sharpe is below that bound, or the drawdown is worse than 2× the backtest max drawdown on a scaled basis.
  - Candidates that pass at 26 weeks move to a larger size, if the Read-Only API is lifted and real paper fills exist.
- **Slippage vs assumed.** From `fills`, the half-spread paid is |fill - mid| / mid for each fill (and in bp of DV01 for DI1). Compare it with the backtest assumptions:
  - 6L: 3 bp one-way
  - DI1: 0.5 bp in rate plus R$2.5
  - B3 stocks: 10 bp
  - SPX legs: model half-spread
  - bonds: quoted spread
  - A book whose realised costs are above 2× the assumption fails on implementability, even if its P&L is fine.
- **Signal fidelity.** Recompute each rebalance's backtest signal on the same data date. The positions must match the backtest's sign and weights, apart from contract rounding and the caps.
- **Box (measurement).** Take the median over at least 20 logged days of `mid_minus_tsy_bp` and `touch_minus_tsy_bp`. It passes if the mid is within ±25 bp of the study's +55 bp. Report whether any day offered touch ≥ UST.
- **Controls.** These are expected to do roughly nothing. A control that matches its backtest is informative about the method, not a promotion signal. `b3_factor_combo` is judged on active return vs BOVA11 over the same dates. `credit_equity_residz` is judged net of 3%/yr borrow on the short side.
- **Caveats.**
  - Delayed/frozen quotes make shadow fills optimistic or pessimistic at random.
  - Fills after the close are closing-price fills.
  - Cash earns 0 in NAV, so compare excess returns.
  - Real paper fills (Read-Only off) are needed for a slippage verdict.
