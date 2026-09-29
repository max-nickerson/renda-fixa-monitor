# Issuer-curve and cross-indexer relative value (`issuer_curve_rv`)

**Verdict: a clean negative for portfolios, with a real but tiny edge underneath.** Within an issuer, the bond that is cheap on the curve does beat the rich one: +0.83%/yr per pair (t 6.4, placebo p95 +0.15). A book cannot harvest this, for two reasons:
- P4+Q already holds each issuer's cheap bonds, because the within-issuer carry ranking and the curve ranking mostly agree. Only 4% of its slots would switch.
- The issuer-aware fair value is a much better *spread* model, but a worse *return* signal. The issuer part of the old peer residual is a credit/carry premium, and that premium pays.

Across 12 variants the best one differs from P4+Q by +0.03%/yr (t 0.65, Holm p = 1). The sealed 2026 holdout gives −0.05.

## Method (all point-in-time: same-day cross-sections of fresh marks, executed by the harness at the next trade)

`signals.py` computes the fair values below for each decision day. All values are CDI+-equivalent bps, and a positive value means *cheap*.

| name | fair value |
|---|---|
| `resid_bps` | The existing peer curve: medians by kind × Lei 12.431 flag, bucketed by duration. |
| `e_iss` | `resid_bps` minus the **leave-one-out** mean residual of the issuer's *other* fresh bonds. It is shrunk toward 0 with k = 1 and defined only when the issuer has at least 2 fresh bonds (72% of rows). |
| `e_grp` | The same as `e_iss`, but by economic group, using the parent-ticker root from `equity_map.csv` (a static map). |
| `e_isec` | The issuer effect shrunk toward the sector mean, excluding the issuer itself. Defined for every bond. |
| `r_sec` | A sector-aware curve with no issuer information. |
| `e_slope` | `e_iss` plus a same-day pooled within-issuer duration-slope correction. Riskier issuers get steeper curves. |

Each of these also has a robust z-score, `*_z`.

The cross-indexer comparison (DI+ vs IPCA+ vs Pré) is built in, because all spreads are converted to CDI+ equivalents. The analysis breaks results down by pair type:
- `curve`: same kind and same tax status;
- `xidx`: different indexers;
- `xinc`: incentivada vs taxable.

Stages in `run.py`:
1. **Spread convergence.** The 13-week change in the bond's own CDI+ spread is regressed on each residual, both across issuers and within issuer.
2. **Return ICs** on `fwd_63` and `fwd_126`, including within-issuer ICs.
3. **Within-issuer pair trades**: long the cheap bond and short the rich bond of the same issuer, on monthly dates, with a random-pair placebo.
4. **Market basis timing**: the IPCA-vs-DI basis and the incentivada-vs-taxable basis.
5. **Harness backtests**: monthly decisions, 126-bday tranches, 25 bps. These include `compare()` with Holm across all 12 variants, 50 bps, rec40, hold 63 and a placebo.
6. **Holdout**, run once.

## Results (pre-2026)

### 1. Which fair value is best? It depends on the target

13-week spread convergence on the common sample of multi-bond issuers (88k obs):

| residual | cross-section β (t) | pooled R² | within-issuer β (t) | within R² |
|---|---|---|---|---|
| resid_bps (current peer) | −0.11 (−5.6) | 2.1% | −0.37 (−13.8) | 11.8% |
| **e_iss (issuer LOO)** | **−0.23 (−11.4)** | **5.3%** | −0.31 (−14.3) | 11.8% |
| e_grp | −0.23 (−11.8) | 5.5% | −0.32 | 11.7% |
| e_isec | −0.23 (−11.7) | 5.3% | −0.31 | 11.8% |
| r_sec (sector-aware) | −0.11 (−6.3) | 2.1% | −0.37 | 11.9% |
| e_slope | −0.23 (−11.3) | 5.2% | −0.30 | 11.4% |
| raw cdi_bps | −0.07 (−3.7) | 1.6% | −0.28 | 9.8% |

- **An issuer-aware fair value is a 2.5× better spread model.** It doubles the convergence speed and more than doubles R².
- **Within an issuer, the residual reverts about 37% in 13 weeks**, a half-life of about 20 weeks.
- **Sector awareness adds nothing** over the kind × incentive peer curve.

IC vs `fwd_126` (monthly, universe):

| | resid_bps | e_iss | e_isec | r_sec | e_slope | cdi_bps |
|---|---|---|---|---|---|---|
| IC (t) | **0.209 (11.5)** | 0.166 (8.4) | 0.175 (12.4) | 0.176 (12.0) | 0.162 (9.8) | 0.142 (1.5) |
| within-issuer IC | 0.130 (6.4) | 0.130 | 0.130 | 0.130 | 0.127 | 0.124 (1.9) |

**The better spread model is the worse return signal.** The issuer component that `e_iss` removes is persistent credit/carry premium. Spreads do not converge it away; they pay it as carry. That is what P4 harvests.

### 2. Within-issuer switches (cheap minus rich bond, same issuer, gross, before switching costs)

| sort / horizon | all | curve | cross-indexer | incentivada vs taxable |
|---|---|---|---|---|
| resid, 126d, %/yr (t) | **+0.83 (6.4)**, n = 5,499, hit 59% | +0.83 (5.5) | +1.03 (3.2) | −2.36 (−1.3), n = 166 |
| resid, 63d | +1.43 (8.7) | +1.49 (5.2) | +1.41 (2.5) | −2.06 |
| cdi_bps (carry), 126d | +0.57 (1.4) | +0.93 (2.7) | −0.26 (−0.2) | **+3.80 (3.2)**, n = 135 |
| random pair, 126d (placebo) | −0.01 (p5 −0.18, p95 +0.15) | | | |

- The mean carry gap between the cheap and rich legs is 55 bps. So only about 0.3%/yr of the +0.83 is convergence beyond carry.
- The effect is stable across halves: +0.74 in 2022–23 and +1.01 in 2024–25.
- **Incentivada vs taxable.** Buying the higher CDI+-equivalent bond pays: that is usually the *taxable* one, since the peer curve is split by tax status. The incent-vs-taxable residual sort fails. This rests on only about 135 pairs, so treat it as a hint.

### 3. Books (monthly decisions, 126-bday tranches, 25 bps; paired vs P4+Q; Holm across the 12 variants)

| variant | exCDI | vs U | vs P4+Q | t | Holm p | 22–23 / 24–25 vs P4+Q |
|---|---|---|---|---|---|---|
| P4Q_issrich (rich uses e_isec_z) | 2.23 | 1.13 | +0.01 | 0.49 | 1 | +0.01 / +0.01 |
| P4Q_bothrich | 2.23 | 1.13 | +0.01 | 0.49 | 1 | |
| P4Q_secrich | 2.22 | 1.12 | +0.01 | 0.61 | 1 | |
| P4Q_switch_resid (issuer exposure fixed) | 2.21 | 1.11 | −0.01 | −0.18 | 1 | −0.05 / +0.03 |
| P4Q_switch_eslope | 2.20 | 1.10 | −0.02 | −0.43 | 1 | |
| P4Q_switch_carry (control) | 2.22 | 1.12 | 0.00 | – | – | identical to P4+Q |
| **tilt carry + 0.5·e_iss** (best) | **2.25** | 1.14 | **+0.03** | 0.65 | 1 | +0.02 / +0.04 |
| tilt carry + 1.0·e_iss | 2.23 | 1.13 | +0.02 | 0.24 | 1 | |
| RV top 20% e_iss_z | 1.24 | 0.14 | −0.98 | −1.97 | 0.54 | |
| RV top 20% e_isec_z | 1.79 | 0.69 | −0.43 | −0.99 | 1 | |
| RV top 20% r_sec_z | 1.99 | 0.89 | −0.23 | −0.62 | 1 | |
| RV top 20% resid_z | 1.93 | 0.83 | −0.29 | −0.92 | 1 | |
| P4 / **P4+Q** / U | 1.89 / **2.22** / 1.10 | | | | | |

**Why the switch book does nothing.** P4+Q already picks each issuer's top-carry bonds, and P4Q_switch_carry reproduces it exactly. Re-assigning the slots by curve residual changes only **4%** of them, about 4 names a month. At +0.8%/yr per switched unit, the book-level gain is about 0.03%/yr, which switching costs eat. Likewise, the issuer-aware rich flag disagrees with `resid_z` on only about 5 top-carry bonds per date.

**Best variant (tilt carry + 0.5·e_iss) sensitivities** vs P4+Q on the same settings:

| setting | vs P4+Q | t |
|---|---|---|
| 50 bps | +0.02 | 0.6 |
| rec40 | +0.03 | 0.7 |
| hold 63 | +0.05 | 1.0 |

Placebo: the variant scores 2.25, against random books at 0.74 (p95 0.98). The placebo shows that the carry base works, not that the RV tilt does.

### 4. Market-level basis timing (diagnostic)

| basis | range, bps | last value pre-2026 | sign trade on expanding z-score |
|---|---|---|---|
| IPCA+ minus DI+ (CDI+ equivalent) | −159 to −52 | −145 | +0.99%/yr, t 1.0 |
| incent minus taxable | −241 to −43 | −111 | −0.90%/yr, t −1.2 |

Neither is usable: there are only 44 overlapping months. The series is saved in `market_basis.csv`.

### 5. Sealed holdout (2026-01 to 2026-09, run once)

| variant | exCDI | vs P4+Q | t | vs U |
|---|---|---|---|---|
| tilt carry + 0.5·e_iss | +0.04 | −0.05 | −2.0 | +3.3 |
| P4Q_switch_resid | −0.10 | −0.18 | −2.8 | +3.2 |
| P4Q_issrich | +0.15 | +0.07 | 1.8 | +3.4 |

The holdout confirms there is no edge over P4+Q.

## Insights

1. **Split the residual in two.** The peer residual `resid_bps` has two parts:
   - an issuer component, which is persistent credit premium: it pays as carry and does not converge;
   - a bond-within-issuer component, which is true relative value: it converges with a half-life of about 20 weeks.

   Mixing them makes the current `resid_z` a good *return* signal and a poor *fair-value* model. For desk RV screens (the "is this bond mispriced" monitor), use `e_iss`: its convergence R² is 2.5× higher.
2. **Within-issuer switches are real but small.** They earn about +0.8%/yr per unit switched at 126d and +1.4%/yr at 63d. Most of it is carry; only about 0.3%/yr is convergence. This is an execution-desk tool (when rolling or adding to a name, buy its cheapest bond), not a portfolio alpha.
3. **Cross-indexer (DI+ vs IPCA+) switches work as well as same-curve switches** once spreads are converted to CDI+ equivalents: +1.0%/yr, t 3.2.
4. **Incentivada vs taxable within an issuer.** The taxable, higher-spread bond outperformed (+3.8%/yr, t 3.2, but only 135 pairs). The market seems to over-price the tax exemption relative to the peer curve.

## Caveats

- **Marks.** Marks are smooth and stale, so part of the within-issuer convergence may be mark noise. The t-stats are inflated.
- **Pair trades.**
  - The short leg is not implementable, so pair returns are gross and describe switch value only.
  - Pair-type sample sizes are small for `xinc`.
- **Reference data.** Kind, incentive, maturity and group come from static reference data (today's SND table and `equity_map.csv`).
- **No rating levels.** No point-in-time rating levels exist in the repo, so a truly rating-aware peer curve could not be tested. Sector plus issuer effects are the proxy.
- **Multiple testing.** 12 book variants were tested and Holm-adjusted. The ICs, convergence regressions and pair tables are additional exploratory tests and are not Holm-adjusted.
- **Holdout.** It is only 9 months long.

## Reusable signal

- **Function:** `research/nightly/issuer_curve_rv/signals.py`, specifically `signals(panel)` and `attach(panel)`. They work on any harness panel.
- **Cache:** `data/history/nightly/issuer_curve_rv/signals_{W,M}.pkl`, loaded with `load_cached(freq)`.
  - It holds all dates, including 2026. The values are same-day cross-sections, so they carry no future information.
  - Columns: `day, codigo, cnpj8, grp, e_iss, e_grp, e_isec, r_sec, e_slope` and their `_z` versions, plus `resid_bps_z, n_iss_other` and the pair-type flags.

## Rerun

```
PYTHONPATH=. PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/Scripts/python.exe research/nightly/issuer_curve_rv/run.py
```

It takes about 15 minutes and writes `results.json`, `equity_total_return.png`, `cum_excess.png`, `pair_trades.png` and `market_basis.csv`.
