# Factor zoo: cross-sectional bond and issuer factors vs P4+Q

Slug `factor_zoo`. This study runs on harness v4, with monthly decisions, 126-bday tranches, 25 bps costs and a 10% issuer cap. Everything was chosen on data before 2026. The holdout was read once, at the end.

## Rerun

```
PYTHONPATH=. PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/Scripts/python.exe research/nightly/factor_zoo/run.py [--rebuild] [--holdout]
```

- The run takes about 8 minutes. Building the features takes about 35 s and is cached in `data/history/nightly/factor_zoo/features_v4.pkl`.
- `--holdout` adds the sealed 2026 evaluation. It was run once, and the results.json in this folder comes from that run.

## What was built: 110 point-in-time candidate features

All features are known at the close of the decision day. The harness executes at the next trade after that close.

| group | features |
|---|---|
| spread momentum / reversal | `ds_5/21/63/126/252` (Δ CDI+ spread), `dres_21/63`, `drz_21/63` (Δ residual vs peer curve), `ds_63_vs_peer`, `sec_ds_63`, `iss_ds_63`, `d_spread_1w/4w` |
| spread risk / drawdown | `svol_63/126` (std of 5d spread changes), `s_from_peak_252`, `s_from_trough_252`, `s_ownz_252`, `own_z` |
| distance to peer curve at several smoothings | `resid_bps`, `resid_z`, `rz_ma5/21/63`, `rb_ma21`, `rz_minus_ma63`, issuer average residual `iss_resid_bps`, `bond_minus_iss_resid`, `iss_resid_disp`, `cs_minus_iss_cs`, `sec_resid` |
| price / realised hedged return | `ratio_ch_21/63`, `ratio_from_hi_252`, `par_dist`, `rmom_21/63/126/252`, `rmom_252_21`, `rvol_126`, `rdd_252`, `rmin_126`, `iss_rmom_126`. Uses `R[..d-1]` only. |
| liquidity (SND trade files) | `ntr_21/63`, `tdays_63`, `log_vol_63`, `ticket_63` (log R$ per trade), `vol_trend`, `amihud_126`, `days_since_trade`, `trades_30d` |
| registry (SND, issuance-time fields only) | `log_issue_size`, `guar_real/quiro` (Garantia/Especie), `is_476`, `call_clause`, `amortizing`, `coupon_freq_m`, `emission_no`, `bond_age_y`, `ttm_orig_y`, `life_frac`, `y_to_amort`, `iss_n_issues` (issues dated ≤ d) |
| terms | `contract`, `incent`, `is_ipca`, `dur`, `years_to_mat`, `carry_per_dur`, `cs_x_dur` |
| rating / news / CVM | `rat_down_365`, `rat_days_since_down`, `n_rating_90d`, `n_fact_90d`, `n_distress_90d`, `n_deb_mtg_90d`, `news_any_30d`, `press_neg_30d`, `days_since_distress` |
| equity (PIT ticker) | `eq_r21/63/126/252`, `eq_mom_12_1`, `eq_idio63` (vs Ibov), `eq_vol63`, `eq_dd252`, `log_eq_adtv`, `eq_vol_trend` |
| fundamentals (strict CVM dates) | `f_lev`, `f_cov`, `f_cash_st`, `f_eq_ratio`, `f_gde`, `f_margin`, `f_rev_g`, `f_d_lev`, `f_d_cov`, `f_size`, `f_quality`, `f_age_days` |

## Method

1. **ICs.** Per-date ICs on the eligible universe at 21, 63, 126 and 252 bdays, using patched rate-hedged excess from the entry trade. Newey-West lag = horizon in months. Four versions were computed:
   - **raw:** Spearman IC;
   - **within-peer:** ranks taken inside day × peer;
   - **partial:** the normal scores of both the feature and the target are residualised on carry (`cdi_bps`), RV (`resid_z`), `dur`, IPCA and incentivada;
   - **inside P4+Q:** the partial IC computed only on the P4+Q names. This is the incremental question.
2. **Clustering.** The per-date Spearman correlation matrix is averaged over dates. Features are then clustered by average linkage on 1-|ρ|, with the cut at |ρ| = 0.5, which gives 49 clusters.
3. **Survivors.** They were chosen on decisions from 2021-03 to 2023-06 only, so their 6m labels end before 2024. A feature had to pass all of these:
   - partial IC 6m |t| ≥ 2;
   - the same sign at 3m;
   - the same sign inside P4+Q;
   - coverage ≥ 30%;
   - it is the best of its cluster;
   - at most 8 survivors are kept;
   - carry and RV variants are excluded.
4. **Validation.** The ICs were validated on decisions from 2024-01 to 2025-06.
5. **Composite.** An equal-weight mean of per-date normal scores of the signed survivors.
6. **Backtests.** 11 variants, all in one Holm family vs P4+Q.

## Results

**Survivors, frozen from the selection period**, with their sign:

| survivor | sign |
|---|---|
| `ds_5` (1-week spread change) | − |
| `days_since_distress` | − |
| `eq_r21` | + |
| `eq_r126` | + |
| `bond_age_y` | + |
| `dres_21` (1-month change in residual) | − |
| `age` (days since the bond's last trade) | + |
| `eq_vol63` | − |

For reference only, an in-sample selection on all data before 2026 gives: `eq_r126`, `eq_r21`, `ds_5`−, `eq_vol63`−, `n_fact_90d`−, `dres_21`−, `f_quality`, `f_d_lev`−.

**Composite IC (6m).** The composite only works once carry is held fixed. Unconditionally its IC is slightly negative, because good names pay less carry.

| measure | IC | t |
|---|---|---|
| raw | −0.03 | −1.1 |
| partial, selection period | +0.12 | 6.7 |
| partial, validation 2024-01 to 2025-06 | +0.077 | 6.9 |
| inside P4+Q, all pre-2026 | +0.11 | 3.8 |
| inside P4+Q, validation | +0.11 | 3.4 |

**Backtests, pre-2026.** Excess over CDI in %/yr, 25 bps. **11 variants were tried.** Holm p is taken across all 11 plus P4.

| variant | exCDI | vs U (t) | vs P4+Q | t | Holm p | 22–23 / 24–25 vs P4+Q | vol | Sharpe | maxDD | names |
|---|---|---|---|---|---|---|---|---|---|---|
| **zoo_screen20** (P4+Q minus worst 20% of composite) | **2.58** | +1.47 (2.9) | **+0.36** | 2.54 | 0.13 | +0.32 / +0.39 | 0.88 | 2.94 | −0.77 | 82 |
| zoo_screen33 | 2.63 | +1.53 (3.1) | +0.42 | 2.54 | 0.13 | +0.38 / +0.45 | 0.87 | 3.03 | −0.66 | 69 |
| zoo_tilt (carry + 0.5·composite, P4-sized) | 2.11 | +1.01 | −0.11 | −1.7 | 0.58 | −0.08 / −0.14 | | | | 130 |
| zoo_is_screen20 (in-sample spec) | 2.56 | +1.46 | +0.35 | 2.32 | 0.18 | +0.24 / +0.45 | | | | 82 |
| group_eq_screen20 (equity factors only) | 2.50 | +1.40 | +0.29 | 2.11 | 0.25 | +0.24 / +0.33 | | | | 93 |
| group_markdyn_screen20 (ds_5, dres_21) | 2.34 | +1.24 | +0.12 | 1.46 | 0.72 | +0.19 / +0.05 | | | | 84 |
| group_other_screen20 (age, bond age, distress) | 2.32 | +1.22 | +0.10 | 0.72 | 1.0 | +0.17 / +0.03 | | | | 82 |
| single eq_r21 screen | 2.44 | +1.34 | +0.22 | 2.46 | 0.14 | | | | | 93 |
| single ds_5 screen | 2.31 | | +0.09 | 1.08 | 1.0 | | | | | 85 |
| single days_since_distress screen | 2.52 | | +0.30 | 0.68 | 1.0 | | | | | 16 |
| P4 minus worst 20% of composite (no quality screen) | 2.33 | | +0.11 | 0.88 | 1.0 | | | | | 103 |
| P4 | 1.89 | +0.79 | −0.33 | −2.3 | | | | | | 130 |

**Robustness of zoo_screen20**, as the difference vs P4+Q on the same settings:

| setting | vs P4+Q | t |
|---|---|---|
| 50 bps | +0.33 | 2.3 |
| rec40 | +0.36 | 2.5 |
| hold 63 | +0.43 | 2.3 |
| weekly tranches | +0.40 | 2.7 |

**Placebo.** Randomly dropping 20% of the P4+Q names, 30 runs, gives a mean of −0.04 and a 95th percentile of +0.01. The actual +0.36 has an empirical p of 0.

**Sealed holdout (2026-01 to 2026-09, 9 months, frozen spec, read once):**
- zoo_screen20 earned +0.20%/yr over CDI.
- **Vs P4+Q it was +0.12%/yr** (t 1.3), and it beat P4+Q in 8 of 9 months.
- Vs the universe it was +3.5%/yr (t 4.2).

**Conclusion.** The increment keeps its sign in the holdout but roughly a third of its size, and it is not significant. It is also **not significant after Holm before 2026 (p 0.13)**.

## Insights

1. **Equity momentum carries most of the increment.**
   - `eq_r126` has a partial 6m IC of +0.107 (t 6.8), +0.117 in validation, and +0.18 inside P4+Q (t 7.4).
   - `eq_dd252`, `eq_r252` and `eq_idio63` are the same cluster.
   - The equity-only sub-composite delivers +0.29 of the +0.36.
   - Stock prices lead debenture marks for listed issuers. Because the raw IC is only about 0.04, this is invisible without conditioning on carry.
2. **Equity volatility works as a risk flag inside the high-carry bucket.**
   - `eq_vol63` has a partial IC of −0.09 at 6m and −0.14 at 12m.
   - It is essentially orthogonal to momentum (a separate cluster).
3. **Short-horizon mark momentum.**
   - A spread that widened over the last week predicts underperformance over 6–12 months: `ds_5` has a partial IC of −0.048 (t −5.5), and `dres_21` −0.036.
   - Longer spread momentum (`ds_63` to `ds_252`) is about 0 once carry is held fixed. This is slow price discovery in stale marks, not a medium-term momentum premium.
   - As a stand-alone screen it adds only +0.09 to +0.12.
4. **Factor decay rises with horizon for almost every surviving factor.** Partial ICs at 21 / 63 / 126 / 252 bdays:

   | factor | 21 | 63 | 126 | 252 |
   |---|---|---|---|---|
   | `eq_r126` | 0.03 | 0.07 | 0.11 | 0.13 |
   | `eq_vol63` | −0.02 | −0.05 | −0.09 | −0.14 |

   These are credit-drift factors, not short-term alpha. Reacting fast does not pay, which matches earlier labs, but a slow screen does.
5. **Several factors look good in raw IC but add nothing after carry and RV.**
   - `rz_ma5` / `rz_ma21` / `resid_bps`: the raw IC is +0.21 (t 11), but the partial IC is only 0.03 to 0.06. That is the known smooth-mark RV effect, already in P4's rich filter.
   - The issuer-level residual `iss_resid_bps` has a raw IC of +0.18 and a partial IC of −0.01.
   - Liquidity (Amihud, volume, trade counts), guarantee type, 476 vs 400 offerings, call clause, coupon frequency and issue size all have |partial t| < 2, or are unstable across periods.
6. **Some selection-period winners flipped sign in validation, which is a warning.**
   - `days_since_distress`: selection t −7.5, validation IC +0.04.
   - `bond_age_y`: selection t +3.5, validation −0.01.
   - `is_476`: t +4.5, then −3.7.
   - A 2.5-year selection window picks up regime-specific artefacts, so trust only the equity block and, weakly, `ds_5`.
7. **Carry is not substitutable.**
   - Tilting the carry rank with the composite (zoo_tilt) loses −0.11 vs P4+Q.
   - The factors only help as a **screen inside the top-carry bucket**.
   - Without the fundamentals quality screen (P4 minus the composite), the gain drops to +0.11. Quality and the zoo composite are complementary.

## Reusable signal

- **`research/nightly/factor_zoo/features.py`:**
  - `build_features()` returns about 60 new PIT features for every M and W decision date, keyed on (`day`, `codigo`).
  - `attach(panel)` merges them into a harness panel.
  - `signals(panel)` returns the frozen composite as (`day`, `codigo`, `value`). Higher is better. The spec is read from `results.json` `composite_spec`.
- **Cache:** `data/history/nightly/factor_zoo/zoo_composite.pkl` has columns `day, codigo, cnpj8, value, freq (M|W)`. It covers universe rows on all dates up to 2026-09, with the spec frozen before 2026.
- **Recommended use:** drop the bottom 20% of `value` among P4+Q names. The equity-only block (`eq_r21`, `eq_r126`, `eq_vol63`−) is the most robust part.

## Caveats

- The survivor selection used only 28 monthly dates. Four of the 8 survivors are unstable. The increment is not significant after Holm (p 0.13) and shrank in the holdout (+0.12, t 1.3, 9 months).
- Smooth and stale marks inflate the t and Sharpe figures. The `age` (days since trade) survivor may partly be a mark-staleness artefact.
- Registry fields come from the 2026-09 SND snapshot, so repactuated bonds carry their current maturity and amortisation terms. Outstanding-quantity fields were deliberately not used.
- Equity features cover about 63% of universe rows. For unlisted issuers the composite averages fewer components, and those issuers are where survivorship bias lives.
- Results are tranche books at marks with no market impact. The screen cuts the book to about 82 names, which raises concentration but is still diversified.

## Files

- Code: `run.py`, `features.py`.
- Results: `results.json`, which holds the factor table, decay, clusters, compare table, robustness, placebo and holdout.
- Charts:
  - `equity_total_return.png`: total return, CDI × (1+excess), vs CDI, U, P4, P4+Q, IDA-DI and Ibov;
  - `cum_excess.png`: cumulative excess vs the universe;
  - `factor_ics.png`: partial IC in the selection period, in validation, and inside P4+Q;
  - `factor_decay.png`.
