# w2_expected_loss_carry: point-in-time distress hazard model and expected-loss-adjusted carry (harness v4)

**Bottom line (negative).** A walk-forward 12-month distress-hazard model gets a debenture-issuer AUC of 0.66–0.69, rising from about 0.6 in 2022 to 0.8–0.85 in 2024–25. That is barely better than the spread level alone, which scores 0.66.

Ranking the universe on expected-loss carry (cdi_bps − PD × LGD × 10⁴) **loses to P4+Q**:
- by −0.25 to −0.33%/yr;
- by −0.9 to −1.0%/yr in 2022–23.

Subtracting PD halves carry's IC, from 0.25 to 0.09–0.12.

As a quintile screen, the PD model does less well than the Q quality screen it was meant to replace:
- P4 − PD quintile scores −0.11 / −0.25 vs P4+Q.
- Adding it to P4+Q gives +0.04 / +0.09 (t ≤ 0.55).
- Adding it to P7 **lowers** P7: −0.12 (LGBM) and −0.15 (logit) vs P7, in both halves, at 50 bps and under harsh survivorship. It is also no better than dropping the same number of random names (placebo p 0.76).

The "best" variant, C_lgbm = P7 + PD screen, shows +0.38 vs P4+Q (t 2.2, Holm p 0.27 over 10 variants). **All of that comes from P7** (+0.50 on its own).

**Identification.** The model is **under-identified for hard defaults**. The debenture universe has 1 / 2 / 14 / 10 / 10 hard-event issuer onsets in 2021–25. Hard events are bias_audit distress, severe rating actions and CVM RJ/default filings. That is below 30 in every year. Adding bond-loss episodes gives 7 / 11 / 46 / 23 / 34 onsets.

## What was built

| file | what |
|---|---|
| `build_data.py` | Builds the events and the broad fundamentals panel (below). |
| `model.py` | The issuer-month panel (76k rows, 1,456 issuers, 2019-04 → 2026-09) and walk-forward logit and monotone-LightGBM PDs (`pd_oos.pkl`). Writes `model_eval.json`. |
| `run.py` | The 10 portfolio variants, the 50 bps runs, rec40, harsh survivorship, the placebo and the charts (`results.json`, `compare_pre2026.csv`, `equity_pre2026.png`). `--holdout` reads 2026 once (`equity_total_return.png`). |
| `diag_capture.py` | Event capture inside P4: PD flags vs worstQ vs the P7 equity-health screen (`capture.json`). |
| `plot_auc.py` | Draws `auc_by_year.png`. |

### Events

The events file is `data/history/nightly/w2_expected_loss_carry/events.pkl`, keyed by `cnpj8, date, type, onset`. A new episode starts 365 days after the issuer's previous event. It combines five sources:

- **(a) bias_audit PIT distress list:** 33 issuers.
- **(b) Severe rating actions:** 15 actions at 5 issuers, taken from `rating_events.pkl` by a text rule (to CCC / C / D / RD, "calote", default, RJ). The file has **no rating levels**, so the "≥ 3 notches" rule in the spec cannot be computed.
- **(c) Bond-loss episodes:**
  - **Path:** the harness v4 patched return path, market-neutralised against the same-day median of its kind and dated on the day the mark shows the loss.
  - **Threshold:** a fall below the trailing 126-bday high of −10% for DI+ floaters, and −20% for IPCA and PRE bonds.
  - **Why IPCA / PRE use −20%:** at −10%, 220 IPCA issuers are flagged and only 10% of them have any hard event. That is rate-hedge noise; for DI+ the overlap is 62%.
- **(d) CVM IPE filings, 2019–26, for every CVM issuer (new data):**
  - **What counts:** the first "Informações de Companhias em Recuperação Judicial ou Extrajudicial" document, or a Fato Relevante / Comunicado announcing RJ, RE or a default.
  - **Filter:** a counterparty-mention filter (the filing must sit within 400 days of the issuer's own RJ document, or the registered name must say "EM RECUPERAÇÃO", "FALIDA" or "LIQUIDAÇÃO"). This removes Vale, Klabin, Engie, Natura and others that only disclosed exposure to someone else's RJ.
  - **Dating:** by `Data_Entrega`.
  - **Download:** the 2019–20 zips were downloaded to the cache folder.

### Broad fundamentals

The file is `fund_all.pkl`: CVM ITR/DFP for **all 933 CVM filers** from 2019. It is built by importing the unmodified `research/data_fundamentals.py` functions, so it covers far more firms than the universe's 435, and it gives the pre-2021 training panel RJ labels. It is point-in-time on `available_date_strict`, and rows more than 460 days stale are dropped.

### Features, known at the decision close

- **Fundamentals:** ND/EBITDA (negative EBITDA mapped to 20), log interest coverage, log cash/ST debt, equity ratio, gross debt/equity, EBITDA margin and its 4-quarter change, Δ ND/EBITDA, Δ coverage, revenue growth, log assets, ROA, EBIT/TA, cash/TA, ST-debt share, and negative equity / EBITDA flags.
- **Equity:** `eq_dd252`, `eq_vol63`, `eq_r126`, and structural_credit `sc_dd`.
- **Bond:** median and max log spread, residual vs peer, min ratio, max `ds_5` / `ds_21` / `ds_63` / `dres_21` (factor_zoo), `rdd_252`, `ratio_from_hi_252`, negative press 30d, CVM distress filings 90d, and number of bonds.
- **Ratings and events:** log days since downgrade, downgrades in the last 365d, event in the past 365d, and log days since the last event.
- **Other:** coverage flags; sector dummies (logit only).

### Models

- **Label:** any event in (d, d + 365d].
- **Training:** rows with label_end ≤ d only, so there is a full 12-month embargo, refit every month.
- **Logit:** L2 with C = 0.05 on standardised features, plus missing-value flags.
- **LightGBM:** monotone signs on every economic feature, 7 leaves, 150 trees, min_child 80, n_jobs 2.

## Model quality

The table shows out-of-sample AUC on debenture issuers. Outcomes stop at 2025-12-31, so 2025 is censored.

| year | issuers with event | logit | LGBM | listed LGBM | unlisted LGBM | no-fundamentals LGBM |
|---|---|---|---|---|---|---|
| 2021 | 13 | 0.46 | 0.67 | 0.47 | 0.71 | 0.50 |
| 2022 | 51 | 0.61 | 0.66 | 0.64 | 0.64 | 0.56 |
| 2023 | 59 | 0.63 | 0.66 | 0.65 | 0.65 | 0.60 |
| 2024 | 56 | 0.73 | 0.81 | 0.89 | 0.75 | 0.69 |
| 2025 (censored) | 36 | 0.78 | 0.85 | 0.92 | 0.82 | 0.79 |
| pooled | | 0.655 | 0.686 | 0.688 | 0.674 | 0.626 |

- Spread level alone scores 0.658 pooled; the past-event flag scores 0.61; ND/EBITDA alone, on covered issuers, scores 0.61.
- On "clean" issuer-months (no event in the prior year) the models reach only 0.56 / 0.60, so much of the AUC is persistence of distress.
- The broad CVM panel (no bonds) scores AUC 0.93, but that is persistence of RJ status, not prediction.
- The top LGBM gains are:
  1. days since the last event;
  2. the bond's ratio vs its 252-day high;
  3. `eq_dd252`;
  4. cash/ST debt;
  5. equity ratio;
  6. coverage.

## Portfolios

Pre-2026 results use monthly decisions, 126-bday tranches and 25 bps. Holm is taken across the 10 variants tried.

| variant | exCDI | vs U (t) | vs P4+Q | t | Holm p | 22–23 / 24–25 | 50 bps vs P4+Q | vs own base |
|---|---|---|---|---|---|---|---|---|
| A logit EL-carry, LGD 0.6 | 1.91 | +0.81 (2.2) | −0.31 | −0.91 | 1 | −0.96 / +0.35 | −0.31 | vs P4 +0.02 |
| A logit, LGD 0.4 | 1.88 | +0.78 | −0.33 | −1.07 | 1 | −0.89 / +0.23 | −0.33 | vs P4 −0.01 |
| A LGBM, LGD 0.6 | 1.96 | +0.86 | −0.26 | −0.66 | 1 | −0.98 / +0.45 | −0.26 | vs P4 +0.07 |
| A LGBM, LGD 0.4 | 1.97 | +0.87 | −0.25 | −0.67 | 1 | −0.88 / +0.39 | −0.24 | vs P4 +0.08 |
| B logit P4 − PD quintile | 1.97 | +0.86 | −0.25 | −1.08 | 1 | −0.57 / +0.06 | −0.24 | vs P4 +0.08 |
| B LGBM P4 − PD quintile | 2.11 | +1.01 | −0.11 | −0.49 | 1 | −0.50 / +0.27 | −0.11 | vs P4 +0.22 (t 0.9) |
| D logit P4+Q − PD quintile | 2.26 | +1.16 | +0.04 | 0.24 | 1 | −0.24 / +0.33 | +0.03 | |
| D LGBM P4+Q − PD quintile | 2.31 | +1.21 | +0.09 | 0.55 | 1 | −0.20 / +0.39 | +0.08 | |
| C logit P7 + PD screen | 2.57 | +1.47 | +0.35 | 1.66 | 0.87 | +0.17 / +0.54 | +0.32 | **vs P7 −0.15 (t −1.5)** |
| **C LGBM P7 + PD screen** ("best") | 2.60 | +1.50 (3.5) | +0.38 | 2.21 | 0.27 | +0.17 / +0.60 | +0.35 | **vs P7 −0.12 (t −1.0)** |
| *ref: P7* | 2.72 | +1.62 | +0.50 | 3.29 | | +0.37 / +0.63 | | |
| *ref: P4* | 1.89 | +0.79 | −0.33 | −2.27 | | | | |

### Robustness of the frozen best (C LGBM)

- **rec40:** +0.35 vs P4+Q (t 1.7).
- **bias_audit harsh survivorship** (60 bonds that stopped trading below 0.98 of par are marked down to 40%): +0.68 vs P4+Q (t 2.9), while P7 itself scores +0.76.
- **Placebo:** a random screen dropping the same number of P7-base names averages −0.08 vs P7 (p95 +0.04). The PD screen scores −0.12, so p = 0.76.
- **Metrics:** Sharpe 2.82, maxDD −0.98%. Marks are smooth, so Sharpe and t are inflated.

## Insights

1. **The PD model sees the events but not the losses.** Inside P4, the LGBM PD flag catches **37%** of the issuer-months that have an event within 12 months, at a 16% flag rate. The Q screen catches 38% at 18%; the P7 equity-health screen catches 36% at 15%. So recall is the same for all three. The flagged names' mean 6-month forward return differs sharply, though:

   | screen | flagged | not flagged |
   |---|---|---|
   | PD | +0.66% | +1.33% |
   | Q | +0.40% | +1.42% |
   | equity-health | **−0.44%** | +1.54% |

   The hazard model flags the "soft" episodes, whose marks dip and recover. It is late or absent on the few hard blow-ups where the money is lost.
2. **Expected-loss carry is a worse ranking variable than raw carry.** PD is almost orthogonal to carry (Spearman −0.01 logit, +0.10 LGBM). So subtracting PD × LGD mostly injects model noise into the carry ranking:

   | ranking variable | IC vs fwd_126 |
   |---|---|
   | carry | 0.25 |
   | EL-carry | 0.09 (LGBM) / 0.12 (logit) |
   | −PD alone | 0.00 (LGBM) / 0.03 (logit) |

   All A books lose about −0.9%/yr to P4+Q in 2022–23 and gain in 2024–25. The market already prices most of the default risk that a 12-month PD can see: spread-level AUC is 0.66 against 0.69 for the model.
3. **Fundamentals cannot fix the unlisted gap.** Unlisted issuers have a pooled AUC of 0.67. Debenture issuers with no fundamentals (unregistered SPVs) have 0.63, and 0.50–0.60 in 2021–23. Those issuers get their PD from bond-market features alone.
4. **Labels are too few and too soft.** Only 37 hard-event onsets hit debenture issuers in 2021–25. The model is trained mostly on bond-loss episodes (defined by marks) and on RJ persistence in the broad CVM panel. Its PD therefore turns into a "recent mark weakness" detector, and the rich rule and the equity screen already cover most of that.
5. **AUC improves with training depth:** LGBM goes from 0.66 in 2022–23 to 0.81–0.85 in 2024–25. On listed issuers it reaches 0.89–0.92. The model may become useful as labels accumulate, but not in this sample.

## Sealed holdout (2026-01 → 2026-09, read once, for the frozen C_lgbm only)

- C_lgbm scores +1.19%/yr ex-CDI, and **+1.11 vs P4+Q (t 2.9)**.
- P7 alone scores +0.68 vs P4+Q (t 2.2).
- P4+Q itself scores +0.08.

So in the holdout the PD screen *added* about +0.43 to P7, the opposite of the pre-2026 sign. The holdout PDs were trained on labels that include the 2025 events (Ambipar, Braskem, Raízen downgrades, Oncoclínicas), and 2026 has the most bond-loss onsets of any year (85). This is 9 months and one credit episode. **It does not change the pre-2026 verdict.** It is the only hint that the object gets better as it learns more events.

## Caveats

- **Weak labels.**
  - Bond-loss episodes come from smooth, stale marks.
  - The IPCA threshold (−20%) was chosen by looking at how often it overlapped with the hard events.
  - The rating file has no levels.
  - The IPE rule is a regex with a counterparty filter.
- **PD is mis-specified as default probability.** It is the probability of *any* distress event, most of which are ≈10–20% mark drawdowns. So PD × LGD with LGD 0.4 / 0.6 overstates expected loss. The LGD sensitivity (0.4 vs 0.6) barely changes the results.
- **The broad CVM panel mixes populations.** Non-debenture firms supply mostly RJ-persistence labels.
- **Sample and marks.** Only about 2 credit episodes fall before 2026. Smooth marks inflate Sharpe and t, and Holm p ≥ 0.27 everywhere.

## Reusable

- **`pd_oos.pkl`** (`data/history/nightly/w2_expected_loss_carry/`) holds the walk-forward out-of-sample PDs `pd_logit` and `pd_lgbm`, keyed by `(cnpj8, day)`.
  - **Dates:** monthly harness days from 2021-03; first business day of the month from 2020-07.
  - **PIT rule:** trained only on labels ending ≤ day.
  - **2026:** 2026 rows are holdout scores.
- **`events.pkl`** holds the PIT issuer distress events with an `onset` flag. **`ipe_events_raw.pkl`** holds the CVM RJ / default filings for all CVM issuers, 2019–26. **`bond_loss_events.pkl`** holds the bond-level market-neutral loss episodes.
- **`fund_all.pkl`** holds CVM fundamentals for all 933 CVM filers, point-in-time on `available_date_strict`. **`issuer_month.pkl`** is the hazard panel with features and labels.
- **Functions:**
  - `model.fundamentals()`
  - `model.build_panel()`
  - `run.load(holdout)`, which gives the panel with `pd_*`, `pdQ_*` and `p4q_pd_*`
  - `run.harsh_R()`, which returns the bias_audit harsh-survivorship return matrix

## Rerun

```
set PYTHONPATH=. PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
.venv/Scripts/python.exe research/nightly/w2_expected_loss_carry/build_data.py    # ~4 min (IPE download, CVM all filers)
.venv/Scripts/python.exe research/nightly/w2_expected_loss_carry/model.py         # ~4 min
.venv/Scripts/python.exe research/nightly/w2_expected_loss_carry/run.py           # ~2 min
.venv/Scripts/python.exe research/nightly/w2_expected_loss_carry/diag_capture.py
.venv/Scripts/python.exe research/nightly/w2_expected_loss_carry/plot_auc.py
.venv/Scripts/python.exe research/nightly/w2_expected_loss_carry/run.py --holdout # sealed holdout, once
```

The task text named both `expected_loss_carry/` and `w2_expected_loss_carry/`. Everything lives in `research/nightly/w2_expected_loss_carry/` and `data/history/nightly/w2_expected_loss_carry/`.
