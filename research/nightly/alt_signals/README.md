# alt_signals: alternative and supply signals as overlays on P4+Q

**Bottom line.** Two new point-in-time signals helped P4+Q before 2026: an issuer launching a new debenture offer, and a sector's commodity nowcast turning bad. Removing the P4+Q names flagged by either one added **+0.34%/yr vs P4+Q** (NW t 2.94, both halves +0.35 / +0.33). The random-exclusion placebo p95 is +0.07. The gain survives 50 bps (+0.31) and rec40 (+0.32).

It is **not a clean win**:
- The combined rule was defined after looking at the IC stage.
- Holm across all 20 variants tried gives p = 0.064. It is 0.042 only within the 11-variant main family.
- In the sealed 2026 holdout it did −0.09 vs P4+Q over about 9 months.

Every other alternative signal was null as an overlay: ratings/outlooks, hot-sector supply, ANEEL tariff resets, the hand-built margin nowcast alone, and positive tilts.

Harness v4, monthly decisions, 126-bday tranches, 25 bps, issuer cap 10%. The sealed holdout was touched once, at the end.

## Data built (all public, cached in `data/history/nightly/alt_signals/`)
| dataset | source | coverage | PIT rule |
|---|---|---|---|
| debenture offers (`offers_deb.pkl`) | CVM `OFERTA/DISTRIB/oferta_distribuicao.zip` (legacy ICVM 400/476 plus RCVM 160) | 2015–2026-09, about 600–650 debenture offers/yr, cnpj8 of the issuer | legacy: `Data_Inicio_Oferta + 7d`; RCVM 160: `Data_requerimento + 1d` (automatic rite, public on filing). Expired requests dropped. Counts only; the value field fills in later, so it is not used |
| ANEEL tariff resets (`aneel_resets.pkl`) | dadosabertos.aneel.gov.br `tarifas-homologadas-distribuidoras` | 2010–2026, B1 residential TE+TUSD per distributor CNPJ | `DatInicioVigencia` (the REH is published a few days earlier) |
| ANEEL tariff flags | `bandeira-tarifaria-acionamento.csv` | 2015–2026 monthly | usable from the first day of the competence month (announced the previous month) |
| sector nowcasts (`sector_nowcasts.pkl`) | `data/history/commodities.pkl` (already carries `available_date`) plus the flags | daily 2019–2026, 16 sectors | 6-month log change ÷ expanding past-only std. The reservoir anomaly is measured against past years' same-month mean |
| ratings | existing `rating_events.pkl`, titles parsed for outlook/watch | 2021–2026 | action date + 1d |

Tried and dropped:
- **ANTT toll revenue:** it is annual only, so it is useless. Monthly traffic exists only for 2024+ in consolidated form.
- **CCEE PLD:** the site is WAF-blocked (known), so ONS CMO was used instead.
- **Google Trends:** needs a non-simple API, so it was skipped.
- **ANAC:** no airline issuers in the universe.

## Signals (reusable)
`research/nightly/alt_signals/signals.py::signals(panel)` adds these columns to any harness panel: `sup_iss_90d, sup_iss_365d, sup_sec_mom, sup_mkt_mom, sup_inc_mom, rat_neg_180d, rat_out_neg_180d, rat_pos_180d, nc_margin, nc_expo, tar_adj_last, tar_days_since`.

A cached long file of the monthly panel is at `data/history/nightly/alt_signals/alt_signals_M.pkl` (codigo, cnpj8, day, sector plus the columns above).

- `nc_expo`: the exposure-weighted nowcast. It uses the existing `research/data/sector_exposures.csv` weights, with commodity and power variables only (no ipca/selic/ibc).
- `nc_margin`: a hand-built margin/parity nowcast. The formulas are in `sector_nowcasts()`:
  - sugar + ethanol − diesel;
  - Brazil diesel retail − ULSD import parity;
  - iron ore;
  - reservoir anomaly for hydro;
  - −anomaly + PLD for thermal;
  - −flag for distribution;
  - −cattle for meatpackers;
  - urea − gas for fertilizers.

## Results (pre-2026, 48 months)
### Information coefficients / conditional forward excess (fwd_126, vs the rest of the date)
| signal | IC univ (t) | IC within P4+Q (t) | flagged − rest, P4+Q, %/yr (t) |
|---|---|---|---|
| issuer new offer 90d | −0.033 (−1.2) | **−0.090 (−4.8)**, 17 dates | **−1.68 (−2.5)** |
| issuer offer 365d | −0.022 (−0.7) | −0.069 (−2.3) | |
| negative rating action 180d | −0.036 (−2.0) | −0.073 (−2.3), 6 dates | −0.97 (−0.8) |
| negative outlook/watch only | −0.033 (−1.5) | too few | univ −2.07 (−2.3) |
| positive rating action | +0.014 (1.7) | +0.097 (3.5), 5 dates | +0.03 |
| sector supply momentum | +0.011 (1.0) | −0.012 (−0.5) | +0.68 (1.1) |
| nc_margin (exposed sectors) | −0.004 (−0.1) | +0.067 (1.0) | < −1: **−1.60 (−2.4)** |
| nc_expo (exposed sectors) | +0.010 (0.4) | +0.048 (1.2) | < −1: **−1.56 (−2.9)** |
| ANEEL last tariff adj. (distribution) | −0.011 (−0.4) | n/a | |

### Books (overlay on P4+Q; Holm within the 11-variant family)
| variant | exCDI | vs U | vs P4+Q | t | Holm p | 22–23 / 24–25 | names |
|---|---|---|---|---|---|---|---|
| P4Q ex nc_margin < −1 | 2.23 | 1.13 | +0.02 | 0.6 | 1 | +0.02 / +0.01 | 101 |
| P4Q ex nc_expo < −1 | 2.31 | 1.21 | +0.09 | 1.5 | 1 | +0.23 / −0.04 | 97 |
| P4Q ex issuer new offer 90d | 2.43 | 1.33 | +0.21 | 1.9 | 0.59 | +0.07 / +0.35 | 84 |
| P4Q only issuers with an offer in 365d | 1.91 | 0.81 | −0.30 | −1.3 | 1 | +0.15 / −0.76 | 62 |
| P4Q ex negative rating 180d | 2.25 | 1.14 | +0.03 | 0.7 | 1 | | 100 |
| P4Q ex negative outlook | 2.22 | 1.12 | +0.00 | 0.9 | 1 | | 103 |
| P4Q ex hot sector supply | 2.17 | 1.07 | −0.05 | −0.5 | 1 | | 89 |
| P4Q ex tariff cut | 2.22 | 1.12 | +0.00 | 1.3 | 1 | | 103 |
| P4Q ex all alt (nc_margin, rating, tariff) | 2.27 | 1.17 | +0.05 | 1.0 | 1 | | 98 |
| **P4Q ex (new offer 90d OR nc_expo < −1)** * | **2.56** | **1.45** | **+0.34** | **2.94** | **0.042** | **+0.35 / +0.33** | 79 |
| P4Q plus strong-nowcast carry names | 2.21 | 1.11 | −0.01 | −0.7 | 1 | | 106 |
| P4+Q (baseline) | 2.22 | 1.12 | 0 | | | | 103 |
| P4 | 1.89 | 0.79 | −0.33 | −2.3 | | | 130 |

\* Added after the IC stage.

For the best variant:
- **Other metrics:** vol 0.98, Sharpe 2.61 (inflated by smooth marks), maxDD −0.94 (P4+Q −1.39), turnover 3.15×/yr.
- **Costs and recovery:** 50 bps +0.31 (t 2.7); rec40 +0.32 (t 2.8).

### Robustness of the best rule (`robustness.json`)
- **Supply window:**

  | window | vs P4+Q | t | names |
  |---|---|---|---|
  | 30d | +0.14 | 1.9 | |
  | 60d | +0.26 | 2.9 | |
  | 90d | +0.34 | 2.9 | |
  | 180d | +0.48 | 2.9 | 62 |
  | 365d | +0.63 | 2.0 | 37 (too concentrated) |

  The effect is monotone in the look-back. That makes it look like a **"frequent borrower / debt-growth" quality signal**, not a short-lived supply-pressure effect.
- **Nowcast threshold:** −0.5 gives +0.25 (t 1.8), −1.5 gives +0.28 (t 2.7). Using `nc_margin` instead gives +0.24 (t 2.3).
- **Engines:**

  | engine | vs P4+Q | t |
  |---|---|---|
  | hold 63 | +0.40 | 2.3 |
  | hold 252 | +0.14 | 1.5 |
  | weekly 126-bday tranches | +0.33 | 2.7 |
  | cohort 3m | +0.45 | 2.8 |
  | cohort 6m | +0.43 | 6.0 |
  | cohort 12m | +0.16 | 3.2 |
- **Concentration:** keeping the 5 most-excluded issuers in the book still leaves +0.28 (t 2.8).
- **Where the supply leg works:** only **outside regulated infra**. Non-infra only gives +0.21 (t 2.2); infra-only (power, sanitation, tolls) gives −0.02. For infra issuers, repeat issuance is simply capex funding and carries no signal.
- **Placebos:**
  - Random exclusion of the same number of P4+Q names (20 draws): mean −0.03, p95 +0.07, real +0.34 (p = 0/20).
  - Sector-permutation of the nowcast leg: real +0.094 vs placebo mean +0.011, p95 +0.067 (0/20).
  - Supply offers shifted ±1 year: −0.10 / −0.11 (t −1.5). The timing matters.
- **Holm across all 20 variants tried** (the 11 above, 7 sensitivities and 2 splits): the best raw p is 0.0032, which gives **Holm p 0.064**.

### Supply event study (`supply_event_study.png`, `event_study.json`)
850 offer events, 2021-06 to 2025. The measure is the cumulative excess of the issuer's existing bonds vs the universe:
- **Before the offer:** −0.14% over the 63 days before (t −1.7).
- **After the offer:** −0.30% over the 126 days after (t −2.3).
- **P4+Q issuers:** only about −0.2% before and ≈0 after.

So the book gain is **not** a clean price-impact effect around the deal. It comes from the kind of issuer that keeps coming back to market: non-infra, growing leverage. That cannot be read off a single event window.

### Sealed holdout (2026-01 → 2026-09, reported once)
Best variant: exCDI −0.01%/yr, **−0.09 vs P4+Q** (t −0.7), +3.25 vs U. A few months only; it neither confirms nor refutes the result.

## Insights
1. **An issuer's own recent debenture issuance is the strongest new alternative signal.** Among P4+Q names it has IC −0.09 (t −4.8), and flagged names underperform by −1.7%/yr on the 6-month label. It is free, public, point-in-time from the CVM registry, and orthogonal to carry/rich/fundamentals.
2. The supply signal works **only for non-infrastructure issuers**. For concession/utility SPVs, new offers are routine capex funding.
3. **Sector commodity nowcasts matter only inside P4+Q.** Among high-carry names in commodity-exposed sectors, a bad 6-month commodity move (nowcast z < −1) predicts −1.6%/yr. Across the whole universe the IC is ≈0: high-carry credits are the ones whose spreads respond to fundamentals.
4. **Ratings, outlooks, tariff resets and sector-level supply add nothing to P4+Q.** The rating news is already in the price/`rich`/quality filters, and outlook changes are too rare in P4+Q (26 rows).

## Caveats
- The combined best variant was chosen after seeing the ICs. Holm across all 20 tries gives 0.064, so this is a borderline candidate, not a proven improvement. The holdout is short and slightly negative.
- Legacy offers before 2023 use start date + 7d. If the CVM file was compiled later, early rows may be slightly more complete than was knowable (a survivorship-free registry, but the values are not used).
- Smooth, stale marks inflate every t. The exclusion shrinks P4+Q from 103 to 79 names, and the 180/365d versions to 62/37, which concentrates the book.
- Sector mapping and exposure weights are static and judgmental (`issuer_sectors.csv`, `sector_exposures.csv`). World Bank monthly series are latest-vintage.
- Matching is by exact cnpj8 only, so group-level issuance (holding vs SPV) is missed.

## Files
- `download.py`: raw data.
- `signals.py`: PIT signal builders and `signals(panel)`.
- `run.py`: IC, books, placebo, charts, holdout.
- `robustness.py`
- `event_study.py`
- Outputs:
  - `results.json`, `robustness.json`, `event_study.json`;
  - `equity_total_return.png`, `cum_excess.png`, `supply_event_study.png`.

## Rerun
```
PYTHONPATH=. PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/Scripts/python.exe research/nightly/alt_signals/download.py
PYTHONPATH=. PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/Scripts/python.exe research/nightly/alt_signals/run.py --stage ic
PYTHONPATH=. PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/Scripts/python.exe research/nightly/alt_signals/run.py --stage books
PYTHONPATH=. PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/Scripts/python.exe research/nightly/alt_signals/robustness.py
PYTHONPATH=. PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/Scripts/python.exe research/nightly/alt_signals/event_study.py
```
The holdout was computed once, via `run.stage_holdout('P4Q_ex_supply_or_nc_expo')`. Its result is stored in results.json. The `--stage holdout` flag re-runs it; do not use it for selection.
