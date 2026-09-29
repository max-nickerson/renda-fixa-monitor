# Networks and contagion (nightly agent `network_contagion`)

**Short answer: no network filter beats P4+Q.** Contagion from shocked neighbours to their connected bonds is weak or absent once you look inside the P4+Q book. That covers corporate-group mates, sibling bonds of the same issuer, sector mates, issuers with similar exposures, and issuers with common fund holders.

One universe-level effect is robust: **sector 3-month return momentum.**
- Bonds in sectors whose other issuers did badly over the last 3 months keep underperforming over the next 1–3 months.
- The top-minus-bottom rank slope is about −1.6%/yr at 63 days, NW t −4.3.
- Both halves agree: t −3.0 in 2022–23 and −3.5 in 2024–25.
- It survives skipping the first month (t −4.2) and controlling for the bond's own 3-month return.
- It is gone by 126 days.
- It does **not** add value on top of P4+Q. P4+Q's carry and rich screens already pick names where it is weak (FM t −1.4 inside P4+Q), and the filter adds +0.00%/yr in the book.

All numbers are pre-2026 on the harness v4: monthly decisions, 126-bday tranches, 25 bps, 48 months.

## Data built (point-in-time)

Caches live in `data/history/nightly/network_contagion/`.

| dataset | source | coverage | PIT rule |
|---|---|---|---|
| `fre_edges.pkl` | CVM FRE `posicao_acionaria` 2020–2026 | 2,982 controller edges, 686 companies | direct controlling shareholders only (no nested chain rows); edge valid from the FRE receipt date `DT_RECEB`; the latest document ≤ date decides the stake |
| `cda_holdings.pkl` | CVM CDA fund portfolios, BLC_4 "Debêntures", quarter ends 2021-03 → 2026-06 | 1.9k–4.2k funds, 850–2,150 bonds per quarter; 93% of rows match SND codes | a snapshot is used from ref month end **+100 days** (covers the 90-day confidentiality lag) |
| `fund_flows.pkl` | CVM `inf_diario` 2021-01 → 2026-09, funds that held debentures | 6.4M fund-days, 6,887 funds | 21-bday net flow / NAV, **lagged 2 bdays** |
| `features_v1.pkl` | all of the above plus the harness return matrix and panels | 295k (decision date, bond) rows, every W and M panel date | everything known at the decision close |

**Corporate groups** (`net.groups_asof`) are the union of three sources:
- equity_map ticker roots;
- curated brand tokens in issuer names;
- FRE majority control (≥ 50%) where the parent is itself an issuer.

Result: 68–77 multi-issuer groups covering 260–290 issuers. Examples: Energisa 14, Motiva 15, Ecorodovias 14, Equatorial 14, EDP 10. 46% of universe rows belong to a multi-issuer group.

JV links (20–50% stakes) are too sparse to test: they cover only 5% of rows. The static maps (equity_map, brands, sectors, exposures) carry today's structure back in time.

**Node shocks** are measured per issuer-week, and the own issuer is always excluded from its neighbours' stress.

| shock | definition | frequency |
|---|---|---|
| spread blow-out | median traded bond −3% (rate-hedged) over 20 bdays | 2.3% |
| rating downgrade | within 30 days | 0.4% |
| negative press | ≥ 3 items in 7 days | 1.5% |
| CVM distress filing | within 30 days | 1.1% |
| own stock crash | ≤ −20% over 21 days | 1.1% |
| any shock | any of the above | 6.1% |
| hard shock | any except press | 4.7% |

**Neighbour channels:**
- `sib_*`: other bonds of the same issuer (min 4-week return; gap vs own).
- `grp_*`: group mates.
- `sec_*`: sector mates (share shocked, median 4-week and 13-week return).
- `exp_*`: exposure-similar issuers in *other* sectors (cosine ≥ 0.5 on `sector_exposures.csv` / overrides).
- `hold_*`: fire-sale channel. Holder-weighted share of the holders' debenture books sitting in shocked issuers outside the own group.
- `flow21`: holder-weighted 21-day net flows of the funds holding the issuer.

## Results

### 1) Fama-MacBeth

Setup:
- Weekly cross-sections of the universe, 2021-06 → 2025.
- The feature is rank-transformed and signed so that + means more stress.
- Controls: carry, resid_z, duration, own 4-week return, own shock.
- Slope is annualised, top vs bottom rank; NW t.

| feature | fwd_21 slope %/yr (t) | fwd_63 slope %/yr (t) | halves fwd_63 (t) | inside P4+Q fwd_63 (t) |
|---|---|---|---|---|
| **sector 13w return (momentum)** | −2.26 (−4.1) | **−1.60 (−4.3)** | −3.0 / −3.5 | −0.97 (−1.4) |
| sector 4w return | −1.20 (−2.8) | −1.23 (−3.3) | −2.4 / −2.8 | −0.42 (−0.8) |
| common holders exposed to any shock | −1.18 (−2.7) | −1.48 (−2.4) | −1.9 / −1.6 | −0.53 (−0.9) |
| exposure-similar share shocked | −0.65 (−1.3) | −0.72 (−1.4) | −0.7 / −3.3 | −1.48 (−2.8) |
| exposure-similar 4w return | −0.43 (−1.1) | −0.69 (−2.2) | −0.9 / −2.5 | −0.40 (−0.9) |
| group mate hard-shocked | −1.23 (−0.9) | −1.08 (−1.9) | −0.2 / −2.4 | too few |
| group 4w return | −0.29 (−0.6) | −0.34 (−1.2) | | too few |
| sibling bond min 4w return | −0.40 (−1.0) | +0.05 (0.1) | +1.2 / −3.3 | −1.63 (−1.9) |
| sibling gap vs own | −0.68 (−1.5) | −0.16 (−0.4) | +0.8 / −3.2 | −1.74 (−1.8) |
| sector share hard-shocked | +0.65 (1.0) | +0.64 (1.4) | | −1.66 (−2.2) |
| holder loss (spread shocks only) | −0.29 (−0.5) | +0.31 (0.5) | | +0.41 (0.4) |
| holder outflows (−flow21) | +0.53 (0.8) | +1.01 (1.3) | | +0.59 (0.6) |

The diagnostics (`diag.json`) are about sector momentum:
- With the own 13-week return as an extra control: t −4.6.
- On the skip-month target (days 21→63): −1.34%/yr, t −4.2.
- At 126 days: t −1.3.

It is a 1–3 month lead-lag between sector mates, not own-bond momentum. The sibling lead-lag shows up only in 2024–25.

### 2) Event study

The table shows forward excess vs the date mean, for non-shocked bonds.

| event | n | fwd_21 %/yr (t) | fwd_63 %/yr (t) |
|---|---|---|---|
| group mate hard-shocked | 4,041 | +0.13 (0.4) | −0.56 (−0.5) |
| group mate blew out (spread shock) | 2,668 | +0.11 (0.2) | −1.17 (−0.7) |
| sector stress top decile | 16,307 | +0.13 (0.5) | +0.20 (0.6) |
| sibling −3% while own flat | 12,402 | −0.36 (−1.1) | **−0.92 (−2.1)** |
| holder loss top decile | 14,633 | +0.42 (0.7) | −0.13 (−0.2) |
| holder outflow bottom decile | 14,005 | +0.62 (1.2) | +0.38 (0.9) |
| own shock (reference) | 9,378 | −1.06 (−1.6) | −1.82 (−1.6) |

Contagion to group mates is economically small and insignificant. Even the own shock itself is only about −1.8%/yr over 63 days, which is consistent with "reacting to news doesn't pay".

### 3) Portfolios: filters on P4+Q

Setup: monthly 126-day tranches, 25 bps, compared with P4+Q (+2.22 exCDI). **12 variants were tried** (10 in `run.py` plus 2 at hold 63), and Holm is applied across them.

| variant | names dropped | exCDI | vs U | vs P4+Q | t | p Holm | 22–23 / 24–25 | 50 bps vs P4+Q | rec40 vs P4+Q |
|---|---|---|---|---|---|---|---|---|---|
| avoid_exposure_stress_q5 (**best**) | 21.9 | 2.29 | +1.18 | **+0.07** | 0.92 | 1.00 | +0.00 / +0.13 | +0.04 | +0.06 |
| avoid_sector_stress_q5 | 26.8 | 2.28 | +1.18 | +0.07 | 0.54 | 1.00 | −0.01 / +0.14 | +0.03 | +0.07 |
| sector_rev_drop_top_q | 19.9 | 2.27 | +1.17 | +0.05 | 0.47 | 1.00 | +0.23 / −0.13 | +0.03 | +0.05 |
| avoid_group_retshock | 1.0 | 2.23 | +1.13 | +0.01 | 2.23 | 0.26 | +0.01 / +0.02 | +0.01 | +0.01 |
| avoid_group_shock | 2.0 | 2.23 | +1.13 | +0.01 | 1.02 | 1.00 | | +0.01 | +0.01 |
| avoid_sibling_drop | 3.4 | 2.23 | +1.13 | +0.01 | 0.37 | 1.00 | | +0.01 | +0.01 |
| sector_mom_drop_bottom_q | 17.6 | 2.22 | +1.12 | +0.00 | 0.02 | 1.00 | −0.19 / +0.19 | −0.03 | −0.01 |
| avoid_any_neighbour | 22.1 | 2.19 | +1.09 | −0.03 | −0.28 | 1.00 | +0.17 / −0.22 | −0.07 | −0.02 |
| avoid_holder_loss_d10 | 17.1 | 2.18 | +1.08 | −0.04 | −0.41 | 1.00 | +0.17 / −0.24 | −0.07 | −0.03 |
| avoid_holder_outflow_d10 | 13.7 | 2.14 | +1.04 | −0.08 | −1.54 | 1.00 | | −0.10 | −0.07 |
| hold 63: sector_mom_drop_bottom_q vs P4+Q h63 | | 2.31 | | +0.04 | 0.41 | | | | |
| hold 63: avoid_exposure_stress_q5 vs P4+Q h63 | | 2.38 | | +0.12 | 1.34 | | | | |
| P4 (ref.) | | 1.89 | +0.79 | −0.33 | −2.28 | | | | |

**Drop-placebo for the best variant.** From P4+Q, drop the same number of names at random, 20 draws:
- random drops average +2.19, with p95 +2.28;
- the best variant scores +2.29, so 5% of draws are at or above it;
- the best variant beats random removal by about +0.10%/yr, which is borderline and **not** significant after Holm.

**Sealed holdout (2026-01 → 2026-09, run once, after all choices were frozen):**

| | exCDI | vs P4+Q | vs U |
|---|---|---|---|
| avoid_exposure_stress_q5 | +0.02%/yr | −0.06 (t −0.8) | +3.28 |
| avoid_any_neighbour | +0.13%/yr | +0.05 (t 0.2) | +3.39 |
| P4+Q | +0.08%/yr | | |
| U | −3.26%/yr | | |

Charts:
- `equity_total_return.png`: total return CDI × (1 + excess) vs CDI, U, P4, P4+Q, IDA-DI and Ibov, plus cumulative excess vs U;
- `cum_excess.png`: all variants vs U and vs P4+Q.

## Insights

1. **The P4+Q screen already does the contagion avoidance.** Its no-negative-press rule, rich exit and quality screen leave few stressed-network names: group-shock filters remove only 1–2 of about 100 names per date. What remains shows no measurable spillover.
2. **Contagion inside groups is not there in the data.** Group mates of a hard-shocked issuer earn −0.6%/yr over 63 days (t −0.5). Corporate families are mostly regulated SPVs (Energisa, Equatorial, Motiva, Ecorodovias, EDP), ring-fenced by concession. The market prices them as separate credits.
3. **Sector momentum is real but lives in the wrong part of the universe.** It is a robust 1–3 month lead-lag (t −4.3; t −4.2 skipping month 1; both halves), consistent with stale marks and slow repricing across sector mates. It is concentrated outside the high-carry names, and it fades before the 126-day holding period.
4. **The fire-sale / common-holder channel is weak.** The fund-flow channel is the wrong sign: bonds whose holders see outflows do slightly *better*, maybe because they get sold cheap and then recover. The holder-loss share tied to spread shocks shows nothing. Only the broad "holders exposed to any shock" (press included) is negative in the universe (t −2.4), and it disappears inside P4+Q.
5. **Siblings of the same issuer show a lead-lag in 2024–25 only.** When a sibling bond falls 3% while the own bond is flat, the own bond does −0.9%/yr over 63 days (t −2.1). The halves disagree (2022–23 +, 2024–25 −3.2 t), and the book effect is +0.01.
6. **Exposure-similar stress inside P4+Q is the best lead** (FM t −2.8 inside P4+Q), but as a filter it is only +0.07%/yr (t 0.9). It fails the sealed holdout (−0.06).

## Reusable signals

- `research/nightly/network_contagion/features.py::signals(panel)` merges all network features on (day, codigo) for any harness panel. It is PIT and cached in `data/history/nightly/network_contagion/features_v1.pkl`, keyed by (codigo, day, dpos, cnpj8). Key columns: `i_sec_r13`, `i_sec_r4m`, `i_exp_any`, `i_exp_r4`, `i_hold_any`, `i_hold_ret`, `i_flow21`, `i_grp_hard`, `i_grp_r4`, `sib_min_r4`, `sib_gap`, the own-issuer shock flags `i_ret/down/press/cvm/eq/any/hard`, and `i_n_holders`.
- `net.py`:
  - `groups_asof(day)`: PIT corporate groups;
  - `affiliates_asof(day)`: JV links;
  - `exposure_sim()`: issuer exposure cosine similarity;
  - `cda_holdings()`: fund × bond holdings with an `avail` date.
- `fund_flows.pkl`: daily flows of the funds that hold debentures, useful to other agents.

## Caveats

- Group, sector and exposure maps are static (today's structure). Only the FRE controller edges and the CDA holdings are dated.
- CDA holdings of 2021–22 come from the annual history files. The 100-day lag is conservative, but confidential positions may be missing before release.
- Shock thresholds (−3% over 4 weeks, ≥ 3 press items, −20% stock) were fixed a priori and not tuned. Quantile cuts (q5 / d10) were fixed a priori too.
- Marks are smooth and stale, so t-stats in the FM are inflated for everything. The sector-momentum lead-lag is partly a stale-mark phenomenon: it is not tradeable as a short, and debentures cannot be shorted.
- The illiquid-subsample split in `diag.py` came out empty: universe bonds all have > 2 trades per 30 days by construction.
- 12 variants were tried. None survives Holm. The best variant is selected on in-sample performance, which biases its +0.07 upward.

## Rerun

```
PYTHONPATH=. PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/Scripts/python.exe research/nightly/network_contagion/download.py
PYTHONPATH=. PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/Scripts/python.exe research/nightly/network_contagion/download2.py
PYTHONPATH=. PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/Scripts/python.exe research/nightly/network_contagion/build_flows.py
PYTHONPATH=. PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/Scripts/python.exe research/nightly/network_contagion/features.py --force   # ~15 min
PYTHONPATH=. PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/Scripts/python.exe research/nightly/network_contagion/run.py              # ~5 min -> results.json, charts
PYTHONPATH=. PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/Scripts/python.exe research/nightly/network_contagion/diag.py             # diag.json
PYTHONPATH=. PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/Scripts/python.exe research/nightly/network_contagion/extra_h63.py        # extra_h63.json
PYTHONPATH=. PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/Scripts/python.exe research/nightly/network_contagion/holdout.py          # holdout.json (sealed, once)
```

The downloads and `build_flows.py` are one-offs. Everything is cached afterwards.
