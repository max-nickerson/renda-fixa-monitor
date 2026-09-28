# Point-in-time issuer fundamentals (`data/history/fundamentals_pit.pkl`)

Built by `research/data_fundamentals.py`. Rebuild:

```
PYTHONPATH=. PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe research/data_fundamentals.py [--refresh] [--fundamentus]
```

(`--refresh` re-downloads current-year CVM zips if older than 1 day; `--fundamentus` prints the spot check below.)
The file is a pickle because pyarrow is not installed; the script writes `.parquet` automatically if an engine is available.

One row per `(cnpj8, available_date)`. Join to a daily panel with
`pd.merge_asof(daily.sort_values("date"), fund.sort_values("available_date"), left_on="date", right_on="available_date", by="cnpj8")`.
Amounts are in **R$ millions**.

## Sources (priority order)

| source | issuers | rule |
|---|---|---|
| CVM ITR/DFP open data (`dados.cvm.gov.br/.../DOC/{ITR,DFP}/DADOS`, 2019-2026 zips cached in `data/history/cvm_fin/`) | 391 | Consolidated (`con`) when the filing has a consolidated balance sheet, else individual (`ind`). `source` = `cvm_{itr,dfp}_{con,ind}`. |
| brapi `balanceSheetHistoryQuarterly / incomeStatementHistoryQuarterly / financialDataHistoryQuarterly` (raw JSON in `data/history/brapi_fund/`) | 44 gap-fill + pre-2019/pre-registration history for direct tickers | Used only for issuers with no CVM filings (mostly `parent` mappings, `is_parent=True`: numbers are the listed parent's) and, for `direct` tickers, for periods before the issuer's first CVM filing. 9 tickers 404 on brapi (delisted: ARTR3, CIEL3, CRFB3, ENBR3, KRSA3, NEOE3, SRNA3, STBP3, ZAMP3). |
| Fundamentus | - | Spot cross-check only (below). |

## Point-in-time rules

* **CVM**: `available_date` = `DT_RECEB` of the **first** version of the filing (from the `itr_/dfp_cia_aberta_YYYY.csv` index, which lists every version). The statement files only contain the **latest** version's values, so a restated filing carries its restated numbers from the original receipt date (small look-ahead). `available_date_strict` = receipt date of the latest version (fully PIT-safe); it is later for 9.4% of rows (median +15 days). `cvm_versao` = latest version number.
* LTM rows also require the prior-year DFP, so `available_date = max(filing receipt, prior DFP receipt)`.
* **brapi**: no filing date -> `available_date = period_end + 45d` (Q1-Q3) or `+ 90d` (Q4).
* Late/back-filed periods: after sorting by `available_date`, a row whose `period_end` is not newer than a period already public for that issuer is dropped (so the as-of row never goes backwards in time); if several periods arrive the same day the latest is kept.
* Observed lag `available_date - period_end`: min 7d, median 45d, max 454d (never negative).

## CVM account map (standard non-financial template)

| field | account(s) |
|---|---|
| total_assets | `1` Ativo Total |
| cash | `1.01.01` Caixa e Equivalentes + `1.01.02` Aplicações Financeiras (current) |
| st_debt / gross_debt | `2.01.04` (current) / `2.01.04 + 2.02.01` Empréstimos e Financiamentos (includes debentures and financing leases booked there; IFRS-16 lease liabilities in 2.0x.05 "Outras obrigações" are excluded). Accepted only if the description contains "emprest/financ" (so bank/insurer templates give NaN). |
| equity | `2.03` Patrimônio Líquido (consolidated incl. minorities) |
| revenue | `3.01` |
| ebit | `3.05` Resultado antes do resultado financeiro e dos tributos |
| fin_exp | `3.06.0x` with "despesa" in the description, sign flipped (positive = expense). If both 3.06.01 and 3.06.02 are zero and only the net `3.06` is reported (e.g. Sabesp), the **net** financial result is used and `fin_exp_is_net=True` (3% of CVM rows). |
| net_income | `3.11` (fallback `3.09`) |
| D&A | Sum of DFC (indirect) `6.01.01.xx` lines matching deprec/amortiz/exaust, excluding transaction-cost/interest/debenture amortisation lines. |

**LTM**: flows are taken year-to-date (earliest `DT_INI_EXERC`); `LTM = YTD + prior fiscal-year annual (DFP) - prior-year YTD (PENÚLTIMO column of the same filing)`; DFP rows use the annual figure. This equals the sum of the last 4 quarters and handles non-calendar fiscal years.
`ebitda_ltm = ebit_ltm + da_ltm`; if D&A is unavailable: brapi `financialData.ebitda` for brapi rows, else EBIT with `ebitda_is_ebit=True` (7.7% of CVM rows, 2.5% brapi).

**brapi**: debt = `loansAndFinancing + longTermLoansAndFinancing` (fallback `totalDebt`), cash = `cash + shortTermInvestments`, LTM = sum of 4 contiguous standalone quarters, EBITDA = `financialData.ebitda`.

**Ratios** (raw, not winsorised; ±inf -> NaN): `net_debt_ebitda`, `gross_debt_equity`, `interest_coverage = ebitda_ltm / fin_exp_ltm`, `cash_to_st_debt`, `equity_ratio`, `ebitda_margin`, `revenue_growth_yoy` (LTM vs LTM of the period one year earlier), `d_net_debt_ebitda_4q`, `d_interest_coverage_4q` (vs period one year earlier, same source). Negative EBITDA gives negative leverage (1% of rows below -35).

## Coverage (lab_daily, `eligible==True`: 943 issuers, 1,005,134 bond-days)

* 11,100 rows, 435 issuers (391 CVM + 44 brapi-only).
* **85.1% of eligible bond-days** have a fundamentals row at that date (CVM 81.2%, brapi 3.9%); 84.2% have `net_debt_ebitda`. By year 2021-2026: 84.2 / 84.3 / 84.3 / 85.6 / 86.2 / 85.0%.
* Median staleness (bond-day minus period_end) 91d, p90 132d.
* Not covered: 508 issuers (~15% of bond-days) — unlisted SPVs/subsidiaries not registered at CVM with no listed parent mapping (`mapping_type=none`) or delisted parents.

## Median ratios by period-end year

| year | ND/EBITDA | gross debt/equity | int. cov. | cash/ST debt | equity ratio | rev growth | EBITDA margin |
|---|---|---|---|---|---|---|---|
| 2019 | 2.55 | 0.99 | 2.46 | 1.40 | 0.32 | 0.11 | 0.26 |
| 2020 | 2.25 | 1.11 | 2.36 | 1.68 | 0.31 | 0.07 | 0.26 |
| 2021 | 1.92 | 1.01 | 2.97 | 1.90 | 0.32 | 0.17 | 0.28 |
| 2022 | 2.17 | 1.12 | 1.93 | 1.74 | 0.31 | 0.22 | 0.27 |
| 2023 | 2.38 | 1.19 | 1.79 | 1.43 | 0.31 | 0.10 | 0.28 |
| 2024 | 2.40 | 1.22 | 1.96 | 1.74 | 0.30 | 0.09 | 0.30 |
| 2025 | 2.50 | 1.23 | 1.84 | 1.82 | 0.30 | 0.09 | 0.30 |
| 2026 | 2.60 | 1.28 | 1.77 | 2.00 | 0.30 | 0.06 | 0.30 |

## Spot checks (latest row, 2026-06-30 ITR, R$ mn)

| issuer | available | gross debt | net debt | EBITDA LTM | ND/EBITDA | int. cov. |
|---|---|---|---|---|---|---|
| Localiza 16670085 | 2026-08-06 | 44,636 | 34,487 | 15,005 | 2.30 | 2.29 |
| Energisa 00864214 | 2026-08-06 | 42,684 | 33,229 | 8,941 | 3.72 | 1.22 |
| Sabesp 43776517 | 2026-08-12 | 51,658 | 34,217 | 15,478 | 2.21 | 6.9 (net fin. result, flagged) |
| Eneva 04423567 | 2026-08-12 | 22,363 | 19,826 | 6,240 | 3.18 | 1.98 |
| ISA Energia 02998611 | 2026-08-03 | 17,726 | 15,968 | 5,041 | 3.17 | 2.32 |

Localiza's CVM EBIT+D&A LTM (15,005) equals brapi's `financialData.ebitda` exactly.

## Cross-checks

* **brapi vs CVM** (direct tickers, same period_end, 2021+): share of rows within 2% — total assets 94.9%, equity 95.0%, gross debt 95.0%, LTM revenue 91.8%; median abs. error 0.
* **Fundamentus** (latest balance 30/06/2026, RENT3, SBSP3, ENEV3, ENGI11, ISAE4): gross debt, net debt and LTM revenue match ours **exactly** for all 5. Fundamentus "EBIT" differs by 3-14% (it is gross profit less selling/G&A, excluding other operating items/equity income; ours is CVM line 3.05).

## Known problems

* Latest-version values are used from the first receipt date (use `available_date_strict` for zero look-ahead).
* Banks/insurers/leasing companies use a different CVM template: debt/EBIT are NaN for them.
* D&A tagging is by description regex; ~7.7% of CVM rows have no D&A found (EBITDA = EBIT, flagged).
* `fin_exp_is_net` rows understate gross interest expense (coverage biased up).
* brapi parent numbers (`is_parent=True`) describe the group, not the issuing subsidiary; brapi dates are assumed (deadline-based), not observed.
