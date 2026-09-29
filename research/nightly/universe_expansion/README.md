# Universe expansion and new sleeves (CRI/CRA, bank paper, FIDC, NTN-B, eurobonds)

**Bottom line.** Adding sleeves to the debenture P4+Q book does not beat P4+Q net of costs in any statistically defensible way. The best mix, 80% P4+Q with 20% FIDC senior quotas picked by P4Q, adds +0.14%/yr (t 0.70, Holm p = 1.0). That sleeve can't really be bought, and its returns are self-reported. NTN-B sleeves cost 1–6%/yr against CDI over 2022–25. Listed CRI funds (FII de papel) are the one investable, market-priced credit sleeve we found. There the P4 logic backfires: a high trailing distribution yield signals distress, not carry.

Harness v4. Monthly stats run over 2022-01 → 2025-12 (48 months) at 25 bps unless stated. **28 variants** went through `H.compare`, so Holm corrects across all 28.

## Data feasibility (honest)

| sleeve | free point-in-time source found | what it gives | usable for a backtest? |
|---|---|---|---|
| **NTN-B / pre / LFT** | ANBIMA index history (public S3 xls: IMA-B5, IMA-B, IMA-B5+, IRF-M, IMA-S, IDKA-IPCA-2A) and Tesouro Direto prices since 2002 | daily index levels from 2002 | **yes**, as an ETF proxy with a 0.20%/yr fee deducted |
| **CRI/CRA: securitisation reports** | CVM `SECURIT/DOC/INF_MENSAL_CRI|CRA`, 2019–2026: 4,628 certificates since 2022 (CRI ~2.5k/yr, CRA ~0.8k/yr), plus `Data_Entrega` (median filing lag 58 days) | book (curve) value, contract rate text, arrears status, debtor/cedent CNPJ | **no returns**. Values are stale (20% of rows unchanged month on month) and `Rentabilidade` has unusable units. It works as a credit-event registry |
| **CRI/CRA: market prices** | ANBIMA public CRI/CRA files: 404. ANBIMA Feed API: credentials present but not approved for production. B3: no free trade file | – | no |
| **CRI (listed wrapper)** | FII de papel: B3 closes plus distributions (brapi) and NAV per quota from CVM FII informe mensal, available at `Data_Entrega` | 89 tickers, ~33 liquid names per month (ADTV ≥ R$500k), P/NAV coverage 85% | **yes**, but survivorship-biased: only tickers brapi still resolves; 196 CVM credit-FII candidates, several dozen of them not resolvable |
| **CRA (listed wrapper)** | Fiagro on brapi | history starts ~2025-10 (≤ 226 days) | no (too short) |
| **FI-Infra** (incentivised debentures) | brapi | prices OK, distributions badly incomplete (13–17 events vs ~55 expected) | no |
| **FIDC** | CVM `FIDC/DOC/INF_MENSAL`, 2013–2026 (downloaded 2020–26): 20.7k senior classes | self-reported monthly return per class, PL, delinquency, subordination | **paper only**: 84% closed-end, 50% of rows report no return. Open data has no delivery date, so availability is assumed at ref month-end + 45d |
| **Bank paper** (LF/CDB/LCI/LCA) | CVM CDA fund holdings (BLC_5), read-only from the fund_flows cache; BCB SGS CDB averages | position counts and contract spreads (LF DI+ median 0.99%, p90 2.35% in 2025-06), 108 LF issuers, R$621bn LF held by funds | **no market returns** (accrual only; CDA has a ~90-day confidentiality lag and no secondary prices) |
| **Offshore Brazil eurobonds** | FRED ICE BofA EM / LatAm corporate indices | total-return and OAS series, but FRED truncates them to a rolling ~3y window starting **2023-09-29** | no 2022–23, so no comparable backtest. USD TR 2024–25: LatAm corp +8.9%/yr, EM HY corp +10.7%/yr |

## Method

- **Index sleeves.** Daily excess over CDI on the harness grid, following the `H.index_excess` convention, minus a 0.20%/yr ETF fee.
- **FII-CRI sleeve.** A monthly book on the first grid day, with signals known at close d and a buy at close d+1. Weights drift with buy-and-hold inside the month, costs are |Δw| × cost/2, and missing prints earn 0.
  - Point-in-time rules:
    - distributions count from their ex-date (the day after `lastDatePrior`);
    - NAV is used only once CVM's `Data_Entrega` ≤ d (first version filed);
    - split units are fixed for both prices and distributions.
  - Universe: ADTV ≥ R$500k, 12-month distribution yield > 2%, CRI share ≥ 50%.
  - P4 analogue: top 30% by 12-month distribution yield (carry), not rich (excluding the top-20% P/NAV), and **Q** removes the worst-quintile 12-month NAV change (realised credit losses).
- **FIDC sleeve.** Senior classes only, non-exclusive, PL > R$50m.
  - Carry is the mean excess of the last 3 reports known at decision time (ref ≤ m−2). The quality filter drops the worst-quintile delinquency ratio.
  - Months without a report earn CDI (optimistic). A pessimistic `missing0` variant earns 0% instead.
  - Returns are winsorised at +20%; losses are kept down to −100%.
- **CRI/CRA arrears link.** Drops from P4+Q the bonds whose issuer's cnpj8 is the debtor or cedent of a certificate reported "Em atraso" in the previous 365 days (by `Data_Entrega`).
- **Allocations.** Monthly-rebalanced fixed weights (daily approximation), with a 10 bps round trip on sleeve switches. Trend rules use 126-day trailing excess at decision d, applied from d+2. Inverse-vol weights use 126-day vols.
- **De-smoothed Sharpe.** Geltner AR(1) on monthly excess, because debenture marks are smooth and sleeves priced on B3 are not.

## Results, pre-2026 (25 bps; paired against P4+Q tranche-126; Holm across 28 rows)

| variant | exCDI %/yr | vs U | vs P4+Q | t | Holm p | vol | Sharpe | de-smoothed Sharpe | maxDD | corr with P4+Q | vs P4+Q 22–23 / 24–25 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| P4+Q (reference) | +2.22 | +1.12 | 0 | | | 1.04 | 2.13 | 1.55 | −1.39 | 1 | |
| P4 | +1.89 | +0.79 | −0.33 | −2.3 | 0.21 | 1.05 | 1.81 | 1.34 | −1.56 | 0.97 | −0.30 / −0.36 |
| NTN-B IMA-B5 | −2.65 | −3.75 | −4.87 | −4.1 | 0.00 | 2.21 | −1.20 | −1.05 | −11.9 | 0.01 | −3.2 / −6.5 |
| NTN-B IMA-B | −4.28 | | −6.49 | −3.0 | 0.05 | 4.38 | −0.98 | | −16.4 | −0.12 | |
| NTN-B IDKA-2A | −2.74 | | −4.95 | −4.0 | 0.00 | 2.42 | −1.13 | | −12.3 | 0.00 | |
| Pre IRF-M | −1.50 | | −3.72 | −2.1 | 0.22 | 2.97 | −0.51 | | −8.3 | −0.04 | |
| LFT IMA-S | +0.02 | | −2.20 | −3.1 | 0.03 | 0.16 | 0.13 | | −0.2 | 0.12 | |
| IDA-IPCA (unhedged debentures) | −3.39 | | −5.61 | −2.5 | 0.16 | 5.13 | −0.66 | | −15.5 | 0.30 | |
| FII-CRI universe | −5.16 | | −7.38 | −2.1 | 0.22 | 8.15 | −0.63 | | −27.2 | 0.13 | |
| FII-CRI carry (top 30% yield) | −7.95 | | −10.16 | −2.3 | 0.21 | 13.1 | −0.60 | | −35.2 | | |
| FII-CRI P4 | −8.89 | | −11.11 | | | 13.8 | | | −38.0 | | |
| FII-CRI P4Q | −9.21 | | −11.42 | −2.5 | 0.16 | 13.6 | −0.68 | | −37.4 | 0.13 | |
| FII-CRI quality only | −4.36 | | −6.58 | | | 7.8 | | | −25.1 | | |
| FII-CRI cheap (low P/NAV) | −9.68 | | −11.90 | | | 14.5 | | | −41.4 | | |
| FIDC-sr universe | −1.11 | | −3.33 | −4.4 | 0.00 | 0.76 | −1.47 | | −5.9 | 0.09 | |
| FIDC-sr P4 | +2.39 | | +0.17 | 0.2 | 1.0 | 1.21 | 1.97 | 1.58 | −1.6 | −0.14 | |
| **FIDC-sr P4Q** | +2.95 | +1.85 | **+0.73** | 0.7 | 1.0 | 0.95 | 3.10 | 2.16 | −0.94 | −0.11 | +1.05 / +0.41 |
| FIDC-sr P4Q, missing report = 0% | +1.31 | | −0.91 | −0.6 | 1.0 | 1.42 | 0.93 | | −3.9 | | |
| MIX 80 P4+Q / 20 IMA-B5 | +1.25 | +0.14 | −0.97 | −4.1 | 0.00 | 0.95 | 1.31 | 1.18 | −1.2 | 0.88 | −0.65 / −1.30 |
| MIX 80 / 20 IMA-B5, trend-timed | +1.62 | +0.52 | −0.59 | −4.0 | 0.00 | 0.85 | 1.91 | 1.34 | −1.2 | 0.99 | −0.60 / −0.59 |
| MIX 80 / 20 FII-CRI P4Q | −0.15 | | −2.37 | −2.6 | 0.13 | 2.92 | −0.05 | | −5.6 | | |
| MIX 70/15/15 P4+Q / IMA-B5 / FII | −0.29 | | −2.51 | −3.5 | 0.01 | 2.26 | −0.13 | | −4.7 | | |
| MIX 60/15/10/15 incl. FIDC | +0.41 | | −1.81 | −3.3 | 0.02 | 1.62 | 0.25 | | −3.0 | | |
| MIX 80 / 20 best-trend (IMA-B5 or FII) | +0.89 | | −1.33 | −4.7 | 0.00 | 1.05 | 0.84 | | −1.7 | | |
| MIX inverse-vol (P4+Q, IMA-B5, FII) | −0.54 | | −2.76 | −5.0 | 0.00 | 1.36 | −0.40 | | −3.7 | | |
| **MIX 80 P4+Q / 20 FIDC-sr P4Q** (best mix) | **+2.36** | +1.26 (t 2.4) | **+0.14** | 0.70 | 1.0 | 0.83 | 2.83 | 2.17 | −0.86 | 0.97 | +0.21 / +0.08 |
| P4+Q minus CRI/CRA-arrears issuers | +2.22 | | 0.00 | | | | | | | | |

- **50 bps.** The best mix is +0.22 vs P4+Q at 50 bps (t 1.06). The 80/20 IMA-B5 mix is −0.90 (t −3.7).
- **Placebos.**
  - FII-CRI P4Q (−9.2) is **below** the mean of 30 random same-size books (−7.7; p = 0.77). The selection actively hurts.
  - FIDC P4Q (+2.9) beats 30 random books (mean −1.1, p95 −0.3; p < 0.03). The carry persists in the reported returns.
- **Selection inside each sleeve** (see `cum_excess.png`, right panel):
  - FII P4Q vs the FII universe: −4.0%/yr (t −1.9), negative in both halves.
  - FIDC P4Q vs the FIDC universe: +4.1%/yr (t 4.9), positive in both halves.

## Sealed holdout (2026-01 → 2026-09, reported once)

| | exCDI %/yr | vs P4+Q |
|---|---|---|
| P4+Q | +0.08 | |
| MIX 80 P4+Q / 20 FIDC-sr P4Q | +0.50 | +0.42 (t 2.4, 9 months) |
| FIDC-sr P4Q vs FIDC universe | | +3.0 |
| NTN-B IMA-B5 | +0.60 | |
| FII-CRI P4Q vs FII universe | −26.1 exCDI | **−12.2** |

The FII losses come from CACR11 −60%, HCTR11 −36% and DEVA11 −32%: high-yield CRI funds in distress.

## Insights

1. **Carry works only where marks are smooth and self-reported.** "Top carry, not rich, minus worst quality" holds up in debentures and in FIDC senior quotas: +4.1%/yr over the FIDC universe, t 4.9, clean placebo, and it held in the holdout. The same rule, applied to the one CRI vehicle priced daily on an exchange (FII de papel), loses 4%/yr against its own universe, and 12%/yr in 2026. The likely reason is that the stated carry is being priced as default risk there. This is the strongest evidence here that part of the debenture/FIDC P4 edge is a mark-smoothing artefact, not harvestable credit carry.
2. **FII price-to-NAV is not a cheapness signal.** Cheap-to-NAV funds (−9.7%/yr) do worse than the FII universe (−5.2). Discounts to NAV anticipate NAV write-downs; `nav_chg12` then realises them.
3. **NTN-B diversifies but doesn't pay in 2022–25.**
   - Monthly correlation with P4+Q is 0.01 and with IDA-IPCA 0.48. IMA-B5 lost 2.6%/yr against CDI, so every static NTN-B mix lowered return and Sharpe.
   - A 126-day trend filter on the NTN-B satellite recovered +0.38%/yr of the damage (−0.97 → −0.59 vs P4+Q), but the mix still trails P4+Q.
   - The de-smoothed Sharpe of P4+Q (1.55, ρ = 0.31) is still far above any market-priced sleeve, so risk-parity or inverse-vol mixing only dilutes it.
4. **CRI/CRA arrears as a debenture filter is a no-op.** 303 CRI/CRA debtor/cedent issuers were ever in arrears. On average only 0.21 bonds in the debenture universe and 0.02 bonds in P4+Q belong to one of them, so the two credit universes barely overlap. The signal is exposed for completeness.
5. **Structural data gap.** No free source gives market prices for CRI/CRA, LF/CDB or eurobonds before 2023-09. Expanding the universe properly needs a paid ANBIMA Feed (CRI/CRA/LF marks) or ICE/Bloomberg history. The only zero-cost start is to begin archiving ANBIMA's daily public files today; they keep only ~10 business days online.

## Caveats

- **FIDC** returns are self-reported and smooth (ρ 0.2–0.7). Most senior classes are closed-end and not investable; availability is assumed at month-end + 45d because open data has no delivery date. Treating missing reports as CDI is optimistic: with missing = 0%, P4Q is −0.9 vs P4+Q. Winsorising at +20% removes some unit errors but not all.
- **FII-CRI** has survivorship bias (only tickers alive on brapi; delisted funds are missing), which biases the sleeve *upwards*, and the result is still negative. Distributions and split units come from heuristic repair. Returns are pre-tax, although FII distributions are tax-exempt for individuals, so an after-tax comparison would favour FIIs by about 15% of the CDI coupon.
- **Index sleeves** are ETF proxies: the fee is deducted, but tracking error and bid/ask are ignored.
- **Mixes** use fixed daily weights, a small approximation to monthly rebalancing.
- **Multiple testing.** 28 variants went into Holm. None survives against P4+Q with a positive sign.

## Reusable signals

- `data/history/nightly/universe_expansion/signal_crisec_arrears.pkl`: columns `(cnpj8, date = Data_Entrega, value = 1)`. The issuer is the debtor or cedent of a CRI/CRA reported in arrears. Point-in-time by filing date.
- `research/nightly/universe_expansion/run.py`:
  - `fidc_series(rule)`: daily excess of the FIDC-senior sleeve;
  - `fii_targets(rule)` with `sleeve_book(...)`: the FII-CRI sleeve;
  - `index_sleeves()`: NTN-B, pre and LFT index excess.
- Caches in `data/history/nightly/universe_expansion/`:
  - `fii_panel.pkl`: day, ticker, dy12, pnav, nav_chg6/12, mom126, vol63, adtv;
  - `fidc_panel.pkl`, `crisec_panel.pkl`, `index_sleeves.pkl`, `series_pre2026.pkl`.

## Files and rerun

- `probe.py`: source probes.
- `download.py`, `download_listed.py`, `download_listed2.py`: data pulls; brapi uses the token from `.env`, which is never printed.
- `build.py`: point-in-time datasets.
- `run.py`: backtests, pre-2026 only → `results_pre2026.json`.
- `feasibility.py`: coverage → `feasibility.json`.
- `final.py`: charts and the one-time holdout → `results.json`, `equity_total_return.png`, `cum_excess.png`.

```
PYTHONPATH=. PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/Scripts/python.exe research/nightly/universe_expansion/build.py && \
PYTHONPATH=. PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/Scripts/python.exe research/nightly/universe_expansion/run.py && \
PYTHONPATH=. PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe research/nightly/universe_expansion/feasibility.py && \
PYTHONPATH=. PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/Scripts/python.exe research/nightly/universe_expansion/final.py
```

Run `download*.py` first on a fresh machine.
