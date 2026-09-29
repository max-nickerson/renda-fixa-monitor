# w2_unlisted_issuer_health: extending P7's equity-health screen to unlisted issuers (harness v4)

**Bottom line.** Extending the screen to unlisted issuers does not add to P7 in a way we can detect.
- **Coverage gap:** 64% of P7's weight (54% of P4+Q rows) sits in issuers with no equity series, so the screen never sees them.
- **Parent mapping:** mapping those issuers to listed parents through CVM FRE control data recovers only 2.5% of rows. The unlisted tail is really unlisted: Aegea, Cagece, BRK, Iguá, Metrô SP, Solví, V2i, Transbrasiliana and similar.
- **Proxies:** proxy screens cover the rest (98%) and beat P4+Q about as well as P7 does. Against P7 itself they add only +0.02 to +0.13%/yr (t ≤ 0.9, Holm p = 1).
- **Unlisted outperformance:** it is real before costs, but about half of it is a stale-mark, survivorship and cost artefact.

All numbers are pre-2026: monthly decisions, 126-bday tranches, 25 bps, 48 months. The holdout was read once, at the end.

## 1. Coverage (pre-2026, universe rows)
| | universe rows | P4+Q rows | P7 weight |
|---|---|---|---|
| own equity (harness: direct 41.7% + parent 24.7% of U) | 66.3% | 43.1% | 32.8% |
| + FRE parent (new, PIT) | 2.5% | 2.5% | 3.6% |
| + name-token parent (new, static, hand-verified) | 0.0% (all already covered or priceless) | 0.0% | 0.0% |
| still no equity | 31.2% | 54.4% | **63.6%** |
| of which has a sector / fundamentals / bond-mark proxy | 30.6% | 52.1% | 61.2% |

There are 539 unlisted issuers in U and 343 in P4+Q. The largest by P4+Q rows are Aegea, RDA, VIX Logística, Cagece, Solví Essencis, Giga+ Fibra, V2i, Skinstore, Transbrasiliana, BRK Ambiental and Metrô SP.

## 2. PIT parent / controller map (`build_map.py`)
- **Control edges** come from CVM FRE files for 2021–2026:
  - `participacao_sociedade`: the filer's subsidiaries, where the filer holds ≥ 50%;
  - `posicao_acionaria`: the filer's controlling legal-person shareholders, with `Acionista_Controlador = S` and ≥ 50%, walked through `Acionista_Relacionado`.
- **Availability and chain:** each edge is available from `DT_RECEB`, the receipt date of that FRE version. The latest version per filer ≤ the decision day is used. The chain is followed breadth-first up to 4 levels, until it reaches a CNPJ that owns one of the 161 harness equity series (equity_map direct rows plus FCA `valor_mobiliario` tickers).
- **Result:** 100 issuers mapped, e.g. Alupar, EDP, Equatorial and ISA transmission SPEs, Auren/AES holdings, Motiva concessions, Serena, Rede D'Or, Oncoclínicas, TAG→Engie, MRS→CSN (dropped, 18.75%).
- **Manual check:** all 110 candidate pairs were reviewed (`fre_map_sample.csv`). About 95% are plausible. Minority members of a control group (Itaúsa 8.5% of NTS, Eletrobras 15% of Norte Energia, CSN 18.75% of MRS) were removed by the ≥ 50% rule. Ligga→Copel (stale 2022 FRE) remains as a known error.
- **Name-token fallback:** 22 candidates, of which the manual check accepted 10 (Neoenergia, Ecorodovias, Serena, Raízen, AXIA, Localiza, Unidas, Corpóreos). Rejected: LINHA→Vamos, BRAZIL→Cyrela, IMOBILIARIOS→Multiplan, TENDA ATACADO→Construtora Tenda and others. Precision 45%, so token matching is unusable without a human.
- **Parents that are listed but outside the 161 series:** only 27 issuers, about 4% of unlisted rows (Iguá, Invepar, Oi→V.tal, AES Operações, Rede Energia, Brisanet, GPS). Not worth downloading.

## 3. Proxies for issuers with no parent (`features.py`)
| proxy | definition | Spearman vs true EQH on listed rows | precision of bottom-20% flag vs true EQH bottom-20% (random 0.20) | IC fwd_126 on unlisted rows |
|---|---|---|---|---|
| px_sector | sector medians of eq_dd252, −eq_vol63, eq_r126 (≥ 3 listed issuers) | **0.66** | **0.63** | +0.011 (t 0.3) |
| px_fund | harness f_quality (CVM strict dates) | 0.13 | 0.31 | +0.040 (t 2.1) |
| px_bond | issuer mean −ds_63, rmom_126, ratio_from_hi_252 | −0.02 | 0.26 | +0.024 (t 0.5) |
| cascade | mean of the available proxies | 0.46 | 0.51 | +0.023 (t 0.7) |

- The true EQH composite itself has **no rank IC** (+0.012, t 0.3 on listed rows). P7's screen works through the tail (blow-up avoidance), not through a monotone signal.

## 4. P7-extended vs P7 and P4+Q (6 variants incl. P7; Holm across the rows)
| variant | exCDI | vs P4+Q (t) | Holm p | vs P7 (t) | 22–23 / 24–25 vs P4+Q | 50 bps vs P4+Q | rec40 | liq costs vs P7 | harsh+liq vs P7 |
|---|---|---|---|---|---|---|---|---|---|
| P7 (reproduced exactly, max abs 0.0) | 2.72 | +0.50 (3.29) | 0.002 | | +0.37 / +0.63 | +0.48 | +0.46 | | |
| X1 + FRE/name parents | 2.79 | +0.57 (3.58) | 0.0015 | +0.07 (0.72) | +0.38 / +0.76 | +0.54 | +0.59 | +0.06 | +0.06 |
| X3 + sector proxy (**selected by the pre-set rule: max t vs P4+Q**) | 2.74 | +0.52 (4.11) | <0.001 | +0.02 (0.13) | +0.45 / +0.59 | +0.47 | +0.53 | −0.02 | −0.02 |
| X4 + fundamentals proxy | = X1 (the Q filter already removes every flagged name) | | | | | | | | |
| X5 + bond-mark proxy | 2.79 | +0.57 (3.45) | 0.002 | +0.07 (0.45) | +0.45 / +0.68 | +0.53 | +0.60 | +0.08 | +0.10 |
| X6 cascade | 2.84 | +0.63 (3.87) | <0.001 | +0.13 (0.88) | +0.43 / +0.82 | +0.59 | +0.66 | +0.10 | +0.12 |

- **Placebo** (drop the same number of unscreened P4+Q names at random, 20 draws):
  - X6 vs P7: +0.125 against a placebo mean of −0.00 (p95 +0.06), p = 0.00;
  - X3: +0.02 against −0.04 (p95 +0.04).
  - So the cascade flags are better than random, but the gain is small and not significant against P7 itself.
- **X3 vs P7 by year:** 2022 +0.22, 2023 −0.07, 2024 −0.32, 2025 +0.25. Excluding the 3 best months gives −0.14.
- **Charts:** `equity_total_return.png` and `cum_vs_P7.png`.

## 5. Is the unlisted / orphan outperformance a stale-mark artefact? (bias_audit scenarios, same on both sides)
| pair, %/yr (t) | base | harsh survivorship | liquidity-bucket costs | harsh + liq |
|---|---|---|---|---|
| U unlisted − U listed | **+0.89 (1.97)**; 22–23 +0.10, 24–25 +1.69 | +0.69 (1.66) | +0.66 (1.43) | **+0.46 (1.08)** |
| P4+Q unlisted − listed | +0.95 (1.18) | +0.86 | +0.54 | +0.45 (0.52); 22–23 −0.51 |
| U orphan − held (fund_flows nh = 0) | +0.47 (1.12) | +0.59 | **−0.40** | −0.27 |
| P4+Q orphan − held | +1.16 (1.48) | +1.57 | +1.08 | +1.50 (1.78) |

- **About half the unlisted premium is an artefact.** Unlisted rows are 2.8× more likely to belong to bonds that later go silent below 0.98 (1.86% vs 0.67%). They also trade less (median 41 vs 53 trades in 30 days).
- **Carry explains part of the rest.** Unlisted carry is 141 vs 104 bps. Within carry deciles the premium is +0.78%/yr (t 1.8).
- **Where it lives.** All of it comes from 2024–25. In 2022–23 it is zero or negative once costs are realistic.
- **Orphans.** Universe orphans lose their edge under realistic costs. Inside P4+Q, orphans stay ahead, but this is about 9% of the book, a few names, and t < 2.
- The P7 placebo's "+0.16 tilt toward unlisted" is therefore mostly carry plus the 2024–25 regime, with a survivorship-flattered level.

## Sealed holdout 2026-01..09 (read once)
| book | exCDI | vs P4+Q (t) | vs P7 (t) |
|---|---|---|---|
| P7 | +0.76 | +0.68 (2.25) | |
| X3 sector proxy (selected) | +0.92 | +0.84 (2.67) | +0.16 (1.24) |
| X6 cascade | +0.87 | +0.79 (4.08) | +0.11 (0.27) |
| P4+Q | +0.08 | | |

Inside P4+Q, unlisted minus listed was +1.99%/yr (t 2.2) in the holdout. Listed names were hit harder in 2026.

## Caveats
- **Static inputs.**
  - The sector map (`issuer_sectors.csv`) and the name map are static.
  - FRE describes the group as of its reference date, although it is dated by receipt.
  - `equity_map`'s parent rows (harness) are static hand-built mappings.
- **Weak fundamentals proxy.** The fundamentals proxy covers only 42% of unscreened P4+Q rows, and the Q filter already absorbs its tail.
- **Selection rule.** The pre-set rule (max t vs P4+Q) picked X3, whose gain vs P7 is about zero. X6 has the largest gain vs P7 but was not selected. Neither gain is significant.
- **Harness caveats.** Smooth, stale marks inflate t. The orphan flag inherits fund_flows' CDA-confidentiality lag leak.

## Files and rerun
- **Code:**
  - `build_map.py`: the PIT FRE map and the name map;
  - `features.py`: `attach(P)` adds `xq_*`, `eq_tier`, `px_sector`, `px_fund` and `px_bond`;
  - `run.py`: stage 1 by default; `--holdout` runs stage 2.
- **Results:** `results.json`, `compare_pre2026.csv`, and the manual-check tables `fre_map_sample.csv` and `name_map_static.csv`.
- **Caches** (in `data/history/nightly/unlisted_issuer_health/`):
  - `raw/` (FRE zips, cad_cia_aberta);
  - `fre_edges.pkl`, `parent_map_fre_pit.pkl`, `name_map_static.pkl`;
  - `panel_M_ext_all.pkl` (the monthly panel with the extended columns, including 2026).

To rerun:

```
PYTHONPATH=. PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/Scripts/python.exe research/nightly/w2_unlisted_issuer_health/build_map.py
PYTHONPATH=. PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/Scripts/python.exe research/nightly/w2_unlisted_issuer_health/run.py
```

`run.py` takes about 5 minutes. `build_map.py` takes about 3 minutes, plus the download.
