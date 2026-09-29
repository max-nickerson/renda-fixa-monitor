# final: frozen model and charts for the PM report

**Decision.** The final model is **P7, unchanged**. The rule, fixed before running anything here, was: add a wave-2 idea only if it adds to P7 itself with a pre-2026 NW t of at least 2 *and* it survived verification. Nothing met both conditions:

| wave-2 candidate | pre-2026 effect | t | why it was not adopted |
|---|---|---|---|
| unlisted-issuer health, X6 cascade | +0.13 vs P7 | 0.88 | not significant |
| unlisted-issuer health, X1 FRE parents | +0.07 vs P7 | 0.72 | not significant |
| flag vote FV2 (p7_adversarial) | +0.10 vs P7 | 1.29 | not significant |
| expected-loss (PD) screen | −0.12 / −0.15 vs P7 | | makes P7 worse |
| primary-market sleeve A2 | +1.31 vs P4+Q | | depends on how never-traded bonds are marked (−0.29 if marked at cash); the verification cohort replica gave +0.51 (t 0.98) |

- **Run:** `PYTHONPATH=. PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/Scripts/python.exe research/nightly/final/run.py` (about 40 s).
- **Outputs:**
  - `results.json`: tables for pre-2026 and the 2026 holdout (25 / 50 bps and honest), plus the per-year returns.
  - `per_year_total_return.csv`
  - charts `a_total_return.png`, `b_cum_excess.png`, `c_drawdowns.png`, `d_what_predicts.png` and `e_per_year_table.png`.
- **Honest scenario:** liquidity-bucket costs plus harsh survivorship, taken from bias_audit. Here it uses every stop known by 2026-09. That is harsher before 2026 than `combined`'s version, which only used stops known by 2026-01.
- **The report** is in `../REPORT.md`.
