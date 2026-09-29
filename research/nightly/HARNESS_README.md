# Nightly common evaluation harness (`research/nightly/harness.py`)

Every nightly agent should load its data, run its backtests and compute its statistics through this module. That way results are comparable: they share the same targets, execution, costs, statistics, baselines and sealed holdout.

**Self-test:** `python research/nightly/harness.py`.

**Validation** (reproduces the published labs and writes `research/nightly/harness/validation.json` and `baselines.png`):

```
PYTHONPATH=. PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=2 .venv/Scripts/python.exe research/nightly/harness/validate.py
```

## 30-second usage

```python
from research.nightly import harness as H

P  = H.load_panel("M")            # monthly decision cross-sections, pre-2026 only (holdout sealed)
PW = H.load_panel("W")            # weekly version

# 1) a selection rule = function of one decision cross-section x (universe rows of that date)
my = lambda x: x["p4q"].to_numpy() & (x["eq_r63"].fillna(0) > -0.10).to_numpy()   # bool mask
r  = H.backtest(my, freq="M", hold=126)            # 126-bday overlapping monthly tranches, 25 bps, cap 10%

# a score: the top `top_frac` of the universe is taken
r2 = H.backtest(lambda x: x["cdi_bps"] - 50 * x["worstQ"], top_frac=0.3)

# weights (as_weights=True), or a DataFrame [day, codigo, weight|score|select] produced by your model
# sig = pd.DataFrame({"day": ..., "codigo": ..., "score": ...});  H.backtest(sig, freq="W", hold=126)

# 2) stats vs CDI and paired stats vs a benchmark (monthly, Newey-West)
b = H.baseline("P4Q")                                  # also "U", "P4", "P4_live", "CDI"
print(H.stats(r["daily"], bench=b["daily"]))           # ann excess, vol, sharpe, maxDD, t_nw, diff vs bench...

# 3) one table for all your variants: exCDI, excess vs U, paired vs P4+Q (NW t, p, Holm across rows), halves
tab = H.compare({"mine": r, "score": r2, "P4": H.baseline("P4")}, bench="P4Q")

# 4) costs / recovery / horizons / placebo
H.backtest(my, cost_bps=50);  H.backtest(my, scenario="rec40");  H.backtest(my, hold=63)
pl = H.placebo(my, n=20)      # the same number of names drawn at random from the universe each date

# 5) curves: total return CDI x (1+excess), plus cumulative excess vs universe, with CDI/U/P4/P4Q/IDA-DI/Ibov refs
H.plot_curves({"mine": r}, "research/nightly/<slug>/equity.png", title="...")
tr = H.total_return_curve(r["daily"])                  # base-100 total-return index

# 6) the sealed holdout: ONCE, at the very end, after every choice is frozen
rh = H.backtest(my, holdout=True);  bh = H.baseline("P4Q", holdout=True)
print(H.stats(rh["daily"], bench=bh["daily"], holdout="only"))    # months >= 2026-01 only
```

Walk-forward model training on the panel. A label is known at grid position `lab_end_H`. Train only on rows with `lab_end_H <= dpos` of the date you predict:

```python
for p in sorted(P["dpos"].unique()):
    tr = P[(P["lab_end_126"] <= p) & P["fwd_126"].notna() & P["univ"]]
    te = P[(P["dpos"] == p) & P["univ"]]
    ...fit on tr, score te...
```

## API

| function | what it does |
|---|---|
| `load_panel(freq="M"\|"W", holdout=False, universe_only=False)` | Point-in-time panel with one row per (decision day, bond on the daily grid), at the **close** of the first grid day of each month or week. Decisions start 2021-03 so models have training data; backtests start 2022-01. `holdout=False` drops decisions ≥ 2026-01-01 **and** sets to NaN any label whose window reaches 2026. |
| `backtest(signal, freq="M", rebalance=None, hold=126, top_frac=0.2, as_weights=False, issuer_cap=0.10, cost_bps=25, overlay=None, exit_signal=None, scenario="base", holdout=False, start="2022-01-01", panel=None, min_names=5, universe="univ", name=None)` | Returns a dict: `daily` (net excess over CDI per grid day), `gross`, `cost`, `exposure`, `turnover_ann` (one-way, × book per year), `cost_ann_%`, `holdings` (day, codigo, weight, entry_day), `n_avg`, `n_decisions`, `runtime_s`, `params`. |
| `baseline(name, **backtest_kw)` | `U`, `P4`, `P4Q`, `P4_live` or `CDI`, run on the same engine. Results are memoised per kwargs. |
| `stats(series, bench=None, lag=6, holdout=False)` | Statistics on **calendar-month** excess: `ann_excess_%`, `vol_%`, `sharpe`, `max_dd_%`, `worst_month_%`, `hit`, `skew`, `t_nw`, `h1_2022_23_%`, `h2_2024_25_%`, `cum_%`. With `bench` it adds `diff_ann_%`, `diff_t_nw`, `diff_p`, `diff_h1_%`, `diff_h2_%`, `diff_hit` and `diff_te_%`. `holdout`: `False` truncates at 2026, `True` uses everything, `'only'` uses ≥ 2026. |
| `compare(results, bench="P4Q", universe="U", lag=6, cost_bps=25)` | DataFrame of exCDI, excess vs U (with t), paired vs bench (t, p, **Holm across the rows given**), vol, Sharpe, maxDD, halves, turnover. **Put every variant you tried in it** so Holm counts them all. |
| `holm(pvals)`, `nw_t(x, lags)`, `monthly(daily)`, `halves(daily)` | Statistical helpers. |
| `ic(panel, feature, target="fwd_126")` | Per-date Spearman IC on the universe, `dok` dates only. NW lag = horizon in months. Returns `{mean, t_nw, n_dates, series}`. |
| `cohort_excess(signal, H=126, freq="M", ...)` | Selection-lab style: per-date H-bday tranche return minus the universe tranche. It needs no daily book, so it is fast. |
| `placebo(signal, n=20, **backtest_kw)` | Null distribution: the same number of names and the same weights, drawn randomly from each date's universe. |
| `total_return_curve(excess)`, `index_excess("IDADI"\|"IBOV"\|"IDAGERAL"\|"IDAIPCA")`, `index_levels()`, `ida_regime()`, `cdi_daily()` | Curves and reference series on the grid. |
| `plot_curves({label: result_or_series}, path, title, refs=("CDI","U","P4","P4Q","IDADI","IBOV"))` | Two-panel PNG: total return (log scale) and cumulative excess vs the universe. |
| `exec_pos(p, max_wait=20)`, `cap_weights(issuers, cap, w0)`, `days()` | Low-level building blocks. |
| constants | `HOLDOUT=2026-01-01`, `START=2022-01-01`, `SPLIT=2024-01-01`, `HZ=(21,63,126,252)`, `ENTRY_MAX=20`, `ISSUER_CAP=0.10`. |

### Signal conventions

`signal` is one of three things:
- a callable `f(x)` over the universe rows of one date, returning one of:
  - a `bool` mask: the selected names get equal weight, subject to the issuer cap;
  - a float score: the top `top_frac` is taken and NaN is never selected;
  - weights, when `as_weights=True`;
- a panel column name, e.g. `"p4q"` or `"cdi_bps"`;
- a DataFrame with columns `day, codigo` and one of `weight` / `score` / `select`.

With fewer than 5 names on a date, that tranche is left in cash.

`universe="univ"` is the default: `eligible & cdi_bps.notna()`. Eligible means a fresh mark (≤ 7 days old), a mark between 0.9 and 1.1 of par, and duration ≥ 0.5. Pass `universe="all"` to see every grid row.

### Engines

**`hold=int` (default 126): overlapping tranches, the HY-lab book.**
- Each decision opens a tranche. Each bond is bought at its **entry**: the first trade dated after the decision day, 1 to 20 bdays later. With no trade in that window, that weight stays in cash (0 excess).
- The tranche is held `hold` bdays with buy-and-hold drift.
- The daily book is the sum of live tranche P&L divided by min(M, #tranches), where M = hold / decision spacing.
- Cost is book-level: |Δ averaged tranche weights| × cost/2 at each decision. `cost_bps` is the round trip.
- `exit_signal` (e.g. `"rich"`) sells a held bond at its next fresh trade after it is flagged on a later panel date. The slot then goes to cash, and the sale costs cost/2.

**`hold=None`: rebalancing book, "live" style.**
- Orders execute at each bond's next fresh trade:
  - buys wait up to 20 bdays, then expire;
  - sells wait up to 20 bdays, otherwise they execute at the last mark on day +20.
- Gross exposure is capped at 1: buys only use available cash.
- NAV compounds.
- Without `exit_signal`, the book is replaced by the target at each rebalance. Names are kept if they stay in the target, and are re-sized only outside a ±25% band. Bonds that leave the universe (for example, not fresh) are sold.
- With `exit_signal`, the book is **sticky**, as in the original daily P4. Names are held until flagged or until they go off the grid. New target names are added, and the whole book is re-targeted to capped equal weight.

**`overlay`:** `"ida"` is P4's regime overlay: invested only while IDA-DI's 21d excess momentum over CDI is > 0. A `pd.Series` of 0..1 on grid days also works. The value at close t applies to the t→t+1 return. Each switch costs |Δ| × exposure × cost/2. Execution is at the mark and instantaneous, which is optimistic for debentures.

**`scenario="rec40"`:** the 55 bonds that stopped trading more than 30 days before the data end, before maturity and with a last mark < 0.90 jump to 40% of par at their last mark.

### Targets and returns

- **Daily returns** are the PATCHED rate-hedged excess returns over CDI from the selection lab (`data/history/sellab_returns.pkl`). They include:
  - moves greater than 20%;
  - multi-day gaps with carry;
  - with isolated spike prints removed.
- **After a bond's last grid row** the return is 0, i.e. cash at CDI at the last mark (see survivorship below).
- **`fwd_H` / `fwd_H_rec40`** are compounded from the entry trade over H ∈ {21, 63, 126, 252} bdays. They are NaN when the bond was never executed; portfolio functions treat that as cash, i.e. 0.
- **`dok_H`** is true when every bond's label for that date is complete.
- **`lab_end_H`** is the grid position at which the label becomes known.

### Panel columns (M: 48k rows × 113; W: 211k rows × 113, all dates incl. 2026)

- **Bond / lab_daily:**
  - identifiers and terms: `codigo, day, dpos, b, cnpj8, kind, contract, incent`;
  - marks and spread: `ratio, cdi_bps, cdi_pct` (0 = highest carry), `dur, bench_rate, age, fresh`;
  - relative value: `peer, resid_bps, resid_z`;
  - flags: `eligible, univ`;
  - news and events: `press_neg_7d/30d, distress_2d, fact_2d, rat_days_since_down`;
  - equity and market: `eq_ret_1w/4w, mkt_mom_21`.
- **Weekly-lab extras**, as of the last Monday ≤ day, built from info before that Monday:
  - trading and spreads: `trades_30d, own_z, d_spread_1w/4w, d_ratio_4w, carry_per_dur`;
  - CVM filing counts: `n_fact_30d/90d, n_distress_30d/90d, n_deb_mtg_*, n_rating_*, n_oficio_*`;
  - distress and news: `days_since_distress, news_any_30d, distress_90d, mkt_distress_z`;
  - maturity: `maturity, years_to_mat`.
- **Fundamentals** (strict CVM availability dates, rows stale after 460d dropped, same construction as the selection lab): `f_lev, f_cov, f_cash_st, f_eq_ratio, f_gde, f_margin, f_rev_g, f_d_lev, f_d_cov, f_size, f_is_parent, f_age_days, f_source, covered, f_quality` (composite, ranked within the universe), `worstQ`.
- **Equity**, from the issuer's own ticker or its listed parent, chosen point-in-time as in the HY lab: `eq_ticker, eq_map_type, eq_confidence, eq_r5/21/63/126/252, eq_vol63, eq_dd252` (vs the 252d high), `eq_adtv` (R$), `listed`.
- **Market:** `idadi_x21/63, ibov_x21/63` (excess over CDI), `ida_regime, cdi_ann, univ_cdi_med, univ_cdi_iqr`.
- **Sector:** `sector` from `research/data/issuer_sectors.csv`, `unknown` if missing.
- **Rule flags:** `p4f, p4, p4q, rich`.
- **Targets:** `entry_pos, executed, fwd_{21,63,126,252}[_rec40], lab_end_H, dok_H`.
- **`dist_stop_LOOKAHEAD`:** a look-ahead flag for diagnostics only. **Never use it as a feature.**

## Baselines, pre-2026 (sealed-holdout mode)

> **Superseded by the Audit section at the end (harness `_VERSION = 4`).** The tables below are the pre-audit (v3) numbers, kept for reference.

Monthly decisions 2022-01 → 2025-12, 126-bday tranches, 25 bps, issuer cap 10%. Statistics are on months 2022-01 → 2025-12 (48 months), with NW lag 6.

| baseline | exCDI %/yr | vs U | t | vs P4+Q | t | vol | Sharpe | maxDD | vs P4+Q 22–23 / 24–25 | turnover/yr | avg names |
|---|---|---|---|---|---|---|---|---|---|---|---|
| U (universe) | +1.15 | 0 | | −1.17 | −2.5 | 1.53 | 0.75 | −3.59 | −1.84 / −0.51 | 1.4 | 699 |
| P4 | +1.98 | +0.84 | 1.6 | −0.34 | −2.3 | 1.03 | 1.93 | −1.52 | −0.33 / −0.34 | 2.3 | 130 |
| **P4+Q** | **+2.32** | **+1.17** | 2.5 | 0 | | 1.00 | 2.32 | −1.32 | | 2.5 | 103 |
| P4_live (weekly sticky, sell when rich, IDA overlay) | +1.41 | +0.27 | 0.3 | −0.91 | −1.8 | 0.99 | 1.43 | −1.07 | +0.07 / −1.89 | 2.7 | 128 |

**Sensitivities.** The P4+Q minus P4 gap is positive in every run: +0.23 to +0.40.

| run | U | P4 | P4+Q | P4+Q − P4 |
|---|---|---|---|---|
| 50 bps | +0.96 | +1.69 | +2.00 | +0.32 |
| rec40 | +1.12 | +1.89 | +2.29 | +0.40 |
| hold 63 | +1.31 | +2.08 | +2.34 | +0.26 |
| hold 252 | +1.19 | +2.25 | +2.49 | +0.23 |
| weekly tranches (126) | +1.12 | +2.16 | +2.40 | +0.24 |

**Cohort level (selection-lab style, pre-2026).**

| horizon | P4 − U | P4+Q − U | P4+Q vs P4, t |
|---|---|---|---|
| 3m | +1.01 | +1.30 | 2.6 |
| 6m | +1.01 | +1.40 | 2.6 |
| 12m | +1.02 | +1.31 | 3.8 |

**Placebo.** P4+Q scores +2.32. Random books of the same size score +0.91 (p95 +1.03).

**Engine variants of P4**, vs P4 tranche +1.98:

| variant | exCDI | vs P4 tranche | cost %/yr | turnover/yr |
|---|---|---|---|---|
| monthly replace-the-book | +1.71 | −0.27 | 1.10 | 8.8 |
| monthly sticky, sell when rich | +2.00 | +0.01 | | |
| weekly sticky, sell when rich | +1.66 | −0.32 (t −1.8) | | |
| tranche + IDA overlay | +1.78 | −0.20 | 1.00 | |
| tranche + weekly rich exit | +2.03 | +0.05 | | |

- Replace-the-book loses because rank churn at the 30% cut is expensive: 8.8×/yr turnover.
- The IDA overlay loses because its switching costs are 1.0%/yr of the book at 25 bps.

### Reproduction of the published labs (includes 2026, which those labs already published)

With `holdout=True` the harness reproduces the HY-lab book table exactly:

| | U | P4 | P4+Q | P4+Q − P4 | P4 vol / maxDD |
|---|---|---|---|---|---|
| harness | +0.452 | +1.551 | +1.963 | +0.412 (t 2.94) | 1.09 / −1.52 |
| published HY lab | +0.45 | +1.53 | +1.94 | +0.41 (t 2.9) | 1.09 / −1.52 |

The selection-lab cohort P4 − U is +1.227 / +1.099 / +1.066 at 3 / 6 / 12m, against the published +1.23 / +1.10 / +1.07. The small +0.02 gap on the HY book comes from month-boundary handling of the first partial month.

**Why the pre-2026 numbers differ from the published ones.** The published numbers run to 2026-09. Here the window stops at 2025-12, and labels reaching into 2026 are masked. Ex-CDI levels are therefore higher for every book, and the P4 − U spread is smaller: +0.84 vs +1.08.

**Why `P4_live` (+1.41) is far below the old `lab_daily` "P4 semanal" (+2.91 exCDI).** Three reasons:
1. **Returns:** patched returns replace the unpatched `ret`, which dropped moves over 20% and gaps.
2. **Weighting:** buy-and-hold drift replaces the fixed-weight daily mean. The HY lab measured that bias at +1.0–1.5%/yr.
3. **Costs:** the regime overlay's switching costs (about 1%/yr) are now charged. The overlay also hurt in 2024–25.

**Do not use P4_live as the main benchmark.** Use P4+Q (tranche 126) as the bar to beat and report P4 as well.

## Runtime and memory

Measured with ~10 other agents running, on 2 threads.

**Runtime, first build (cached afterwards):**

| step | time |
|---|---|
| core arrays | ~1 min |
| monthly panel | ~19 s |
| weekly panel | ~20 s |

**Runtime, warm:**

| call | time |
|---|---|
| `load_panel` | 0.1–0.4 s |
| `baseline` / `backtest`, monthly tranches | 1–3 s (the universe takes ~3 s) |
| `backtest`, weekly tranches | 6–15 s |
| `backtest`, weekly sticky | ~5 s |
| `cohort_excess` | ~1 s |
| `placebo(n=20)` | ~25 s |
| `plot_curves` | ~6 s (it computes its own baselines) |

**Memory.** Peak working set is ~1.1 GB while building the weekly panel, since it loads the 0.6 GB `lab_daily`. The full validation run peaks at ~1.4 GB. In memory:
- monthly panel: 43 MB;
- weekly panel: 178 MB;
- core: return matrices 1479 days × 2987 bonds (float32 cached, float64 in use), about 35 MB each.

**Caches** live in `data/history/nightly/harness/`: `core.pkl`, `panel_M.pkl`, `panel_W.pkl`, `indices.pkl`.
- To rebuild, bump `_VERSION` in `harness.py`. **Do not delete these files while other agents are running.**
- IDA-DI comes from `rfmonitor.history.ida`.
- Ibovespa is read from `research/out/hy_indices.pkl` (read-only), with brapi as the fallback.

## Caveats (read before believing a t-stat)

1. **Smooth, stale marks.** Debenture marks are smooth and stale, so vol, Sharpe and t are inflated for every book. Monthly aggregation helps but does not fix this. Prefer paired differences vs P4+Q and the halves over levels.
2. **Survivorship.**
   - After a bond's last print it earns 0 (cash). 1,403 bonds stop trading before maturity, and 55 of them do so below 0.90.
   - Always report `scenario="rec40"` too.
   - Rarely-marked, uncovered, unlisted issuers look best, and that is exactly where this bias lives.
3. **Execution.**
   - Entry is at the mark of the first trade after the decision. That is the published SND price with no market impact, beyond `cost_bps` (a round trip of 25 bps, with 50 as the check).
   - Exits that find no trade within 20 bdays execute at the stale last mark, which is optimistic.
   - The IDA overlay switches instantly at the mark.
4. **Holdout.**
   - With `holdout=False` nothing dated ≥ 2026-01-01 is visible: no decisions, no labels reaching 2026, and daily series truncated at 2025-12-31.
   - Use `holdout=True` only once, for the final report, with `stats(..., holdout="only")`.
   - Features such as `f_*` and `eq_*` are point-in-time, but lab_daily's `resid_z`, `cdi_pct` and similar are same-day cross-sections: known at the close, never used for same-day execution.
5. **Panel quirks.**
   - Fundamentals come from CVM (latest restatement at `available_date_strict`) and, for parents, from brapi. The brapi parent rows describe the group, not the SPV.
   - Equity data is ~68% listed coverage, and 22% of that is parent-only.
   - `trades_30d` and other weekly extras come from Monday snapshots, so on a weekly panel they are up to 4 days older than the decision close.
6. **Multiple testing.** Pass every variant you tried to `compare()` (or `holm()`) and state how many you tried.
7. **Engines differ.** Tranche books (hold = 126) and live books (hold = None) are not directly comparable. Compare a strategy with the baseline run on the **same engine and kwargs**, e.g. `baseline("P4Q", hold=None, freq="W")`.

## Audit (adversarial review, harness v4)

Reviewer: audit agent. Files live in `research/nightly/harness_audit/`:
- the pre-audit copy, `harness_original_snapshot.py`;
- the runner, `audit_run.py`;
- results in `audit_after.json`, `audit_oldR.json`, `validation_before.json` and `validate_*.log`.

`harness.py` is now `_VERSION = 4`. Cache files are versioned (`core_v4.pkl`, `panel_{M,W}_v4.pkl`) and written atomically, so agents still running v3 code keep reading the old `core.pkl` / `panel_*.pkl`. **Do not delete the v3 files while agents are running.** Results computed before the audit (v3) are not comparable with v4. Rerun them before comparing.

### Checked and found clean
- **Point-in-time features:**
  - **lab_daily grid membership:** the bond must have traded before, have a mark ≤ 14 days old, and not have matured. It never depends on later survival.
  - **`resid_bps` / `resid_z`:** built from a per-day, per-peer curve over that day's fresh marks. The curve is never pooled or fitted on future data.
  - **`cdi_pct`:** ranked among that day's eligible bonds.
  - **CVM events:** dated by `Data_Entrega`, counted in (d-w, d].
  - **Press:** dated by publication date, counted in [d-w, d].
  - **Ratings:** dated by action date.
  - **Equity returns:** closes <= d.
  - **`mkt_mom_21`, `idadi_x*`, `ibov_x*`:** index closes <= d.
  - **lab_weekly extras:** the Monday row <= d, built from marks and filings before that Monday.
  - **Fundamentals:** CVM `available_date_strict` (receipt date of the latest version, so restatements cannot leak forward). brapi uses deadline dates, +45d or +90d. Rows more than 460d stale are dropped.
  - **Equity ticker choice:** needs a price at d and at least 150 prices in the prior 252 days.
- **Execution:** a bond is entered at the first grid row whose trade date is after the decision day, within 20 bdays. `R[t]` is the move of the mark from t to t+1 (checked in `run_selection_lab.build_returns`). A tranche entered at k therefore earns `R[k:]`, which is correct.
- **Universe survivorship:** the SND reference table keeps excluded (dead or matured) bonds: 5,213 "Excluido" and 4,796 "Registrado". Only 1 traded code, with 3 trades, is missing from it. The bond universe is not filtered by survivorship.
- **Holdout masking:** with `holdout=False`:
  - decisions from 2026 on are dropped;
  - labels with `lab_end >= 2026-01-01` become NaN;
  - books are truncated at the grid position of 2026-01-01;
  - orders and exits scheduled after that point never execute.
- **Statistics:**
  - `monthly()` compounds (1+CDI+x)/(1+CDI) within each month.
  - `ann_excess_%` is 12 × the arithmetic mean of monthly excess.
  - Newey-West uses Bartlett weights, and the Holm step-down is correct.
  - Paired differences are taken on monthly excess.
- **Placebo:** random books of the same size, drawn from the same day's universe.

### Real bugs fixed in `harness.py`
1. **Gap returns were booked before they were observable (look-ahead, and a holdout leak).**
   - **The bug:**
     - A bond leaves the grid after more than 14 days without a trade.
     - The patched `r_patch` of the last row before the gap carries the whole move to the next trade: 18,321 segments, gaps of up to 1,223 bdays, moves from -73% to +100%.
     - That move was booked on the day before the gap. Books and labels "knew" the move in advance, and tranches that ended inside the gap still earned it.
     - 368 of these segments straddle 2026-01-01, with moves from -53% to +28%. They leaked holdout information into 2025 returns and labels.
   - **Fix:**
     - The move is booked at `pos(next row) - 1`, so it is realised at the next row's mark.
     - `fwd_H` labels are now mark-to-market at `entry + H`. A move not realised by then is excluded, so `lab_end_H` is honest.
2. **Exits could escape a gap at a stale mark.** This became reachable once fix 1 was in place.
   - **New `sell_pos(q)`:** sells at the first fresh trade within 20 bdays. Otherwise:
     - if the bond still has a mark <= 14 days old on day +20, it sells at that mark;
     - if the bond is in a gap, it sells when the bond next trades;
     - if the bond never trades again, it sells at the mark on day +20 (its return is 0 afterwards anyway).
   - The rebalancing engine and tranche early exits use it.
   - A tranche whose natural end falls inside a gap is closed when the bond next trades.
3. **The IDA regime overlay traded at the same close it observed.**
   - **The bug:** the value at close t was applied to `R[t]` (t to t+1), but IDA-DI is published after the close.
   - **Fix:** the overlay is lagged 1 day, so the value known at close t applies to the t+1 to t+2 return. This also applies to user Series overlays.
4. **Tranche-book costs were under-charged.**
   - **The bug:**
     - The initial purchase was free, because `prev=None`.
     - During the ramp-up, the cost book was divided by M while the P&L was divided by min(M, #tranches).
     - Early-exit costs were also divided by M rather than the P&L divisor.
   - **Fix:** costs now use the same divisor as the P&L. Decision costs are charged at p+1, when trades execute and the series starts, so the first decision's cost is no longer sliced off.
5. **Cache races between concurrent agents.**
   - **Fix:** cache files are versioned and written as a temporary file followed by a rename.

### Before / after (pre-2026: monthly decisions, 126-bday tranches, 25 bps, 48 months)
| | v3 (before) | v4, engine fixes only (old R) | **v4, all fixes** |
|---|---|---|---|
| U exCDI | +1.15 | +1.11 | **+1.10** |
| P4 exCDI | +1.98 | +1.92 | **+1.89** (vs U +0.79, t 1.6) |
| P4+Q exCDI | +2.32 | +2.26 | **+2.22** (vs U +1.12, t 2.4) |
| P4 - P4+Q | -0.34 (t -2.3) | | **-0.33 (t -2.3)**, halves -0.30 / -0.36 |
| P4_live | +1.41 | +1.39 | **+1.32** |
| P4 tranche + IDA overlay | +1.78 | +1.67 | **+1.58** (-0.31 vs P4) |
| P4+Q turnover/yr | 2.47 | | 2.94 (cost 0.37%/yr) |

**v4 sensitivities:**

| run | U | P4 | P4+Q | P4+Q - P4 |
|---|---|---|---|---|
| 50 bps | +0.89 | +1.54 | +1.84 | +0.30 |
| rec40 | +1.07 | +1.80 | +2.19 | +0.39 |
| hold 63 | +1.28 | +2.01 | +2.26 | +0.25 |
| hold 252 | +1.14 | +2.15 | +2.37 | +0.22 |
| weekly tranches | +1.07 | +2.05 | +2.28 | +0.23 |

**v4 engine variants of P4, vs the P4 tranche:**

| variant | vs P4 tranche |
|---|---|
| replace-the-book | -0.26 |
| monthly sticky, sell when rich | +0.13 |
| weekly sticky | -0.28 |
| tranche + weekly rich exit | +0.06 |

**v4 placebo:** P4+Q scores +2.22. Random books of the same size score +0.82 (p95 +0.95).

**v4 cohort level (selection-lab style).** This moved the most, because the old labels contained post-horizon gap moves.

| horizon | P4 - U (v3) | P4 - U (v4) | P4+Q - U (v4) | P4+Q vs P4, t (v4) |
|---|---|---|---|---|
| 3m | +1.01 | +0.79 | +0.99 | 1.7 |
| 6m | +1.01 | +0.84 | +1.19 | 2.4 |
| 12m | +1.02 | +0.88 | +1.15 | 3.8 |

**Reproduction including 2026** (comparison with the published labs only; no choices are made from it). The v4 harness deliberately no longer matches the published labs exactly, because they share bug 1.

| | published HY lab | v4 |
|---|---|---|
| U | +0.45 | +0.41 |
| P4 | +1.53 | +1.48 |
| P4+Q | +1.94 | +1.88 |
| P4+Q - P4 | +0.41 | +0.40 (t 3.0) |

**Conclusion unchanged:**
- P4+Q beats P4 by about +0.3%/yr in every run.
- P4 beats U by about +0.8%/yr.
- All levels are about 0.05-0.10%/yr lower than in v3.
- The cohort-level P4 - U is about 0.15-0.2%/yr smaller.

### Known limitations (not fixable in the harness; interpret accordingly)
- **Static reference data:**
  - Contract rate, maturity, kind and incentive flag come from today's SND table. Repactuated or extended bonds carry their current terms back in time.
  - `duration_factors` comes from the latest ANBIMA file.
  - `equity_map.csv` and `issuer_sectors.csv` are static, hand-built mappings.
- **Spike cleaning uses one future trade.** `ratio_c` replaces a print that moves more than 10% when the NEXT trade reverts it. This can clean the entry price too. The effect is small and mostly conservative.
- **Execution at marks:**
  - Tranche ends, exits and entries use SND prints or marks <= 14 days old, with no market impact.
  - The IDA overlay is still executed at the mark for the whole book, which is optimistic.
  - `fwd_H` labels are mark-to-market at `entry + H`, while the tranche book holds through a gap. Labels and books can differ for gap bonds.
- **Rebalancing engine band test:** it sizes decisions at close t on weights that already include `R[t]` (the t to t+1 marks). This only affects the ±25% band test. The effect is negligible, so it was left as is.
- **`ic()` drops never-executed bonds** (NaN label), so the IC is measured on the tradable subset.
- **`resid_bps` IC of +0.21 (t 11.5):** it is point-in-time, since entry is a later trade than the signal's mark. It is mostly smooth-mark mean reversion, which a book only partly captures: the P4 rich filter earns about 0.1%/yr in books. Do not read the t-stat at face value.
