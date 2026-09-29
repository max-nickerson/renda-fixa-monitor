# Event-driven opportunities in debentures (nightly, slug `event_driven`, harness v4)

## Rerun
```
PYTHONPATH=. PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/Scripts/python.exe research/nightly/event_driven/events.py   # event sets (≈10 s)
PYTHONPATH=. PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/Scripts/python.exe research/nightly/event_driven/final.py    # everything else (≈5 min)
```
Delete `data/history/nightly/event_driven/event_signals_*.pkl` if you rebuild the events. The log is in `data/history/nightly/event_driven/final.log`.

## Files
| file | what it does |
|---|---|
| `events.py` | Builds the point-in-time event sets. Outputs `bonds_ref.pkl` (bond terms from the SND registry) and `events_issuer.pkl` (cnpj8, date, etype, text) |
| `study.py` | Event studies on the harness weekly panel (fwd_H labels from harness v4) |
| `run.py` | Overlay and sleeve rules. **`signals(panel)`** is the reusable signal, cached as `data/history/nightly/event_driven/event_signals_{M,W}.pkl` |
| `final.py` | Full pipeline: event studies → 12 variants → Holm → 50 bps → robustness → placebo → charts → sealed holdout (run once) |
| `results.json` | Every number below |
| `event_studies.png`, `equity_total_return.png`, `cum_excess.png` | Charts |

## Data built (point-in-time)
- **CVM IPE filings 2021–2026** (already cached zips). Events are dated by `Data_Entrega`. An event is used only from the first decision day **strictly after** that date. Events are classified by regex on Categoria/Tipo/Assunto:
  - `agd`: any debenture-holder meeting (AGDEB), 2,028 filings, 283 issuers;
  - `agd_waiver`: the waiver/consent subset, 577;
  - `resgate`: early-redemption or redemption-offer notices, 474;
  - `deb_buyback`: issuer buys back its own debentures, 90;
  - `equity_raise`, 2,109;
  - `ma`: M&A, reorganisation, control change or OPA, 2,609;
  - `rj`: recuperação judicial or extrajudicial, 1,250;
  - `ipe_new_deb`: start of a debenture offer, 697.
- **Rating actions** from `rating_events.pkl`: 145 upgrades and 185 downgrades.
- **SND registry** (`snd_caracteristicas.tsv`), per bond: distribution start, whether it is callable (`Resgate Antecipado`), 476 vs 400 offering, incentivised flag, and size.
  - `supply_new_issue`: the issuer starts distributing a new bond. There are 3,229 issuer-days; the event applies to the issuer's *existing* bonds.
  - New issue: the bond's first appearance on the grid within 4 months of its distribution start.
- The registry does not give usable early-exit dates. "Data de Vencimento" is overwritten with the exit date, and none of the 130 registry-flagged "RESGATE TOTAL ANTECIPADO" bonds since 2021 ever traded on SND. Redemptions are therefore studied only through the IPE announcements.

## Event studies (weekly decisions 2021-03 → 2025-12, universe bonds; AR_c = carry, duration, kind and incentive adjusted; NW t by date)
| event (issuer's bonds, first decision after the filing) | n events | AR_c 21d | AR_c 63d (t) | AR_c 126d (t) | pre-63d AR |
|---|---|---|---|---|---|
| rating downgrade | 60 | −0.94 | −0.95 (−1.8) | **−1.54 (−3.2)** | −0.31 |
| M&A / reorganisation / control change | 358 | −0.13 | −0.16 (−1.3) | **−0.64 (−2.8)** | −0.13 |
| debenture-holder meeting (AGD) | 209 | −0.35 | −0.65 (−1.6) | −0.95 (−1.5) | −0.32 |
| AGD with waiver/consent language | 85 | | −0.16 (−0.6) | +0.05 (0.1) | −0.57 |
| early-redemption notice | 194 | −0.27 | −0.33 (−1.6) | −0.35 (−1.4) | −0.05 |
| debenture buyback by issuer | 37 | | −1.15 (−1.5) | −2.20 (−2.3) | +0.06 |
| equity raise | 274 | 0.00 | +0.18 (1.5) | −0.01 (−0.1) | +0.42 |
| rating upgrade | 75 | +0.15 | +0.22 (1.9) | **+0.56 (3.0)** | +0.14 |
| issuer supply (new bond of the same issuer) | 798 | −0.05 | +0.03 (0.5) | −0.06 (−0.7) | +0.03 |
| new issue, first weeks on the grid | 1,486 | +0.14 | **+0.24 (2.9)** | +0.07 (0.4) | |
| new issue, incentivised (IPCA, Lei 12.431) | 556 | +0.24 | +0.35 (2.2) | **+0.44 (3.7)** | |
| new issue, not incentivised | 940 | +0.06 | +0.10 (1.4) | −0.03 (−0.1) | |
| callable & ratio > 1.01 (negative convexity) | 5,894 obs | | −0.04 (−1.0) | +0.07 (1.6) | |
| redemption notice & bond below par (< 0.995) | 68 | | −0.96 (−1.0) | −1.15 (−1.3) | |

**Seasoning curve.** Carry-adjusted AR_c over 126 days, by months since issue:

| months since issue | 0–3m | 3–6m | 6–12m | 12–24m | 24–48m | 48m+ |
|---|---|---|---|---|---|---|
| AR_c 126d | −0.14 | −0.15 | −0.05 | −0.09 | −0.12 | **+0.45 (t 6.0)** |

The 48m+ bonds are low-carry bonds close to maturity (60 bps). This is most likely pull-to-par or a misspecified carry control, not an event effect.

## Tradeable rules (monthly decisions, 126d tranches, 25 bps, issuer cap 10%, 2022-01 → 2025-12)
12 variants were tried. All of them are in the Holm adjustment.

| variant | exCDI | vs U | vs P4+Q (t) | Holm p | 22–23 / 24–25 vs P4+Q | vs P4+Q at 50 bps | names |
|---|---|---|---|---|---|---|---|
| **P4Q_exNegEvents**: P4+Q minus issuers with M&A 180d, downgrade 180d, AGD 180d, redemption notice 126d or RJ 365d | **2.52** | +1.42 | **+0.30 (1.95)** | 0.51 | +0.13 / +0.47 | +0.27 (1.79) | 75 |
| P4Q_exMA180 | 2.39 | +1.29 | +0.17 (1.51) | 0.92 | +0.15 / +0.20 | +0.16 | 88 |
| P4Q_exAGD180 | 2.32 | +1.22 | +0.10 (1.06) | 1 | −0.08 / +0.28 | +0.09 | 94 |
| P4Q_exResgate126 | 2.28 | +1.18 | +0.06 (2.11) | 0.38 | +0.07 / +0.05 | +0.06 | 96 |
| P4Q_exRatingDown180 | 2.24 | +1.14 | +0.03 (0.71) | 1 | | +0.02 | 100 |
| P4Q_combo (exNeg + new issues + upgrades) | 2.34 | +1.24 | +0.12 (0.62) | 1 | −0.08 / +0.32 | +0.08 | 91 |
| P4Q_plusRatingUp | 2.20 | +1.10 | −0.02 | 1 | | −0.03 | 111 |
| P4Q_plusNewIncent | 2.18 | +1.08 | −0.04 | 1 | | −0.05 | 112 |
| P4Q_plusNewIssue | 2.10 | +1.00 | −0.12 (−1.6) | 0.83 | | −0.14 | 119 |
| Sleeve_NewIncent (standalone) | 1.11 | +0.01 | −1.10 | 1 | | | 10 |
| Sleeve_NewIssue (standalone) | 0.89 | −0.21 | −1.33 (−2.4) | 0.22 | | | 20 |
| Sleeve_NewIssue_W63 (weekly, 63d) | 0.95 | −0.16 | −1.27 | 0.64 | | | 21 |
| *P4+Q* | 2.22 | +1.12 | 0 | | | | 103 |
| *P4* | 1.89 | +0.79 | −0.33 (−2.3) | | | | 130 |
| *U* | 1.10 | 0 | −1.12 | | | | 699 |

**Robustness of P4Q_exNegEvents.** Each run is compared with P4+Q on the same kwargs.

| run | vs P4+Q | t | 22–23 / 24–25 |
|---|---|---|---|
| 50 bps | +0.27 | 1.8 | +0.10 / +0.45 |
| rec40 | +0.32 | 2.0 | +0.13 / +0.51 |
| hold 63 | +0.39 | 2.0 | +0.15 / +0.63 |
| hold 252 | +0.28 | 1.7 | +0.19 / +0.38 |
| issuer cap 5% | +0.28 | 1.8 | +0.13 / +0.44 |
| weekly tranches | +0.32 | 1.9 | +0.18 / +0.46 |

**Placebo.** Each date, the rule drops the **same number of P4+Q issuers, chosen at random**. Over 40 draws that gives −0.02%/yr on average (p95 +0.10). The real rule gives +0.30, and 0 of 40 draws beat it. The gain therefore comes from *which* issuers are dropped, not from concentrating the book.

**Sealed holdout (2026-01 → 2026-09, 9 months, run once after freezing):**

| rule | vs P4+Q | t |
|---|---|---|
| P4Q_exNegEvents | **+0.42%/yr** | 2.2 |
| P4Q_combo | −0.77 | |
| Sleeve_NewIssue | −4.16 vs P4+Q (−0.82 vs U) | |

In the holdout, P4+Q earns +0.08 exCDI.

## Insights
1. **Credit-event drift exists and is slow.** After a downgrade the issuer's bonds lose another −1.5% over 6 months (t −3.2), and after an M&A or reorganisation filing another −0.64% (t −2.8). Most of that is *after* the first 21 days, so a monthly rule still captures it. This does not contradict "reacting fast to news doesn't pay": the edge is in *avoiding* names for months, not in trading quickly.
2. **Abstaining beats adding.** Every exclusion rule improves P4+Q. Every addition (new issues, upgrades) hurts once it is inside the 126d tranche engine. The combined exclusion adds +0.30%/yr (placebo p < 1/40) and held up in the 2026 holdout (+0.42). It still does **not** pass Holm at 5% (p 0.51 across 12 variants), so it is a candidate, not a proven edge.
3. **The new-issue premium is real but tiny and short-lived.** It is about +0.24% carry-adjusted over the first 3 months (t 2.9) and gone by 6 months. That is roughly one round-trip cost. Standalone new-issue sleeves lose to the universe, and in 2026 they lost −4%/yr. Only incentivised IPCA new issues keep a +0.44% 6-month AR, and they are too few (about 10 names) and too volatile for a sleeve.
4. **Buying call or redemption candidates does not pay on SND marks.** After a redemption notice the issuer's bonds underperform (−0.35% over 126d). Below-par bonds of issuers announcing redemptions do not pull to par in the data (−1.15%, n = 68). Callable bonds above par show no negative-convexity penalty (+0.07, t 1.6).
5. **New supply from the same issuer does not cheapen its existing bonds** (≈0 at every horizon, n = 798). Equity raises are neutral for bondholders.

## Caveats
- The event classification is a regex on filing titles. M&A and equity-raise sets are noisy: they include routine reorganisations, and IPE covers only CVM-registered companies.
- There are few rating events (60 downgrades in the universe), gathered from news and Fitch's Wayback pages, so coverage is incomplete.
- The static SND registry supplies the callable flag, incentivised flag and distribution start. The registry snapshot is from 2026-09.
- Marks are smooth and stale and survivorship caveats apply (see the harness README). t-stats are inflated. Prefer the paired differences, the halves and the placebo.
- The early-redemption event study had to rely on announcements. None of the registry-flagged early exits ever traded on SND.
- The holdout is only 9 months, and P4+Q itself is flat in it.

## Reusable signal
Call `research.nightly.event_driven.run.signals(panel)`, or read the cached `data/history/nightly/event_driven/event_signals_{M,W}.pkl`. It returns `codigo, day, cnpj8` plus:
- `ev_<etype>_days`: days since the issuer's last event of that type, dated strictly before `day`, for etype in agd, agd_waiver, resgate, deb_buyback, equity_raise, ma, rj, rating_up, rating_down, ipe_new_deb and supply_new_issue;
- `months_since_issue`, `new_issue_3m`, `callable`.

The rule is `run.neg(x)`, which applies the exNegEvents mask.
