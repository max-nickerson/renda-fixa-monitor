# TEXT & NLP: CVM filings and news headlines as credit signals (nightly, harness v4)

**Verdict: negative.** Portuguese text features built from CVM IPE filings and Google News headlines do not beat P4+Q.
- The best overlay excludes P4+Q issuers that had a zero-shot credit-negative document in the last 90 days. It adds **+0.27%/yr (NW t 1.28, Holm p = 1.0, 9 variants tried)**.
- Filings and headlines arrive **after** bond marks have already moved, so they give no useful lead time.
- A learned embedding model has an IC of about 0.

All research numbers are pre-2026. The sealed holdout was read once, at the end.

## Data built (all point-in-time)

| file | contents | PIT rule |
|---|---|---|
| `data/history/nightly/text_nlp/docs.pkl` | 102k pre-2026 docs (127k incl. 2026): CVM IPE filings 2021–26 for the 384 grid issuers that file with CVM (Categoria/Tipo/Especie/Assunto, first delivery of each protocol; insider-position reports, ballots, bylaws and policies dropped), plus 32k Google-News headlines for 212 issuers (`press_items.pkl`, negative-keyword queries) | usable at decision close d only if `avail < d` (IPE `Data_Entrega` = delivery date, news = pubDate) |
| `emb.npy`, `emb_texts.pkl` | 46k unique texts embedded with `paraphrase-multilingual-MiniLM-L12-v2` (384-d, normalised; digits collapsed in IPE subjects; news plus credit-relevant IPE categories only, to save CPU) | text only |
| `doc_feats.pkl` | per doc: keyword classes `kw_*`, zero-shot cosine `zs_*`, `zs_label`, credit sentiment `sent` | text only, no labels |
| `pdf/<doc_id>.txt` | full text of 700 Fatos Relevantes / Comunicados (2022–25), extracted with a pure-python PDF parser; 98% extracted, median 4.5k chars | same dates as the doc |
| `text_signals_M.pkl` | reusable issuer-date features for the monthly panel | `avail < day` |
| `txt_ml_M.pkl` | walk-forward ridge text score | trains only on labels with `lab_end_63 <= dpos` |

## Method
- **Keyword classes** are accent-folded Portuguese regexes:
  - distress and debt events: rj, default_waiver, liab_mgmt, call_redeem, deb_holders, new_debt;
  - corporate events: equity_raise, mna, capex, guidance_cut, dividend, mgmt_change, litigation;
  - ratings and inquiries: rating_down, rating_up, oficio.
- **Zero-shot classes** are the cosine to centroids of hand-written Portuguese prototype sentences, with threshold 0.55 and a "routine" class. Agreement with the keywords:
  - high for rj (93%), mgmt_change (70%), rating_down (73%) and oficio (67%);
  - poor for default_waiver (1%) and guidance_cut (3%).
- **Sentiment** is cos(doc, negative-credit centroid) − cos(doc, positive-credit centroid). Spot checks were sensible:
  - most negative: "Deferimento do pedido de recuperação judicial", "Ambipar deixa de pagar juros de debêntures…";
  - most positive: capital increases, "lucro recorde".
- **Event studies:**
  - The event is the first doc of a class per issuer after 90 quiet days. Day 0 is the first grid day after `avail`.
  - Returns are the issuer's mean bond daily excess return (harness `R`, gap-corrected) minus the mean over all bonds on the grid, over ±120 bdays.
  - Spreads are the issuer's median CDI+ minus the universe median on the weekly panel, over ±26 weeks.
  - The placebo uses random issuer-dates.
- **Lead time:** 368 spread blow-outs, where the issuer's spread relative to the universe widens by ≥150 bps over 4 weeks (one per issuer per 26 weeks). For each, I checked whether a text flag appeared in the 13 or 26 weeks before, and compared that with random issuer-weeks.
- **Overlays:** harness `backtest` with monthly decisions, 126-bday tranches, 25 bps, issuer cap 10%. I also ran 50 bps and rec40. Holm is across all 9 variants.

## Results

### Overlays on P4+Q (pre-2026, 48 months)
| variant | exCDI | vs U | vs P4+Q | t | Holm p | 22–23 / 24–25 vs P4+Q | vs P4+Q at 50 bps | vs P4+Q rec40 | names |
|---|---|---|---|---|---|---|---|---|---|
| V1 ex any credit-neg keyword 90d | 2.39 | 1.29 | +0.17 | 0.82 | 1 | +0.04 / +0.30 | +0.14 | +0.16 | 78 |
| V2 ex hard events (rj/waiver/liab-mgmt) 180d | 2.19 | 1.09 | −0.03 | −0.62 | 1 | −0.06 / +0.01 | −0.03 | −0.01 | 98 |
| V3 ex IPE credit-neg 90d | 2.40 | 1.30 | +0.18 | 0.93 | 1 | +0.06 / +0.31 | +0.15 | +0.17 | 80 |
| **V4 ex zero-shot negative 90d** | **2.49** | **1.38** | **+0.27** | **1.28** | **1** | +0.06 / +0.48 | +0.24 | +0.25 | 74 |
| V5 ex worst-quintile news sentiment 30d | 2.22 | 1.12 | +0.00 | 0.37 | 1 | 0 / +0.01 | 0 | 0 | 103 |
| V6 ex filing burst (>3× normal) | 2.21 | 1.11 | −0.00 | −0.51 | 1 | 0 / −0.01 | −0.01 | −0.00 | 103 |
| V7 ex bottom-20% learned text score | 2.31 | 1.21 | +0.09 | 1.31 | 1 | +0.01 / +0.17 | +0.07 | +0.10 | 92 |
| V8 tilt 1/(1+neg_kw_90) | 2.31 | 1.21 | +0.09 | 0.83 | 1 | +0.03 / +0.15 | +0.08 | +0.09 | 103 |
| V9 P4+Q + positive-event adds | 2.15 | 1.05 | −0.07 | −1.41 | 1 | +0.04 / −0.17 | −0.07 | −0.07 | 117 |
| P4+Q | 2.22 | 1.12 | 0 | | | | | | 103 |
| P4 | 1.89 | 0.79 | −0.33 | −2.28 | | | | | 130 |

Notes on the best variant (V4):
- Sharpe is 2.52 against 2.32 for P4+Q, and max drawdown is −0.90% against −1.32%. Both are inflated by smooth marks.
- The gain comes almost entirely from 2024–25.
- The names V4 removes still beat the universe by +0.46%/yr at the cohort level (t 0.7). They are merely below average within P4+Q.
- The random-book placebo averages 0.79 (p95 1.07). This only shows that the P4+Q base is good; the placebo that matters is the paired t above.

**Sealed holdout (2026-01 to 2026-09, read once, V4 frozen):** exCDI +0.41%/yr, and **+0.33%/yr vs P4+Q (t 1.53, 9 months)**. The sign is consistent, but the gap is not significant.

### IC (monthly panel, Spearman, NW t)
- **Every text-activity measure has a negative IC, and the generic ones are as strong as the credit-specific ones:**
  - `ipe_one_90` (number of filings): −0.064 (t −2.5);
  - `burst_30`: −0.054;
  - `ipe_new_debt_180`: −0.050;
  - `ipe_dividend_180`: −0.043;
  - `neg_kw_90`: −0.044 (t −3.6);
  - `zs_neg_90`: −0.047.
  - Much of the "text signal" is a proxy for large, listed, frequently-filing issuers with tight spreads.
- **Sharper content:**
  - `news_neg_kw_90` −0.041 (t −5.1);
  - `news_litigation_90` −0.032 (t −4.1);
  - `news_rj_90` −0.026 (t −3.5);
  - `ipe_rj_180` −0.041 at 126d, but only 0.6% of rows are non-zero.
- `ipe_liab_mgmt_180` is **positive** (+0.025, t 3.1): after a reprofiling or renegotiation filing, bonds have already been marked down and then recover.
- The walk-forward ridge on mean embeddings (768-d, IPE and news) has **IC −0.006 (t −0.2)**. Supervised text embeddings learn nothing out of sample here.

### Event studies (abnormal bond return, %)
| event | n | pre −60..−1 | post 0..20 | post 0..60 | spread w+13 vs w−4..−1 |
|---|---|---|---|---|---|
| IPE recuperação judicial | 11 | −2.4 (t −0.9) | −2.7 | −2.3 (t −2.0) | +233 bps (median +61) |
| IPE default/waiver | 108 | −0.4 | −0.1 | +0.6 | +55 bps (t 2.4, median +6) |
| IPE liability mgmt | 62 | **−1.2 (t −2.7)** | −0.2 | −0.2 | −5 |
| IPE M&A | 666 | −0.3 | −0.16 (t −2.8) | **−0.28 (t −2.4)** | +16 bps (t 2.9) |
| IPE guidance cut / impairment | 207 | −0.3 | −0.1 | −0.54 (t −2.2) | +14 |
| IPE ofício / clarification | 519 | −0.26 | 0.0 | −0.24 (t −2.0) | +11 (t 2.1) |
| any Fato Relevante | 859 | −0.14 | −0.14 | −0.27 (t −2.5) | +10 (t 2.0) |
| top-5% negative sentiment | 688 | −0.06 | −0.16 | −0.25 | +10 |
| top-5% *positive* sentiment | 805 | −0.06 | −0.13 | −0.24 (t −2.4) | +5 |
| news RJ headline | 121 | −0.48 | −0.15 | −0.0 | +7 |
| placebo, random dates | 1047 | +0.16 | +0.07 | +0.16 | −1 |

### Lead time before spread blow-outs (368 events, 2021–25)
| flag in prior 13 weeks | blow-outs | random weeks | lift |
|---|---|---|---|
| hard keyword (rj/waiver/liab-mgmt) | 7.1% | 6.8% | 1.04 |
| any credit-negative keyword | 24.7% | 27.1% | 0.91 |
| zero-shot negative | 26.4% | 27.0% | 0.98 |
| top-5% sentiment | 14.9% | 15.1% | 0.99 |
| Fato Relevante | 24.2% | 26.0% | 0.93 |

The median first-flag lead of 74–165 days is no better than chance. **Text does not lead spread blow-outs.** Between 18% and 25% of blow-outs get a flag during or after the widening; disclosure follows the market.

### Full text vs title (700-PDF sample)
- **Hidden events:** 7.2% of Fatos Relevantes / Comunicados whose title looks clean contain hard credit-event language in the body.
- **Title flags hold up:** 88% of hard-flagged titles are confirmed by the body.
- **Weak agreement:** title and body sentiment correlate at only 0.45.
- **The hidden events were already priced.** Filings that are hard in the body but clean in the title (n = 36) show **−5.1% abnormal return in the 60 days before the filing (t −3.1)**, followed by +1.3% (t 0.7) afterwards. Whatever the PDF says, the market had already marked it.
- The top tercile of body sentiment is followed by −0.6% over 60 days (t −2.3, n = 155). That is suggestive, but it is one of several cuts on a small sample.

## Insights
1. **Disclosure is lagging, not leading, for Brazilian debentures.**
   - Liability-management and waiver filings arrive after 1–5% of abnormal underperformance.
   - Spread blow-outs are preceded by text flags no more often than random weeks are.
   - This matches the earlier "reacting fast to news doesn't pay" result, and extends it to CVM filings, full text and embeddings.
2. **The only post-event drift is slow and "soft", and it belongs to event-risk filings, not distress filings.**
   - M&A: −0.28% over 60d, spread +16 bps at 13 weeks, t 2.9.
   - Guidance cut: −0.54%.
   - CVM inquiries (ofício): −0.24%.
   - Fatos Relevantes in general: −0.27% vs a placebo of +0.16%.
   - Bondholders slowly price the leverage or event risk of corporate actions. The size (~0.3–0.5% on the issuer's bonds, rare events) is too small to move a 100-name book much.
3. **"Any news is bad news."**
   - Top-5% *positive* sentiment docs underperform just like negative ones (−0.24%, t −2.4).
   - Generic filing counts have ICs as strong as the credit-specific classes.
   - The portable part of the text signal is mostly *issuer activity or visibility*, not meaning.
4. **Liability management marks the bottom.** `ipe_liab_mgmt_180` has a positive IC (+0.025, t 3.1): post-reprofiling bonds recover. Do not use it as an exclusion filter.
5. **Supervised learning on embeddings fails** (IC −0.006). Zero-shot prototypes plus keywords are as good as it gets with titles.

## Caveats
- The news corpus comes from negative-keyword Google News queries, with relevance-capped recall and a strong 2025–26 density increase. Some headlines are off-topic (e.g. generic court cases).
- IPE covers only CVM-registered issuers (384 of 775 grid issuers, about 83% of universe rows). Unlisted SPVs and infrastructure issuers get no filings, so "no text" ≠ "clean".
- The keyword lists and zero-shot prototypes were written by me before the backtests, but the variant set is informal. 9 variants plus about 40 event classes and about 30 IC features were looked at, so treat every t below 3 as noise.
- Marks are smooth and stale, and some event paths are dominated by one or two gap jumps (RJ, n = 11). Medians are reported for that reason.
- Only a subset of IPE categories was embedded, to save CPU. Routine governance filings are keyword-classified only.
- The full-text PDF extraction is crude. Hex-encoded fonts without CMaps drop text, and 2% of files came out empty.

## Reusable signals
- `research/nightly/text_nlp/features.py: signals(panel) -> DataFrame[cnpj8, day, ...]` works on any harness panel (M or W).
  - Key columns: `zs_neg_90`, `neg_kw_90`, `news_neg_kw_90`, `hard_180`, `ipe_liab_mgmt_180`, `ipe_mna_180`, `sent_news_30`, `burst_30`, `ipe_one_90`.
- Cached versions: `data/history/nightly/text_nlp/text_signals_M.pkl`, and `txt_ml_M.pkl` for the learned score (not useful).
- The doc-level table `doc_feats.pkl` can be used for other windows or decays.

## Files
- `build_docs.py`: builds the document table.
- `classify.py`: embeddings, keyword classes, zero-shot classes and sentiment.
- `fulltext.py`: PDF sample and extractor.
- `features.py`: point-in-time features and `signals()`.
- `run.py`: event studies, lead time, IC, learned model, overlays and holdout.
- `results.json`, `equity_total_return.png`, `cum_excess.png`, `event_studies.png`, `run.log`.

## Rerun
```
cd renda-fixa-monitor
set PYTHONPATH=. & set PYTHONIOENCODING=utf-8 & set OMP_NUM_THREADS=2 & set MKL_NUM_THREADS=2
.venv/Scripts/python.exe research/nightly/text_nlp/build_docs.py
.venv/Scripts/python.exe research/nightly/text_nlp/classify.py     # ~45 min on a busy machine (embeddings cached afterwards)
.venv/Scripts/python.exe research/nightly/text_nlp/fulltext.py     # ~12 min network (cached)
.venv/Scripts/python.exe research/nightly/text_nlp/run.py          # ~13 min
```
