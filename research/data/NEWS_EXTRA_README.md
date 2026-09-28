# Extra point-in-time news events (2021-01 → 2026-09-28)

Built by `research/data_news_extra.py` (`ratings | wayback | regulators | sector | build | all`). Raw downloads are cached under
`data/history/news_extra_raw/` (5,064 Google News RSS JSONs + 5 Wayback CDX pages, 25 MB), so re-runs only fetch missing ranges;
ranges ending within 7 days of today are refetched.
**Date semantics:** Google News historical `pubDate` is a publication DATE (time is a placeholder), so an item is known at the END
of that date (use it from the next day). Fitch slugs end in `-dd-mm-yyyy`, which is used as the event date. The Wayback capture
timestamp (`captured`) is only an upper bound on when the page existed.

## 1. `data/history/rating_events.pkl` [cnpj8, brand, date, direction, agency, title, source, n_articles, captured]
- Universe: `research/out/press_brands.json` (242 cnpj8, 217 brands). 64 generic or ambiguous brands are skipped (e.g. Energetica,
  Transportadora, Norte, Minas). There are aliases for the others (Axia→Eletrobras, Diagnosticos→Dasa, Motiva→CCR, Sendas→Assaí…), which gives 149 queries.
- Google News: `"<brand>" (rebaixa OR eleva OR rating OR Fitch OR "Moody's" OR "S&P")`. Queries run by year. A year with ≥6 items is split into halves, and a half with ≥6 into quarters,
  because Google returns relevance-capped subsets. A title is kept when it names the brand (word boundary) and names an agency or uses credit-rating wording.
  Titles are dropped for foreign affiliates, sell-side "rebaixa para neutro" calls, ESG/S&P indices, and common-word brands (Vale, Vivo, TIM, Light, Valid…) with no agency named.
- Wayback CDX: `fitchratings.com/research/pt/*` (all slugs) plus `research/corporate-finance/*` and `infrastructure-project-finance/*`,
  filtered server-side to slugs naming a covered brand. This gives 8.7k URLs in total. Only rating-action slugs are used (rebaixa/eleva/afirma/downgrades/…).
- Direction: an action verb (rebaixa/corta/downgrade/perde grau ↔ eleva/upgrade) wins. Otherwise an outlook transition ("de negativa para estável" = +1).
  Otherwise outlook/watch words (perspectiva/observação negativa = −1, positiva = +1, which also applies to affirmations that keep a negative outlook). Everything else is 0.
- Repeated coverage is collapsed: same brand/agency/direction within 5 days becomes 1 event (`n_articles`, `source` = gnews / wayback_fitch / both).
  An event is then exploded to every cnpj8 of the brand.

Brand-level events per year (−1/0/+1): 2021 8/99/23 · 2022 8/130/20 · 2023 28/131/14 · 2024 17/132/15 · 2025 33/135/30 · 2026 (to Sep) 75/106/29.
In total there are 1,033 brand events (1,196 cnpj8 rows). By agency: Fitch 836, Moody's 87, S&P 86, unknown 23. By source: 758 events contain Wayback-Fitch, 328 contain Google News, and 53 are in both.
**Coverage:** 133 of 242 issuers have ≥1 event, 100 have ≥1 up/down/outlook move, and 61 have ≥1 downgrade-type event.

## 2. `data/history/regulator_events.pkl` [sector_group, regulator, date, direction, title, source]
There are 10 monthly Google News queries (a negative-leaning and a positive-leaning query per regulator; one each for ANA and ARSESP). The title must name the regulator.
Direction is taken from the issuer-credit side. multa/caducidade/intervenção/suspensão/penalidade/nega/tariff cut = −1. aprova/autoriza/reajuste/aumento/prorroga/renova = +1.
Sector groups: aneel→`utilities`, antt→`toll_roads` or `railways` (keyword-routed, both if unclear), anp→`oil_gas`, ana/arsesp→`sanitation`, anac→`airlines`.
There are 6,334 rows (−1: 1,553 · 0: 2,268 · +1: 2,513). By year: 2021 341, 2022 393, 2023 441, 2024 793, 2025 1,501, 2026 2,865.
By regulator: antt 2,186 · aneel 1,875 · anac 916 · anp 888 · ana 422 · arsesp 47.

## 3. `data/history/sector_news.pkl` [topic, date, n_items, n_negative, query]
There are 16 topics, and their names match `features_ext.TOPIC_SECTORS`. Each query has the form `(crise OR queda) <term>` and runs monthly, split to weeks at ≥95 items (this never happened). The query is kept per row and in `df.attrs["queries"]`.
`n_negative` counts titles that match a negative lexicon. Totals: 27.7k items / 12.9k negative. Items per year: 3.3k, 3.7k, 3.8k, 4.7k, 5.9k, 6.3k.

## Classification spot-check (20 random titles per type, judged by hand)
- Ratings, Google News titles: 15/20 were fully correct (right issuer, right direction). Direction alone was right for 17/20. The errors were an affirmation with a positive outlook scored +1, "pressão sobre nota" scored 0, and three irrelevant or misattributed items. The rules were then patched (for example "tira de observação negativa" is now +1, and Valid is now a common-word brand).
- Ratings, Wayback slugs: 19/20 correct. The one error was florida-power-light, a foreign company matched to Light; foreign filter now added.
- Regulators: 16/20 correct. The errors were consumer rules scored +1 for airlines, bus-line permits routed to toll_roads, "aumenta pedágio" scored 0 (fixed), and a "penalidades" mention scored −1.
- Sector negative flag: 16/20 correct before the last lexicon patch (derrub/estiagem/reduz/escass were added). About 25% of items are off-topic (foreign or generic).

## Caveats / failures
- **Recall is partial.** Google News RSS returns a relevance-capped subset (for example "Fitch rebaixa" for Jan-2023 returns 1 item), and density rises strongly toward 2025-26 (recency bias).
  Use relative or z-scored features (as `features_ext` does). Do not read levels across years.
- Wayback coverage of Fitch is sparse (about 8.7k slugs since 2020, mostly banks and debenture assignments). S&P `/ratings/pt/` and Moody's returned nothing through CDX.
- Regulator RSS on gov.br returned 404 or a redirect (ANEEL/ANTT/ANP), so no native archive is available. Only Google News headlines are used.
- 64 ambiguous brands are skipped, so those issuers have no rating events. There are no Austin/Liberum/SR queries, so smaller local agencies are rare.
- Direction 0 rows (affirmations, new assignments, commentary) are kept on purpose. Features use only direction ≠ 0.
