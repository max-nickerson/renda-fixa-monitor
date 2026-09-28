# Equity history for debenture issuers (built by `research/data_equity.py`)

**Files.** `research/data/equity_map.csv` has columns cnpj8, issuer_name, ticker, mapping_type (direct/parent/none), confidence and note. `data/history/equity_daily.pkl` is a long table with columns ticker, date, close, adj_close, volume and source. Raw inputs are cached in `data/history/equity_raw/`: brapi `<T>.json` files (range=10y, interval=1d) and `cotahist_<Y>.csv.gz`, a B3 COTAHIST spot-market extract. Re-run the script to refresh: brapi data is refetched after 20h, past COTAHIST years are cached permanently, and the current year is refreshed daily.

**Universe and coverage.** The universe is 924 issuers with eligible bond-weeks from 2022 onward (205,246 rows). The map covers 381 issuers: every issuer up to 85% of cumulative weight was reviewed by hand, and the tail was matched automatically on ISIN code plus issuer name.
- Share of eligible bond-weeks with any mapped ticker: **68.1%**. With a price on that week: **65.6%**.
- Priced through a direct listing: 43.6%. Priced only through a listed parent: 22.0%.
- By mapping confidence: high 58.8%, med 6.3%, low 7.0% (the bands overlap).
- The remaining ~32% are issuers with no listed equity. Examples: the Aegea group (Aegea, Corsan, Aguas do Rio, Prolagos...), state water utilities (Saneago, Cagece, Casan), NTS, VLI, Algar Telecom, Vero, BRK Ambiental, Iguá, securitisers, and wind, solar and transmission SPEs of unlisted or foreign sponsors. These appear as `mapping_type=none` rows for the top-weight names.

**Tickers and dates.** The table has 161 tickers with data (162 mapped; ARTR3 has no B3 trades after 2020-06). It spans 2020-06-01 to 2026-09-25 and has 228,702 rows. Today's bar is dropped as partial.
- **Coverage by ticker:** most tickers cover the whole window. Series that start late are post-2020 IPOs: RDOR3 (Dec-20), VAMO3, ASAI3, CMIN3, JALL3, SMFT3, RAIZ4, ONCO3, CBAV3, BRST3, TTEN3 and similar. Other late starts:
  - AURE3 from 2022-03 and VTRU3 from 2024-06.
  - PASS3 from 2026-05 and AZUL3 from 2026-04. Both are too short to use.
- **Delisted, taken from COTAHIST up to the last trade:**
  - NEOE3 (to 2026-05-04), ENBR3 (to 2023-08) and AESB3 (to 2024-10).
  - CESP6 (to 2022-03), LCAM3 (to 2022-07) and BRML3 (to 2023-01).
  - CIEL3 (to 2024-08), SOMA3 (to 2024-07) and KRSA3 (to 2025-04).
  - CRFB3 (to 2025-05), BRFS3 (to 2025-09), STBP3 (to 2025-10), SRNA3 (OMGE3 spliced in, to 2025-11) and ZAMP3 (BKBR3 spliced in, to 2025-12).
  - After a parent delists there is no parent price. This affects the NEOE3 subsidiaries (Coelba, Celpe, Cosern, Elektro, Brasília) after 2026-05 and the EDP subsidiaries after 2023-08.
- **Illiquid own listings** (median volume under 5k shares a day): CEEB3, EKTR4, ENMT4, EQMA3B, CGAS5, GEPA4, MRSA3B and CLSC4. Most of them also have a liquid parent row.

**Mapping decisions.** Every subsidiary was mapped to its current listed controller: Energisa (ENGI11), Equatorial (EQTL3, including CEEE-D, CEA and Echoenergia), Neoenergia (NEOE3), CPFL (CPFE3), Cemig (CMIG4), Copel (CPLE3), AXIA/Eletrobras (AXIA3), EDP (ENBR3), Ecorodovias (ECOR3), Motiva (MOTV3), Localiza (RENT3), Rumo (RAIL3), Ultrapar (UGPA3), Cosan (Comgás, Compass) and Engie (EGIE3). Two points need care:
- **Change of control:** Unidas, CESP, BR Malls, BRF, Soma and AES Operações have two rows, the old direct or parent ticker and the successor (`note` gives the switch date). Choose the row by date to stay point-in-time.
- **Minority stakes and JVs:** JVs without a clear controller are `none` or `low`.

**Data caveats.**
- **brapi validation:** brapi `close` is already split-adjusted and `adj_close` is also dividend-adjusted. Every brapi series was checked against B3 COTAHIST daily returns (median agreement 100%, none under 95%).
- **Wrong spliced history removed:** brapi attaches predecessor history to the current ticker. It was kept only where the predecessor was verified (for example ELET3→AXIA3, CCRO3→MOTV3, TRPL4→ISAE4, MRFG3→MBRF3, GUAR3→RIAA3, NTCO3→NATU3). Three wrong splices were dropped: RDOR3 before its IPO, AURE3 before 2022 (which was CESP6), and ALOS3 before Oct-2023, which was refilled from ALSO3 (`source=cotahist:ALSO3`).
- **COTAHIST series are not adjusted:** `adj_close = close`, with no dividend or split adjustment (KRSA3 has an 8x reverse-split jump on 2024-10-11), and `volume` is in shares.
- **Real price shocks are kept:** AMER3, AMBP3, SEQL3, AZUL3 and IRBR3 show extreme moves.
- **Point-in-time:** a daily close is known at the end of its date, so for weekly features use closes on or before the week's as-of date. The map is based on today's corporate structure, so check the `note` field for control changes.
