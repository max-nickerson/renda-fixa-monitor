# Commodity / macro point-in-time panel

Build: `.venv\Scripts\python.exe research\data_commodities.py [--refresh]` → `data/history/commodities.pkl`
(long: `series, period_date, available_date, value, unit, source`). Raw files cached in `data/history/commod_raw/`
(live files re-fetched after 20h, closed ONS years cached forever; ≥1s between requests). Legacy `.xls` (CEPEA, EIA)
needs `xlrd`, auto-installed with `pip --target` into `commod_raw/_pylib` (the venv is not touched). Coverage 2018-01 → 2026-09-28
(2018 = warm-up). **Join rule: a value may be used on date t only if `available_date <= t`.** Latest-vintage values throughout.

| series | source (URL) | freq | range obtained | available_date rule |
|---|---|---|---|---|
| gas_hh, brent, wti | EIA spot xls `eia.gov/dnav/{ng,pet}/hist_xls/{RNGWHHD,RBRTE,RWTC}d.xls` (= FRED DHHNGSP/DCOILBRENTEU/DCOILWTICO; FRED tried first) | D | 2018-01-02 → 2026-09-22 | next business day |
| ulsd_ny, gasoline_nyh, propane_mb | EIA spot NYH ULSD, NYH RBOB, Mont Belvieu propane (extras; naphtha/diesel proxies) | D | 2018-01-02 → 2026-09-22 | next business day |
| gas_eu, urea, dap, potash, sugar, soy, corn, wheat (US HRW), iron_ore, coal (Australian) + gas_us_wb, lng_japan, brent_wb, tsp, phosphate_rock, soymeal, soyoil, coal_sa, beef_wb, chicken_wb, coffee_arabica, cotton, palm_oil, rubber, aluminum, copper, nickel, zinc | World Bank Pink Sheet `CMO-Historical-Data-Monthly.xlsx` (link scraped from worldbank.org/en/research/commodity-markets) | M | 2018-01 → 2026-08 | 1st business day of M+1 + 3 business days (~35d) |
| sugar_br | CEPEA açúcar cristal SP `cepea.org.br/br/indicador/series/acucar.aspx?id=53` | D | 2018-01-02 → 2026-09-25 | next business day |
| cattle_br | CEPEA boi gordo `.../boi-gordo.aspx?id=2` | D | 2018-01-02 → 2026-09-25 | next business day |
| ethanol_br, ethanol_anidro_br | CEPEA etanol hidratado combustível (id=103) / anidro (id=104), SP, weekly dated Friday | W | 2018-01-05 → 2026-09-25 | next business day (Mon) |
| pld (proxy) | ONS CMO semi-horário SE/CO, daily mean `ons-aws-prod-opendata.s3.amazonaws.com/dataset/cmo_tm/` | D | 2020-01-01 → 2026-09-28 | day + 1 |
| cmo_se_weekly | ONS CMO semanal SE/CO (DECOMP) `.../dataset/cmo_se/` ; period = operative week start (Sat) | W | 2018-01-06 → 2026-09-26 | = week start (ex-ante plan published Thu/Fri before) |
| reservoir, reservoir_br | ONS EAR diário por subsistema `.../dataset/ear_subsistema_di/` (SE/CO %; SIN = Σverif/Σmax) | D | 2018-01-01 → 2026-09-27 | day + 1 |
| diesel_br, diesel_s10_br, gasoline_br, ethanol_retail_br, glp_br, gnv_br | ANP SLP `gov.br/anp/.../shlp/semanal/semanal-brasil-desde-2013.xlsx`, Brazil avg retail; period = week-end Saturday | W | 2018-01-06 → 2026-09-26 | week end + 3 days |
| usdbrl | BCB SGS 1 (PTAX venda) `api.bcb.gov.br` | D | 2018-01-02 → 2026-09-28 | next business day |
| selic | BCB SGS 432 (meta) | D | 2018-01-01 → 2026-09-28 | next business day |
| ipca | BCB SGS 433 (% m/m) + IBGE release calendar `servicodados.ibge.gov.br/api/v3/calendario/9256` | M | 2018-01 → 2026-08 | exact IBGE release date + 1 day (104/104 exact) |
| ibc_br, ibc_br_sa | BCB SGS 24363 (NSA), 24364 (SA) | M | 2018-01 → 2026-07 | month end + 55 days (next bday) |

Lag rationale / caveats
- EIA spot prices are market assessments observable same day; EIA itself publishes with a few days' lag (file on 2026-09-28
  ends 09-22), so "next business day" is right for market knowledge but the *download* can lag ~1 week at the live edge.
- Pink Sheet: August 2026 data appeared in the September 2026 Pink Sheet (early Sept). Last 1–2 months are occasionally revised.
- IBC-Br: BCB releases ~45–50 days after month end (occasionally ~50d); +55d chosen to be conservative.
  Both IBC-Br series are revised every release (SA is re-estimated back in history) — latest vintage used, look-ahead risk remains.
- IPCA: released 09:00 on the IBGE date; +1 day keeps it safe for EOD signals. IPCA is not revised.
- `pld` is the **unclipped** SE/CO CMO (DESSEM) daily mean (min 0, max ~3,092 BRL/MWh). True PLD = CMO clipped to the
  ANEEL floor/cap of each year; apply `clip(floor_y, cap_y)` if the exact level matters. DESSEM runs day-ahead, so D+1 is conservative.
- ANP weekly survey covers Sun–Sat and is published around Fri/Sat; week end + 3d is conservative. ANP has no revisions.
- Business days are weekday-based (no holiday calendar); an available_date on a holiday just means "usable next session".

Gaps / failures (tested 2026-09-28)
- **FRED** (`fred.stlouisfed.org`, fredgraph.csv and /data/*.txt): connection reset / timeouts from this network → EIA used
  (identical underlying series). The script still tries FRED first (20s timeout) and falls back automatically.
- **CCEE** (`dadosabertos.ccee.org.br`, `pda-download.ccee.org.br`, `ccee.org.br`): HTTP 403 "Acesso bloqueado" (WAF) → no
  official PLD; ONS CMO used as proxy. Semi-hourly CMO starts 2020, so 2019 power-price signal must come from `cmo_se_weekly`.
- **BCB SGS** intermittently returns an HTML "Requisição inválida" page with HTTP 200 → responses are JSON-validated and retried.
- ammonia: no free keyless source found (WB Pink Sheet has none) — skipped; use gas_hh/gas_eu + urea as proxies.
- naphtha: no free spot series (EIA/WB don't publish it) — proxies: brent, propane_mb, gasoline_nyh.
- steel, used_cars (FIPE): not attempted beyond scoping (FIPE only offers per-model queries; no free steel index) — skipped.
- CEPEA daily hydrated-ethanol indicator not used (weekly SP indicators 103/104 are the long, stable series).
