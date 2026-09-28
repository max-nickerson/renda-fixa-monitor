# Renda Fixa Monitor

Local dashboard + alerts + a quant relative-value strategy for **Brazilian fixed income** (debêntures,
CRI/CRA, Tesouro/títulos públicos) and **eurobonds** — keyed by **ISIN**.

Built for a portfolio manager: **every bond is shown as duration + CDI+ spread** (DI+, %DI, Pré and IPCA+
on one yardstick, using B3's DI×Pré curve and the NTN-B real curve), with **rich/cheap vs its own history and vs peers**,
**fair spread → fair price → upside**, portfolio **CS01/DV01, carry, concentration, stress**, and
**out-of-sample-tested strategies that use point-in-time news** (CVM filings + press). See [Research](#research).

Type an ISIN → the app works out what it is, who the issuer is, which stock to watch, and starts tracking:

| Alert | Source |
|---|---|
| Price / yield moves | ANBIMA daily files (debentures, federal bonds), ANBIMA API (CRI/CRA, history), Tesouro Direto, TradingView MCP or manual/CSV (eurobonds) |
| Spread moves & z-score | vs DI, NTN-B (same reference ANBIMA uses), pre curve (LTN/NTN-F) or US Treasuries (FRED) |
| Issuer stock moves / drawdown | brapi.dev |
| Fundamentals published (ITR/DFP) | CVM open data |
| Material facts & CVM filings | CVM IPE |
| News (negative keywords: *recuperação judicial, default, rebaixamento…*) | Google News RSS (pt-BR + en) |
| Strategy signal flips (BUY/HOLD/SELL) | built-in strategy |

Alerts appear on the dashboard and can be e-mailed (SMTP).

### How fresh is the data?

Each source is polled as often as it actually changes:

| Data | Updated at source | App polls |
|---|---|---|
| Issuer stock (brapi) | ~5 s | every `QUOTES_EVERY_SECONDS` (60 s) during B3 hours, one batched call |
| Eurobond price (TradingView MCP, optional) | intraday, ~15 min delayed | every 5 min during US hours |
| Debentures & títulos públicos (ANBIMA) | once a day, ~20h Brasília | every `COLLECT_EVERY_MINUTES` (15 min); new file picked up within ~10 min |
| Tesouro Direto | once a day | every 15 min |
| CVM filings / material facts | CVM refreshes daily files | every 15 min (cached 6 h) |
| News | continuous | every 15 min |

Debentures and CRI/CRA have no free intraday price anywhere — ANBIMA's end-of-day rate is the market reference.
The dashboard reloads itself every 60 s.

> ⚠️ Research tool, **not investment advice**. Indicative prices are not executable quotes.

## Two tabs

**Minha carteira** — the ISINs you hold or follow (saved in `watchlist.yaml`), with optional quantity and
average price per asset (value and P&L per currency), every alert above, charts, events and the
per-asset strategy signal/backtest.

**Oportunidades (quant)** — ranks the **whole ANBIMA-priced debenture universe** (~1,200 bonds, CRI/CRA too
once the ANBIMA API is enabled) every day and lets you add any bond to your portfolio with one click:

| Factor | Weight | Idea |
|---|---|---|
| Value | 45% | spread above the peer "fair" curve (median spread by duration bucket), robust z-score |
| Carry | 20% | spread per year of duration (how much widening the carry absorbs) |
| Momentum | 15% | 5-day spread change (tightening +, widening −) |
| Quality | 20% | more REUNE trades +, higher dealer dispersion − |
| Penalties | | material fact at CVM in the last 15 days (−1), spread +50 bps in 5 days (−0.5) |

Peers = same indexer **and** same tax status (Lei 12.431 tax-exempt bonds trade structurally tighter).
Excluded: no indicative rate, < 6 months, price < 90% of par, top 1% spreads per group (credit events).
Free data has no ratings — "cheap vs peers" is often "riskier than peers". Use it as a research shortlist.
CLI: `python -m rfmonitor screener --top 20`.

---

## Put it online (Render, one click)

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/max-nickerson/renda-fixa-monitor)

1. Click the button (create a free Render account with your GitHub login when asked).
2. Render reads `render.yaml`: a Docker web service on the **Starter** plan (always on, ≈ US$7/month) with a
   5 GB persistent disk for the database, watchlist and history.
3. Fill in the secrets it asks for: **`BRAPI_TOKEN`** (required), **`ADMIN_KEY`** (any password-like string —
   your edit key), and optionally ANBIMA and SMTP.
4. **Apply**. After the build you get `https://renda-fixa-monitor-xxxx.onrender.com`.

- **Viewing is open to anyone with the URL** (no login). Adding/removing ISINs, positions, manual prices and
  "Atualizar agora" ask for the `ADMIN_KEY` once per browser (🔒 in the menu).
- First boot downloads the research history in the background (~30–60 min); the dashboard works meanwhile.
- The monthly ML ranking column is off on Starter (`ENABLE_ML_SELECTION=false`, it needs ~1–2 GB RAM); P4,
  regime, alerts and everything else run. Switch to a Standard plan and set it to `true` to enable it.
- Every `git push` to `main` redeploys automatically.

## Quick start (any machine with Python 3.10+)

```bash
git clone https://github.com/max-nickerson/renda-fixa-monitor.git
cd renda-fixa-monitor
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env                 # then fill in your keys (Windows: copy .env.example .env)
python -m rfmonitor run              # → http://127.0.0.1:8000
```

The first run copies `watchlist.example.yaml` to `watchlist.yaml` and collects data in the background
(then every `COLLECT_EVERY_MINUTES`). Add ISINs on the dashboard or with `python -m rfmonitor add <ISIN>`.

`.env`, `watchlist.yaml` and `data/` are **git-ignored** — your keys, positions and database never leave your machine.

### Keys (`.env`)

| Variable | Needed for | Where |
|---|---|---|
| `BRAPI_TOKEN` | issuer stock, fundamentals, SELIC/CDI | https://brapi.dev/dashboard |
| `ANBIMA_CLIENT_ID` / `ANBIMA_CLIENT_SECRET` | CRI/CRA prices, **debenture history** beyond ~10 business days | https://developers.anbima.com.br (app with *Preços e Índices*) |
| `SMTP_*`, `ALERT_EMAIL_TO` | e-mail alerts (Gmail → App Password) | https://myaccount.google.com/apppasswords |
| `TRADINGVIEW_MCP=true` | automatic eurobond prices (optional, Essential+ plan) | `pip install "mcp>=1.10"` then `python -m rfmonitor tv-login` |

Everything else (ANBIMA public files, Tesouro Direto, CVM, OpenFIGI, FRED, news) is free and keyless.

## CLI

```bash
python -m rfmonitor run [--no-scheduler]      # dashboard + scheduler
python -m rfmonitor collect [--backfill 180]  # one collect → alerts → e-mail cycle
python -m rfmonitor add BRBRKMDBS0A1          # add to watchlist (11-char ISINs are completed)
python -m rfmonitor resolve USN15516AB8       # show what an ISIN resolves to
python -m rfmonitor signal USN15516AB8        # current strategy signal
python -m rfmonitor backtest USN15516AB8      # walk-forward backtest on stored history
python -m rfmonitor import-prices <ISIN> prices.csv   # CSV: date,price  (or TradingView t,close)
python -m rfmonitor history                   # once per machine: SND trades, B3 curves, IDA (~30 min)
python -m rfmonitor models                    # retrain regime + selection models now (else daily)
python -m rfmonitor test-email
```

JSON API at `/docs` (FastAPI): `/api/assets`, `/api/asset/{isin}/series|signal|backtest`.

## How an ISIN is resolved

```
ISIN ─┬─ BR…DBS… → SND registry (debentures.com.br): CETIP code, issuer CNPJ, indexer, maturity
      ├─ BR…CRI/CRA… → OpenFIGI; set `cetip_code` in watchlist.yaml for ANBIMA prices
      ├─ BRSTN… → OpenFIGI (BNTNB/BLTN/…) → NTN-B / LTN / NTN-F / LFT + maturity
      └─ foreign (US…, XS…) → OpenFIGI: issuer, coupon, maturity → eurobond
issuer CNPJ / name → CVM registry (CVM code, parent company) → brapi → most liquid B3 stock
```

Anything can be overridden per asset in `watchlist.yaml` (`stock_ticker`, `cnpj`, `cetip_code`,
`news_query`, `coupon`, `maturity`, `tv_symbol`, per-asset `alerts:` thresholds).

## Strategy

Relative-value credit signal, computed only from trailing data (no look-ahead — see `tests/`):

| Component | Idea | Sign |
|---|---|---|
| **value** | z-score of the spread vs its own 120-day history | wide = cheap = + |
| **momentum** | 20-day spread trend (don't catch falling knives) | still widening = − |
| **equity** | equity–credit divergence: spread change vs what the issuer's stock move implies (rolling beta) | spread wider than stock justifies = + |
| **carry** | spread per unit of duration vs spread volatility (breakeven) | + |

Composite ≥ 0.75 → **BUY**, ≤ −0.75 → **SELL**, else **HOLD**. **Risk filters** turn a BUY into
**HOLD\*** when there was a material fact or negative-keyword news in the last 10 days, net debt/EBITDA
> 4.5× or rising > 0.5× in 90 days, or the issuer stock fell > 25% from its 20-day high.
When a bond has no spread (e.g. distressed paper without an ANBIMA rate) the price is used instead.

**Backtest**: long-only, enters at score ≥ entry, exits at ≤ exit, trades on the next day, charges
`cost_bps` per trade, earns CDI when flat (BRL assets). Bond return ≈ carry − duration × Δyield.
The event filter is applied point-in-time.

Honest limitations:
- ANBIMA prices are *indicative*; real bid-ask on illiquid paper can be much wider than 30 bps.
- History is short (public ANBIMA files cover ~10 business days; add ANBIMA API keys to backfill),
  so a backtest has few trades and weak statistics. Use it as a sanity check, not as proof.
- Defaults, amortisation schedules and restructurings are not modelled.

## Data sources & terms

ANBIMA (public files / Feed API), SND – debentures.com.br, CVM dados abertos, Tesouro Transparente,
OpenFIGI, FRED (St. Louis Fed), brapi.dev, Google News RSS (personal, non-commercial use),
TradingView MCP (optional, per TradingView's terms). This project runs locally for personal use.

## CDI+ equivalent spread

| Indexer | CDI+ spread (multiplicative, like ANBIMA's DI+) |
|---|---|
| DI + x | x |
| p% do DI | (p − 1) × pre(D) |
| Pré y | (1+y) / (1+pre_DI(D)) − 1 |
| IPCA + y | (1+y) / (1+real_NTN-B(D)) − 1 — the inflation breakeven cancels |

`pre_DI` comes from B3's daily TaxaSwap file (DI×Pré — matches LTN within ~2 bps); the real curve is NTN-B
(Tesouro Direto / ANBIMA). B3's `DIC` vertices were tested and rejected (avg −1.6 pp vs NTN-B, corrupted days
such as 2026-02-20). Eurobonds are shown as UST+.

## Research

Everything is **walk-forward out-of-sample with costs**; details and charts on the dashboard's **Pesquisa** tab
(`research/out/`). Reproduce: `python -m rfmonitor history` (≈30 min, once), then
`python research/run_timing.py` and `python research/run_selection.py`.

**A · Credit-regime timing** (IDA-DI / IDA-IPCA vs CDI & IMA-B, 2013–2026 OOS, 15 bps per switch)

| | Sharpe | Max DD | vs buy & hold |
|---|---|---|---|
| DI credit — 21d momentum rule | 2.16 | −3.0% | B&H 1.35 / −7.5% |
| IPCA credit (duration-hedged) — logistic | 0.62 | −5.5% | B&H 0.23 / −10.1% |

The edge comes mostly from sidestepping 2–3 credit crises → used as a **regime / risk indicator**. Gradient
boosting overfit and is not used.

**B · Bond selection** (658k SND trades 2021–2026, implied CDI+ spreads, monthly, 2023–2026 OOS)

Realistic execution: enter at the first trade *after* the rebalance date (the naive version, entering at the
signal's own trade, looked 5× better — almost all bid-ask bounce). Buy the top 20%, hold until a bond leaves the
top 50%:

| cost / turnover | Blend (heuristic + Ridge) vs universe | Max DD (universe −3.5%) |
|---|---|---|
| 0 bps | +1.7% a.a. | −1.7% |
| 25 bps | +1.1% a.a. | −2.0% |
| 50 bps | +0.5% a.a. | −2.3% |

Most robust single signal: **spread above the peer curve** (IC 0.11, positive in 95% of months) — the
dashboard's "cheap vs peers". Edge is real but small after costs; mainly lower drawdowns. Used live as the
**Modelo** column (Oportunidades) and the **Ação** column (carteira), retrained daily.

**C · Strategy lab with point-in-time news** (weekly 2022–2026, rate-hedged "pure credit" returns, decision
Monday with data to Sunday, execution at next week's trades, 25 bps; robustness at 2-week lag and 50 bps).
Reproduce: `python research/build_lab.py`, `python research/build_press.py`, then
`RET=ret_hedged python research/run_lab2.py data/history/press_weekly.pkl` (or `python research/run_all.py`).

News sources tested (12): **CVM IPE** (286k filings: material facts, RJ, waivers, early maturity, debenture-holder
meetings, CVM/B3 inquiries) and **Google News RSS** with `after:/before:` (32k negative headlines for the 217 brands
of the 250 most-traded issuers + market "recuperação judicial"/"calote" gauge) are used. GDELT (thin mid-cap
coverage, throttled), B3 Plantão (~12 months only), sitemaps/Wayback (backup), Bing and paid news APIs (no free
history), SND events (stale) were rejected — see the Pesquisa tab.

| Strategy (23 tested; excerpt) | vs universe | +2w lag | 50 bps | Sharpe | Max DD |
|---|---|---|---|---|---|
| **P4 carry top 30% CDI+, never rich, no entry with negative press 30d, CDI when momentum regime off** | **+1.22%** | +1.05% | +1.17% | **3.22** | **−0.6%** |
| C3 same without the press filter | +1.23% | +1.10% | +1.18% | 3.14 | −0.7% |
| M3 Ridge (CVM news) + news exit + regime | +1.34% | +1.12% | +0.80% | 1.03 | −4.7% |
| P1 hold everything, sell on negative-press spike | −0.07% | −0.07% | −0.29% | 0.85 | −2.9% |
| B3 buy after a >50 bps widening (no news) | −0.90% | −1.88% | −3.04% | 0.10 | −7.1% |
| RV4 only bonds rich vs peers (diagnostic) | −3.82% | −3.89% | −4.71% | −0.97 | −14.1% |
| P10 only issuers with a negative-press spike (diagnostic) | −1.51% | −1.76% | −2.73% | −0.22 | −4.7% |

Findings: news **predicts** losses (press-spike issuers −1.5% a.a.) but **selling after** the news doesn't help —
use news as an **entry filter**; spread-widening stops hurt (short-term reversal); rich-vs-peers is the strongest
sell signal; adding news features to ML didn't help. Sharpe figures are on hedged credit excess returns
(annual vol ≈ 0.4%), so they are high in absolute terms — compare them across rows, not with equity Sharpes.
**P4 runs live** in Oportunidades/carteira (daily Google News check for the covered issuers).

## Development

```bash
python -m pytest -q
```

MIT License.
