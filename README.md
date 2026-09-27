# Renda Fixa Monitor

Local dashboard + alerts + a quant relative-value strategy for **Brazilian fixed income** (debêntures,
CRI/CRA, Tesouro/títulos públicos) and **eurobonds** — keyed by **ISIN**.

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

---

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

## Development

```bash
python -m pytest -q
```

MIT License.
