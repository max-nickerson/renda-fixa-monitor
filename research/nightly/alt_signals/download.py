"""Download raw public alternative data for the alt_signals study (cached; re-run is a no-op).

Sources (all public, keyless):
  - CVM offers registry  dados.cvm.gov.br/dados/OFERTA/DISTRIB/DADOS/oferta_distribuicao.zip
  - ANTT toll revenue by federal concessionaire (monthly)  dados.antt.gov.br receita-pedagio
  - ANEEL tariffs approved per distributor (resolutions)  dadosabertos.aneel.gov.br tarifas-distribuidoras
  - ANEEL tariff flags (bandeiras) acionamento
Cache dir: data/history/nightly/alt_signals/raw/
"""
from __future__ import annotations
import time
from pathlib import Path
import httpx

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "data" / "history" / "nightly" / "alt_signals" / "raw"
RAW.mkdir(parents=True, exist_ok=True)

URLS = {
    "oferta_distribuicao.zip": "https://dados.cvm.gov.br/dados/OFERTA/DISTRIB/DADOS/oferta_distribuicao.zip",
    "antt_receita_pedagio.csv": "https://dados.antt.gov.br/dataset/2e466a71-bb1b-4ecb-8f30-276af559292f/resource/4d772b06-daae-43ca-ae63-9a6f1ff66eac/download/receita_de_pedagio.csv",
    "aneel_tarifas_homologadas.csv": "https://dadosabertos.aneel.gov.br/dataset/5a583f3e-1646-4f67-bf0f-69db4203e89e/resource/fcf2906c-7c32-4b9b-a637-054e7a5234f4/download/tarifas-homologadas-distribuidoras-energia-eletrica.csv",
    "aneel_bandeiras.csv": "https://dadosabertos.aneel.gov.br/dataset/7f43a020-6dc5-44b8-80b4-d97eaa94436c/resource/0591b8f6-fe54-437b-b72b-1aa2efd46e42/download/bandeira-tarifaria-acionamento.csv",
}


def fetch(name: str, url: str, force: bool = False) -> Path:
    out = RAW / name
    if out.exists() and out.stat().st_size > 0 and not force:
        return out
    tmp = out.with_suffix(out.suffix + ".part")
    with httpx.Client(timeout=httpx.Timeout(60, read=300), follow_redirects=True) as c:
        with c.stream("GET", url) as r:
            r.raise_for_status()
            with open(tmp, "wb") as f:
                for chunk in r.iter_bytes(1 << 20):
                    f.write(chunk)
    tmp.replace(out)
    time.sleep(1)
    return out


if __name__ == "__main__":
    for n, u in URLS.items():
        try:
            p = fetch(n, u)
            print(n, p.stat().st_size)
        except Exception as e:  # noqa: BLE001
            print(n, "FAILED", type(e).__name__, str(e)[:200])
