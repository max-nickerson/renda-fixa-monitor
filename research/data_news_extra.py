"""Extra POINT-IN-TIME news events (2021-01 → today) for the debenture study.

    python research/data_news_extra.py ratings     # Google News rating headlines per brand (cached)
    python research/data_news_extra.py wayback     # Wayback CDX of fitchratings.com/research/pt/* (cached pages)
    python research/data_news_extra.py regulators  # ANEEL/ANTT/ANP/ANA/ARSESP/ANAC decisions per month
    python research/data_news_extra.py sector      # negative sector/commodity headline volume per month
    python research/data_news_extra.py build       # parse caches → data/history/{rating_events,regulator_events,sector_news}.pkl
    python research/data_news_extra.py all

Date semantics: Google News historical <pubDate> carries the publication DATE (time is a placeholder) → an item is
known at the END of that date. Fitch slugs end in -dd-mm-yyyy = publication date of the action (used as the event
date); the Wayback capture timestamp is only an upper bound on when the page existed (kept as `captured`).
All raw downloads are cached under data/history/news_extra_raw/ so re-runs only fetch missing ranges.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import time
import unicodedata
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rfmonitor.config import DATA_DIR  # noqa: E402
from rfmonitor.http import get  # noqa: E402
from rfmonitor.ml.press import _fetch  # noqa: E402

H = DATA_DIR / "history"
RAW = H / "news_extra_raw"
START, END = date(2021, 1, 1), date.today() + timedelta(days=1)  # END exclusive
SLEEP = 1.0


def norm(s: str) -> str:
    s = (s or "").replace("’", "'").replace("‘", "'")
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()


def add_months(m: date, n: int) -> date:
    k = m.month - 1 + n
    return date(m.year + k // 12, k % 12 + 1, 1)


def periods(step_months: int):
    m = START
    while m < END:
        yield m, min(add_months(m, step_months), END)
        m = add_months(m, step_months)


# ------------------------------------------------------------------ Google News (own cache)
def gnews(query: str, a: date, b: date, split_at: int = 95) -> list[dict]:
    """Items dated in [a, b). Cached under news_extra_raw/gnews. If ≥split_at items → split (months, then weeks)."""
    d = RAW / "gnews"
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{hashlib.sha1(query.encode()).hexdigest()[:12]}_{a:%Y%m%d}_{b:%Y%m%d}.json"
    fresh = b <= date.today() - timedelta(days=7)
    if path.exists() and fresh:
        return json.loads(path.read_text(encoding="utf-8"))
    try:
        items = _fetch(query, a - timedelta(days=1), b)
    except Exception as e:  # noqa: BLE001
        print(f"  ! gnews fail {a}..{b} {query[:60]}: {e}", flush=True)
        return []
    time.sleep(SLEEP)
    items = [i for i in items if a.isoformat() <= i["date"] < b.isoformat()]
    if len(items) >= split_at and (b - a).days > 7:
        subs = []
        if (b - a).days > 40:
            m = a
            while m < b:
                subs.append((m, min(add_months(m, 1), b)))
                m = add_months(m, 1)
        else:
            m = a
            while m < b:
                subs.append((m, min(m + timedelta(days=7), b)))
                m += timedelta(days=7)
        seen, out = set(), []
        for sa, sb in subs:
            for it in gnews(query, sa, sb, split_at):
                k = (it["date"], it["title"])
                if k not in seen:
                    seen.add(k)
                    out.append(it)
        items = out
    path.write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")
    return items


# ------------------------------------------------------------------ 1. RATINGS
SKIP = {"Acucareira", "Aeroporto", "Aguas", "Amazonense", "Araguaia", "Bolsa", "Cli Sul", "Colombo", "Complexo",
        "Concessao", "Construtora", "Cutia", "Delta", "Distribuicao", "Eletric.", "Empreendimentos", "Energetica",
        "Goias", "Guaraciaba", "Helio", "Horizon", "Hospital", "Hospitalar", "Instituicao", "Integradas", "Interior",
        "International", "Jose Maria", "Malls", "Manaus", "Mata Santa", "Matrincha", "Minas", "Modas", "Norte",
        "Opea Securitizadora", "Paranagua", "Paranaiba", "Potengi", "Publicos", "Rda Importacao", "Rede Neutra",
        "Regenera", "Rio Parana", "Rio Paranapanema", "Sao Manoel", "Securitizadora", "Siderurgica", "Sistema",
        "Tecnologia", "Terminal", "Transportadora", "Transportes", "Travessia", "Tvv Terminal", "Uniao", "Usina",
        "V2I", "Vale Sertao", "Ventos", "Vert Securitizadora", "Viario", "Auto Raposo", "Cea"}
ALIAS = {  # brand → up to 2 search/match names (first = primary)
    "Axia": ["Eletrobras", "Axia Energia"], "Axia Sul": ["Eletrosul", "Axia Sul"], "Belo Monte": ["Norte Energia"],
    "Brk Ambiental": ["BRK Ambiental", "BRK"], "Btg Pactual": ["BTG Pactual", "BTG"], "Ceee": ["CEEE"],
    "Comgas": ["Comgás"], "Cpfl Comercializacao": ["CPFL"], "Cpfl Transmissao": ["CPFL"],
    "Csn Mineracao": ["CSN Mineração", "CSN"], "Db3 Telecomunicacoes": ["DB3"], "Diagnosticos": ["Dasa"],
    "Edp Espirito": ["EDP"], "Edp Sao": ["EDP"], "Eletropaulo": ["Enel SP", "Eletropaulo"], "Enel Green": ["Enel"],
    "Gas Minas": ["Gasmig"], "Guararapes": ["Guararapes", "Riachuelo"], "Hidrovias": ["Hidrovias do Brasil"],
    "Igua Rio": ["Iguá"], "Igua S.A": ["Iguá"], "Iochpe-Maxion": ["Iochpe", "Maxion"], "Irb Resseguros": ["IRB"],
    "Isa": ["ISA CTEEP", "ISA Energia"], "Itapoa": ["Itapoá"], "Itausa": ["Itaúsa"], "Janauba": ["Janaúba"],
    "Jsl S.A": ["JSL"], "Log Commercial": ["LOG Commercial"], "Motiva": ["Motiva", "CCR"], "Mrs Logistica": ["MRS Logística"],
    "Mrv Engenharia": ["MRV"], "Origem": ["Origem Energia"], "Paulista": ["CPFL Paulista"],
    "Petroreconcavo": ["PetroReconcavo"], "Piratininga": ["CPFL Piratininga"], "Prio Forte": ["Prio", "PetroRio"],
    "Raia Drogasil": ["Raia Drogasil", "RD Saúde"], "Raizen": ["Raízen"], "Randoncorp": ["Randon"],
    "Rge Sul": ["RGE"], "Rota Bandeiras": ["Rota das Bandeiras"], "Rumo Malha": ["Rumo"], "Santos": ["Santos Brasil"],
    "Sao Martinho": ["São Martinho"], "Sbf Comercio": ["Grupo SBF", "Centauro"], "Sendas": ["Assaí", "Sendas"],
    "Smartfit": ["Smart Fit", "Smartfit"], "Solvi": ["Solví"], "Tag": ["TAG"], "Telefonica": ["Telefônica", "Vivo"],
    "Tim S.A": ["TIM"], "Vix Logistica": ["Vix Logística"], "Vli Multimodal": ["VLI"], "Zamp": ["Zamp"],
    "Argo Transmissao": ["Argo Energia", "Argo Transmissão"], "Aura Almas": ["Aura Minerals"], "Gsh Corp": ["GSH"],
    "Wiz Solucoes": ["Wiz"], "Rede D'Or": ["Rede D'Or"], "Anima": ["Ânima"], "Colinas": ["Rodovias das Colinas"],
    "Assurua": ["Assuruá"], "Equipav": ["Equipav"], "Energisa": ["Energisa"],
}
RAT_TERMS = 'rebaixa OR eleva OR rating OR Fitch OR "Moody\'s" OR "S&P"'


def brand_names() -> dict[str, list[str]]:
    brands = json.loads((ROOT / "research/out/press_brands.json").read_text(encoding="utf-8"))
    return {b: ALIAS.get(b, [b]) for b in sorted(set(brands.values())) if b and b not in SKIP}


def rating_query(names: list[str]) -> str:
    nm = " OR ".join(f'"{n}"' for n in names)
    return f"({nm}) ({RAT_TERMS})" if len(names) > 1 else f"{nm} ({RAT_TERMS})"


def fetch_ratings():
    qs = sorted({rating_query(v) for v in brand_names().values()})
    print(f"ratings: {len(qs)} queries x years (split when busy)", flush=True)
    part, nparts = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else (0, 1)
    qs = qs[part::nparts]
    for k, q in enumerate(qs):
        n = 0
        for a, b in periods(12):  # yearly; relevance-capped results → halves → quarters raise recall
            items = gnews(q, a, b)
            n += len(items)
            if len(items) < 6:
                continue
            for ha in (a, add_months(a, 6)):
                hb = min(add_months(ha, 6), b)
                if ha >= b or len(gnews(q, ha, hb)) < 6:
                    continue
                for qa in (ha, add_months(ha, 3)):
                    if qa < hb:
                        gnews(q, qa, min(add_months(qa, 3), hb))
        print(f"  [{k + 1}/{len(qs)}] {q[:50]} {n}", flush=True)


AGENCIES = [("Fitch", r"\bfitch"), ("Moody's", r"\bmoody"), ("S&P", r"s ?& ?p\b|standard ?& ?poor"),
            ("Austin", r"austin rating"), ("Liberum", r"\bliberum")]
RATING_CTX = (r"\brating|nota de credito|nota de risco|notas? da|classificacao de risco|grau de investimento|"
              r"downgrade|upgrade|rebaix|perspectiva|outlook|\bidr|\(bra\)|br\.|escala nacional|observacao")
VERB_DOWN = (r"rebaix|\bcorta|\bcortou|\bcortam|\bcorte (na|da|em|de) (sua |a )?(nota|rating)|downgrad|\blowers?\b|"
             r"\bcuts?\b|reduz[a-z]* (a |o |as |os |sua |seu |suas |seus )?(nota|rating)|perde grau de investimento|"
             r"perda do grau de investimento|default restrito|para 'd'|'rd'")
VERB_UP = (r"\beleva(m|r|do|da|dos|das|u)?\b|\belevou\b|\bsobe\b|upgrad|\braises?\b|melhora (a |o |as |os |sua |seu )?"
           r"(nota|rating)|elevacao (de|do|da|das|dos) (rating|nota)|(recupera|conquista|obtem|ganha|volta ao|retoma) (o )?"
           r"grau de investimento")
UP_TRANS = (r"(negativ[ao]|observacao negativa) (para|to) (estavel|stable|positiv)|estavel para positiva|"
            r"stable to positive|negative to stable|\b(re)?tira\b.*observacao negativa|removes? .*negative watch")
DOWN_TRANS = r"(estavel|stable|positiv[ao]) (para|to) negativ|positiva para estavel|positive to stable"
OUTLOOK_DOWN = (r"observacao negativa|implicacoes negativas|watch negative|negative watch|perspectiva (e |para |de )?"
                r"negativa|\bpara negativa|to negative|outlook (to )?negative|negative outlook|"
                r"revisao para (possivel )?rebaix|\bpiora")
OUTLOOK_UP = (r"perspectiva (e |para |de )?positiva|\bpara positiva|observacao positiva|positive outlook|to positive|"
              r"outlook (to )?positive|positive watch|watch positive|revisao para (possivel )?elevacao")


FOREIGN = (r"peru|chile|colombia|mexic|argentin|russia|spain|espana|italia|italy|idesa|uruguay|paraguay|ecuador|"
           r"bolivia|guatemala|panama|costa.rica|\bindia|moviles|irb.infra|germany|deutschland|\buk\b|reino unido|"
           r"portugal|connecticut|florida|light.power|power.light|nextera|\beua\b")
# equity-research up/downgrades and ESG/sustainability rankings are not credit-rating actions
NOT_CREDIT = (r"\bbofa\b|j\.? ?p\.? ?morgan|goldman|itau bba|\bbtg (eleva|rebaixa|corta|da|faz)|\bxp (eleva|rebaixa|corta)|"
              r"\bsafra\b|\bciti\b|morgan stanley|\bubs\b|bradesco bbi|jefferies|barclays|\bhsbc\b|recomendac|"
              r"preco.alvo|\bacoes\b.*\b(compra|venda)\b|sustentav|sustainability|yearbook|\besg\b|indice s&p|"
              r"s&p 500|vale a pena|vale o |vale mais")
CREDIT_CTX = r"\brating|nota de credito|notas de credito|classificacao de risco|grau de investimento|agencias? de"
BROKER_CALL = r"para (neutr|compra|venda|underperform|outperform|market)|\b(santander|bradesco|itau|genial|ativa)\b.*rebaix"
COMMON_WORD_BRANDS = {"Valid", "Vale", "Telefonica", "Claro", "Tim S.A", "Light", "Tag", "Origem", "Vero", "Delta", "Isa"}


def _first(pd_, pu):
    return (-1 if pd_.start() < pu.start() else 1) if pd_ and pu else (-1 if pd_ else (1 if pu else 0))


def classify_rating(title: str) -> int:
    """Action verbs (rebaixa/corta/eleva…) win; else outlook transitions; else outlook/watch level words."""
    t = norm(title)
    v = _first(re.search(VERB_DOWN, t), re.search(VERB_UP, t))
    if v:
        return v
    if re.search(UP_TRANS, t):
        return 1
    if re.search(DOWN_TRANS, t):
        return -1
    return _first(re.search(OUTLOOK_DOWN, t), re.search(OUTLOOK_UP, t))


def agency_of(title: str) -> str:
    t = norm(title)
    return next((a for a, p in AGENCIES if re.search(p, t)), "")


def _name_rx(n: str) -> str:
    return r"(?<![a-z0-9])" + re.escape(norm(n)) + r"(?![a-z0-9])"


def rating_items() -> pd.DataFrame:
    names = brand_names()
    rows = []
    for b, nm in names.items():
        q = rating_query(nm)
        rx = "|".join(_name_rx(n) for n in nm)
        seen = set()
        for p in (RAW / "gnews").glob(f"{hashlib.sha1(q.encode()).hexdigest()[:12]}_*.json"):
            for it in json.loads(p.read_text(encoding="utf-8")):
                k = (it["date"], it["title"])
                if k in seen:
                    continue
                seen.add(k)
                head = re.sub(r"\s+-\s+[^-]+$", "", it["title"])  # drop ' - Outlet'
                t = norm(head)
                if not re.search(rx, t) or not (agency_of(head) or re.search(RATING_CTX, t)) or re.search(FOREIGN, t):
                    continue
                if re.search(NOT_CREDIT, t) or (b in COMMON_WORD_BRANDS and not agency_of(head)):
                    continue
                if not agency_of(head) and (not re.search(CREDIT_CTX, t) or re.search(BROKER_CALL, t)):
                    continue  # no agency named → need explicit credit-rating wording (drops equity 'rebaixa p/ neutro')
                rows.append({"brand": b, "date": it["date"], "direction": classify_rating(head),
                             "agency": agency_of(head), "title": it["title"], "source": "gnews"})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ 1b. Wayback: Fitch PT research slugs
def _slug(n: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", norm(n)).strip("-")


WAYBACK_SETS = [("fitchratings.com/research/pt/", "fitch_pt", False),
                ("fitchratings.com/research/corporate-finance/", "fitch_cf", True),
                ("fitchratings.com/research/infrastructure-project-finance/", "fitch_ipf", True)]


def fetch_wayback_all():
    for prefix, tag, filt in WAYBACK_SETS:
        fetch_wayback(prefix, tag, filt)


def fetch_wayback(prefix: str = "fitchratings.com/research/pt/", tag: str = "fitch_pt", brand_filter: bool = False):
    d = RAW / "wayback"
    d.mkdir(parents=True, exist_ok=True)
    key, page = None, 0
    while True:
        path = d / f"{tag}_{page:04d}.txt"
        if path.exists():
            lines = path.read_text(encoding="utf-8").splitlines()
        else:
            url = (f"https://web.archive.org/cdx/search/cdx?url={prefix}&matchType=prefix&collapse=urlkey"
                   f"&fl=timestamp,original&limit=3000&showResumeKey=true&from=2020")
            if brand_filter:  # server-side regex: only slugs naming a covered brand
                alt = "|".join(sorted({_slug(n) for v in brand_names().values() for n in v}))
                url += "&filter=original:.*-(" + alt.replace("|", "%7C") + ")s?-.*"
            if key:
                url += f"&resumeKey={key}"
            txt = None
            for k in range(6):
                try:
                    r = get(url, retries=0, timeout=180)
                    if r.status_code == 200:
                        txt = r.text
                        break
                    print(f"  wayback {r.status_code}, backoff", flush=True)
                except Exception as e:  # noqa: BLE001
                    print(f"  wayback err {e}, backoff", flush=True)
                time.sleep(20 * (k + 1))
            if txt is None:
                print("  wayback: giving up", flush=True)
                return
            path.write_text(txt, encoding="utf-8")
            lines = txt.splitlines()
            time.sleep(5)
        # resume key = last non-empty line after a blank line
        if "" in lines and lines[-1].strip() and len(lines[-1].split()) == 1:
            key = lines[-1].strip()
            page += 1
            print(f"  wayback page {page} ({len(lines)} lines)", flush=True)
        else:
            print(f"  wayback done: {page + 1} pages", flush=True)
            return


SLUG_DOWN = (r"rebaix|downgrad|para-negativa|perspectiva-negativa|observacao-negativa|to-negative|negative-outlook|"
             r"watch-negative|negative-watch|rating-watch-negative|corta|lowers|cuts")
SLUG_UP = (r"eleva-|upgrad|para-positiva|perspectiva-positiva|observacao-positiva|to-positive|positive-outlook|"
           r"watch-positive|positive-watch|raises")
SLUG_ACTION = SLUG_DOWN + "|" + SLUG_UP + r"|afirma|affirm|atribui|assign|revisa|revises|coloca|places|retira|withdraw|rates"


def wayback_items() -> pd.DataFrame:
    names = brand_names()
    rows, seen = [], set()
    for p in sorted((RAW / "wayback").glob("fitch_*.txt")):
        for ln in p.read_text(encoding="utf-8").splitlines():
            parts = ln.split(" ", 1)
            if len(parts) != 2:
                continue
            ts, url = parts
            slug = norm(re.sub(r"[?#].*$", "", url).rstrip("/").split("/")[-1])
            m = re.search(r"-(\d{2})-(\d{2})-(20\d{2})$", slug)
            if not m or not re.search(SLUG_ACTION, slug):
                continue
            try:
                dt = date(int(m.group(3)), int(m.group(2)), int(m.group(1)))
            except ValueError:
                continue
            if dt < START or slug in seen or re.search(FOREIGN, slug):
                continue
            seen.add(slug)
            if re.search(r"(negativ[ao]|negative)-(para|to)-(estavel|stable)|estavel-para-positiva|stable-to-positive", slug):
                dr = 1
            elif re.search(r"(estavel|stable)-(para|to)-negativ", slug):
                dr = -1
            else:
                dd, uu = re.search(SLUG_DOWN, slug), re.search(SLUG_UP, slug)
                dr = (-1 if dd.start() < uu.start() else 1) if dd and uu else (-1 if dd else (1 if uu else 0))
            text = "-" + slug + "-"
            for b, nm in names.items():
                if any(re.search("-" + re.escape(_slug(n)) + "s?-", text) for n in nm):
                    rows.append({"brand": b, "date": dt.isoformat(), "direction": dr, "agency": "Fitch",
                                 "title": slug, "source": "wayback_fitch", "captured": ts[:8]})
    return pd.DataFrame(rows)


def build_ratings() -> pd.DataFrame:
    brands = json.loads((ROOT / "research/out/press_brands.json").read_text(encoding="utf-8"))
    it = pd.concat([rating_items(), wayback_items()], ignore_index=True)
    if it.empty:
        return it
    it["date"] = pd.to_datetime(it["date"])
    it = it.sort_values(["brand", "date"]).reset_index(drop=True)
    # collapse repeated coverage: same brand/direction/agency within 5 days → one event (earliest date)
    keep, n_art = [], []
    for (_, _, _), g in it.groupby(["brand", "direction", "agency"], sort=False):
        last = None
        for i, r in g.iterrows():
            if last is not None and (r["date"] - it.at[last, "date"]).days <= 5:
                n_art[-1] += 1
                if r["source"] not in it.at[last, "source"]:
                    it.at[last, "source"] += "+" + r["source"]
                continue
            keep.append(i)
            n_art.append(1)
            last = i
    ev = it.loc[keep].copy()
    ev["n_articles"] = n_art
    # unknown-agency events within 5 days of a known-agency event of same brand/direction are duplicates
    known = ev[ev["agency"] != ""]
    drop = []
    for i, r in ev[ev["agency"] == ""].iterrows():
        k = known[(known["brand"] == r["brand"]) & (known["direction"] == r["direction"])]
        if ((k["date"] - r["date"]).abs().dt.days <= 5).any():
            drop.append(i)
    ev = ev.drop(index=drop)
    inv = {}
    for c, b in brands.items():
        inv.setdefault(b, []).append(c)
    out = ev.assign(cnpj8=ev["brand"].map(inv)).explode("cnpj8")
    cols = ["cnpj8", "brand", "date", "direction", "agency", "title", "source", "n_articles", "captured"]
    return out[cols].sort_values(["date", "cnpj8"]).reset_index(drop=True)


# ------------------------------------------------------------------ 2. REGULATORS
REG_Q = {
    "aneel": ['ANEEL (multa OR caducidade OR intervenção OR "revisão tarifária" OR penalidade OR suspende OR fiscalização)',
              'ANEEL (aprova OR reajuste OR "reajuste tarifário" OR autoriza OR prorrogação OR renovação)'],
    "antt": ['ANTT (multa OR caducidade OR intervenção OR "revisão tarifária" OR penalidade OR suspende OR "reequilíbrio")',
             'ANTT (aprova OR reajuste OR pedágio OR autoriza OR prorrogação OR renovação)'],
    "anp": ['ANP (multa OR interdição OR suspende OR autuação OR penalidade OR cassa OR revoga)',
            'ANP (aprova OR autoriza OR leilão OR concede OR reajuste OR libera)'],
    "ana": ['"Agência Nacional de Águas" (multa OR norma OR tarifa OR regulação OR aprova OR saneamento)'],
    "arsesp": ['ARSESP (Sabesp OR tarifa OR reajuste OR multa OR revisão OR aprova OR Comgás)'],
    "anac": ['ANAC (multa OR suspende OR penalidade OR cassa OR interdição OR autuação)',
             'ANAC (aprova OR autoriza OR concessão OR reajuste OR libera)'],
}
REG_MATCH = {"aneel": r"\baneel\b", "antt": r"\bantt\b", "anp": r"\banp\b", "ana": r"agencia nacional de aguas|\bana\b",
             "arsesp": r"\barsesp\b", "anac": r"\banac\b"}
REG_NEG = (r"\bmulta|multad|caducidade|interven|suspen|penalidad|\bpune|puni[cr]|autua|interdi|\bcassa|revoga|"
           r"rejeit|\bnega\b|negou|indefer|proib|descumpr|infra[cç]|reajuste negativo|revisao tarifaria negativa|"
           r"reduz[a-z]* (a |as )?(tarifa|conta|pedagio)|reducao (da|na|de) (tarifa|conta|pedagio)|"
           r"(tarifa|conta|pedagio)s? (mais barat|menor|cai|caem|recua)|queda (da|na) (tarifa|conta)|"
           r"ameaca|irregular|investiga|processo contra|sancao|sancion")
REG_POS = (r"aprova|autoriza|reajuste|aumento|aumenta|\bsobe|\bsobem|mais car|\beleva|prorroga|renova|reequilibrio|"
           r"concede|libera|revisao tarifaria")


def reg_sector(reg: str, t: str) -> list[str]:
    if reg == "aneel":
        return ["utilities"]
    if reg == "antt":
        if re.search(r"ferrov|\btrem|\btrens|malha|\brumo\b|\bmrs\b|\bvli\b|trilho", t):
            return ["railways"]
        if re.search(r"rodov|pedag|\bbr-?\d|concessionaria|estrada|via ", t):
            return ["toll_roads"]
        return ["toll_roads", "railways"]
    return {"anp": ["oil_gas"], "ana": ["sanitation"], "arsesp": ["sanitation"], "anac": ["airlines"]}[reg]


def classify_reg(title: str) -> int:
    t = norm(title)
    if re.search(REG_NEG, t):
        return -1
    return 1 if re.search(REG_POS, t) else 0


def fetch_regulators():
    for reg, qs in REG_Q.items():
        for q in qs:
            n = sum(len(gnews(q, a, b)) for a, b in periods(1))
            print(f"  {reg}: {q[:60]} → {n}", flush=True)


def build_regulators() -> pd.DataFrame:
    rows, seen = [], set()
    for reg, qs in REG_Q.items():
        for q in qs:
            for a, b in periods(1):
                for it in gnews(q, a, b):
                    head = re.sub(r"\s+-\s+[^-]+$", "", it["title"])
                    t = norm(head)
                    k = (reg, t)
                    if k in seen or not re.search(REG_MATCH[reg], t):
                        continue
                    seen.add(k)
                    if reg == "arsesp" and re.search(r"\bcomgas|\bgas\b", t):
                        continue  # ARSESP also regulates piped gas; keep sanitation only
                    for sg in reg_sector(reg, t):
                        rows.append({"sector_group": sg, "regulator": reg, "date": it["date"],
                                     "direction": classify_reg(head), "title": it["title"], "source": "gnews"})
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["date", "regulator"]).reset_index(drop=True)


# ------------------------------------------------------------------ 3. SECTOR NEWS
SEC_Q = {
    "fertilizantes": "(crise OR queda) fertilizantes", "ureia": "(crise OR queda) ureia",
    "gas": '(crise OR queda) "gás natural"', "etanol": "(crise OR queda) etanol", "acucar": "(crise OR queda) açúcar",
    "setor_eletrico": '(crise OR queda) ("setor elétrico" OR PLD)',
    "reservatorios": "(crise OR queda OR seca) (reservatórios OR hidrelétricas)",
    "aereas": '(crise OR queda) (aéreas OR "companhias aéreas")', "locadoras": "(crise OR queda) locadoras",
    "varejo": "(crise OR queda) varejo", "construtoras": "(crise OR queda) construtoras",
    "agro": '(crise OR queda OR "recuperação judicial") agronegócio', "frigorificos": "(crise OR queda) frigoríficos",
    "siderurgia": "(crise OR queda) siderurgia", "celulose": "(crise OR queda) celulose",
    "saneamento": "(crise OR queda) saneamento",
}
SEC_NEG = (r"crise|queda|derrub|estiagem|reduz|escass|\bcai\b|caem|caiu|recua|recuo|despenc|tomb|prejuizo|recupera(cao|coes) judicia|falencia|seca|"
           r"escassez|colapso|\brisco|inadimpl|demiss|fecha|perda|\bpior|baixa|retrac|desacelera|apagao|racionamento|"
           r"deficit|alerta|dificuldade|endivid|calote|rebaix|pressao|falta de")


def fetch_sector():
    order = list(SEC_Q.items())
    if len(sys.argv) > 2 and sys.argv[2] == "rev":  # lets a second worker start from the other end
        order = order[::-1]
    for topic, q in order:
        n = sum(len(gnews(q, a, b)) for a, b in periods(1))
        print(f"  {topic}: {n}", flush=True)


def build_sector() -> pd.DataFrame:
    rows = []
    for topic, q in SEC_Q.items():
        seen = set()
        for a, b in periods(1):
            for it in gnews(q, a, b):
                k = (it["date"], it["title"])
                if k in seen:
                    continue
                seen.add(k)
                head = re.sub(r"\s+-\s+[^-]+$", "", it["title"])
                t = re.sub(r"crise climatica|mudancas? climaticas?", "", norm(head))
                rows.append({"topic": topic, "date": it["date"], "neg": bool(re.search(SEC_NEG, t))})
    it = pd.DataFrame(rows)
    it["date"] = pd.to_datetime(it["date"])
    df = it.groupby(["topic", "date"]).agg(n_items=("neg", "size"), n_negative=("neg", "sum")).reset_index()
    df["query"] = df["topic"].map(SEC_Q)
    df.attrs["queries"] = SEC_Q
    return df


def build():
    r = build_ratings()
    r.to_pickle(H / "rating_events.pkl")
    print(f"rating_events: {len(r)} rows, {r['cnpj8'].nunique()} issuers", flush=True)
    g = build_regulators()
    g.to_pickle(H / "regulator_events.pkl")
    print(f"regulator_events: {len(g)} rows", flush=True)
    s = build_sector()
    s.to_pickle(H / "sector_news.pkl")
    print(f"sector_news: {len(s)} rows, {s['n_items'].sum()} items", flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd in ("ratings", "all"):
        fetch_ratings()
    if cmd in ("wayback", "all"):
        fetch_wayback_all()
    if cmd in ("regulators", "all"):
        fetch_regulators()
    if cmd in ("sector", "all"):
        fetch_sector()
    if cmd in ("build", "all"):
        build()
