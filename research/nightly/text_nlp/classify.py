"""Step 2 - embed every unique document text with a multilingual sentence-transformer and classify credit events.

  * embeddings: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 (384-d, L2-normalised), unique texts only,
    cached in data/history/nightly/text_nlp/emb.npy (+ emb_texts.pkl).
  * keyword classifier (accent-folded regex, Portuguese) -> kw_<event> booleans
  * zero-shot classifier: cosine similarity of the doc to a centroid of hand-written Portuguese prototype sentences per
    event -> zs_<event> in [-1, 1]; zs label = argmax if its cosine > ZS_T and beats the "routine" centroid.
  * sentiment axis: cos(doc, negative-credit centroid) - cos(doc, positive-credit centroid) -> sent (higher = worse)

No labels/returns are used anywhere in this file (pure text -> features), so it cannot leak the future.
Output: data/history/nightly/text_nlp/doc_feats.pkl  (docs + kw_* + zs_* + sent + zs_label)
"""
from __future__ import annotations

import re
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "data" / "history" / "nightly" / "text_nlp"
MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

KW = {
    "rj": r"recuperacao judicial|recuperacao extrajudicial|falencia|protecao (contra|de) credores|tutela cautelar|"
          r"tutela de urgencia|pedido de protecao|chapter 11|companhias em recuperacao|mediacao (com|de|junto) credores|"
          r"plano de recuperacao",
    "default_waiver": r"vencimento antecipado|inadimpl|waiver|covenant|descumpr|nao pagamento|\bdefault\b|standstill|"
                      r"perdao|dispensa de|nao declarar|indice financeiro|limite de alavancagem|nao cumprimento",
    "liab_mgmt": r"reperfil|alongamento|repactua|renegocia|reestruturacao (de|da|do|das|dos) (divida|endividamento|passivo|"
                 r"obrigac|debentures|notas)|reestruturacao financeira|liability management|troca de (divida|notas)|"
                 r"exchange offer|reescalonamento|standstill|plano de reestruturacao",
    "call_redeem": r"resgate antecipado|amortizacao extraordinaria|oferta de (resgate|recompra)|tender offer|"
                   r"recompra (de|das) (debentures|notas|bonds|titulos)|pre-?pagamento|pre pagamento|quitacao antecipada",
    "deb_holders": r"debenturistas|titulares (das|de) debentures|assembleia geral de titulares|agd\b",
    "new_debt": r"emissao de debentures|emissao de notas|notas comerciais|senior notes|\bbonds?\b|captacao|"
                r"distribuicao publica|anuncio de inicio|anuncio de encerramento|escritura|financiamento|"
                r"\bemissao\b|debentures incentivadas",
    "equity_raise": r"aumento (do|de) capital|follow.?on|oferta (publica )?(primaria|de acoes)|subscricao|aporte de capital|"
                    r"capitalizacao",
    "mna": r"aquisicao|alienacao|venda (de|da|do) (ativo|participac|controle|subsidiaria|empresa|carteira)|incorporacao|"
           r"\bfusao\b|\bcisao\b|combinacao de negocios|desinvestimento|\bopa\b|oferta publica de aquisicao|"
           r"reorganizacao societaria|memorando de entendimento|exclusividade|compra e venda|joint venture|"
           r"transferencia de controle",
    "capex": r"investimento|expansao|capex|novo projeto|construcao|leilao|arremat|vencedor|nova planta|"
             r"inicio da operacao|entrada em operacao|ampliacao",
    "guidance_cut": r"guidance|projecoes|prejuizo|impairment|baixa contabil|reapresentacao das? demonstr|reenvio|"
                    r"republicacao|inconsistencia|erro (nas|contab)|queda (do|de|da|no|na) (lucro|receita|resultado|ebitda)|"
                    r"reducao (do|de|da) (lucro|receita|guidance|producao)|ressalva|abstencao de opiniao|"
                    r"continuidade operacional",
    "dividend": r"dividend|juros sobre (o )?capital|\bjcp\b|proventos|distribuicao de lucros|bonificacao|"
                r"reducao de capital",
    "rating_down": r"rebaix|downgrade|perspectiva negativa|observacao negativa|creditwatch negativ|corta (a )?nota|"
                   r"corta (o )?rating|rating .*negativ",
    "rating_up": r"eleva (a |o )?(nota|rating|classificacao)|upgrade|perspectiva positiva|observacao positiva|"
                 r"elevacao (do|da) (rating|nota|classificacao)",
    "rating": r"rating|classificacao de risco|agencia (de )?(classificacao|rating)|fitch|moody|standard ?& ?poor|s&p",
    "mgmt_change": r"renuncia|destituicao|substituicao (do|de) (diretor|ceo|cfo|presidente)|diretor presidente|"
                   r"diretor financeiro|\bceo\b|\bcfo\b|novo presidente|eleicao de diretor",
    "oficio": r"oficio|esclarecimento|questionamento|solicitacao de (esclarecimento|informacao)|noticia veiculada|"
              r"materia veiculada|oscilacao atipica",
    "litigation": r"processo|acao (judicial|civil)|arbitra|multa|autuacao|auto de infracao|decisao judicial|liminar|"
                  r"\bstf\b|\bstj\b|\btcu\b|operacao (da )?policia|investigac|fraude|denuncia|improbidade|busca e apreensao|"
                  r"cade|condenac|penhora|bloqueio",
    "results": r"resultado|release|demonstracoes financeiras|\bitr\b|\bdfp\b|apresentacao|teleconferencia|"
               r"previa operacional|relatorio da administracao|informacoes trimestrais",
}
NEG_EVENTS = ["rj", "default_waiver", "liab_mgmt", "rating_down", "guidance_cut", "litigation", "oficio"]
POS_EVENTS = ["rating_up", "equity_raise", "new_debt"]

PROTO = {
    "rj": ["A companhia ajuizou pedido de recuperação judicial", "Pedido de recuperação extrajudicial e plano com credores",
           "Tutela cautelar para suspender cobranças de credores antes da recuperação judicial",
           "Company filed for bankruptcy protection"],
    "default_waiver": ["A companhia não pagou os juros das debêntures e há risco de vencimento antecipado",
                       "Aprovação de waiver para não declarar o vencimento antecipado por descumprimento de covenant",
                       "Descumprimento de índice financeiro de alavancagem previsto na escritura",
                       "Inadimplemento de obrigação financeira"],
    "liab_mgmt": ["Proposta de reperfilamento e alongamento do prazo das dívidas",
                  "Renegociação das condições das debêntures com os credores",
                  "Oferta de recompra e resgate antecipado de notas e debêntures",
                  "Reestruturação do endividamento da companhia"],
    "mna": ["Aquisição de participação societária em outra empresa", "Alienação de ativos e venda de controle",
            "Incorporação e reorganização societária entre empresas do grupo", "Assinatura de memorando para fusão"],
    "capex": ["Plano de investimentos e expansão da capacidade", "A companhia venceu o leilão de concessão",
              "Início da construção de novo projeto", "Entrada em operação comercial de nova usina"],
    "guidance_cut": ["Revisão para baixo do guidance e das projeções", "A companhia registrou prejuízo e queda da receita",
                     "Reconhecimento de impairment e baixa contábil", "Reapresentação das demonstrações financeiras por erro"],
    "dividend": ["Declaração de dividendos e juros sobre capital próprio", "Pagamento de proventos aos acionistas",
                 "Distribuição de dividendos intermediários"],
    "rating_down": ["Agência rebaixa o rating da companhia", "Fitch rebaixa nota de crédito e coloca perspectiva negativa",
                    "Moody's coloca rating em observação negativa"],
    "rating_up": ["Agência eleva o rating da companhia", "S&P eleva nota de crédito com perspectiva positiva",
                  "Fitch eleva classificação de risco"],
    "equity_raise": ["Aumento de capital com oferta primária de ações", "Aporte de capital pelos acionistas controladores",
                     "Follow-on para reforçar a estrutura de capital"],
    "new_debt": ["Emissão de debêntures simples para captação de recursos", "Captação de financiamento de longo prazo",
                 "Anúncio de início da distribuição pública de debêntures"],
    "litigation": ["Decisão judicial desfavorável e multa milionária", "Operação da Polícia Federal investiga fraude na empresa",
                   "Processo administrativo sancionador e autuação fiscal", "Condenação em ação civil pública"],
    "mgmt_change": ["Renúncia do diretor presidente", "Eleição de novo diretor financeiro", "Troca de CEO da companhia"],
    "oficio": ["Esclarecimentos sobre notícia veiculada na imprensa em resposta a ofício da B3",
               "Resposta a ofício da CVM sobre oscilação atípica"],
    "routine": ["Ata da reunião do conselho de administração", "Aprovação das demonstrações financeiras trimestrais",
                "Edital de convocação da assembleia geral ordinária", "Apresentação de resultados trimestrais",
                "Remuneração dos administradores", "Eleição de membros do conselho"],
}
SENT_NEG = PROTO["rj"] + PROTO["default_waiver"] + PROTO["rating_down"] + PROTO["guidance_cut"] + PROTO["litigation"] + [
    "Crise financeira e dificuldades de liquidez", "Calote e perdas para credores", "Queda forte das ações após notícia negativa"]
SENT_POS = PROTO["rating_up"] + PROTO["equity_raise"] + [
    "Lucro recorde e forte geração de caixa", "Redução da alavancagem e desalavancagem", "Resultado acima do esperado",
    "Melhora da liquidez e alongamento do perfil da dívida com custo menor"]
ZS_T = 0.55
EMB_CATS = ["Fato Relevante", "Comunicado ao Mercado", "Aviso aos Debenturistas", "Escrituras e aditamentos de debêntures",
            "Informações de Companhias em Recuperação Judicial ou Extrajudicial", "Comunicação sobre demandas societárias",
            "Informação Prestada às Bolsas Estrangeiras", "Aviso aos Acionistas",
            "Comunicação sobre Transação entre Partes Relacionadas"]


def fold(s: pd.Series) -> pd.Series:
    return s.fillna("").map(lambda x: unicodedata.normalize("NFKD", x).encode("ascii", "ignore").decode().lower())


def embed(texts: list[str], batch=128) -> np.ndarray:
    import torch
    torch.set_num_threads(2)
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(MODEL, device="cpu")
    m.max_seq_length = 128
    return m.encode(texts, batch_size=batch, normalize_embeddings=True, show_progress_bar=False,
                    convert_to_numpy=True).astype(np.float32)


def get_embeddings(texts: list[str]) -> tuple[np.ndarray, dict]:
    ep, tp = OUT / "emb.npy", OUT / "emb_texts.pkl"
    if ep.exists() and tp.exists():
        E, T = np.load(ep), pd.read_pickle(tp)
        idx = {t: i for i, t in enumerate(T)}
        miss = [t for t in texts if t not in idx]
    else:
        E, T, idx, miss = np.zeros((0, 384), np.float32), [], {}, list(texts)
    if miss:
        t0 = time.time()
        print(f"embedding {len(miss)} texts ...", flush=True)
        # sort by length for speed
        miss = sorted(set(miss), key=len)
        T = list(T)
        for i in range(0, len(miss), 4096):
            part = miss[i:i + 4096]
            E = np.vstack([E, _enc(part)])
            T += part
            np.save(ep, E)
            pd.to_pickle(T, tp)
            print(f"  {i + len(part)}/{len(miss)}  {time.time() - t0:.0f}s", flush=True)
        idx = {t: i for i, t in enumerate(T)}
    return E, idx


_M = {}


def _enc(texts):
    if "m" not in _M:
        import torch
        torch.set_num_threads(2)
        from sentence_transformers import SentenceTransformer
        _M["m"] = SentenceTransformer(MODEL, device="cpu")
        _M["m"].max_seq_length = 64
    return _M["m"].encode(texts, batch_size=128, normalize_embeddings=True, show_progress_bar=False,
                          convert_to_numpy=True).astype(np.float32)


def centroid(sents):
    v = _enc(sents).mean(0)
    return v / np.linalg.norm(v)


def main():
    docs = pd.read_pickle(OUT / "docs.pkl")
    # the embedded text: for IPE drop the boiler-plate category prefix when an Assunto exists
    docs["etext"] = np.where(docs["assunto"].str.len() > 3,
                             (docs["tipo"].where(docs["src"] == "ipe", "") + ": " + docs["assunto"]).str.strip(": "),
                             docs["text"]).astype(str)
    docs["etext"] = docs["etext"].str.slice(0, 400)
    # IPE: collapse numbers (quarters, issue numbers, dates) -> far fewer unique texts, same semantics
    docs.loc[docs["src"] == "ipe", "etext"] = docs.loc[docs["src"] == "ipe", "etext"].str.replace(r"\d+", "0", regex=True)
    # only embed news + credit-relevant IPE categories (routine governance minutes/AGMs/financial statements are
    # still keyword-classified, but not embedded: CPU budget on a shared machine)
    emb_ok = (docs["src"] == "news") | docs["cat"].isin(EMB_CATS)
    uniq = docs.loc[emb_ok, "etext"].unique().tolist()
    E, idx = get_embeddings(uniq)
    rows_all = np.array([idx.get(t, -1) for t in docs["etext"]])
    rows_all[~emb_ok.to_numpy()] = -1
    has = rows_all >= 0
    rows = np.where(has, rows_all, 0)
    # keyword classes on folded full text
    ft = fold(docs["text"])
    for k, pat in KW.items():
        docs[f"kw_{k}"] = ft.str.contains(pat, regex=True)
    # rating mentions only count as up/down when a direction word is present; plain "rating" kept separately
    # zero-shot
    names = list(PROTO)
    C = np.vstack([centroid(PROTO[k]) for k in names])
    S = E @ C.T                                   # unique-text x classes
    for j, k in enumerate(names):
        docs[f"zs_{k}"] = np.where(has, S[rows, j], np.nan).astype(np.float32)
    lab = np.array(names)[S.argmax(1)]
    ok = (S.max(1) > ZS_T) & (lab != "routine")
    docs["zs_label"] = np.where(has & ok[rows], lab[rows], "none")
    neg, pos = centroid(SENT_NEG), centroid(SENT_POS)
    docs["sent"] = np.where(has, E[rows] @ neg - E[rows] @ pos, np.nan).astype(np.float32)
    docs["emb_row"] = rows_all
    docs.drop(columns=["etext"]).to_pickle(OUT / "doc_feats.pkl")
    # quick diagnostics
    kwc = [c for c in docs if c.startswith("kw_")]
    print(docs.groupby("src")[kwc].mean().T.round(3))
    print(docs.groupby("src")["zs_label"].value_counts().unstack(0).fillna(0).astype(int))
    agree = {}
    for k in names:
        if f"kw_{k}" in docs:
            z = docs["zs_label"] == k
            kk = docs[f"kw_{k}"]
            agree[k] = {"kw": int(kk.sum()), "zs": int(z.sum()), "both": int((z & kk).sum()),
                        "zs_precision_vs_kw": round(float((z & kk).sum() / max(z.sum(), 1)), 3)}
    print(pd.DataFrame(agree).T)
    pd.to_pickle({"agree": agree, "ZS_T": ZS_T, "model": MODEL}, OUT / "classify_meta.pkl")
    # show the most negative / positive sentiment examples for a sanity check
    for src in ("ipe", "news"):
        d = docs[docs["src"] == src].drop_duplicates("text")
        print(f"\n--- {src}: most negative")
        print(d.nlargest(8, "sent")[["sent", "text"]].to_string(max_colwidth=110))
        print(f"--- {src}: most positive")
        print(d.nsmallest(8, "sent")[["sent", "text"]].to_string(max_colwidth=110))


if __name__ == "__main__":
    main()
