"""Optional step - full text of a SAMPLE of Fatos Relevantes / Comunicados (CVM RAD PDFs).

No PDF library is installed (and we must not pip install), so this is a minimal pure-python PDF text extractor:
FlateDecode streams + Tj/TJ operators, literal strings decoded as cp1252, hex strings mapped with any ToUnicode CMaps
found in the file (merged). It works for most Word-generated PDFs; scanned PDFs give nothing (reported as 'empty').

Sample: universe issuers, Fato Relevante + Comunicado ao Mercado, delivered 2022-01..2025-12 (never the holdout),
stratified: every doc whose TITLE is flagged by a negative keyword class + an equal-size random draw of the rest.
Cache: data/history/nightly/text_nlp/pdf/<doc_id>.txt ; table fulltext.pkl [doc_id, n_chars, text]
"""
from __future__ import annotations

import re
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "data" / "history" / "nightly" / "text_nlp"
PDF = OUT / "pdf"
PDF.mkdir(parents=True, exist_ok=True)

_ESC = {b"n": b"\n", b"r": b"\r", b"t": b"\t", b"b": b"\b", b"f": b"\f", b"(": b"(", b")": b")", b"\\": b"\\"}


def _literal(s: bytes) -> bytes:
    out, i = bytearray(), 0
    while i < len(s):
        c = s[i:i + 1]
        if c == b"\\" and i + 1 < len(s):
            n = s[i + 1:i + 2]
            if n in _ESC:
                out += _ESC[n]; i += 2; continue
            m = re.match(rb"[0-7]{1,3}", s[i + 1:i + 4])
            if m:
                out.append(int(m.group(), 8) & 255); i += 1 + len(m.group()); continue
            i += 2; continue
        out += c; i += 1
    return bytes(out)


def _cmaps(streams: list[bytes]) -> dict:
    cm = {}
    for t in streams:
        if b"beginbf" not in t:
            continue
        for blk in re.findall(rb"beginbfchar(.*?)endbfchar", t, re.S):
            for a, b in re.findall(rb"<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>", blk):
                try:
                    cm[int(a, 16)] = bytes.fromhex(b.decode()).decode("utf-16-be", "ignore")
                except Exception:
                    pass
        for blk in re.findall(rb"beginbfrange(.*?)endbfrange", t, re.S):
            for a, b, c in re.findall(rb"<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>", blk):
                lo, hi, st = int(a, 16), int(b, 16), int(c, 16)
                for k in range(lo, min(hi, lo + 512) + 1):
                    try:
                        cm[k] = chr(st + k - lo)
                    except Exception:
                        pass
    return cm


def pdf_text(b: bytes) -> str:
    raw = re.findall(rb"stream\r?\n(.*?)\r?\nendstream", b, re.S)
    streams = []
    for s in raw:
        try:
            streams.append(zlib.decompress(s))
        except Exception:
            try:
                streams.append(zlib.decompressobj().decompress(s))
            except Exception:
                streams.append(s)
    cm = _cmaps(streams)
    out = []
    tok = re.compile(rb"\((?:\\.|[^\\)])*\)|<[0-9A-Fa-f\s]*>|-?\d+\.?\d*|T[Jj]|'|\"|T[dD*]|Tm|ET", re.S)
    for t in streams:
        if b"BT" not in t or (b"Tj" not in t and b"TJ" not in t):
            continue
        for m in tok.finditer(t):
            x = m.group()
            if x.startswith(b"("):
                out.append(_literal(x[1:-1]).decode("cp1252", "ignore"))
            elif x.startswith(b"<"):
                h = re.sub(rb"\s", b"", x[1:-1])
                if len(h) % 4 == 0 and cm:
                    out.append("".join(cm.get(int(h[i:i + 4], 16), "") for i in range(0, len(h), 4)))
                else:
                    try:
                        out.append(bytes.fromhex(h.decode()).decode("cp1252", "ignore"))
                    except Exception:
                        pass
            elif x in (b"Td", b"TD", b"T*", b"Tm", b"ET", b"'", b'"'):
                out.append(" ")
            elif re.fullmatch(rb"-\d+\.?\d*", x) and float(x) < -180:
                out.append(" ")
    txt = re.sub(r"\b(pt-BR|en-US|pt-PT|es-ES)\b", " ", "".join(out))
    txt = re.sub(r"[^\w\s.,;:%$()/\-–ºª°&]", " ", txt)
    return re.sub(r"\s+", " ", txt).strip()


def main(n_extra: int = 250, max_docs: int = 700):
    import httpx
    from research.nightly.text_nlp.classify import NEG_EVENTS, KW, fold
    d = pd.read_pickle(OUT / "docs.pkl")
    ft = fold(d["text"])
    for k in NEG_EVENTS:
        d[f"kw_{k}"] = ft.str.contains(KW[k], regex=True)
    d = d[(d["src"] == "ipe") & d["cat"].isin(["Fato Relevante", "Comunicado ao Mercado"])
          & (d["avail"] >= "2022-01-01") & (d["avail"] < "2026-01-01")]
    negf = d[[f"kw_{k}" for k in NEG_EVENTS]].any(axis=1)
    rng = np.random.default_rng(0)
    neg = d[negf]
    neg = neg.sample(min(len(neg), max_docs - n_extra), random_state=0)
    rest = d[~negf].sample(n_extra, random_state=0)
    samp = pd.concat([neg, rest]).sample(frac=1.0, random_state=1)  # interleave so a partial run is balanced
    print(f"sample {len(samp)} ({len(neg)} flagged + {len(rest)} random)", flush=True)
    rows, t0 = [], time.time()
    with httpx.Client(timeout=40, follow_redirects=True, headers={"User-Agent": "Mozilla/5.0 research"}) as cl:
        for i, r in enumerate(samp.itertuples()):
            fp = PDF / f"{r.doc_id}.txt"
            if fp.exists():
                txt = fp.read_text(encoding="utf-8")
            else:
                try:
                    resp = cl.get(r.link)
                    txt = pdf_text(resp.content) if resp.content[:4] == b"%PDF" else ""
                except Exception:
                    txt = ""
                fp.write_text(txt, encoding="utf-8")
                time.sleep(0.3)
            rows.append({"doc_id": r.doc_id, "n_chars": len(txt), "text": txt[:6000], "flagged": bool(negf.loc[r.Index])})
            if i % 50 == 0:
                print(i, f"{time.time() - t0:.0f}s", flush=True)
    ft = pd.DataFrame(rows)
    ft.to_pickle(OUT / "fulltext.pkl")
    print("non-empty (>200 chars):", (ft["n_chars"] > 200).mean().round(3))


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    main()
