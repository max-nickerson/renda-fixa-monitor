"""Step 3 - point-in-time issuer-level text features + reusable signal.

For a decision at the close of day d, a document counts only if avail < d (avail <= d-1).

signals(panel) -> DataFrame [cnpj8, day, <features>] aligned to panel rows (by cnpj8 & day).
Feature families (windows in calendar days):
  ipe_<ev>_<w>   count of IPE filings flagged by keyword class <ev> in (d-w, d-1]
  news_<ev>_<w>  same for Google-News headlines
  zs_<ev>_<w>    count of docs (both sources) whose zero-shot label is <ev>
  neg_kw_<w>     count of docs with ANY credit-negative keyword class (rj, default_waiver, liab_mgmt, rating_down,
                 guidance_cut, litigation, oficio)
  hard_<w>       rj | default_waiver | liab_mgmt (the "credit-event" family) in IPE or news
  sent_news_30 / sent_ipe_90   mean embedding-sentiment (higher = more negative) of the docs in the window
  sent_max_30    max sentiment over any doc in 30d
  burst_30       IPE filings in 30d / (IPE filings in prior 365d / 12)  (abnormal disclosure activity)
  n_ipe_365, n_news_365   coverage counts
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "data" / "history" / "nightly" / "text_nlp"
NEG_EVENTS = ["rj", "default_waiver", "liab_mgmt", "rating_down", "guidance_cut", "litigation", "oficio"]
HARD = ["rj", "default_waiver", "liab_mgmt"]
EVS = ["rj", "default_waiver", "liab_mgmt", "call_redeem", "deb_holders", "new_debt", "equity_raise", "mna", "capex", "guidance_cut",
       "dividend", "rating_down", "rating_up", "mgmt_change", "oficio", "litigation"]


def load_docs() -> pd.DataFrame:
    d = pd.read_pickle(OUT / "doc_feats.pkl")
    from research.nightly.text_nlp.classify import KW, fold      # keyword classes are recomputed from text (cheap)
    ft = fold(d["text"])
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for k, pat in KW.items():
            d[f"kw_{k}"] = ft.str.contains(pat, regex=True)
    d["neg_kw"] = d[[f"kw_{k}" for k in NEG_EVENTS]].any(axis=1)
    d["hard"] = d[[f"kw_{k}" for k in HARD]].any(axis=1)
    # the deb_holders class alone is mostly routine (AGD notices); count it as hard only with a waiver/liab word
    for k in EVS:
        d[f"zsl_{k}"] = d["zs_label"] == k
    d["zs_neg"] = d["zs_label"].isin(NEG_EVENTS)
    d["one"] = True
    return d


def _window_sums(docs: pd.DataFrame, cols: list[str], keys: pd.DataFrame, w: int, vals=None) -> pd.DataFrame:
    """sum of docs[cols] with avail in [day-w, day-1] for each key row (cnpj8, day). Vectorised per issuer."""
    out = np.zeros((len(keys), len(cols)))
    kd = keys["day"].to_numpy("datetime64[D]")
    kgrp = keys.groupby("cnpj8").indices
    for c8, g in docs.groupby("cnpj8"):
        if c8 not in kgrp:
            continue
        ix = kgrp[c8]
        a = g["avail"].to_numpy("datetime64[D]")
        o = np.argsort(a, kind="stable")
        a = a[o]
        V = g[cols].to_numpy(float)[o]
        cs = np.vstack([np.zeros((1, len(cols))), np.nancumsum(V, axis=0)])
        hi = np.searchsorted(a, kd[ix], side="left")                       # avail < day
        lo = np.searchsorted(a, kd[ix] - np.timedelta64(w, "D"), side="left")
        out[ix] = cs[hi] - cs[lo]
    return pd.DataFrame(out, columns=cols, index=keys.index)


def signals(panel: pd.DataFrame, docs: pd.DataFrame | None = None) -> pd.DataFrame:
    d = load_docs() if docs is None else docs
    keys = panel[["cnpj8", "day"]].drop_duplicates().reset_index(drop=True)
    ipe, news = d[d["src"] == "ipe"], d[d["src"] == "news"]
    F = keys.copy()
    kwcols = [f"kw_{k}" for k in EVS] + ["neg_kw", "hard", "one"]
    for w in (30, 90, 180):
        a = _window_sums(ipe, kwcols, keys, w).add_prefix("ipe_").rename(columns=lambda c: c.replace("kw_", "") + f"_{w}")
        b = _window_sums(news, kwcols, keys, w).add_prefix("news_").rename(columns=lambda c: c.replace("kw_", "") + f"_{w}")
        z = _window_sums(d, [f"zsl_{k}" for k in EVS] + ["zs_neg"], keys, w)
        z.columns = [c.replace("zsl_", "zs_") + f"_{w}" for c in z.columns]
        F = pd.concat([F, a, b, z], axis=1)
        F[f"neg_kw_{w}"] = F[f"ipe_neg_kw_{w}"] + F[f"news_neg_kw_{w}"]
        F[f"hard_{w}"] = F[f"ipe_hard_{w}"] + F[f"news_hard_{w}"]
    # sentiment means
    for src, frame, w in (("news", news, 30), ("news", news, 90), ("ipe", ipe, 90)):
        fr = frame[frame["sent"].notna()]
        s = _window_sums(fr.assign(sv=fr["sent"].astype(float)), ["sv", "one"], keys, w)
        F[f"sent_{src}_{w}"] = np.where(s["one"] > 0, s["sv"] / s["one"].clip(lower=1), np.nan)
    # max sentiment 30d (loop-light: use top-quantile flag counts instead of a true max)
    q90 = float(d["sent"].quantile(0.95))
    s = _window_sums(d.assign(hot=(d["sent"] > q90)), ["hot"], keys, 30)
    F["sent_hot_30"] = s["hot"]
    n365 = _window_sums(ipe, ["one"], keys, 365)["one"]
    F["n_ipe_365"] = n365
    F["n_news_365"] = _window_sums(news, ["one"], keys, 365)["one"]
    F["burst_30"] = F["ipe_one_30"] / (n365 / 12.0 + 1.0)
    return F


def attach(panel: pd.DataFrame, F: pd.DataFrame) -> pd.DataFrame:
    return panel.merge(F, on=["cnpj8", "day"], how="left")


def build_cache(freq: str = "M"):
    from research.nightly import harness as H
    P = H.load_panel(freq)
    F = signals(P)
    F.to_pickle(OUT / f"text_signals_{freq}.pkl")
    return F
