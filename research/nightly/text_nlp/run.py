"""TEXT & NLP nightly study - event studies, lead time, IC and P4+Q overlays (pre-2026 only; holdout once at the end).

Run order (see README):
  build_docs.py -> classify.py -> [fulltext.py] -> run.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from research.nightly import harness as H                       # noqa: E402
from research.nightly.text_nlp import features as TF            # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = TF.OUT
RES: dict = {"notes": []}
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)


# =====================================================================================================================
# 1) Event studies on bond returns and spreads
# =====================================================================================================================
def issuer_map_pre2026() -> np.ndarray:
    P = H.load_panel("W")                      # pre-2026
    m = P.drop_duplicates("b", keep="last").set_index("b")["cnpj8"]
    arr = np.array(["?"] * len(H._core()["codes"]), dtype=object)
    arr[m.index.to_numpy()] = m.to_numpy()
    return arr


def event_table(docs: pd.DataFrame, flag: str, src: str | None, cool: int = 90) -> pd.DataFrame:
    """First doc of class `flag` per issuer after `cool` days without one -> one event (cnpj8, avail)."""
    d = docs[docs[flag]] if src is None else docs[docs[flag] & (docs["src"] == src)]
    d = d.sort_values(["cnpj8", "avail"])
    gap = d.groupby("cnpj8")["avail"].diff().dt.days
    return d[gap.isna() | (gap > cool)][["cnpj8", "avail"]].reset_index(drop=True)


def event_study(ev: pd.DataFrame, pre: int = 120, post: int = 120, iss=None, name="") -> dict:
    """Issuer-average abnormal return path (bond daily excess R minus the all-bond average of that day),
    event day 0 = first grid day AFTER avail (the first day a trade on the news is possible)."""
    C = H._core()
    R, TD, dd = H._Rmat(), C["TD"], C["days"]
    hp = H._hpos()
    on = TD >= 0
    Rm = np.where(on, R, np.nan)
    mkt = np.nanmean(Rm[:hp], axis=1)
    iss = issuer_map_pre2026() if iss is None else iss
    b_of = pd.Series(np.arange(len(iss))).groupby(iss).apply(lambda s: s.to_numpy()).to_dict()
    paths, meta = [], []
    for r in ev.itertuples():
        bs = b_of.get(r.cnpj8)
        if bs is None:
            continue
        p0 = int(dd.searchsorted(r.avail + pd.Timedelta(days=1)))
        if p0 - pre < 0 or p0 + post >= hp:
            continue
        seg = Rm[p0 - pre:p0 + post, :][:, bs]
        if not np.isfinite(seg[pre - 20:pre + 20]).any():      # issuer must have a live bond around the event
            continue
        ab = np.nanmean(seg, axis=1) - mkt[p0 - pre:p0 + post]
        ab = np.nan_to_num(ab)
        paths.append(np.cumsum(ab))
        meta.append((r.cnpj8, r.avail))
    if not paths:
        return {"n": 0}
    A = np.vstack(paths) * 100
    A = A - A[:, [pre - 1]]                   # rebase to 0 at day -1
    m, se = A.mean(0), A.std(0, ddof=1) / np.sqrt(len(A))

    def win(a, b):
        x = A[:, pre + b - 1] - A[:, pre + a - 1] if a > -pre else A[:, pre + b - 1] - A[:, 0]
        return {"mean_%": round(float(x.mean()), 3), "t": round(float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))), 2),
                "median_%": round(float(np.median(x)), 3)}
    return {"n": len(A), "path_mean": m.round(4).tolist(), "path_se": se.round(4).tolist(),
            "pre_-60_-1": win(-60, 0), "pre_-120_-1": win(-119, 0), "post_0_20": win(0, 20), "post_0_60": win(0, 60),
            "post_0_120": win(0, post - 1), "name": name}


def spread_study(ev: pd.DataFrame, PW: pd.DataFrame, pre=26, post=26) -> dict:
    """Weekly issuer median CDI+ spread minus universe median, event week 0 = first weekly decision after avail."""
    x = PW[PW["cdi_bps"].notna()]
    wk = np.sort(x["day"].unique())
    med = x.groupby("day")["cdi_bps"].median()
    iss = x.groupby(["cnpj8", "day"])["cdi_bps"].median().sub(med, level="day").unstack("day").reindex(columns=wk)
    paths = []
    for r in ev.itertuples():
        if r.cnpj8 not in iss.index:
            continue
        k = int(np.searchsorted(wk, np.datetime64(r.avail + pd.Timedelta(days=1))))
        if k - pre < 0 or k + post >= len(wk):
            continue
        v = iss.loc[r.cnpj8].to_numpy()[k - pre:k + post]
        if np.isnan(v[pre - 2:pre + 1]).all():
            continue
        v = pd.Series(v).ffill().to_numpy()
        base = np.nanmean(v[pre - 4:pre])
        if not np.isfinite(base):
            continue
        paths.append(v - base)
    if not paths:
        return {"n": 0}
    A = np.vstack(paths)

    def win(i):
        c = A[:, pre + i]
        c = c[np.isfinite(c)]
        return {"mean_bps": round(float(c.mean()), 1), "median_bps": round(float(np.median(c)), 1),
                "t": round(float(c.mean() / (c.std(ddof=1) / np.sqrt(len(c)))), 2), "n": int(len(c))}
    return {"n": len(A), "path_mean": np.nanmean(A, 0).round(2).tolist(),
            "w-26": win(-26), "w-13": win(-13), "w-4": win(-4), "w+4": win(4), "w+13": win(13), "w+25": win(25)}


def lead_time(docs: pd.DataFrame, PW: pd.DataFrame) -> dict:
    """Spread blow-outs: issuer median spread (vs universe median) widens by >= 150 bps over 4 weeks (first such week
    per issuer after a 26-week cool-off). For each blow-out: was there a text flag in the prior 13 / 26 weeks, and how
    many days before the blow-out week was the first flag? Compared with the unconditional flag rate (random weeks)."""
    x = PW[PW["cdi_bps"].notna()]
    med = x.groupby("day")["cdi_bps"].median()
    iss = x.groupby(["cnpj8", "day"])["cdi_bps"].median().sub(med, level="day").unstack("day").sort_index(axis=1)
    iss = iss.ffill(axis=1, limit=4)
    ch = iss - iss.shift(4, axis=1)
    rows = []
    for c8, s in ch.iterrows():
        last = None
        for day, v in s[s >= 150].items():
            if last is None or (day - last).days > 182:
                rows.append((c8, day, v))
                last = day
    bo = pd.DataFrame(rows, columns=["cnpj8", "day", "jump_bps"])
    bo["start"] = bo["day"] - pd.Timedelta(days=28)            # the widening window starts 4 weeks earlier
    out = {"n_blowouts": int(len(bo))}
    flags = {"hard (rj/waiver/liab-mgmt kw)": "hard", "any credit-negative kw": "neg_kw", "zero-shot negative": "zs_neg",
             "news any": "news_any", "ipe fato relevante": "fr", "top-5% sentiment": "hot"}
    d = docs.copy()
    d["news_any"] = d["src"] == "news"
    d["fr"] = (d["src"] == "ipe") & (d["cat"] == "Fato Relevante")
    d["hot"] = d["sent"] > d["sent"].quantile(0.95)
    rng = np.random.default_rng(0)
    alld = np.sort(x["day"].unique())
    ctrl = pd.DataFrame({"cnpj8": rng.choice(bo["cnpj8"].to_numpy(), 3000) if len(bo) else [],
                         "start": pd.to_datetime(rng.choice(alld, 3000)) if len(bo) else []})
    for lab, f in flags.items():
        g = d[d[f]].groupby("cnpj8")["avail"].apply(lambda s: np.sort(s.to_numpy("datetime64[D]"))).to_dict()

        def stats(frame):
            has13, has26, leads = [], [], []
            for r in frame.itertuples():
                a = g.get(r.cnpj8)
                st = np.datetime64(r.start, "D")
                if a is None:
                    has13.append(False); has26.append(False); continue
                w = a[(a < st) & (a >= st - np.timedelta64(182, "D"))]
                has26.append(len(w) > 0)
                has13.append(bool(((a < st) & (a >= st - np.timedelta64(91, "D"))).any()))
                if len(w):
                    leads.append(int((st - w.min()).astype(int)))
            return np.mean(has13), np.mean(has26), leads
        h13, h26, leads = stats(bo)
        c13, c26, _ = stats(ctrl)
        # after the start: flag inside the widening window or within 4 weeks after (reactive disclosure)
        post = []
        for r in bo.itertuples():
            a = g.get(r.cnpj8)
            st = np.datetime64(r.start, "D")
            post.append(bool(a is not None and ((a >= st) & (a < st + np.timedelta64(56, "D"))).any()))
        out[lab] = {"blowouts_with_flag_prior_13w": round(float(h13), 3), "random_weeks_prior_13w": round(float(c13), 3),
                    "blowouts_with_flag_prior_26w": round(float(h26), 3), "random_weeks_prior_26w": round(float(c26), 3),
                    "lift_13w": round(float(h13 / c13), 2) if c13 > 0 else None,
                    "median_lead_days_first_flag_26w": float(np.median(leads)) if leads else None,
                    "flag_during_or_after_(0,+8w)": round(float(np.mean(post)), 3)}
    bo["year"] = bo["day"].dt.year
    out["by_year"] = bo.groupby("year").size().to_dict()
    return out


# =====================================================================================================================
# 2) Learned text model: walk-forward ridge on issuer mean-embeddings -> 63d forward excess (issuer-demeaned by date)
# =====================================================================================================================
def embed_window_means(docs: pd.DataFrame, keys: pd.DataFrame, w: int = 90) -> np.ndarray:
    E = np.load(OUT / "emb.npy", mmap_mode="r")
    out = np.zeros((len(keys), 2 * E.shape[1] + 2), np.float32)
    kd = keys["day"].to_numpy("datetime64[D]")
    kg = keys.groupby("cnpj8").indices
    for j, src in enumerate(("ipe", "news")):
        dd = docs[(docs["src"] == src) & (docs["emb_row"] >= 0)]
        for c8, g in dd.groupby("cnpj8"):
            if c8 not in kg:
                continue
            ix = kg[c8]
            g = g.sort_values("avail")
            a = g["avail"].to_numpy("datetime64[D]")
            V = np.asarray(E[g["emb_row"].to_numpy()], np.float64)
            cs = np.vstack([np.zeros((1, V.shape[1])), np.cumsum(V, 0)])
            hi = np.searchsorted(a, kd[ix], "left")
            lo = np.searchsorted(a, kd[ix] - np.timedelta64(w, "D"), "left")
            n = (hi - lo)[:, None]
            out[ix, j * E.shape[1]:(j + 1) * E.shape[1]] = np.where(n > 0, (cs[hi] - cs[lo]) / np.maximum(n, 1), 0)
            out[ix, 2 * E.shape[1] + j] = np.log1p(n[:, 0])
    return out


def learned_score(P: pd.DataFrame, docs: pd.DataFrame, H_=63, alpha=30.0) -> pd.Series:
    from sklearn.linear_model import Ridge
    keys = P[["cnpj8", "day"]].drop_duplicates().reset_index(drop=True)
    X = embed_window_means(docs, keys)
    kidx = pd.MultiIndex.from_frame(keys)
    U = P[P["univ"]].copy()
    U["y"] = U[f"fwd_{H_}"] - U.groupby("day")[f"fwd_{H_}"].transform("mean")
    U["y"] = U.groupby("day")["y"].rank(pct=True) - 0.5              # rank target, robust to gap jumps
    U["row"] = kidx.get_indexer(pd.MultiIndex.from_frame(U[["cnpj8", "day"]]))
    U["has"] = X[U["row"].to_numpy(), -2:].sum(1) > 0
    score = pd.Series(np.nan, index=U.index)
    for p in sorted(U["dpos"].unique()):
        tr = U[(U[f"lab_end_{H_}"] <= p) & U["y"].notna() & U["has"]]
        te = U[(U["dpos"] == p) & U["has"]]
        if len(tr) < 2000 or te.empty or tr["day"].nunique() < 6:
            continue
        m = Ridge(alpha=alpha).fit(X[tr["row"].to_numpy()], tr["y"].to_numpy())
        score.loc[te.index] = m.predict(X[te["row"].to_numpy()])
    return score


def fulltext_study(docs: pd.DataFrame, iss) -> dict:
    """Title vs full text on the PDF sample (fulltext.py). Does the body reveal credit-negative content the title hides,
    and do such docs move bonds?"""
    files = sorted((OUT / "pdf").glob("*.txt"))
    if not files:
        return {"status": "no fulltext sample"}
    from research.nightly.text_nlp.classify import KW, fold, centroid, SENT_NEG, SENT_POS, _enc
    rows = []
    for f in files:
        t = f.read_text(encoding="utf-8")
        rows.append({"doc_id": int(f.stem), "n_chars": len(t), "text": t[:6000]})
    ft = pd.DataFrame(rows)
    ft = ft.merge(docs[["doc_id", "cnpj8", "avail", "cat", "text"] +
                                       [f"kw_{k}" for k in TF.NEG_EVENTS]].rename(columns={"text": "title"}),
                                  on="doc_id", how="inner")
    ok = ft["n_chars"] > 200
    out = {"n_sample": int(len(ft)), "extract_ok_share": round(float(ok.mean()), 3),
           "median_chars": float(ft.loc[ok, "n_chars"].median()) if ok.any() else 0}
    ft = ft[ok].copy()
    body = fold(ft["text"])
    for k in TF.NEG_EVENTS:
        ft[f"body_{k}"] = body.str.contains(KW[k], regex=True)
    ft["title_neg"] = ft[[f"kw_{k}" for k in TF.NEG_EVENTS]].any(axis=1)
    ft["flagged"] = ft["title_neg"]
    ft["body_hard"] = ft[[f"body_{k}" for k in TF.HARD]].any(axis=1)
    ft["title_hard"] = ft[[f"kw_{k}" for k in TF.HARD]].any(axis=1)
    rnd = ft[~ft["flagged"]]
    out["random_titleclean_docs"] = int(len(rnd))
    out["random_titleclean_body_hard_share"] = round(float(rnd["body_hard"].mean()), 3) if len(rnd) else None
    out["flagged_title_hard_confirmed_by_body"] = round(float(ft.loc[ft["title_hard"], "body_hard"].mean()), 3)         if ft["title_hard"].any() else None
    for k in TF.NEG_EVENTS:
        out[f"share_body_{k}_(random)"] = round(float(rnd[f"body_{k}"].mean()), 3) if len(rnd) else None
    # sentiment of the body: mean over up to 4 chunks of ~50 words
    try:
        neg, pos = centroid(SENT_NEG), centroid(SENT_POS)
        chunks, owner = [], []
        for i, t in enumerate(ft["text"]):
            w = t.split()
            for j in range(0, min(len(w), 200), 50):
                chunks.append(" ".join(w[j:j + 50])); owner.append(i)
        Eb = _enc(chunks)
        sc = pd.Series(Eb @ neg - Eb @ pos).groupby(np.array(owner)).mean()
        ft["body_sent"] = sc.reindex(range(len(ft))).to_numpy()
        tsent = docs.set_index("doc_id")["sent"]
        ft["title_sent"] = ft["doc_id"].map(tsent)
        out["corr_title_vs_body_sent"] = round(float(ft[["title_sent", "body_sent"]].corr().iloc[0, 1]), 3)
    except Exception as e:
        out["sent_err"] = str(e)[:100]
    # event studies inside the sample
    for nm, m in {"body_hard & title clean": ft["body_hard"] & ~ft["title_hard"],
                  "title hard": ft["title_hard"], "body clean & title clean (random)": ~ft["body_hard"] & ~ft["flagged"],
                  "body_sent top tercile": ft.get("body_sent", pd.Series(0, index=ft.index)) >
                  ft.get("body_sent", pd.Series(0, index=ft.index)).quantile(2 / 3)}.items():
        ev = ft.loc[m, ["cnpj8", "avail"]].drop_duplicates()
        es = event_study(ev, iss=iss, name=nm)
        out[f"ES {nm}"] = {kk: vv for kk, vv in es.items() if not kk.startswith("path")}
    return out


# =====================================================================================================================
# 3) Overlays on P4+Q
# =====================================================================================================================
def main():
    docs = TF.load_docs()
    docs = docs[docs["avail"] < H.HOLDOUT]                  # nothing from the holdout is even loaded for research
    log("docs", len(docs))
    P = H.load_panel("M")
    PW = H.load_panel("W")
    F = TF.signals(P, docs)
    F.to_pickle(OUT / "text_signals_M.pkl")
    P = TF.attach(P, F)
    log("features", F.shape)

    # ---------- event studies
    iss = issuer_map_pre2026()
    ES, SS = {}, {}
    ev_defs = {f"ipe:{k}": (f"kw_{k}", "ipe") for k in TF.EVS}
    ev_defs.update({f"news:{k}": (f"kw_{k}", "news") for k in ["rj", "default_waiver", "liab_mgmt", "rating_down",
                                                                 "litigation", "guidance_cut", "mna", "rating_up"]})
    ev_defs.update({f"zs:{k}": (f"zsl_{k}", None) for k in TF.EVS})
    ev_defs["ipe:fato_relevante"] = ("fr", "ipe")
    docs["fr"] = docs["cat"].eq("Fato Relevante")
    docs["hot"] = docs["sent"] > docs["sent"].quantile(0.95)
    docs["cold"] = docs["sent"] < docs["sent"].quantile(0.05)
    ev_defs["any:sent_top5%"] = ("hot", None)
    ev_defs["any:sent_bottom5%"] = ("cold", None)
    for nm, (flag, src) in ev_defs.items():
        ev = event_table(docs, flag, src)
        ES[nm] = event_study(ev, iss=iss, name=nm)
        SS[nm] = spread_study(ev, PW)
        log("event", nm, ES[nm].get("n"), ES[nm].get("post_0_60"), SS[nm].get("w+13"))
    # placebo: random issuer-dates from the same issuers
    rng = np.random.default_rng(1)
    base = event_table(docs, "one", "ipe")
    pl = pd.DataFrame({"cnpj8": rng.choice(base["cnpj8"], 1500),
                       "avail": pd.to_datetime(rng.choice(pd.date_range("2021-08-01", "2025-06-30").to_numpy(), 1500))})
    ES["placebo:random_dates"] = event_study(pl, iss=iss, name="placebo")
    SS["placebo:random_dates"] = spread_study(pl, PW)
    RES["event_study_returns"] = {k: {kk: vv for kk, vv in v.items() if not kk.startswith("path")} for k, v in ES.items()}
    RES["event_study_spreads"] = {k: {kk: vv for kk, vv in v.items() if not kk.startswith("path")} for k, v in SS.items()}
    pd.to_pickle({"ES": ES, "SS": SS}, OUT / "event_paths.pkl")
    RES["lead_time"] = lead_time(docs, PW)
    log("lead time", RES["lead_time"])
    RES["fulltext_sample"] = fulltext_study(docs, iss)
    log("fulltext", RES["fulltext_sample"])

    # ---------- IC of the features (monthly panel, fwd_63 and fwd_126)
    feats = ["neg_kw_90", "hard_180", "hard_90", "ipe_neg_kw_90", "news_neg_kw_90", "zs_zs_neg_90", "sent_news_30",
             "sent_news_90", "sent_ipe_90", "sent_hot_30", "burst_30", "ipe_one_90", "news_one_90", "ipe_liab_mgmt_180",
             "ipe_default_waiver_180", "ipe_rj_180", "news_rj_90", "ipe_mna_180", "ipe_new_debt_180", "ipe_dividend_180",
             "ipe_rating_down_180", "news_rating_down_90", "ipe_capex_180", "ipe_guidance_cut_180", "ipe_oficio_90",
             "ipe_equity_raise_180", "ipe_mgmt_change_90", "ipe_deb_holders_90", "news_litigation_90"]
    for c in ["zs_neg_90"]:
        if c in P and "zs_zs_neg_90" not in P:
            P["zs_zs_neg_90"] = P[c]
    IC = {}
    for f in feats:
        if f not in P:
            continue
        r63 = H.ic(P, f, "fwd_63")
        r126 = H.ic(P, f, "fwd_126")
        cov = float((P.loc[P["univ"], f].fillna(0) != 0).mean())
        IC[f] = {"ic63": round(r63["mean"], 4), "t63": round(r63["t_nw"], 2), "ic126": round(r126["mean"], 4),
                 "t126": round(r126["t_nw"], 2), "nonzero_share": round(cov, 3)}
    RES["ic"] = IC
    log("IC done")

    # ---------- learned embedding model (walk-forward)
    P["txt_ml"] = learned_score(P, docs)
    r = H.ic(P, "txt_ml", "fwd_63")
    RES["ic"]["txt_ml (walk-forward ridge on embeddings)"] = {"ic63": round(r["mean"], 4), "t63": round(r["t_nw"], 2),
                                                              "n_dates": r["n_dates"]}
    log("learned", RES["ic"]["txt_ml (walk-forward ridge on embeddings)"])
    P[["cnpj8", "day", "codigo", "txt_ml"]].dropna().to_pickle(OUT / "txt_ml_M.pkl")

    # ---------- overlays on P4+Q (monthly decisions, 126d tranches, 25 bps)
    def pctl(x, c):
        return x[c].rank(pct=True)

    def ml_excl(x):
        s = x["txt_ml"]
        cut = s[x["p4q"]].quantile(0.2) if s[x["p4q"]].notna().sum() > 10 else -np.inf
        return (x["p4q"] & ~(s < cut)).to_numpy()

    V = {
        "V1 P4Q ex any credit-neg kw 90d": lambda x: (x["p4q"] & (x["neg_kw_90"].fillna(0) == 0)).to_numpy(),
        "V2 P4Q ex hard events 180d": lambda x: (x["p4q"] & (x["hard_180"].fillna(0) == 0)).to_numpy(),
        "V3 P4Q ex IPE credit-neg 90d": lambda x: (x["p4q"] & (x["ipe_neg_kw_90"].fillna(0) == 0)).to_numpy(),
        "V4 P4Q ex zero-shot neg 90d": lambda x: (x["p4q"] & (x["zs_neg_90"].fillna(0) == 0)).to_numpy(),
        "V5 P4Q ex worst news sentiment 30d": lambda x: (x["p4q"] & ~(pctl(x, "sent_news_30") > 0.8)).to_numpy(),
        "V6 P4Q ex filing burst": lambda x: (x["p4q"] & ~(x["burst_30"].fillna(0) > 3)).to_numpy(),
        "V7 P4Q ex bottom-20% learned text score": ml_excl,
        "V8 P4Q tilt 1/(1+neg_kw_90)": lambda x: (x["p4q"] / (1.0 + x["neg_kw_90"].fillna(0))).to_numpy(float),
        "V9 P4Q + positive-event adds (top40% carry, equity raise/rating up 180d)": lambda x: (
            x["p4q"] | ((x["cdi_pct"] < 0.4) & (x["resid_z"] > -1.5) & ~x["worstQ"].fillna(False).astype(bool)
                        & ((x["ipe_equity_raise_180"] + x["ipe_rating_up_180"] + x["news_rating_up_90"]).fillna(0) > 0)
                        & (x["neg_kw_90"].fillna(0) == 0))).to_numpy(),
    }
    R = {}
    for nm, f in V.items():
        R[nm] = H.backtest(f, as_weights=nm.startswith("V8"), panel=P, name=nm)
        log(nm, round(H.stats(R[nm]["daily"], bench=H.baseline("P4Q")["daily"])["diff_ann_%"], 3))
    ref = {"P4Q": H.baseline("P4Q"), "P4": H.baseline("P4"), "U": H.baseline("U")}
    tab = H.compare({**R, **ref}, bench="P4Q")
    n_tried = len(V)
    RES["overlays_25bps"] = json.loads(tab.round(4).to_json(orient="index"))
    RES["n_variants_tried"] = n_tried
    log("\n" + tab[["exCDI_%", "exU_%", "vs_bench_%", "t_vs_bench", "p_holm", "h1_vs_bench", "h2_vs_bench",
                    "n_avg"]].round(3).to_string())
    # 50 bps and rec40 for all variants
    R50 = {nm: H.backtest(f, as_weights=nm.startswith("V8"), panel=P, cost_bps=50) for nm, f in V.items()}
    t50 = H.compare({**R50, "P4Q": H.baseline("P4Q", cost_bps=50)}, bench="P4Q", cost_bps=50)
    RES["overlays_50bps"] = json.loads(t50.round(4).to_json(orient="index"))
    Rrec = {nm: H.backtest(f, as_weights=nm.startswith("V8"), panel=P, scenario="rec40") for nm, f in V.items()}
    trec = H.compare({**Rrec, "P4Q": H.baseline("P4Q", scenario="rec40")}, bench="P4Q", scenario="rec40")
    RES["overlays_rec40"] = json.loads(trec.round(4).to_json(orient="index"))
    # the excluded names themselves: cohort excess of the bonds each filter removes from P4+Q
    EXC = {}
    for nm, f in V.items():
        if nm.startswith(("V8", "V9")):
            continue
        g = lambda x, f=f: (x["p4q"].to_numpy() & ~f(x))
        try:
            ce = H.cohort_excess(g, H=126, panel=P)
            EXC[nm] = {"mean_ann_%": round(float(ce.mean() / 0.5 * 100), 3), "n_dates": int(ce.notna().sum()),
                       "t_nw": round(H.nw_t(ce.dropna(), 6), 2)}
        except Exception as e:  # fewer than 5 excluded names on most dates
            EXC[nm] = {"err": str(e)[:80]}
    RES["excluded_names_cohort_vs_U"] = EXC
    log("excluded cohorts", EXC)

    # pick the best variant by pre-2026 paired diff (the choice itself is part of the multiple-testing count)
    best = tab.loc[list(V), "vs_bench_%"].astype(float).idxmax()
    RES["best_variant"] = best
    pl = H.placebo(V[best], n=20, panel=P) if not best.startswith("V8") else None
    if pl:
        RES["placebo_best"] = {"mean": pl["mean"], "p95": pl["p95"]}
    # halves / stats of best
    RES["best_stats"] = H.stats(R[best]["daily"], bench=H.baseline("P4Q")["daily"])
    # curves
    H.plot_curves({best.split(" ")[0] + " (text overlay)": R[best]}, str(HERE / "equity_total_return.png"),
                  title=f"TEXT & NLP: {best} vs refs (pre-2026, 25 bps)")
    cum_plot(R[best], best)
    event_plot(ES, SS)

    # ---------- sealed holdout, once, frozen choice
    Ph = H.load_panel("M", holdout=True)
    docs_all = TF.load_docs()
    Fh = TF.signals(Ph, docs_all)
    Ph = TF.attach(Ph, Fh)
    if best.startswith("V7"):
        Ph["txt_ml"] = learned_score(Ph, docs_all)
    rh = H.backtest(V[best], as_weights=best.startswith("V8"), panel=Ph, holdout=True)
    bh = H.baseline("P4Q", holdout=True)
    RES["holdout_2026_best"] = H.stats(rh["daily"], bench=bh["daily"], holdout="only")
    log("holdout", RES["holdout_2026_best"])
    (HERE / "results.json").write_text(json.dumps(RES, indent=1, default=str), encoding="utf-8")
    log("done")


def cum_plot(r, name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    u = H.baseline("U")["daily"]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for lab, s in ((name, r["daily"]), ("P4+Q", H.baseline("P4Q")["daily"]), ("P4", H.baseline("P4")["daily"])):
        x = s.reindex(u.index).fillna(0) - u
        ax.plot(((1 + x).cumprod() - 1) * 100, label=lab, lw=1.6 if lab == name else 1.1)
    x = H.baseline("P4Q")["daily"].reindex(u.index).fillna(0)
    ax.plot(((1 + r["daily"].reindex(u.index).fillna(0) - x).cumprod() - 1) * 100, label=f"{name.split(' ')[0]} minus P4+Q",
            color="k", ls="--", lw=1)
    ax.axhline(0, color="grey", lw=0.6)
    ax.set_ylabel("cumulative excess vs universe, %")
    ax.set_title("Cumulative excess vs the eligible universe (pre-2026, 25 bps)")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(HERE / "cum_excess.png", dpi=130)
    plt.close(fig)


def event_plot(ES, SS):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    show = ["ipe:rj", "ipe:default_waiver", "ipe:liab_mgmt", "news:rj", "news:rating_down", "ipe:mna", "ipe:new_debt",
            "any:sent_top5%", "placebo:random_dates"]
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.8))
    for k in show:
        e = ES.get(k, {})
        if e.get("n", 0) >= 10:
            pm = np.array(e["path_mean"])
            axs[0].plot(np.arange(len(pm)) - 120, pm, label=f"{k} (n={e['n']})", lw=1.8 if "placebo" not in k else 1,
                        ls="--" if "placebo" in k else "-")
        s = SS.get(k, {})
        if s.get("n", 0) >= 10:
            pm = np.array(s["path_mean"])
            axs[1].plot(np.arange(len(pm)) - 26, pm, label=f"{k} (n={s['n']})", ls="--" if "placebo" in k else "-")
    axs[0].axvline(0, color="grey", lw=0.6); axs[1].axvline(0, color="grey", lw=0.6)
    axs[0].set_title("Cumulative abnormal bond excess return (%, vs all bonds), day 0 = first day after filing")
    axs[0].set_xlabel("business days"); axs[1].set_xlabel("weeks")
    axs[1].set_title("Issuer CDI+ spread vs universe median (bps, rebased to weeks -4..-1)")
    axs[0].legend(fontsize=7, frameon=False); axs[1].legend(fontsize=7, frameon=False)
    for a in axs:
        a.title.set_fontsize(9)
    fig.tight_layout()
    fig.savefig(HERE / "event_studies.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
