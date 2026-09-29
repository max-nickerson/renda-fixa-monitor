"""Adversarial verification of text_nlp V4 (P4+Q ex zero-shot credit-negative doc in 90d). Pre-2026 only."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[4]; sys.path.insert(0, str(ROOT))
from research.nightly import harness as H
from research.nightly.text_nlp import features as TF
HERE = Path(__file__).parent
NEG = TF.NEG_EVENTS
docs0 = TF.load_docs(); docs0 = docs0[docs0["avail"] < H.HOLDOUT].copy()
P0 = H.load_panel("M")
B = H.baseline("P4Q")
OUT = {}

def zsneg_feature(docs, name="f"):
    keys = P0[["cnpj8", "day"]].drop_duplicates().reset_index(drop=True)
    s = TF._window_sums(docs.assign(v=docs["flag"].astype(float)), ["v"], keys, 90)["v"]
    k = keys.assign(**{name: s.values})
    return k

def run(docs, nm):
    k = zsneg_feature(docs, "f")
    P = P0.merge(k, on=["cnpj8", "day"], how="left")
    sig = lambda x: (x["p4q"] & (x["f"].fillna(0) == 0)).to_numpy()
    r = H.backtest(sig, panel=P, name=nm)
    st = H.stats(r["daily"], bench=B["daily"])
    res = {k2: round(float(st[k2]), 3) for k2 in ("diff_ann_%", "diff_t_nw", "diff_h1_%", "diff_h2_%")}
    res["n_avg"] = round(float(r["n_avg"]), 1)
    print(nm, res, flush=True); OUT[nm] = res
    return r, P

# 1) reproduce
d = docs0.assign(flag=docs0["zs_neg"])
r4, P4 = run(d, "V4 reproduce")
# 2) extra availability lag (IPE after-close/night deliveries, news timezone / late indexing)
for lag in (1, 3, 7):
    run(d.assign(avail=d["avail"] + pd.Timedelta(days=lag)), f"V4 avail+{lag}d")
# 3) source split
run(d.assign(flag=d["zs_neg"] & (d["src"] == "ipe")), "V4 IPE-only")
run(d.assign(flag=d["zs_neg"] & (d["src"] == "news")), "V4 news-only")
# 4) threshold sensitivity (recompute label from saved cosines)
zc = [c for c in docs0.columns if c.startswith("zs_") and c not in ("zs_label", "zs_neg") and docs0[c].dtype != bool]
cls = [c[3:] for c in zc]
S = docs0[zc].to_numpy(float); has = ~np.isnan(S).all(1)
Sf = np.where(np.isnan(S), -9, S); am = Sf.argmax(1); mx = Sf.max(1); lab = np.array(cls)[am]
for T in (0.50, 0.60):
    lbl = np.where(has & (mx > T) & (lab != "routine"), lab, "none")
    run(docs0.assign(flag=np.isin(lbl, NEG)), f"V4 threshold {T}")
# 5) content placebo: shuffle zs_neg flags among embedded docs (same doc timing/activity, random meaning), 20 draws
emb = docs0["emb_row"] >= 0 if "emb_row" in docs0 else has
rng = np.random.default_rng(7); pl = []
fl = docs0["zs_neg"].to_numpy()
idx = np.where(emb)[0]
for i in range(20):
    f2 = fl.copy(); f2[idx] = rng.permutation(fl[idx])
    k = zsneg_feature(docs0.assign(flag=f2), "f")
    P = P0.merge(k, on=["cnpj8", "day"], how="left")
    r = H.backtest(lambda x: (x["p4q"] & (x["f"].fillna(0) == 0)).to_numpy(), panel=P)
    pl.append(float(H.stats(r["daily"], bench=B["daily"])["diff_ann_%"]))
    print("shuffle", i, round(pl[-1], 3), flush=True)
OUT["content_shuffle_placebo"] = {"mean": round(np.mean(pl), 3), "p95": round(np.percentile(pl, 95), 3),
                                  "share_ge_V4": float(np.mean(np.array(pl) >= OUT["V4 reproduce"]["diff_ann_%"])), "draws": pl}
# 6) activity control: exclude P4Q issuers with any embedded-category doc (FR/Comunicado/news) in 90d
run(docs0.assign(flag=emb.to_numpy() if hasattr(emb, "to_numpy") else emb), "control: any embedded doc 90d")
run(docs0.assign(flag=(docs0["cat"] == "Fato Relevante")), "control: any Fato Relevante 90d")
# 7) random P4Q issuer exclusion matched to V4's per-date issuer count, 20 draws
k = zsneg_feature(d, "f"); P = P0.merge(k, on=["cnpj8", "day"], how="left")
pr = []
for s in range(20):
    rs = np.random.default_rng(100 + s)
    def sig(x, rs=rs):
        m = x["p4q"].to_numpy().astype(bool)
        flagged = set(x.loc[m & (x["f"].fillna(0) > 0).to_numpy(), "cnpj8"])
        iss = x.loc[m, "cnpj8"].unique()
        drop = set(rs.choice(iss, min(len(flagged), len(iss)), replace=False)) if len(flagged) else set()
        return m & ~x["cnpj8"].isin(drop).to_numpy()
    r = H.backtest(sig, panel=P)
    pr.append(float(H.stats(r["daily"], bench=B["daily"])["diff_ann_%"])); print("rand", s, round(pr[-1], 3), flush=True)
OUT["random_matched_exclusion"] = {"mean": round(np.mean(pr), 3), "p95": round(np.percentile(pr, 95), 3),
                                   "share_ge_V4": float(np.mean(np.array(pr) >= OUT["V4 reproduce"]["diff_ann_%"])), "draws": pr}
(HERE / "verify_results.json").write_text(json.dumps(OUT, indent=1), encoding="utf-8")
print(json.dumps({k: v for k, v in OUT.items() if "draws" not in v}, indent=1))
