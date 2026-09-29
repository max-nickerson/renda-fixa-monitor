"""Part 3b: leave-top-k-contributing-issuers-out (both books), fixed signal construction."""
import json
import numpy as np, pandas as pd
from pathlib import Path
from research.nightly import harness as H
from research.nightly.ml_ranking import run as RUN
OUT = Path("research/nightly/ml_ranking/verify_leakage")
P = pd.read_pickle("data/history/nightly/ml_ranking/features_M.pkl")
pr = pd.read_pickle("data/history/nightly/ml_ranking/pred_lgb_peer.pkl")["pred"]
b = H.baseline("P4Q")
res = json.load(open(OUT / "v3_results.json"))
top = list(res["top_issuers_%yr"].keys())
for k in (0, 1, 3, 5):
    drop = set(top[:k])
    Pk = P[~P["cnpj8"].isin(drop)]
    sT = RUN.overlay_signals("lgb_peer", pr[~pr["cnpj8"].isin(drop)], Pk)["tilt"]
    rt = H.backtest(sT)
    rq = H.backtest(lambda x: (x["p4q"] & ~x["cnpj8"].isin(drop)).to_numpy())
    st = H.stats(rt["daily"], bench=rq["daily"])
    res[f"drop_top{k}_issuers"] = {x: st[x] for x in ("diff_ann_%", "diff_t_nw", "diff_h1_%", "diff_h2_%")}
    print(k, res[f"drop_top{k}_issuers"], flush=True)
json.dump(res, open(OUT / "v3_results.json", "w"), indent=1, default=float)
