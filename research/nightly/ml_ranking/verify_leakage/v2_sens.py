"""Part 2: leakage-sensitivity reruns of lgb_peer (pre-2026 only, frozen original hyper-params)."""
import json, sys
import numpy as np, pandas as pd
from pathlib import Path
from research.nightly import harness as H
from research.nightly.ml_ranking import run as RUN, models as M, features as F

OUT = Path("research/nightly/ml_ranking/verify_leakage")
P = pd.read_pickle("data/history/nightly/ml_ranking/features_M.pkl")
orig = pd.read_pickle("data/history/nightly/ml_ranking/pred_lgb_peer.pkl")
PARAMS = orig["params"]
b = H.baseline("P4Q")
FULL = list(F.FEATURES)
PATH = ["br_21", "br_63", "br_126", "bvol_63", "bmdd_126", "bstale_63", "bmin_126", "iss_br63_mean", "iss_br126_min",
        "iss_worst_bmdd", "sec_br63_mean"]
orig_labelled = M.labelled


def run_variant(tag, feats=FULL, embargo=0, seed=7):
    M.FEATURES = feats
    M.labelled = (lambda P_, upto: orig_labelled(P_, upto - embargo))
    class Peer(M.LGBPeer):
        def _mk(self):
            m = super()._mk()
            m.set_params(random_state=seed)
            return m
    M.MODELS["_v"] = Peer
    pr, _, _ = M.walk_forward("_v", P, params_by_year=PARAMS, log=lambda *a: None)
    M.FEATURES = FULL; M.labelled = orig_labelled
    out = {}
    for ov in ("tilt", "veto20"):
        s = RUN.overlay_signals("lgb_peer", pr, P)[ov]
        st = H.stats(H.backtest(s)["daily"], bench=b["daily"])
        out[ov] = {k: st[k] for k in ("diff_ann_%", "diff_t_nw", "diff_h1_%", "diff_h2_%")}
    ic = RUN.ic_table({"lgb_reg": pr, "m": pr}, P)["m"]
    out["ic"] = ic
    print(tag, json.dumps(out), flush=True)
    return out


res = {}
res["base_seed7"] = run_variant("base_seed7")
res["embargo21"] = run_variant("embargo21", embargo=21)
res["embargo63"] = run_variant("embargo63", embargo=63)
res["no_path_feats"] = run_variant("no_path_feats", feats=[f for f in FULL if f not in PATH])
for s in (1, 2, 3, 4, 5):
    res[f"seed{s}"] = run_variant(f"seed{s}", seed=s)
json.dump(res, open(OUT / "v2_results.json", "w"), indent=1, default=float)
