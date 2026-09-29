"""Leakage verification of ml_ranking, part 1: reproduce lgb_peer:tilt and integrity checks (pre-2026 only)."""
import json, time
import numpy as np, pandas as pd
from pathlib import Path
from research.nightly import harness as H
from research.nightly.ml_ranking import run as RUN, models as M, features as F

OUT = Path("research/nightly/ml_ranking/verify_leakage")
P = pd.read_pickle("data/history/nightly/ml_ranking/features_M.pkl")
res = {}
# --- integrity: holdout rows absent, labels NaN if they reach holdout
res["max_day"] = str(P["day"].max().date())
hp = H._hpos()
res["labels_reaching_holdout_nonnull"] = int((P["lab_end_126"] >= hp).sum() and P.loc[P["lab_end_126"] >= hp, "fwd_126"].notna().sum())
# --- preds integrity
pr = pd.read_pickle("data/history/nightly/ml_ranking/pred_lgb_peer.pkl")["pred"]
res["pred_max_day"] = str(pr["day"].max().date())
res["fit_pos_le_dpos"] = bool((pr["fit_pos"] <= pr["dpos"]).all())
# max label end used by each fit
fits = sorted(pr["fit_pos"].unique())
chk = []
for f in fits:
    tr = M.labelled(P, int(f))
    chk.append((int(f), int(tr["lab_end_126"].max()), int(tr["dpos"].max()), len(tr)))
res["fit_checks(fit_pos,max_lab_end,max_train_dpos,n)"] = chk
# feature columns: none of the forbidden
bad = [c for c in F.FEATURES if c.startswith(("fwd_", "lab_end", "dok_", "entry", "executed")) or "LOOKAHEAD" in c]
res["forbidden_features"] = bad
# --- reproduce best
sig = RUN.overlay_signals("lgb_peer", pr, P)["tilt"]
r = H.backtest(sig, name="lgb_peer:tilt")
b = H.baseline("P4Q")
res["repro_tilt_vs_p4q"] = H.stats(r["daily"], bench=b["daily"])
json.dump(res, open(OUT / "v1_results.json", "w"), indent=1, default=str)
print(json.dumps(res, indent=1, default=str)[:4000])
