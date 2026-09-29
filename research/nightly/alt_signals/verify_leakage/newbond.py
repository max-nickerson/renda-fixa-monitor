import json, pandas as pd, numpy as np
from pathlib import Path
from research.nightly import harness as H
from research.nightly.alt_signals import signals as S
Q = pd.read_pickle(S.CACHE / "panel_alt_M.pkl")
PW = H.load_panel("W")  # first appearance on the weekly grid (pre-2026)
fs = PW.groupby("codigo")["day"].min(); del PW
Q["_fs"] = Q["codigo"].map(fs)
Q["_new"] = ((Q["_fs"] > "2021-04-01") & ((Q["day"] - Q["_fs"]).dt.days <= 180)).to_numpy()
B = H.baseline("P4Q")["daily"]; res = {}
U = Q[Q["p4q"] & (Q["day"] >= H.START)]; fl = U["sup_iss_90d"].fillna(0) > 0
res["flagged_p4q_rows"] = int(fl.sum()); res["flagged_share_new_bond"] = float(U.loc[fl, "_new"].mean())
res["unflagged_share_new_bond"] = float(U.loc[~fl, "_new"].mean())
for nm, f in {"supply_seasoned_only": lambda x: x["p4q"].to_numpy() & ~((x["sup_iss_90d"].fillna(0).to_numpy() > 0) & ~x["_new"].to_numpy()),
              "supply_new_only": lambda x: x["p4q"].to_numpy() & ~((x["sup_iss_90d"].fillna(0).to_numpy() > 0) & x["_new"].to_numpy()),
              "ex_all_new_bonds": lambda x: x["p4q"].to_numpy() & ~x["_new"].to_numpy()}.items():
    s = H.stats(H.backtest(f, panel=Q)["daily"], bench=B)
    res[nm] = {k: s[k] for k in ["diff_ann_%", "diff_t_nw", "diff_h1_%", "diff_h2_%"]}
print(json.dumps(res, indent=1)); json.dump(res, open(Path(__file__).parent / "newbond_results.json", "w"), indent=1)
