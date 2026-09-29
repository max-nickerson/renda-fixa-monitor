"""SEALED HOLDOUT, run once after every choice was frozen: best pre-2026 variant (avoid_exposure_stress_q5) and the
combined avoid_any_neighbour filter, decisions >= 2026-01-01, stats on months >= 2026-01 only."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import run as R
from research.nightly import harness as H
R.HOLDOUT = True
PM = R.load("M")
V = R.variants()
bh = H.baseline("P4Q", holdout=True)["daily"]
uh = H.baseline("U", holdout=True)["daily"]
out = {}
for k in ("avoid_exposure_stress_q5", "avoid_any_neighbour"):
    rh = H.backtest(V[k], panel=PM, holdout=True)
    a = H.stats(rh["daily"], bench=bh, holdout="only")
    u = H.stats(rh["daily"], bench=uh, holdout="only")
    out[k] = {"exCDI_%": a["ann_excess_%"], "vs_p4q_%": a["diff_ann_%"], "t_vs_p4q": a["diff_t_nw"],
              "vs_u_%": u["diff_ann_%"], "n_months": None}
out["P4Q"] = {"exCDI_%": H.stats(bh, holdout="only")["ann_excess_%"]}
out["U"] = {"exCDI_%": H.stats(uh, holdout="only")["ann_excess_%"]}
print(out)
(Path(__file__).parent / "holdout.json").write_text(json.dumps(R.jf(out), indent=1))
