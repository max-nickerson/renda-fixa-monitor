"""Diagnostics on the one robust universe-level finding (sector 3m return momentum): is it a stale-mark lead-lag?
Tests the effect on the skip-month target (days 21->63), with own 3m return as an extra control, and on fwd_126.
Rerun: PYTHONPATH=. PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe research/nightly/network_contagion/diag.py"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
import run as R
PW = R.load("W")
PW["own_shock"] = PW["i_any"].fillna(False).astype(float)
PW["fwd_skip"] = (1 + PW["fwd_63"]) / (1 + PW["fwd_21"]) - 1
PW["dok_skip"] = PW["dok_63"]
PW["lab_skip"] = PW["lab_end_63"]
out = {}
ctl = ("cdi_bps", "resid_z", "dur", "r4", "own_shock")
ctl2 = ctl + ("r13",)
for nm in ("sec_r13", "sec_r4", "exp_any", "hold_any", "sib_gap"):
    f = "s_" + nm
    out[nm] = {"fwd_63": R.fama_macbeth(PW, f, "fwd_63", controls=ctl),
               "fwd_63_ctl_own_r13": R.fama_macbeth(PW, f, "fwd_63", controls=ctl2),
               "fwd_126_ctl_own_r13": R.fama_macbeth(PW, f, "fwd_126", controls=ctl2)}
    # skip-month target (21->63): reuse the FM function with a renamed column
    P2 = PW.rename(columns={"fwd_skip": "fwd_42"}).assign(dok_42=PW["dok_63"])
    out[nm]["fwd_21to63_ctl_own_r13"] = R.fama_macbeth(P2, f, "fwd_42", controls=ctl2)
    # stale subsample: own bond did not trade in the last 30 days vs traded
    PW["stale"] = PW["trades_30d"].fillna(0) <= 2
    PW["liquid"] = ~PW["stale"]
    out[nm]["fwd_63_illiquid(<=2 trades/30d)"] = R.fama_macbeth(PW, f, "fwd_63", controls=ctl2, sub="stale")
    out[nm]["fwd_63_liquid(>2 trades/30d)"] = R.fama_macbeth(PW, f, "fwd_63", controls=ctl2, sub="liquid")
    print(nm, pd.DataFrame(out[nm]).T.round(2).to_string(), flush=True)
(Path(__file__).parent / "diag.json").write_text(json.dumps(R.jf(out), indent=1))
