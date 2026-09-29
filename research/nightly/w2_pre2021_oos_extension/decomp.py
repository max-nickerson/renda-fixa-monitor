"""Decomposition of P7 on the rebuilt 2015-2020 grid: equity-health screen alone vs carry weighting alone
(both frozen p7 components; counted as 2 extra variants in the Holm family). Appends to results.json."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from research.nightly import harness as H
from research.nightly.combined import p7
from research.nightly.w2_pre2021_oos_extension import run as RUN

OUT = RUN.OUT
C = pd.read_pickle(OUT / "core.pkl")
H._MEM.clear(); H._MEM["core"] = C; H._MEM["idx"] = pd.read_pickle(H.CACHE / "indices.pkl")
P = pd.read_pickle(OUT / "panel_M.pkl"); P = P[P["day"] <= RUN.OOS1]
S = RUN.OOS0
R = {"P4Q": H.backtest("p4q", panel=P, start=S),
     "P4Q_eqh_EW": H.backtest(p7.make_signal(construct=None), panel=P, start=S, **RUN.KW7),
     "P4Q_carry_noscreen": H.backtest(p7.make_signal(screen=(), aum=None), panel=P, start=S, **RUN.KW7),
     "P7": H.backtest(p7.make_signal(), panel=P, start=S, **RUN.KW7),
     "P7_noliq": H.backtest(p7.make_signal(aum=None), panel=P, start=S, **RUN.KW7)}
out = {}
for nm in ("P4Q_eqh_EW", "P4Q_carry_noscreen"):
    out[nm] = {pk: RUN.mstats(R[nm]["daily"], R["P4Q"]["daily"], a, z) for pk, (a, z) in RUN.PERIODS.items()}
# Holm over the whole 7-variant family vs P4+Q (P4, P7, P7_noliq, ZOO, P4Q_mom63, + these 2)
res = json.loads((RUN.HERE / "results.json").read_text())
fam = {k: res["oos_25bps"][k]["2015_20"]["diff_p"] for k in ("P4", "P7", "P7_noliq", "ZOO", "P4Q_mom63")}
fam.update({k: out[k]["2015_20"]["diff_p"] for k in out})
hp = H.holm(list(fam.values()))
res["decomposition_P7"] = out
res["holm_7_variant_family"] = dict(zip(fam, [round(float(x), 4) for x in hp]))
res["n_variants_vs_P4Q"] = 7
(RUN.HERE / "results.json").write_text(json.dumps(RUN.jsonable(res), indent=1))
print(json.dumps(RUN.jsonable({k: v["2015_20"] for k, v in out.items()}), indent=0)); print(res["holm_7_variant_family"])
for nm in out: print(nm, {pk: out[nm][pk]["diff_%"] for pk in out[nm]})
