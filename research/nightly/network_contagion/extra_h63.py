"""Extra variant (#11): sector-momentum avoid filter on P4+Q with 63-bday tranches (the horizon where the
universe-level sector momentum lives), vs P4+Q on the same engine. Rerun like run.py."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import run as R
from research.nightly import harness as H
PM = R.load("M")
V = R.variants()
r = {k: H.backtest(V[k], panel=PM, hold=63, name=k) for k in ("sector_mom_drop_bottom_q", "avoid_exposure_stress_q5")}
tab = H.compare(r, bench="P4Q", hold=63)
print(tab.round(3).to_string())
(Path(__file__).parent / "extra_h63.json").write_text(json.dumps(R.jf(tab.round(4).to_dict(orient="index")), indent=1))
