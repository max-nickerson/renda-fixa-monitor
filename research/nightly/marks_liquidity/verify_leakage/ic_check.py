import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from research.nightly import harness as H
from research.nightly.marks_liquidity import signals as SG
PM = SG.signals(H.load_panel("M")); PM["noise_part"] = PM["resid_bps"] - PM["kf_resid_bps"]
for f in ("resid_bps", "kf_resid_bps", "vw_resid_bps", "noise_part", "kf_gap"):
    r = H.ic(PM, f, "fwd_126"); print(f, round(r["mean"], 3), round(r["t_nw"], 1), r["n_dates"])
# pre-2024 only (KF params in-sample period) vs 2024-25
for lab, m in (("<2024", PM["day"] < "2024-01-01"), (">=2024", PM["day"] >= "2024-01-01")):
    for f in ("resid_bps", "kf_resid_bps"):
        r = H.ic(PM[m], f, "fwd_126"); print(lab, f, round(r["mean"], 3), round(r["t_nw"], 1))
