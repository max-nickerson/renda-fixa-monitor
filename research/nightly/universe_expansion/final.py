"""Charts, sealed-holdout report (run ONCE, after all choices were frozen on pre-2026 data) and results.json.
Requires run.py (series_pre2026.pkl, results_pre2026.json) and feasibility.py (feasibility.json)."""
import json
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from research.nightly import harness as H  # noqa: E402
from research.nightly.universe_expansion import run as R  # noqa: E402

OUTD = Path("research/nightly/universe_expansion")
D = Path("data/history/nightly/universe_expansion")
BEST = "MIX_80deb_20fidc"  # frozen from the pre-2026 table (best paired diff vs P4+Q); see README

ser = pickle.load(open(D / "series_pre2026.pkl", "rb"))
S, A = ser["S"], ser["A"]
pre = json.load(open(OUTD / "results_pre2026.json"))
feas = json.load(open(OUTD / "feasibility.json"))

# ---------------- charts (pre-2026)
show = {"MIX 80% P4+Q / 20% FIDC-sr P4Q": A["MIX_80deb_20fidc"],
        "MIX 80% P4+Q / 20% IMA-B5 (trend)": A["MIX_80deb_20ntnb_trend"],
        "MIX 70/15/15 P4+Q/IMA-B5/FII-CRI": A["MIX_70deb_15ntnb_15fii"],
        "NTN-B IMA-B5 (ETF fee)": S["NTNB_IMAB5"],
        "FIDC senior P4Q (paper)": S["FIDC_P4Q"],
        "FII-CRI P4Q (listed)": S["FII_P4Q"]}
H.plot_curves(show, OUTD / "equity_total_return.png",
              title="Universe expansion: sleeves and multi-sleeve mixes vs debenture refs (pre-2026, net 25 bps)")

U = H.baseline("U")["daily"]
fig, axs = plt.subplots(1, 2, figsize=(15, 5.6), dpi=110)
idx = U.index[(U.index >= H.START) & (U.index < H.HOLDOUT)]
for nm, s in {**show, "P4+Q": H.baseline("P4Q")["daily"], "P4": H.baseline("P4")["daily"]}.items():
    s = s.reindex(idx).fillna(0)
    ls = "--" if nm in ("P4+Q", "P4") else "-"
    axs[0].plot(idx, ((1 + s - U.reindex(idx).fillna(0)).cumprod() - 1) * 100, ls, lw=1.5, label=nm)
axs[0].set_title("Cumulative excess vs debenture universe (%)", fontsize=10)
pairs = {"FII-CRI: P4Q vs own universe": (S["FII_P4Q"], S["FII_U"]),
         "FII-CRI: carry vs own universe": (S["FII_carry"], S["FII_U"]),
         "FII-CRI: quality-only vs own universe": (S["FII_Qonly"], S["FII_U"]),
         "FIDC-sr: P4Q vs own universe": (S["FIDC_P4Q"], S["FIDC_U"]),
         "FIDC-sr: P4Q (missing=0%) vs U (missing=0%)": (S["FIDC_P4Q_missing0"], S["FIDC_U_missing0"])}
for nm, (a, b) in pairs.items():
    a, b = a.reindex(idx).fillna(0), b.reindex(idx).fillna(0)
    axs[1].plot(idx, ((1 + a - b).cumprod() - 1) * 100, lw=1.5, label=nm)
axs[1].set_title("Does P4-style selection work inside the new sleeves? (cum. excess vs sleeve universe, %)", fontsize=10)
for ax in axs:
    ax.axhline(0, color="k", lw=0.6); ax.grid(alpha=0.25); ax.legend(fontsize=7, frameon=False)
fig.tight_layout(); fig.savefig(OUTD / "cum_excess.png"); plt.close(fig)

# ---------------- sealed holdout: ONCE
P4Qh = H.baseline("P4Q", holdout=True)["daily"]
fidc_h = R.fidc_series("P4Q", holdout=True)[0]
fidc_u_h = R.fidc_series("U", holdout=True)[0]
IS = R.index_sleeves(end=None)
mix_h = R.mix({"deb": P4Qh, "ntnb": IS["NTNB_IMAB5"], "fii": P4Qh * 0, "fidc": fidc_h},
              {"deb": .8, "ntnb": 0, "fii": 0, "fidc": .2})
X = R.fii_matrix()
fii_h = R.sleeve_book(X, R.fii_targets("P4Q", holdout=True), 25, end=None)
fii_u_h = R.sleeve_book(X, R.fii_targets("U", holdout=True), 25, end=None)
hold = {"note": "decisions >= 2026-01-01 only; reported once; FIDC months through 2026-08 (report lag)",
        BEST: H.stats(mix_h, bench=P4Qh, holdout="only"),
        "P4Q": H.stats(P4Qh, holdout="only"),
        "FIDC_P4Q_vs_FIDC_U": H.stats(fidc_h, bench=fidc_u_h, holdout="only"),
        "NTNB_IMAB5": H.stats(IS["NTNB_IMAB5"], holdout="only"),
        "FII_P4Q_vs_FII_U": H.stats(fii_h, bench=fii_u_h, holdout="only")}

tab = pd.DataFrame(pre["table"]).T
b = tab.loc[BEST]
res = {"slug": "universe_expansion", "harness_version": 4, "best_variant": BEST,
       "n_variants_tried": pre["n_variants_tried"],
       "best_pre2026": {k: b[k] for k in tab.columns},
       "best_50bps_vs_P4Q50": pre.get(f"{BEST}_50bps_vs_P4Q50"),
       "pre2026": pre, "holdout": hold, "feasibility": feas}
json.dump(res, open(OUTD / "results.json", "w"), indent=1, default=str)
print(json.dumps(hold, indent=1, default=str))
print(tab.loc[BEST])
