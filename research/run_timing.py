"""Study A — credit timing. Writes research/out/timing_*.png and timing_results.json."""
import sys as _sys
from pathlib import Path as _P
_sys.path.insert(0, str(_P(__file__).resolve().parent.parent))  # run from anywhere

import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rfmonitor.ml import timing

warnings.filterwarnings("ignore")
OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)
COLORS = {"Buy & hold crédito": "#667085", "Momentum 21d > 0": "#b54708", "Logistic": "#1d4ed8",
          "Gradient boosting": "#067647"}

results = timing.run_all()
fig, axes = plt.subplots(len(results), 1, figsize=(11, 4.2 * len(results)), dpi=110)
report = {}
for ax, res in zip(axes, results):
    eq = (1 + res.daily).cumprod()
    for c in eq.columns:
        ax.plot(eq.index, (eq[c] - 1) * 100, label=c, color=COLORS.get(c), lw=1.6 if c != "Buy & hold crédito" else 1.2)
    ax.set_title(f"{res.name} — retorno acumulado acima do CDI (fora da amostra, após custos)", fontsize=11)
    ax.set_ylabel("% acima do CDI")
    ax.axhline(0, color="#999", lw=0.6)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", fontsize=9, frameon=False)
    print(f"\n=== {res.name} ===")
    print(res.stats.round(2).to_string())
    print("latest:", res.latest["date"], res.latest["prob"], res.latest["position"])
    report[res.name] = {"stats": res.stats.round(3).to_dict(orient="index"), "latest": res.latest}
fig.tight_layout()
fig.savefig(OUT / "timing_equity_curves.png")
(OUT / "timing_results.json").write_text(json.dumps(report, indent=2, default=str))
print("\nsaved", OUT / "timing_equity_curves.png")
