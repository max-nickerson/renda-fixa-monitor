"""Study B — bond selection. Writes research/out/selection_*.png, selection_results.json, and the panel cache."""
import sys as _sys
from pathlib import Path as _P
_sys.path.insert(0, str(_P(__file__).resolve().parent.parent))  # run from anywhere

import json
import sys
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from rfmonitor.config import DATA_DIR
from rfmonitor.ml import selection as sel

warnings.filterwarnings("ignore")
OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)
SNAP = DATA_DIR / "history" / "snapshots.pkl"

t = time.time()
if SNAP.exists() and "--rebuild" not in sys.argv:
    snap = pd.read_pickle(SNAP)
else:
    panel = sel.build_panel()
    print(f"panel {panel.shape} {panel['date'].min().date()}..{panel['date'].max().date()} ({time.time() - t:.0f}s)")
    snap = sel.snapshots(panel)
    snap.to_pickle(SNAP)
print(f"snapshots {snap.shape}, labelled {snap['y'].notna().sum()}, months {snap['rebalance'].nunique()} ({time.time() - t:.0f}s)")
print(snap.groupby("kind")["cdi_bps"].describe().round(0))

naive = sel.walk_forward(snap, cost_bps=25, label="y_naive")
print("\n=== NAIVE label (from the feature trade — contaminated by bid-ask bounce), 25 bps ===")
print(naive["stats"].round(2).to_string())
results = {}
for cost in (0, 25, 50):
    res = sel.walk_forward(snap, cost_bps=cost, exit_q=0.5, blend=True)
    results[cost] = res
    print(f"\n=== REALISTIC (entry at next trade), buy top 20% / hold until out of top 50%, cost {cost} bps ===")
    print(res["stats"].round(2).to_string())

res = results[25]
m = res["monthly"]
cols = [c for c in ["Universo (EW)", "Heurística (screener)", "Ridge", "Gradient boosting", "Blend (heurística+Ridge)"]
        if c in m]
colors = {"Universo (EW)": "#667085", "Heurística (screener)": "#b54708", "Ridge": "#1d4ed8",
          "Gradient boosting": "#067647", "Blend (heurística+Ridge)": "#7a2e9e"}
fig, ax = plt.subplots(2, 1, figsize=(11, 8.5), dpi=110, gridspec_kw={"height_ratios": [2, 1]})
for c in cols:
    eq = (1 + m[c].fillna(0)).cumprod()
    ax[0].plot(eq.index, (eq - 1) * 100, label=c, color=colors[c], lw=2.2 if c.startswith("Blend") else 1.3)
ax[0].set_title("Seleção de debêntures — compra top 20%, mantém até sair do top 50%\n"
                "retorno acima do CDI, fora da amostra, execução no próximo negócio, custo 25 bps/giro", fontsize=11)
ax[0].set_ylabel("% acima do CDI")
ax[0].grid(alpha=0.25)
ax[0].legend(frameon=False, fontsize=9)
for c in cols[1:]:
    rel = (m[c] - m["Universo (EW)"]).fillna(0).cumsum() * 100
    ax[1].plot(rel.index, rel, label=f"{c} − universo", color=colors[c])
ax[1].axhline(0, color="#999", lw=0.6)
ax[1].set_title("Excesso acumulado vs universo (pontos %)")
ax[1].grid(alpha=0.25)
ax[1].legend(frameon=False, fontsize=9)
fig.tight_layout()
fig.savefig(OUT / "selection_equity_curves.png")

# Feature information coefficients (what actually predicts next-month excess return).
el = sel.eligible(snap[snap["y"].notna()])
ic = {}
for f in sel.FEATURES:
    v = [g[f].rank().corr(g["y"].rank()) for _, g in el.groupby("rebalance") if g[f].notna().sum() > 30 and g[f].nunique() > 3]
    if len(v) > 5:
        ic[f] = (pd.Series(v).mean(), pd.Series(v).mean() / (pd.Series(v).std() / len(v) ** 0.5))
ic = pd.DataFrame(ic, index=["IC", "t"]).T.sort_values("IC")
labels = {"resid_z": "spread vs curva de pares (z)", "resid_bps": "spread vs pares (bps)", "cdi_bps": "spread CDI+ (nível)",
          "carry_per_dur": "carry / duration", "issuer_resid": "emissor: spread vs pares", "mom_1m": "Δ spread 1m (abriu)",
          "mom_3m": "Δ spread 3m", "own_z": "spread vs próprio histórico", "dur": "duration", "T": "prazo",
          "log_trades_21d": "liquidez (nº negócios)", "days_since_trade": "dias sem negociar", "ratio": "PU / PU curva",
          "issuer_n": "nº de emissões do emissor"}
fig2, ax2 = plt.subplots(figsize=(9, 6), dpi=110)
ax2.barh([labels.get(i, i) for i in ic.index], ic["IC"],
         color=["#067647" if (v > 0 and t > 2) else "#b42318" if (v < 0 and t < -2) else "#98a2b3"
                for v, t in zip(ic["IC"], ic["t"])])
ax2.axvline(0, color="#666", lw=0.6)
ax2.set_title("O que prevê o excesso de retorno do mês seguinte (IC de Spearman médio, 2021–2026)\n"
              "verde/vermelho = |t| > 2", fontsize=10)
fig2.tight_layout()
fig2.savefig(OUT / "selection_feature_ic.png")
json.dump({str(k): v["stats"].round(3).to_dict(orient="index") for k, v in results.items()},
          open(OUT / "selection_results.json", "w"), indent=2, default=str)
print("saved", OUT / "selection_equity_curves.png")
