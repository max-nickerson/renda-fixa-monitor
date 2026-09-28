"""Equity curves for the strategy lab (rounds 1+2). Writes research/out/lab_equity_curves.png."""
import sys as _sys
from pathlib import Path as _P
_sys.path.insert(0, str(_P(__file__).resolve().parent.parent))  # run from anywhere

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from rfmonitor.config import DATA_DIR

OUT = Path(__file__).parent / "out"
tag = _sys.argv[1] if len(_sys.argv) > 1 else ""
c1 = pd.read_pickle(DATA_DIR / "history" / "lab_curves.pkl")
c2 = pd.read_pickle(DATA_DIR / "history" / f"lab2_curves{tag}.pkl")
cur = pd.concat([c1, c2.drop(columns=[c for c in c2 if c in c1])], axis=1)

groups = {
    "Segurar e sair quando preciso": ["U0 Universo: compra e segura", "U1 Segura + sai se spread abre >100bps/4s",
                                      "U2 Segura + sai com notícia ruim (CVM)", "U6 Segura + sai se ML(com notícia) risco top 5%",
                                      "A2 A1 + só compra se z≥0 (não compra caro/justo-caro)"],
    "Comprar barato / vender caro / contrarian": ["U0 Universo: compra e segura", "RV1 Barato vs pares (z≥1.5) → vende no justo",
                                                  "B1 Contrarian: barato (z≥1.5) COM notícia 30d",
                                                  "B3 Contrarian: spread abriu >50bps em 4s SEM notícia",
                                                  "MR1 Spread alto vs próprio histórico → vende na média",
                                                  "RV4 Só os CAROS (diagnóstico)"],
    "Carry, ML e regime (melhores)": ["U0 Universo: compra e segura", "C2 Carry alto e não-caro", "C3 C2 + regime momentum",
                                      "ML1 Ridge sem notícia (top20/sai>50%)", "ML2 Ridge com notícia (top20/sai>50%)",
                                      "M3 M1 + sai com notícia ruim + regime"],
}
fig, axes = plt.subplots(3, 1, figsize=(12, 14), dpi=105)
for ax, (title, cols) in zip(axes, groups.items()):
    for c in cols:
        if c not in cur:
            continue
        eq = (1 + cur[c].fillna(0)).cumprod()
        best = c.startswith(("C3", "M3"))
        ax.plot(eq.index, (eq - 1) * 100, label=c, lw=2.4 if best else (1.0 if c.startswith("U0") else 1.4),
                color="#667085" if c.startswith("U0") else None, ls="--" if c.startswith("U0") else "-")
    ax.set_title(f"{title} — excesso sobre CDI, semanal, fora da amostra 2022–26, execução 1 semana depois, 25 bps",
                 fontsize=10)
    ax.axhline(0, color="#999", lw=0.6)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, frameon=False, loc="lower left")
    ax.set_ylabel("% acima do CDI")
fig.tight_layout()
fig.savefig(OUT / f"lab_equity_curves{tag}.png")
print("saved")
