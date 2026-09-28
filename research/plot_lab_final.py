"""Final lab chart (rate-hedged, CVM + press news). Writes research/out/lab_final_equity_curves.png."""
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
cur = pd.read_pickle(DATA_DIR / "history" / "lab2_curves_gdelt_hedged.pkl")
groups = {
    "1 · Segurar tudo e sair quando preciso": ["U0 Universo: compra e segura", "A1 Segura tudo, vende quando fica CARO (z≤−1.5)",
        "P1 Universo + sai com pico de notícia ruim (imprensa)", "P2 Universo + sai imprensa OU CVM",
        "A3 A1 + regime momentum", "P6 Universo + regime só imprensa (índice RJ/calote)"],
    "2 · Comprar barato / contrarian / notícias": ["U0 Universo: compra e segura", "B1 Contrarian: barato (z≥1.5) COM notícia 30d",
        "B3 Contrarian: spread abriu >50bps em 4s SEM notícia", "P9 Contrarian: barato c/ imprensa negativa, sem distress CVM",
        "P10 Só emissores com pico de imprensa (diagnóstico)", "RV4 Só os CAROS (diagnóstico)"],
    "3 · Carry, ML e os vencedores": ["U0 Universo: compra e segura", "C2 Carry alto e não-caro", "C3 C2 + regime momentum",
        "P4 C3 + não compra c/ imprensa negativa 30d", "M3 M1 + sai com notícia ruim + regime",
        "P8 P7 + regime momentum"],
}
fig, axes = plt.subplots(3, 1, figsize=(12, 14), dpi=105)
for ax, (title, cols) in zip(axes, groups.items()):
    for c in cols:
        if c not in cur:
            continue
        eq = (1 + cur[c].fillna(0)).cumprod()
        star = c.startswith(("P4", "C3"))
        ax.plot(eq.index, (eq - 1) * 100, label=c, lw=2.6 if c.startswith("P4") else 2.0 if star else
                (1.1 if c.startswith("U0") else 1.4), color="#667085" if c.startswith("U0") else None,
                ls="--" if c.startswith("U0") else "-")
    ax.set_title(f"{title}\nexcesso sobre CDI, crédito puro (juros hedgeados), semanal, fora da amostra 2022–26, "
                 "execução 1 semana depois, 25 bps", fontsize=10)
    ax.axhline(0, color="#999", lw=0.6)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax.set_ylabel("% acima do CDI")
fig.tight_layout()
fig.savefig(OUT / "lab_final_equity_curves.png")

# summary table for the dashboard
res = json.load(open(OUT / "lab2_results_gdelt_hedged.json"))
rows = {}
for k, v in res.items():
    name, cfg = k.split(" | ")
    rows.setdefault(name, {})[cfg] = v
table = []
for name, cfgs in rows.items():
    a, b, c = cfgs.get("lag1 25bps", {}), cfgs.get("lag2 25bps", {}), cfgs.get("lag1 50bps", {})
    table.append({"strategy": name, "vs_u0": a.get("vs_bench_ann_%"), "vs_u0_lag2": b.get("vs_bench_ann_%"),
                  "vs_u0_50bps": c.get("vs_bench_ann_%"), "excess": a.get("excess_ann_%"), "sharpe": a.get("sharpe"),
                  "max_dd": a.get("max_dd_%"), "n": a.get("avg_bonds")})
json.dump(table, open(OUT / "lab_summary.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=float)
print("saved")
