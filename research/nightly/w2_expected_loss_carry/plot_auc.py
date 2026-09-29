"""AUC by year (pre-2026 outcomes; 2025 censored at 2025-12-31) for the walk-forward PD models."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
e = json.load(open("research/nightly/w2_expected_loss_carry/model_eval.json"))["eval"]
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
for a, sub in zip(ax, ["debenture_issuers", "debenture_listed", "debenture_unlisted"]):
    d = {int(k): v for k, v in e[sub].items() if k != "all"}
    yrs = sorted(d)
    for m, c in (("pd_logit", "#1f77b4"), ("pd_lgbm", "#d62728")):
        a.plot(yrs, [d[y][f"auc_{m}"] for y in yrs], "o-", color=c, label=m)
    for y in yrs:
        a.annotate(f"{d[y]['pos_issuers']} iss", (y, 0.36), ha="center", fontsize=8, color="grey")
    a.axhline(0.5, color="k", lw=0.6)
    a.axhline(e["benchmarks_debenture_issuers"]["auc_spread_level"], color="grey", ls="--", lw=1,
              label="spread level alone (pooled)")
    a.set_title(sub.replace("_", " ") + " (12m distress, OOS)")
    a.set_ylim(0.3, 1.0)
    a.grid(alpha=0.3)
ax[0].set_ylabel("AUC")
ax[0].legend(fontsize=8)
fig.tight_layout()
fig.savefig("research/nightly/w2_expected_loss_carry/auc_by_year.png", dpi=110)
