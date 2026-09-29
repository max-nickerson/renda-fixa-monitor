"""Stage 4: robustness of the best pre-2026 sleeve (A2: primary, conc > 50 bps, press/EQH/Q screens).

Every row here counts as an additional variant tried (reported with a Holm adjustment over ALL sleeve variants).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from rfmonitor.sources import snd
from research.nightly import harness as H
from research.nightly.w2_primary_market_concession import portfolio as PF

RES = PF.RES


def main():
    ni = PF.load(False)
    V = PF.variants(ni)
    m = V["A2_prim_conc50_scr"]
    sel = ni[m]
    b = H.baseline("P4Q")["daily"]
    out = {}

    def st(x, bench=b):
        s = H.stats(x["daily"] if isinstance(x, dict) else x, bench=bench)
        return {k: s[k] for k in ["ann_excess_%", "diff_ann_%", "diff_t_nw", "diff_p", "diff_h1_%", "diff_h2_%",
                                  "vol_%", "max_dd_%"]}

    # 1) where does the P&L come from: printed vs never-printed (model-marked) positions
    printed = sel["on_grid"] & (sel["fp_pos"] < sel["settle_pos"] + PF.HOLD)
    rows = {"A2_base": PF.sleeve(sel),
            "A2_printed_only(LOOKAHEAD split)": PF.sleeve(sel[printed]),
            "A2_unprinted_only(LOOKAHEAD split)": PF.sleeve(sel[~printed]),
            "A2_noprint_cash": PF.sleeve(sel, noprint="cash"),
            "A2_noprint_carry": PF.sleeve(sel, noprint="carry"),
            "B2_sec_conc50_scr": PF.sleeve(sel, mode="secondary"),
            "A2_no_screens": PF.sleeve(ni[ni["elig"] & (ni["conc"] > 50)]),
            "A2_conc_peer_only>50": PF.sleeve(ni[ni["elig"] & ni["screen_ok"] & (ni["conc_peer"] > 50)]),
            "A2_conc>100": PF.sleeve(ni[ni["elig"] & ni["screen_ok"] & (ni["conc"] > 100)]),
            "A2_conc>25": PF.sleeve(ni[ni["elig"] & ni["screen_ok"] & (ni["conc"] > 25)]),
            "A2_hold63": None, "A2_hold252": None}
    for hh in (63, 252):
        PF.HOLD = hh
        rows[f"A2_hold{hh}"] = PF.sleeve(sel)
    PF.HOLD = 126
    # carry-matched secondary comparison: spread-only selection of new issues (no concession): s_iss >= A2's median
    med = float(sel["s_iss"].median())
    rows["carry_matched_new_issues(s_iss>=A2 median, any conc)"] = PF.sleeve(ni[ni["elig"] & ni["screen_ok"] & (ni["s_iss"] >= med)])
    # carry-matched secondary book from the harness universe: P4Q names with cdi_bps >= A2 median issue spread
    hb = H.backtest(lambda x: x["p4q"].to_numpy() & (x["cdi_bps"] >= med).to_numpy(), name="P4Q_carry>=A2med")
    rows["harness: P4Q & cdi_bps >= A2 median spread"] = hb
    rows["harness: U & cdi_bps >= A2 median spread (not rich, press ok)"] = H.backtest(
        lambda x: (x["cdi_bps"] >= med).to_numpy() & (x["resid_z"].fillna(0) > -1.5).to_numpy()
        & (x["press_neg_30d"].fillna(0) == 0).to_numpy())
    for k, v in rows.items():
        out[k] = st(v)
        if isinstance(v, dict) and "n_pos" in v:
            out[k]["n_pos"] = v["n_pos"]
    out["_A2_median_issue_spread_bps"] = med
    out["_A2_share_printed_within_hold"] = float(printed.mean())
    # 2) exits of the never-printed A2 series (registry reason for leaving, as of today = outcome only)
    t = snd.table().drop_duplicates("Codigo do Ativo").set_index("Codigo do Ativo")
    mot = t.reindex(sel["codigo"])["Motivo de Saida"].astype(str).str[:30]
    out["_exit_reason_printed"] = mot[printed.to_numpy()].value_counts().head(8).to_dict()
    out["_exit_reason_unprinted"] = mot[~printed.to_numpy()].value_counts().head(8).to_dict()
    # 3) concentration: drop the top-10 issuers by contribution
    contrib = {}
    for c, g in sel.groupby("cnpj8"):
        contrib[c] = len(g)
    top = sel["cnpj8"].value_counts().head(10).index
    out["A2_ex_top10_issuers_by_count"] = st(PF.sleeve(sel[~sel["cnpj8"].isin(top)]))
    # 4) by year of the difference
    d = H.monthly(rows["A2_base"]["daily"]) - H.monthly(b.reindex(rows["A2_base"]["daily"].index).fillna(0))
    out["_A2_vsP4Q_by_year_%"] = (d.groupby(d.index.year).mean() * 1200).round(3).to_dict()
    # 5) placebo: random new issues with the same count per month AND the same spread distribution
    #    (draw from eligible new issues in the same s_iss tercile of the month)
    rng = np.random.default_rng(7)
    E = ni[ni["elig"]].copy()
    E["per"] = E["settle"].dt.to_period("M")
    E["sq"] = E.groupby("per")["s_iss"].transform(lambda s: pd.qcut(s.rank(method="first"), 3, labels=False))
    S2 = E.loc[E["codigo"].isin(sel["codigo"])]
    res = []
    for i in range(30):
        picks = []
        for (per, q), c in S2.groupby(["per", "sq"]).size().items():
            pool = E[(E["per"] == per) & (E["sq"] == q)]
            picks.append(pool.sample(n=min(c, len(pool)), random_state=int(rng.integers(1e9))))
        res.append(H.stats(PF.sleeve(pd.concat(picks))["daily"], bench=b)["diff_ann_%"])
    out["_placebo_spread_matched_random_new_issues_vsP4Q"] = {"mean": float(np.mean(res)), "p95": float(np.percentile(res, 95)),
                                                              "actual": out["A2_base"]["diff_ann_%"]}
    R = json.loads((RES / "results.json").read_text(encoding="utf-8"))
    R["robustness_A2"] = out
    (RES / "results.json").write_text(json.dumps(R, indent=1, default=str), encoding="utf-8")
    df = pd.DataFrame({k: v for k, v in out.items() if not k.startswith("_")}).T
    pd.set_option("display.width", 250)
    print(df.round(3).to_string())
    for k, v in out.items():
        if k.startswith("_"):
            print(k, v)


if __name__ == "__main__":
    main()
