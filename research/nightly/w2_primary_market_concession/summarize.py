"""Global Holm across EVERY variant tried (portfolio family + A2 robustness + harness seasoning), then (stage
'holdout') the sealed 2026 read, once.  Also the blended P4+Q + 20% A2 curves."""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
from scipy import stats as sst

from research.nightly import harness as H
from research.nightly.w2_primary_market_concession import portfolio as PF
from research.nightly.w2_primary_market_concession import seasoning as SE

RES = PF.RES


def global_holm():
    R = json.loads((RES / "results.json").read_text(encoding="utf-8"))
    rows = []
    for r in R["pre2026_table"]:
        if r["variant"].startswith(("A", "B")):
            rows.append((r["variant"], r["vsP4Q_%"], r["t_vsP4Q"], r["p"]))
    for k, v in R["robustness_A2"].items():
        if k.startswith("_") or k == "A2_base":
            continue
        rows.append(("rob:" + k, v["diff_ann_%"], v["diff_t_nw"], v["diff_p"]))
    for r in R["seasoning_harness"]["table"]:
        rows.append(("seas:" + r["variant"], r["vs_bench_%"], r["t_vs_bench"], r["p_vs_bench"]))
    df = pd.DataFrame(rows, columns=["variant", "vsP4Q_%", "t", "p"])
    df["holm_global"] = H.holm(df["p"].astype(float).to_numpy())
    R["global_holm"] = {"n_variants_tried": int(len(df)), "table": df.round(4).to_dict("records")}
    (RES / "results.json").write_text(json.dumps(R, indent=1, default=str), encoding="utf-8")
    print(df.round(4).to_string())
    return R


def blended_curves():
    ni = PF.load(False)
    sel = ni[PF.variants(ni)["A2_prim_conc50_scr"]]
    b = H.baseline("P4Q")
    a2 = PF.sleeve(sel)
    a2_50 = PF.sleeve(sel, cost_bps=50)
    b50 = H.baseline("P4Q", cost_bps=50)["daily"]
    out = {}
    for lab, a, fill in [("P4Q+20%A2 (100% fill)", 0.2, 1.0), ("P4Q+20%A2 (30% fill)", 0.2, 0.3)]:
        s = PF.blend(b["daily"], a2, a * fill)
        st = H.stats(s, bench=b["daily"])
        s50 = PF.blend(b50, a2_50, a * fill)
        st50 = H.stats(s50, bench=b50)
        stu = H.stats(s, bench=H.baseline("U")["daily"])
        out[lab] = {"exCDI_%": st["ann_excess_%"], "exU_%": stu["diff_ann_%"], "vsP4Q_%": st["diff_ann_%"],
                    "t": st["diff_t_nw"], "h1": st["diff_h1_%"], "h2": st["diff_h2_%"], "vsP4Q_50bps_%": st50["diff_ann_%"],
                    "sharpe": st["sharpe"], "maxDD_%": st["max_dd_%"]}
    print(pd.DataFrame(out).T.round(3))
    H.plot_curves({"A2 primary sleeve (conc>50, screens), model marks": a2,
                   "A2 never-printed at cash (pessimistic)": PF.sleeve(sel, noprint="cash"),
                   "P4Q + 20% A2": PF.blend(b["daily"], a2, 0.2)},
                  RES / "equity_A2.png", title="Primary-market sleeve A2 vs baselines (pre-2026, 25 bps)")
    return out


def holdout():
    R = json.loads((RES / "results.json").read_text(encoding="utf-8"))
    PF.main("holdout")
    out, _ = SE.main(holdout=True)
    R = json.loads((RES / "results.json").read_text(encoding="utf-8"))
    R["holdout_2026_seasoning"] = out
    # A2 blend in the holdout
    ni = PF.load(True)
    sel = ni[PF.variants(ni)["A2_prim_conc50_scr"]]
    bh = H.baseline("P4Q", holdout=True)["daily"]
    a2 = PF.sleeve(sel, holdout=True)
    s = H.stats(PF.blend(bh, a2, 0.2), bench=bh, holdout="only")
    R["holdout_2026_blend_P4Q+20%A2"] = {"vsP4Q_%": s["diff_ann_%"], "t": s["diff_t_nw"], "exCDI_%": s["ann_excess_%"]}
    sc = H.stats(PF.sleeve(sel, holdout=True, noprint="cash")["daily"], bench=bh, holdout="only")
    R["holdout_2026_A2_noprint_cash"] = {"vsP4Q_%": sc["diff_ann_%"], "t": sc["diff_t_nw"], "exCDI_%": sc["ann_excess_%"]}
    (RES / "results.json").write_text(json.dumps(R, indent=1, default=str), encoding="utf-8")
    print(out, R["holdout_2026_blend_P4Q+20%A2"], R["holdout_2026_A2_noprint_cash"])


if __name__ == "__main__":
    st = sys.argv[1] if len(sys.argv) > 1 else "pre"
    if st == "pre":
        global_holm()
        R = json.loads((RES / "results.json").read_text(encoding="utf-8"))
        R["blends_A2"] = blended_curves()
        (RES / "results.json").write_text(json.dumps(R, indent=1, default=str), encoding="utf-8")
    else:
        holdout()
