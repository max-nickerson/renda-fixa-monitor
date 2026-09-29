import json, pandas as pd
from research.nightly import harness as H
from research.nightly.factor_zoo import features as FZ
from research.nightly.factor_zoo.run import composite, screen_fn
spec = json.load(open("research/nightly/factor_zoo/results.json"))["composite_spec"]
P = composite(FZ.attach(H.load_panel("M")), spec, "zoo")
def md(r, b):
    r = r[r.index < H.HOLDOUT]; return (H.monthly(r) - H.monthly(b.reindex(r.index).fillna(0))).dropna()
out = {}
for nm, iss in {"ex_americanas_00776574": ["00776574"], "ex_12104241": ["12104241"], "ex_top2": ["00776574", "12104241"],
                "ex_top3": ["00776574", "12104241", "61486650"]}.items():
    P2 = P.copy(); m = P2.cnpj8.isin(iss)
    for c in ("univ", "p4q", "p4"): P2.loc[m, c] = False
    for start, lo in (("2022-01-01", None), ("2023-07-01", "2024-01-01")):
        r = H.backtest(screen_fn("zoo", 0.2), panel=P2, start=start)["daily"]
        b = H.backtest("p4q", panel=P2, start=start)["daily"]
        d = md(r, b)
        if lo: d = d[d.index >= lo]
        out[f"{nm}_{'full' if lo is None else 'cleanOOS'}"] = {"ann": round(d.mean()*1200, 3), "t": round(H.nw_t(d, 6), 2),
             "h1": round(d[d.index < H.SPLIT].mean()*1200, 3), "h2": round(d[d.index >= H.SPLIT].mean()*1200, 3)}
    print(nm, out[f"{nm}_full"], out[f"{nm}_cleanOOS"], flush=True)
json.dump(out, open("research/nightly/factor_zoo/verify_robustness/ex_issuers.json", "w"), indent=1)
