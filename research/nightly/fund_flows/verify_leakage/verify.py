"""Adversarial leakage check of fund_flows X4 (P4+Q ex bottom-quintile cshare). Read-only w.r.t. their files."""
import json, time
import numpy as np, pandas as pd
from research.nightly import harness as H
from research.nightly.fund_flows import features as F

OUT = "research/nightly/fund_flows/verify_leakage"
t0 = time.time()
P0 = H.load_panel("M")
B = H.baseline("P4Q")
dd = sorted(pd.to_datetime(P0["day"].unique()))
res = {}

def ex_cshare(x):
    r = x["cshare"].rank(pct=True)
    return x["p4q"].to_numpy() & ~(r <= 0.2).fillna(False).to_numpy()

deb, _ = F._load()
refs = sorted(deb["ref"].unique())
res["cda_refs"] = [str(pd.Timestamp(refs[0]).date()), str(pd.Timestamp(refs[-1]).date()), len(refs)]
# positions per ref (look for the rewrite/confidential gaps)
res["rows_per_ref_tail"] = {str(pd.Timestamp(k).date()): int(v) for k, v in deb.groupby("ref").size().tail(10).items()}

for lag in (60, 90, 120, 180):
    F.CDA_LAG_DAYS = lag
    Bf = F.bond_features(dd)
    P = P0.merge(Bf[["day", "codigo", "cshare", "nh"]], on=["day", "codigo"], how="left")
    r = H.backtest(ex_cshare, panel=P, name=f"X4_lag{lag}")
    s = H.stats(r["daily"], bench=B["daily"])
    r50 = H.backtest(ex_cshare, panel=P, cost_bps=50)
    s50 = H.stats(r50["daily"], bench=H.baseline("P4Q", cost_bps=50)["daily"])
    # orphans X5
    r5 = H.backtest(lambda x: x["p4q"].to_numpy() & (x["nh"].fillna(0) > 0).to_numpy(), panel=P)
    s5 = H.stats(r5["daily"], bench=B["daily"])
    res[f"lag{lag}"] = {"X4_diff": s["diff_ann_%"], "X4_t": s["diff_t_nw"], "h1": s["diff_h1_%"], "h2": s["diff_h2_%"],
                        "X4_diff_50": s50["diff_ann_%"], "X5_orphan_diff": s5["diff_ann_%"], "X5_t": s5["diff_t_nw"],
                        "share_univ_orphan": float((P.loc[P["univ"], "nh"].fillna(0) == 0).mean())}
    print(lag, res[f"lag{lag}"], round(time.time() - t0), flush=True)
    if lag == 60:
        # placebo 50 draws
        pl = []
        rng = np.random.default_rng(1)
        for i in range(50):
            seed = int(rng.integers(1e9))
            def rnd(x, seed=seed):
                base = x["p4q"].to_numpy(); keep = ex_cshare(x); nd = int(base.sum() - (base & keep).sum())
                g = np.random.default_rng(seed + int(x["dpos"].iloc[0])); idx = np.flatnonzero(base); m = base.copy()
                if nd > 0 and len(idx) > nd: m[g.choice(idx, nd, replace=False)] = False
                return m
            pl.append(H.stats(H.backtest(rnd, panel=P)["daily"], bench=B["daily"])["diff_ann_%"])
        pl = np.array(pl)
        res["placebo50"] = {"mean": float(pl.mean()), "sd": float(pl.std()), "p95": float(np.percentile(pl, 95)),
                            "share_ge": float((pl >= s["diff_ann_%"]).mean())}
        print(res["placebo50"], flush=True)
        # orphan survivorship check: last grid date of orphans vs held, fwd_126 vs rec40
        u = P[P["univ"] & P["p4q"]]
        u = u.assign(orph=u["nh"].fillna(0) == 0)
        res["orphan_profile_p4q"] = u.groupby("orph")[["fwd_126", "fwd_126_rec40", "cdi_bps", "ratio"]].mean().round(4).to_dict()
json.dump(res, open(f"{OUT}/results.json", "w"), indent=1, default=float)
print("done", round(time.time() - t0))
