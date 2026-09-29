"""Build point-in-time target weights for every construction variant on the monthly panel.

Candidate set = P4+Q names of the decision date (the alpha source), except 'wide_mv' (optimizer picks from a wider
quality-filtered set). Cached to data/history/nightly/portfolio_construction/weights_{tag}.pkl.
"""
from __future__ import annotations
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

from research.nightly import harness as H
from research.nightly.portfolio_construction import risk as RK, construct as CO
from research.nightly.portfolio_construction.data import build as build_liq, OUT

AUM = 500e6
PART = 0.25          # max position = 25% of the bond's R$ volume over the last 91 calendar days (build in ~a quarter)
MIN_LOT = 0.0025     # R$1.25m on R$500m


def composite(x: pd.DataFrame) -> pd.Series:
    """Simple PIT composite within the universe of the date: 0.5 carry + 0.25 cheapness vs peer curve + 0.25
    fundamental quality (uncovered = neutral). All pct ranks in [0, 1], higher = better."""
    c = x["cdi_bps"].rank(pct=True)
    r = x["resid_bps"].rank(pct=True).fillna(0.5)
    q = x["f_quality"].rank(pct=True).fillna(0.5)
    return 0.5 * c + 0.25 * r + 0.25 * q


def wide_mask(x):
    return ((x["cdi_pct"] <= 0.5) & ~x["rich"].astype(bool) & (x["press_neg_30d"].fillna(0) == 0)
            & ~x["worstQ"].fillna(False).astype(bool))


def liq_wmax(s, aum):
    v = s["vol91_brl"].fillna(0).to_numpy()
    return PART * v / aum


def build(holdout: bool = False, tag: str = "pre", start="2022-01-01"):
    path = OUT / f"weights_{tag}.pkl"
    if path.exists():
        return pd.read_pickle(path)
    P = H.load_panel("M", holdout=holdout)
    L = build_liq()[["codigo", "day", "vol91_brl", "adv_brl"]]
    X = P[P["univ"]].merge(L, on=["codigo", "day"], how="left")
    X["sector"] = X["sector"].astype(str)
    X["cnpj8"] = X["cnpj8"].astype(str)
    X["comp"] = X.groupby("dpos", group_keys=False).apply(composite)
    X = X[X["day"] >= pd.Timestamp(start)]
    rows, meta = [], []
    prev_tc = pd.Series(dtype=float)
    t0 = time.time()
    for p, x in X.groupby("dpos", sort=True):
        day = x["day"].iloc[0]
        bp = RK.block_params(p, x["b"].to_numpy(), x["sector"].to_numpy(), x["cnpj8"].to_numpy())
        s = x[x["p4q"]].reset_index(drop=True)
        wd = x[wide_mask(x) | x["p4q"]].reset_index(drop=True)
        if len(s) < 5:
            continue
        ud = float(x["dur"].mean())
        band = (ud - 1.0, ud + 1.0)
        Sd, vd = RK.cov(p, s, "dts", bp)
        Se, ve = RK.cov(p, s, "emp", bp)
        mu = 0.5 * s["cdi_bps"].to_numpy() / 1e4
        c10 = {"issuer_cap": 0.10}
        c5 = {"issuer_cap": 0.05}
        c5s = {"issuer_cap": 0.05, "sector_cap": 0.25}
        fund = {"issuer_cap": 0.05, "sector_cap": 0.25, "dur_band": band, "wmax": liq_wmax(s, AUM), "min_lot": MIN_LOT}
        W = {}
        n = len(s)
        eq = np.ones(n)
        W["EW_cap10"] = CO.project(s, eq, c10)
        W["EW_cap5"] = CO.project(s, eq, c5)
        W["EW_cap5_sec25"] = CO.project(s, eq, c5s)
        # liquidity / fund constraints (relax liquidity by x1.5 steps if infeasible; the multiplier is recorded)
        for aum_tag, aum in (("fund500m", 500e6), ("fund2bn", 2e9)):
            k, v = 1.0, None
            while v is None and k < 50:
                f2 = dict(fund, wmax=np.minimum(liq_wmax(s, aum) * k, 1.0))
                v = CO.project(s, eq, f2)
                if v is None:
                    k *= 1.5
            W[f"EW_{aum_tag}"] = v
            meta.append({"day": day, "variant": f"EW_{aum_tag}", "liq_relax": k})
        pct = s["comp"].rank(pct=True).to_numpy()
        W["score_tilt"] = CO.project(s, 0.5 + pct, c10)
        W["carry_prop"] = CO.project(s, s["cdi_bps"].to_numpy(), c10)
        top = np.argsort(-s["comp"].to_numpy())[:50]
        m50 = np.zeros(n); m50[top] = 1.0
        W["top50_comp"] = CO.project(s, m50, c10) if n > 50 else W["EW_cap10"]
        W["invvol_dts"] = CO.project(s, 1 / vd, c10)
        W["minvar_emp"] = CO.minvar(s, Se, c5s)
        W["maxdiv_dts"] = CO.maxdiv(s, Sd, vd, c10)
        W["erc_issuer_emp"] = CO.erc_issuer(s, Se, c10)
        W["mv_emp_g20"] = CO.meanvar(s, Se, mu, 20, c5s)
        W["mv_emp_g100"] = CO.meanvar(s, Se, mu, 100, c5s)
        W["mv_dts_g100"] = CO.meanvar(s, Sd, mu, 100, c5s)
        # carry per unit of de-smoothed empirical variance, simple tilt (no optimizer)
        W["carry_over_var_tilt"] = CO.project(s, mu / np.diag(Se), c5)
        # the MV winner under real-fund constraints (liquidity at R$500m, duration band, sector, min lot)
        k, v = 1.0, None
        while v is None and k < 50:
            v = CO.meanvar(s, Se, mu, 20, dict(fund, wmax=np.minimum(liq_wmax(s, AUM) * k, 1.0)))
            if v is None:
                k *= 1.5
        W["mv_emp_g20_fund500m"] = v
        meta.append({"day": day, "variant": "mv_emp_g20_fund500m", "liq_relax": k})
        W["mv_emp_g20_durband"] = CO.meanvar(s, Se, mu, 20, dict(c5s, dur_band=band))
        wp = prev_tc.reindex(s["codigo"]).fillna(0).to_numpy()
        W["mv_emp_g20_tc"] = CO.meanvar(s, Se, mu, 20, c5s, w_prev=wp if prev_tc.size else None, tc=0.002)
        if W["mv_emp_g20_tc"] is not None:
            prev_tc = pd.Series(W["mv_emp_g20_tc"], index=s["codigo"].to_numpy())
        # default-scenario CVaR (6m horizon)
        sc, pdh = RK.default_scenarios(s, bp, Sig=Sd * 0.5 / 0.5, n=2000)
        CO.SOLVER = "HIGHS"
        W["cvar_l1"] = CO.cvar_opt(s, sc, 1.0, c5s)
        W["cvar_l5"] = CO.cvar_opt(s, sc, 5.0, c5s)
        CO.SOLVER = "CLARABEL"
        # wide candidate set: the optimizer does the selection
        Sw, vw = RK.cov(p, wd, "emp", bp)
        muw = 0.5 * wd["cdi_bps"].to_numpy() / 1e4
        ww = CO.meanvar(wd, Sw, muw, 100, c5s)
        # ex-ante tail: common default scenarios on the wide set (superset of the P4+Q candidates)
        scw, _ = RK.default_scenarios(wd, bp, n=4000, seed=11)
        pos_w = pd.Series(np.arange(len(wd)), index=wd["codigo"].to_numpy())
        for nm, w in list(W.items()) + [("wide_mv", ww)]:
            if w is None:
                continue
            cand = wd if nm == "wide_mv" else s
            keep = w > 1e-6
            dfw = pd.DataFrame({"day": day, "codigo": cand["codigo"].to_numpy()[keep], "weight": w[keep],
                                "variant": nm})
            rows.append(dfw)
            idx = pos_w.reindex(dfw["codigo"]).to_numpy()
            ok = np.isfinite(idx)
            loss = -(scw[:, idx[ok].astype(int)] @ dfw["weight"].to_numpy()[ok])
            iw = pd.Series(w[keep], index=cand["cnpj8"].to_numpy()[keep]).groupby(level=0).sum()
            sw = pd.Series(w[keep], index=cand["sector"].to_numpy()[keep]).groupby(level=0).sum()
            vol = cand["vol91_brl"].fillna(0).to_numpy()[keep]
            meta.append({"day": day, "variant": nm, "n": int(keep.sum()), "eff_n_issuer": float(1 / (iw ** 2).sum()),
                         "max_issuer": float(iw.max()), "max_sector": float(sw.max()),
                         "dur": float(cand["dur"].to_numpy()[keep] @ w[keep]),
                         "spread_bps": float(cand["cdi_bps"].to_numpy()[keep] @ w[keep]),
                         "cvar99_6m_%": float(RK.cvar(loss, 0.99) * 100),
                         "exp_loss_6m_%": float(loss.mean() * 100),
                         # liquidity breach at R$500m: weight above 25% of 91d volume
                         "liq_excess_500m": float(np.clip(w[keep] - PART * vol / 500e6, 0, None).sum()),
                         "liq_excess_2bn": float(np.clip(w[keep] - PART * vol / 2e9, 0, None).sum()),
                         "days_to_trade_500m_med": float(np.median(w[keep] * 500e6 /
                                                                   np.maximum(0.2 * vol / 63, 1)))})
        meta.append({"day": day, "variant": "_params", **bp, "n_cand": len(s), "n_wide": len(wd),
                     "pd6m_med": float(np.median(pdh))})
        print(f"{day.date()} n={len(s)} wide={len(wd)} {time.time() - t0:.0f}s", flush=True)
    Wdf = pd.concat(rows, ignore_index=True)
    M = pd.DataFrame(meta)
    pd.to_pickle((Wdf, M), path)
    return Wdf, M


if __name__ == "__main__":
    hold = "--holdout" in sys.argv
    build(holdout=hold, tag="all" if hold else "pre")
