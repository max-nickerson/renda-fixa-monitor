"""marks_liquidity / step 1: point-in-time fair-value marks and liquidity features for every debenture grid day.

Inputs (read-only): SND daily trade aggregates data/history/snd_trades_YYYYMM.csv.gz (qty, #trades, PU medio,
% PU da curva per bond-day), lab_daily.pkl (spread of each trade day, duration, kind, issuer, peer), harness grid days.

Outputs (data/history/nightly/marks_liquidity/):
  trades.pkl   one row per (codigo, trade date): s (CDI+ equiv. spread of that day's print, bps), dur, vol_brl,
               n_trades, ticket, ratio, cnpj8, peer, kind + cleaning flags
  params.json  noise / process variances estimated on trade data dated < 2024-01-01 only (training window)
  marks.pkl    one row per (codigo, grid day) of lab_daily: kf_s / kf_sd (Kalman fair spread, info <= close of day),
               kf_prior (one-step prediction before today's print), vw_s (volume-weighted clean spread of the last
               <= 5 clean prints within 10 bdays), last clean spread, liquidity features over the past 63 bdays
               (adv63, tdays63, ticket63, ntr63), bond-level Roll half-spread (252 bdays).
Everything at day d uses only prints dated <= d (SND publishes the day's aggregates after the close).
"""
from __future__ import annotations

import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from research.nightly import harness as H  # noqa: E402

OUT = ROOT / "data" / "history" / "nightly" / "marks_liquidity"
OUT.mkdir(parents=True, exist_ok=True)
HERE = Path(__file__).resolve().parent
TRAIN_END = pd.Timestamp("2024-01-01")     # parameters estimated on prints before this date only
T0 = time.time()

TICKET_BINS = [0, 5e4, 2e5, 1e6, 5e6, np.inf]
TICKET_LAB = ["<50k", "50-200k", "200k-1m", "1-5m", ">5m"]


def log(*a):
    print(f"[marks +{time.time() - T0:6.1f}s]", *a, flush=True)


# ------------------------------------------------------------------------------------------------ trades table
def build_trades() -> pd.DataFrame:
    fs = sorted(glob.glob(str(ROOT / "data" / "history" / "snd_trades_*.csv.gz")))
    raw = pd.concat([pd.read_csv(f, parse_dates=["date"]) for f in fs], ignore_index=True)
    raw["vol_brl"] = raw["qty"] * raw["pu_avg"]
    raw = raw[["codigo", "date", "qty", "trades", "pu_avg", "vol_brl"]]
    g = pd.read_pickle(ROOT / "data" / "history" / "lab_daily.pkl")
    t = g.loc[g["day"] == g["date"], ["codigo", "date", "ratio", "cdi_bps", "dur", "kind", "peer", "cnpj8", "incent",
                                      "contract"]].copy()
    del g
    t["codigo"] = t["codigo"].astype(str)
    raw["codigo"] = raw["codigo"].astype(str)
    t = t.merge(raw, on=["codigo", "date"], how="left")
    t = t.rename(columns={"cdi_bps": "s", "trades": "n_trades"})
    t["ticket"] = t["vol_brl"] / t["n_trades"].clip(lower=1)
    t["tbin"] = pd.cut(t["ticket"], TICKET_BINS, labels=False, right=False).fillna(0).astype(int)
    t = t.sort_values(["codigo", "date"]).reset_index(drop=True)
    return t


# ------------------------------------------------------------------------------------------------ parameter estimation
def _rv(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return float((1.4826 * np.median(np.abs(v - np.median(v)))) ** 2) if len(v) > 50 else np.nan


def _grp_kind(k):
    return "DI" if k == "DI_SPREAD" else "IPCA_PRE"


def estimate_params(t: pd.DataFrame, dd: pd.DatetimeIndex, F: pd.DataFrame) -> dict:
    """Noise / process variances from prints dated < TRAIN_END only, per indexer group (DI floaters vs IPCA/PRE).
    Changes between two prints of the same bond k = 1..20 bdays apart, net of the peer factor move, in price units
    dp = dur * (ds - dF). ROBUST variance (MAD^2) vs k: Var(k) = 2 R_p + k q_p -> typical print noise R_p (price
    bps^2) and process variance q (spread bps^2 / bday = slope of the spread-unit fit). Heavy tails (the occasional
    wild print) are handled by gating in the filter, and are measured separately by the classic Roll estimator
    (mean-based, tails included) = the economic effective half-spread."""
    x = t[t["date"] < TRAIN_END].copy()
    x["pos"] = dd.get_indexer(x["date"])
    x = x[x["pos"] >= 0].sort_values(["codigo", "pos"])
    Fv = F.reindex(columns=sorted(x["peer"].unique())).fillna(0.0)
    fl = Fv.to_numpy()
    ci = Fv.columns.get_indexer(x["peer"])
    x["F"] = fl[x["pos"].to_numpy(), ci]
    g = x.groupby("codigo")
    x["gp"] = x["pos"] - g["pos"].shift(1)
    x["gn"] = g["pos"].shift(-1) - x["pos"]
    x["dsa"] = (x["s"] - g["s"].shift(1)) - (x["F"] - g["F"].shift(1))
    x["dsa_n"] = g["dsa"].shift(-1)
    x["dpa"] = x["dsa"] * x["dur"]
    x["dpa_n"] = x["dsa_n"] * x["dur"]
    x["grp"] = x["kind"].map(_grp_kind)
    x["ncat"] = np.select([x["n_trades"] <= 1, x["n_trades"] >= 5], [0, 2], 1)
    P = {"groups": {}}
    for gname, z0 in x.groupby("grp"):
        z0 = z0[z0["gp"].between(1, 20) & z0["s"].between(-300, 800)]
        ks, vs, vp = [], [], []
        for k, z in z0.groupby("gp"):
            if len(z) > 100:
                ks.append(k); vs.append(_rv(z["dsa"])); vp.append(_rv(z["dpa"]))
        bs, as_ = np.polyfit(ks, vs, 1)
        bp, ap = np.polyfit(ks, vp, 1)
        z1 = z0[z0["gp"] == 1]
        base = _rv(z1["dpa"])
        mult = {int(c): float(_rv(zz["dpa"]) / base) for c, zz in z1.groupby("ncat")}
        # classic Roll (mean autocovariance, tails included) by ticket bin: E[dp_t dp_{t+1}] = -sigma^2
        y = z0[(z0["gp"] == 1) & (z0["gn"] == 1) & z0["dpa"].abs().lt(1000) & z0["dpa_n"].abs().lt(1000)]
        roll = {}
        for bn, zz in y.groupby("tbin"):
            pr = (zz["dpa"] * zz["dpa_n"]).to_numpy()
            roll[TICKET_LAB[int(bn)]] = float(np.sqrt(max(-pr.mean(), 0))) if len(pr) > 100 else np.nan
        pr = (y["dpa"] * y["dpa_n"]).to_numpy()
        # issuer-common share of the process variance: same-day adjusted changes of two bonds of one issuer
        zz = z0[(z0["gp"] == 1) & z0["dsa"].abs().lt(60)][["cnpj8", "pos", "codigo", "dsa"]]
        pairs = zz.merge(zz, on=["cnpj8", "pos"])
        pairs = pairs[pairs["codigo_x"] < pairs["codigo_y"]]
        cov = float(np.mean(pairs["dsa_x"] * pairs["dsa_y"])) if len(pairs) > 300 else 0.0
        q = max(bs, 0.25)
        P["groups"][gname] = {
            "R_price_bps2_typical": float(max(ap / 2, 1.0)), "q_price_bps2_per_bday": float(bp),
            "q_spread_bps2_per_bday": float(q), "R_spread_intercept_half": float(as_ / 2),
            "noise_mult_by_ntrades(1|2-4|5+)": mult, "robvar_k": {int(k): float(v) for k, v in zip(ks, vs)},
            "roll_half_spread_price_bps_by_ticket": roll,
            "roll_half_spread_price_bps_all": float(np.sqrt(max(-pr.mean(), 0))),
            "issuer_cov_same_day": cov, "issuer_share": float(np.clip(cov / q, 0.05, 0.8)),
            "median_dur": float(z0["dur"].median()), "n_pairs_k1": int(len(z1)), "n_issuer_pairs": int(len(pairs))}
    return P


# ------------------------------------------------------------------------------------------------ peer factor
def peer_factor(t: pd.DataFrame, dd: pd.DatetimeIndex) -> pd.DataFrame:
    """Daily peer factor: median spread change of bonds printed on both t-1 and t within the peer group
    (kind x incentive), >= 5 such bonds else 0; winsorised at +-30 bps. Known at the close of t."""
    x = t[["codigo", "date", "s", "peer"]].copy()
    x["pos"] = dd.get_indexer(x["date"])
    x = x[x["pos"] >= 0].sort_values(["codigo", "pos"])
    g = x.groupby("codigo")
    x["ds"] = x["s"] - g["s"].shift(1)
    x["gap"] = x["pos"] - g["pos"].shift(1)
    x = x[(x["gap"] == 1) & x["ds"].abs().lt(100)]
    f = x.groupby(["pos", "peer"])["ds"].agg(["median", "size"])
    f["df"] = np.where(f["size"] >= 5, f["median"].clip(-30, 30), 0.0)
    F = f["df"].unstack("peer").reindex(range(len(dd))).fillna(0.0)
    return F.cumsum()          # level (bps) relative to day 0


# ------------------------------------------------------------------------------------------------ Kalman filter
def run_kalman(t: pd.DataFrame, grid: pd.DataFrame, dd: pd.DatetimeIndex, F: pd.DataFrame, prm: dict) -> pd.DataFrame:
    """Per issuer, state = [issuer common a_t, bond offsets u_b]; spread_b,t = F_peer(b),t + a_t + u_b,t.
    a, u random walks with variances q_a = rho*q, q_u = (1-rho)*q (q scaled by (max(s,30)/100)^1 per bond level);
    observation noise R = sigma_price^2(ticket bin, #trades) / dur^2 (bps^2); robust gating: |z|>3 -> R*(z/3)^2.
    Output for each (codigo, grid day): kf_s (posterior mean after that day's prints), kf_sd, kf_prior (prediction
    before the day's print, only on print days), kf_z (standardised innovation)."""
    GP = prm["groups"]
    t = t.copy()
    t["pos"] = dd.get_indexer(t["date"])
    t = t[(t["pos"] >= 0) & t["s"].notna() & t["dur"].notna()]
    Fv = np.hstack([F.to_numpy(), np.zeros((len(F), 1))])   # last column = no factor
    fcol = {c: i for i, c in enumerate(F.columns)}
    gpos = dd.get_indexer(grid["day"])
    grid = grid.assign(pos=gpos)
    out = []
    for c8, ti in t.groupby("cnpj8", sort=False):
        codes = list(dict.fromkeys(ti["codigo"]))
        bidx = {c: i + 1 for i, c in enumerate(codes)}
        n = len(codes) + 1
        pmap = ti.drop_duplicates("codigo").set_index("codigo")["peer"]
        peer_i = np.array([fcol.get(pmap[c], Fv.shape[1] - 1) for c in codes])
        gk = GP[_grp_kind(ti["kind"].iloc[0])]     # issuer-level process params from its first bond's group
        qb = np.array([GP[_grp_kind(k)]["q_spread_bps2_per_bday"] for k in
                       ti.drop_duplicates("codigo").set_index("codigo").loc[codes, "kind"]])
        q_a = gk["issuer_share"] * float(np.median(qb))
        q_ub = (1 - gk["issuer_share"]) * qb
        obs_by_pos = {}
        for r in ti.itertuples(index=False):
            gg = GP[_grp_kind(r.kind)]
            nt = r.n_trades if (r.n_trades is not None and not np.isnan(r.n_trades)) else 1
            m = gg["noise_mult_by_ntrades(1|2-4|5+)"][0 if nt <= 1 else (2 if nt >= 5 else 1)]
            R = gg["R_price_bps2_typical"] * m / max(r.dur, 0.5) ** 2
            obs_by_pos.setdefault(int(r.pos), []).append((bidx[r.codigo], float(r.s), R))
        p0 = int(ti["pos"].min())
        p1 = int(min(ti["pos"].max() + 15, len(dd) - 1))
        x = np.zeros(n)
        Pm = np.zeros((n, n))
        Pm[0, 0] = 400.0
        started = np.zeros(n, bool)
        started[0] = True
        lvl = np.full(n, 100.0)          # last known spread level per bond (for q scaling)
        rec_pos, rec_b, rec_s, rec_sd, rec_prior, rec_z = [], [], [], [], [], []
        for p in range(p0, p1 + 1):
            # predict (factor enters deterministically; only variances grow)
            sc = np.clip(lvl[1:], 50, 1000) / 100.0
            Pm[0, 0] += q_a
            Pm[np.arange(1, n), np.arange(1, n)] += np.where(started[1:], q_ub * sc, 0.0)
            obs = obs_by_pos.get(p)
            prior = {}
            if obs:
                for bi, y, R in obs:
                    fac = Fv[p, peer_i[bi - 1]]
                    if not started[bi]:
                        x[bi] = y - fac - x[0]
                        Pm[bi, :] = 0.0
                        Pm[:, bi] = 0.0
                        Pm[bi, bi] = R + 25.0
                        Pm[bi, 0] = Pm[0, bi] = 0.0
                        started[bi] = True
                        lvl[bi] = y
                        prior[bi] = (np.nan, np.nan)
                        continue
                    pred = fac + x[0] + x[bi]
                    Hh = np.zeros(n)
                    Hh[0] = 1.0
                    Hh[bi] = 1.0
                    PH = Pm @ Hh
                    S = Hh @ PH + R
                    v = y - pred
                    zz = v / np.sqrt(S)
                    prior[bi] = (pred, zz)
                    if abs(zz) > 2.5:                 # heavy-tailed prints: bounded influence
                        S = S - R + R * (zz / 2.5) ** 2
                    K = PH / S
                    x = x + K * v
                    Pm = Pm - np.outer(K, PH)
                    lvl[bi] = y
            # record all started bonds
            act = np.nonzero(started[1:])[0] + 1
            if len(act):
                fac = Fv[p, peer_i[act - 1]]
                s_hat = fac + x[0] + x[act]
                var = Pm[0, 0] + Pm[act, act] + 2 * Pm[0, act]
                rec_pos.append(np.full(len(act), p))
                rec_b.append(act)
                rec_s.append(s_hat)
                rec_sd.append(np.sqrt(np.maximum(var, 0)))
                pr = np.full(len(act), np.nan)
                pz = np.full(len(act), np.nan)
                if prior:
                    for j, i in enumerate(act):
                        if i in prior:
                            pr[j], pz[j] = prior[i]
                rec_prior.append(pr)
                rec_z.append(pz)
        if rec_pos:
            cc = np.array(codes, dtype=object)
            out.append(pd.DataFrame({"pos": np.concatenate(rec_pos), "codigo": cc[np.concatenate(rec_b) - 1],
                                     "kf_s": np.concatenate(rec_s), "kf_sd": np.concatenate(rec_sd),
                                     "kf_prior": np.concatenate(rec_prior), "kf_z": np.concatenate(rec_z)}))
    K = pd.concat(out, ignore_index=True)
    K["day"] = dd[K["pos"].to_numpy()]
    return K.drop(columns=["pos"])


# ------------------------------------------------------------------------------------------------ clean / VW marks
def clean_marks(t: pd.DataFrame, grid: pd.DataFrame, dd: pd.DatetimeIndex) -> pd.DataFrame:
    """(b) cleaned marks. A print is 'odd' if it is a single trade below R$200k or total volume below R$50k;
    'outlier' if it deviates from the median of the bond's previous 5 clean prints by more than
    max(40 bps, 4 x their MAD) (backward-looking only). vw_s = volume-weighted spread of the last <= 5 clean prints
    within 10 bdays of the grid day; cl_s = last clean print (<= 14 days old)."""
    x = t[["codigo", "date", "s", "vol_brl", "n_trades", "ticket"]].copy()
    x["odd"] = ((x["n_trades"] <= 1) & (x["vol_brl"] < 2e5)) | (x["vol_brl"] < 5e4)
    x["out"] = False
    res = []
    for c, z in x.groupby("codigo", sort=False):
        s = z["s"].to_numpy()
        odd = z["odd"].to_numpy()
        outl = np.zeros(len(z), bool)
        hist, run, med = [], [], 0.0
        for i in range(len(z)):
            if len(hist) >= 3:
                h = np.array(hist[-5:])
                med = np.median(h)
                mad = np.median(np.abs(h - med)) * 1.4826
                if abs(s[i] - med) > max(40.0, 4 * mad):
                    outl[i] = True
            if not odd[i] and not outl[i]:
                hist.append(s[i])
                run = []
            elif outl[i]:
                # 3 consecutive outliers on the same side = a genuine level shift: restart the history there
                run = run + [s[i]] if (i and outl[i - 1]) else [s[i]]
                if len(run) >= 3 and (np.all(np.array(run) > med) or np.all(np.array(run) < med)):
                    hist = list(run)
                    run = []
        res.append(outl)
    x["out"] = np.concatenate(res)
    x["clean"] = ~x["odd"] & ~x["out"]
    x["pos"] = dd.get_indexer(x["date"])
    xc = x[x["clean"] & (x["pos"] >= 0)]
    gg = grid[["codigo", "day"]].copy()
    gg["gpos"] = dd.get_indexer(gg["day"])
    # last clean print
    a = pd.merge_asof(gg.sort_values("gpos"), xc[["codigo", "pos", "s"]].sort_values("pos").rename(
        columns={"s": "cl_s", "pos": "cl_pos"}), left_on="gpos", right_on="cl_pos", by="codigo",
        direction="backward")
    a.loc[a["gpos"] - a["cl_pos"] > 10, "cl_s"] = np.nan
    # VW of last <= 5 clean prints within 10 bdays
    xc = xc.sort_values(["codigo", "pos"])
    grp = xc.groupby("codigo")
    xc = xc.assign(vs=xc["s"] * xc["vol_brl"].clip(lower=1e4), vv=xc["vol_brl"].clip(lower=1e4))
    lagged = []
    for L in range(5):
        lagged.append(xc[["codigo", "pos"]].assign(vs=xc.groupby("codigo")["vs"].shift(L),
                                                   vv=xc.groupby("codigo")["vv"].shift(L),
                                                   pp=xc.groupby("codigo")["pos"].shift(L)))
    num = sum(np.where((l["pos"] - l["pp"] <= 10) & l["vs"].notna(), l["vs"], 0.0) for l in lagged)
    den = sum(np.where((l["pos"] - l["pp"] <= 10) & l["vv"].notna(), l["vv"], 0.0) for l in lagged)
    xc = xc.assign(vw=num / np.where(den > 0, den, np.nan))
    b = pd.merge_asof(gg.sort_values("gpos"), xc[["codigo", "pos", "vw"]].sort_values("pos"), left_on="gpos",
                      right_on="pos", by="codigo", direction="backward")
    b.loc[b["gpos"] - b["pos"] > 10, "vw"] = np.nan
    a = a.set_index(["codigo", "day"])
    b = b.set_index(["codigo", "day"])
    out = pd.DataFrame({"cl_s": a["cl_s"], "vw_s": b["vw"]})
    return out.reset_index(), x[["codigo", "date", "odd", "out", "clean"]]


# ------------------------------------------------------------------------------------------------ liquidity
def liquidity(t: pd.DataFrame, grid: pd.DataFrame, dd: pd.DatetimeIndex) -> pd.DataFrame:
    """Past-63-bday liquidity (prints dated <= day): adv63 (R$/bday incl. zero days), tdays63 (share of bdays with a
    print), ticket63 (median ticket), ntr63 (#trades); roll_hs: bond Roll half-spread (price bps) from 252 bdays of
    consecutive-day price changes (NaN if < 15 triples or positive autocovariance -> set 0)."""
    x = t[["codigo", "date", "vol_brl", "n_trades", "ticket", "s", "dur"]].copy()
    x["pos"] = dd.get_indexer(x["date"])
    x = x[x["pos"] >= 0]
    codes = pd.Index(sorted(x["codigo"].unique()))
    ND, NB = len(dd), len(codes)
    bi = codes.get_indexer(x["codigo"])
    V = np.zeros((ND, NB), np.float32)
    V[x["pos"].to_numpy(), bi] = x["vol_brl"].fillna(0).to_numpy()
    Tn = np.zeros((ND, NB), np.float32)
    Tn[x["pos"].to_numpy(), bi] = x["n_trades"].fillna(0).to_numpy()
    D = (V > 0).astype(np.float32)
    cV = np.vstack([np.zeros((1, NB)), np.cumsum(V, 0, dtype=np.float64)])
    cT = np.vstack([np.zeros((1, NB)), np.cumsum(Tn, 0, dtype=np.float64)])
    cD = np.vstack([np.zeros((1, NB)), np.cumsum(D, 0, dtype=np.float64)])
    gg = grid[["codigo", "day"]].copy()
    gp = dd.get_indexer(gg["day"])
    gb = codes.get_indexer(gg["codigo"])
    ok = (gp >= 0) & (gb >= 0)
    lo = np.maximum(gp - 62, 0)
    for nm, C in (("adv63", cV), ("ntr63", cT), ("tdays63", cD)):
        v = np.full(len(gg), np.nan)
        v[ok] = (C[gp[ok] + 1, gb[ok]] - C[lo[ok], gb[ok]])
        gg[nm] = v
    gg["adv63"] = gg["adv63"] / 63
    gg["tdays63"] = gg["tdays63"] / 63
    gg["ticket63"] = gg["adv63"] * 63 / gg["ntr63"].replace(0, np.nan)
    # Roll half-spread per bond, rolling 252 bdays (price bps)
    x = x.sort_values(["codigo", "pos"])
    g = x.groupby("codigo")
    x["dp"] = -x["dur"] * (x["s"] - g["s"].shift(1))
    x["gap"] = x["pos"] - g["pos"].shift(1)
    x["dp_n"] = g["dp"].shift(-1)
    x["gap_n"] = g["gap"].shift(-1)
    # the product dp_t * dp_{t+1} is only known at t+1 -> date it at the NEXT print
    x["prod"] = np.where((x["gap"] == 1) & (x["gap_n"] == 1) & (x["dp"].abs() < 500) & (x["dp_n"].abs() < 500),
                         x["dp"] * x["dp_n"], np.nan)
    x["known_pos"] = g["pos"].shift(-1)
    pr = x.dropna(subset=["prod", "known_pos"])
    P = np.zeros((ND, NB))
    N = np.zeros((ND, NB))
    np.add.at(P, (pr["known_pos"].astype(int).to_numpy(), codes.get_indexer(pr["codigo"])),
              np.clip(pr["prod"].to_numpy(), -2e4, 2e4))
    np.add.at(N, (pr["known_pos"].astype(int).to_numpy(), codes.get_indexer(pr["codigo"])), 1)
    cP = np.vstack([np.zeros((1, NB)), np.cumsum(P, 0)])
    cN = np.vstack([np.zeros((1, NB)), np.cumsum(N, 0)])
    lo = np.maximum(gp - 251, 0)
    sp = np.full(len(gg), np.nan)
    sn = np.full(len(gg), np.nan)
    sp[ok] = cP[gp[ok] + 1, gb[ok]] - cP[lo[ok], gb[ok]]
    sn[ok] = cN[gp[ok] + 1, gb[ok]] - cN[lo[ok], gb[ok]]
    cov = sp / np.where(sn > 0, sn, np.nan)
    gg["roll_n"] = sn
    gg["roll_hs"] = np.where(sn >= 15, np.sqrt(np.maximum(-cov, 0)), np.nan)
    return gg


def main():
    dd = H.days()
    log("trades table")
    t = build_trades()
    t.to_pickle(OUT / "trades.pkl")
    log(f"trades: {len(t)} rows, {t['codigo'].nunique()} bonds; vol known {t['vol_brl'].notna().mean():.3f}")
    F = peer_factor(t, dd)
    F.to_pickle(OUT / "peer_factor.pkl")
    prm = estimate_params(t, dd, F)
    log("params", json.dumps(prm, indent=1)[:3000])
    json.dump(prm, open(OUT / "params.json", "w"), indent=1)
    json.dump(prm, open(HERE / "params.json", "w"), indent=1)
    log("grid keys")
    grid = pd.read_pickle(ROOT / "data" / "history" / "lab_daily.pkl")[["codigo", "day", "cnpj8"]]
    grid["codigo"] = grid["codigo"].astype(str)
    grid["cnpj8"] = grid["cnpj8"].astype(str)
    log("kalman")
    K = run_kalman(t, grid, dd, F, prm)
    log(f"kalman rows {len(K)}")
    log("clean marks")
    CM, flags = clean_marks(t, grid, dd)
    flags.to_pickle(OUT / "trade_flags.pkl")
    log("liquidity")
    L = liquidity(t, grid, dd)
    M = grid[["codigo", "day"]].merge(K, on=["codigo", "day"], how="left").merge(CM, on=["codigo", "day"], how="left")
    M = M.merge(L.drop(columns=["gpos"], errors="ignore"), on=["codigo", "day"], how="left")
    for c in M.columns:
        if M[c].dtype == np.float64:
            M[c] = M[c].astype(np.float32)
    M.to_pickle(OUT / "marks.pkl")
    log(f"marks.pkl {M.shape}; kf coverage {M['kf_s'].notna().mean():.3f}; vw {M['vw_s'].notna().mean():.3f}")


if __name__ == "__main__":
    main()
