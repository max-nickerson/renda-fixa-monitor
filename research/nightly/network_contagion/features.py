"""Point-in-time node shocks and neighbour-stress features on every decision date of the harness W and M panels.

Node (issuer) shocks at decision close p (all inputs known at the close of p):
  shock_ret   : the issuer's median bond (among bonds that traded in the last 30d) lost >= 3% (rate-hedged excess,
                patched returns) over the last 20 bdays -> spread blow-out
  shock_down  : rating downgrade in the last 30 days
  shock_press : >= 3 negative press items in the last 7 days
  shock_cvm   : CVM distress filing in the last 30 days
  shock_eq    : own listed stock (direct mapping) down >= 20% over 21 bdays
  shock_any   : any of the above
Neighbour channels (own issuer always excluded):
  sib_*  other bonds of the SAME issuer (cnpj8) — lead-lag between stale and fresh marks
  grp_*  other issuers of the same corporate group (net.groups_asof: equity_map + brands + FRE majority control, PIT)
  aff_*  FRE JV / 20-50% controlling-block links
  sec_*  other issuers of the same sector
  exp_*  exposure-similar issuers (cosine >= 0.5) of OTHER sectors
  hold_* common fund holders (CDA snapshot available at p, ref + 100d): holder-loss exposure to shocked issuers
  flow_* holder-fund net flows over the last 21 bdays (inf_diario, 2-bday publication lag), weighted by holdings
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import net  # noqa: E402
from research.nightly import harness as H  # noqa: E402

OUT = net.CD / "features_v1.pkl"
COLS = ["codigo", "day", "dpos", "b", "cnpj8", "sector", "trades_30d", "rat_days_since_down", "press_neg_7d",
        "n_distress_30d", "eq_r21", "eq_map_type", "univ"]


def _panels():
    W = H.load_panel("W", holdout=True)[COLS]
    M = H.load_panel("M", holdout=True)[COLS]
    X = pd.concat([W, M]).drop_duplicates(["dpos", "codigo"]).reset_index(drop=True)
    X["cnpj8"] = X["cnpj8"].astype(str).str.zfill(8)
    return X


def _flows_matrix():
    """fund x grid-day 21-bday net flow / NAV, known with a 2-bday lag (value at p uses days <= p-2)."""
    path = net.CD / "fund_flows.pkl"
    if not path.exists():
        return None
    F = pd.read_pickle(path)
    F["fund"] = F["fund"].astype(str)
    dd = H.days()
    F = F[F["day"].isin(dd)]
    net_ = (F["captc"].fillna(0) - F["resg"].fillna(0))
    NF = pd.pivot_table(F.assign(n=net_), index="day", columns="fund", values="n", aggfunc="sum").reindex(dd).fillna(0)
    NAV = pd.pivot_table(F, index="day", columns="fund", values="nav", aggfunc="last").reindex(dd).ffill(limit=10)
    fl = NF.rolling(21, min_periods=15).sum() / NAV.shift(21).where(NAV.shift(21) > 1e6)
    fl = fl.clip(-1, 1).shift(2)
    return fl


def build(force: bool = False) -> pd.DataFrame:
    if OUT.exists() and not force:
        return pd.read_pickle(OUT)
    X = _panels()
    dd = H.days()
    LC = H._LC("base")
    iss = net.issuers().set_index("cnpj8")
    S = net.exposure_sim()
    Hd = net.cda_holdings()
    c2i = X.drop_duplicates("codigo").set_index("codigo")["cnpj8"]
    Hd = Hd[Hd["codigo"].isin(c2i.index)].copy()
    Hd["cnpj8"] = Hd["codigo"].map(c2i)
    Hi = Hd.groupby(["ref", "avail", "fund", "cnpj8"], as_index=False)["v"].sum()
    snaps = Hi[["ref", "avail"]].drop_duplicates().sort_values("avail")
    FL = _flows_matrix()
    grp_cache, aff_cache, hold_cache = {}, {}, {}
    out = []
    for p, x in X.groupby("dpos", sort=True):
        p = int(p)
        day = dd[p]
        x = x.copy()
        bb = x["b"].to_numpy()
        lo4, lo13 = max(p - 20, 0), max(p - 63, 0)
        x["r4"] = np.expm1(LC[p, bb] - LC[lo4, bb])
        x["r13"] = np.expm1(LC[p, bb] - LC[lo13, bb])
        traded = x["trades_30d"].fillna(0) > 0
        x["r4t"] = x["r4"].where(traded)
        # ---- issuer-level node shocks
        g = x.groupby("cnpj8")
        I = pd.DataFrame({
            "r4min": g["r4t"].min(), "r4med": g["r4t"].median(), "r13med": g["r13"].median(),
            "down": g["rat_days_since_down"].min() <= 30, "press": g["press_neg_7d"].max() >= 3,
            "cvm": g["n_distress_30d"].max() > 0,
            "eq": g.apply(lambda s: bool(((s["eq_r21"] <= -0.20) & (s["eq_map_type"] == "direct")).any()),
                          include_groups=False),
            "sector": g["sector"].first(), "nb": g.size()})
        I["ret"] = I["r4med"] <= -0.03
        I["any"] = I[["ret", "down", "press", "cvm", "eq"]].any(axis=1)
        I["hard"] = I[["ret", "down", "cvm", "eq"]].any(axis=1)     # without press
        ids = I.index.to_numpy()
        # ---- groups (quarterly cache, PIT)
        qk = (day.year, (day.month - 1) // 3)
        if qk not in grp_cache:
            gq = net.groups_asof(day)
            grp_cache[qk] = gq
            aff_cache[qk] = net.affiliates_asof(day)
        gser = grp_cache[qk]
        I["grp"] = gser.reindex(ids).fillna(pd.Series(["C:" + i for i in ids], index=ids)).to_numpy()
        # group aggregates excluding own issuer: sums minus own
        for k in ("any", "hard", "ret"):
            s = I.groupby("grp")[k].transform("sum") - I[k]
            n = I.groupby("grp")[k].transform("size") - 1
            I[f"grp_{k}"] = np.where(n > 0, s / n.clip(lower=1), np.nan)
            I[f"grp_{k}_n"] = np.where(n > 0, s, np.nan)
        s = I.groupby("grp")["r4med"].transform("sum") - I["r4med"].fillna(0)
        n = I.groupby("grp")["r4med"].transform("count") - I["r4med"].notna()
        I["grp_r4"] = np.where(n > 0, s / n.clip(lower=1), np.nan)
        # affiliates
        aff = aff_cache[qk]
        A = {}
        for a, b_ in aff:
            A.setdefault(a, set()).add(b_)
            A.setdefault(b_, set()).add(a)
        I["aff_any"] = [np.mean([I.at[j, "any"] for j in A.get(i, ()) if j in I.index]) if any(j in I.index for j in A.get(i, ())) else np.nan for i in ids]
        # sector aggregates excluding own and own group
        for k in ("any", "hard", "ret"):
            s = I.groupby("sector")[k].transform("sum") - I[k]
            n = I.groupby("sector")[k].transform("size") - 1
            I[f"sec_{k}"] = np.where(n >= 3, s / n.clip(lower=1), np.nan)
        for k in ("r4med", "r13med"):
            s = I.groupby("sector")[k].transform("sum") - I[k].fillna(0)
            n = I.groupby("sector")[k].transform("count") - I[k].notna()
            I[f"sec_{k[:3]}"] = np.where(n >= 3, s / n.clip(lower=1), np.nan)
        I.loc[I["sector"].isin(["unknown", "other"]), ["sec_any", "sec_hard", "sec_ret", "sec_r4m", "sec_r13"]] = np.nan
        # exposure-similar issuers of other sectors
        Sm = S.reindex(index=ids, columns=ids).fillna(0).to_numpy()
        secv = I["sector"].to_numpy()
        Sm = np.where((Sm >= 0.5) & (secv[:, None] != secv[None, :]), Sm, 0.0)
        den = Sm.sum(1)
        for k in ("any", "ret"):
            v = I[k].to_numpy(float)
            I[f"exp_{k}"] = np.where(den > 0, Sm @ v / np.where(den > 0, den, 1), np.nan)
        r4 = I["r4med"].to_numpy(float)
        ok = np.isfinite(r4)
        Sm2 = Sm * ok[None, :]
        d2 = Sm2.sum(1)
        I["exp_r4"] = np.where(d2 > 0, Sm2 @ np.nan_to_num(r4) / np.where(d2 > 0, d2, 1), np.nan)
        # ---- common fund holders (latest CDA snapshot available at `day`)
        av = snaps[snaps["avail"] <= day]
        I["hold_any"] = np.nan
        I["hold_ret"] = np.nan
        I["flow21"] = np.nan
        I["n_holders"] = np.nan
        if len(av):
            ref = av["ref"].iloc[-1]
            h = Hi[Hi["ref"] == ref]
            h = h[h["cnpj8"].isin(ids)]
            if len(h):
                funds = h["fund"].unique()
                fi = pd.Index(funds)
                ii = pd.Index(ids)
                Wm = np.zeros((len(ii), len(fi)))
                np.add.at(Wm, (ii.get_indexer(h["cnpj8"]), fi.get_indexer(h["fund"])), h["v"].to_numpy())
                hol = Wm / np.where(Wm.sum(1, keepdims=True) > 0, Wm.sum(1, keepdims=True), 1)   # issuer -> holder shares
                book = Wm / np.where(Wm.sum(0, keepdims=True) > 0, Wm.sum(0, keepdims=True), 1)  # fund book weights
                for k in ("any", "ret"):
                    v = I[k].to_numpy(float)
                    # fund f's book share in shocked issuers j; exclude own issuer & own group:
                    # loss_i = sum_f hol[i,f] * sum_j book[j,f] v_j (j not in grp(i))
                    Lf = book.T  # f x j
                    tot = hol @ (Lf @ v)                             # includes own group
                    gcode = pd.factorize(I["grp"].to_numpy())[0]
                    Bg = np.zeros((gcode.max() + 1, book.shape[1]))
                    np.add.at(Bg, gcode, book * v[:, None])           # group-level shocked book share per fund
                    own = (hol * Bg[gcode]).sum(1)
                    I[f"hold_{k}"] = np.where(Wm.sum(1) > 0, tot - own, np.nan)
                I["n_holders"] = (Wm > 0).sum(1)
                if FL is not None:
                    fl = FL.iloc[p].reindex(fi).to_numpy()
                    okf = np.isfinite(fl)
                    wv = hol * okf[None, :]
                    dn = wv.sum(1)
                    I["flow21"] = np.where(dn > 0.3, wv @ np.nan_to_num(fl) / np.where(dn > 0, dn, 1), np.nan)
        # ---- bond level: siblings (other bonds of the same issuer)
        x = x.join(I.drop(columns=["sector", "nb"]).add_prefix("i_"), on="cnpj8")
        gs = x.groupby("cnpj8")["r4t"]
        cnt = gs.transform("count") - x["r4t"].notna()
        # min over siblings excluding own: use sorted two smallest
        def _sib_min(s):
            v = s.to_numpy(float)
            out_ = np.full(len(v), np.nan)
            fin = np.where(np.isfinite(v))[0]
            if len(fin) >= 2:
                o = fin[np.argsort(v[fin])]
                m1, m2 = v[o[0]], v[o[1]]
                out_[:] = m1
                out_[o[0]] = m2
            elif len(fin) == 1 and len(v) > 1:
                out_[:] = v[fin[0]]
                out_[fin[0]] = np.nan
            return pd.Series(out_, index=s.index)
        x["sib_min_r4"] = gs.transform(lambda s: _sib_min(s))
        x["sib_n"] = cnt
        x["sib_gap"] = x["sib_min_r4"] - x["r4"]
        out.append(x)
        if len(out) % 25 == 0:
            print("feat", day.date(), len(x), flush=True)
    F = pd.concat(out, ignore_index=True)
    keep = ["codigo", "day", "dpos", "cnpj8", "r4", "r13", "sib_min_r4", "sib_n", "sib_gap"] + \
        [c for c in F.columns if c.startswith("i_")]
    F = F[keep]
    F.to_pickle(OUT)
    return F


def signals(panel: pd.DataFrame) -> pd.DataFrame:
    """Merge the network features onto a harness panel (by day, codigo). Returns the panel with extra columns."""
    F = build()
    return panel.merge(F.drop(columns=["dpos", "cnpj8"]), on=["day", "codigo"], how="left")


if __name__ == "__main__":
    F = build(force="--force" in sys.argv)
    print(F.shape)
    print(F.describe().T.round(4).to_string())
