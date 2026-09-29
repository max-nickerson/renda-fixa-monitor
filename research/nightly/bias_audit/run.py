"""Independent bias audit of the P4 / P4+Q baselines (harness v4).

Every bias is switched on (a) alone on top of the v4 baseline and (b) cumulatively, to build an 'honest' P4 / P4+Q.
Rerun:
  PYTHONPATH=. PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/Scripts/python.exe research/nightly/bias_audit/build.py
  PYTHONPATH=. PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/Scripts/python.exe research/nightly/bias_audit/run.py
"""
from __future__ import annotations

import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from research.nightly import harness as H
from research.nightly.bias_audit import build as B
from research.nightly.bias_audit import engine as E

OUTD = Path("research/nightly/bias_audit")
CACHE = B.OUT
T0 = time.time()
RES: dict = {"n_variants_tried": 0}


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


C = H._core()
DD, CODES, TD = C["days"], C["codes"], C["TD"]
ND, NB = TD.shape
HP = H._hpos()
P = H.load_panel("M")
R0 = H._Rmat("base")
EX = B.build_exec()
lm = B.lab_min()

# =====================================================================================================================
# 0. helpers
# =====================================================================================================================
SIG = {"U": lambda x: x["univ"].to_numpy(), "P4": lambda x: x["p4"].to_numpy(), "P4Q": lambda x: x["p4q"].to_numpy()}
_TCACHE = {}


def T_of(name, panel=None, key=None):
    k = (name, key)
    if k not in _TCACHE:
        _TCACHE[k] = E.targets(SIG[name] if isinstance(name, str) and name in SIG else name, P if panel is None else panel)
    return _TCACHE[k]


def st(daily, bench=None):
    s = H.stats(daily, bench=bench)
    keep = ["ann_excess_%", "vol_%", "sharpe", "max_dd_%", "t_nw", "h1_2022_23_%", "h2_2024_25_%"]
    out = {k: s[k] for k in keep if k in s}
    if bench is not None:
        out.update({k: s[k] for k in ["diff_ann_%", "diff_t_nw", "diff_p", "diff_h1_%", "diff_h2_%"]})
    return out


def pen(M, sign=1.0):
    return lambda k, b: sign * M[np.minimum(k, ND - 1), b]


# =====================================================================================================================
# 1. diagnostics: holiday accrual, universe PIT, off-grid bonds
# =====================================================================================================================
def diag_holiday():
    out, d = B.build_hol()
    # empirical check: universe rows crossing a holiday earn the fake accrual (regress r on c*extra/252)
    G, _ = pd.read_pickle(H.HIST / "sellab_returns.pkl")
    ho = out[["pos", "b", "extra", "c", "kind"]]
    m = pd.DataFrame({"pos": np.nonzero(TD >= 0)[0]})  # dummy to keep memory low
    del m
    Rh = B.R_hol(R0)
    # mean daily fake accrual on the universe book, %/yr: compare U with and without the fix later (scenario table)
    by_kind = out.groupby("kind").apply(lambda x: float(((1 - x["f"]) * 1e4).mean())).to_dict()
    # empirical test: on DI_SPREAD rows not crossing a holiday vs crossing, mean R minus expected contract accrual
    sel = (ho["kind"] == "DI_SPREAD") & (ho["pos"] < HP)
    r_cross = R0[ho.loc[sel, "pos"].to_numpy(), ho.loc[sel, "b"].to_numpy()]
    fake = (1 - out.loc[sel, "f"].to_numpy())
    d.update({"fake_accrual_bps_per_holiday_row_by_kind": by_kind,
              "DI_rows_crossing_holiday_mean_R_bps": float(np.mean(r_cross) * 1e4),
              "DI_rows_crossing_holiday_mean_fake_bps": float(np.mean(fake) * 1e4)})
    # all DI rows (not crossing) mean R for comparison
    G = G[G["pos"] < HP]
    lk = lm[["codigo", "day", "kind"]].copy()
    G["day"] = DD[G["pos"].to_numpy()]
    G = G.merge(lk, on=["codigo", "day"], how="left")
    di = G[G["kind"] == "DI_SPREAD"]
    key = set(zip(ho["pos"], ho["b"]))
    cross = np.array([(p, b) in key for p, b in zip(di["pos"], di["b"])])
    d["DI_rows_not_crossing_mean_R_bps"] = float(di.loc[~cross, "r_patch"].clip(-0.2, 0.2).mean() * 1e4)
    d["DI_rows_crossing_mean_R_bps_clip"] = float(di.loc[cross, "r_patch"].clip(-0.2, 0.2).mean() * 1e4)
    return d, Rh


def diag_universe():
    U = P[P["univ"]]
    age = (U["day"] - pd.to_datetime(U["date"])).dt.days
    d = {"univ_rows": int(len(U)), "univ_mark_after_decision_day": int((pd.to_datetime(U["date"]) > U["day"]).sum()),
         "univ_mark_age_gt7d": int((age > 7).sum()), "univ_ratio_outside_0.9_1.1": int((~U["ratio"].between(0.9, 1.1)).sum())}
    # entries strictly after the decision day
    bad = 0
    for p in sorted(U["dpos"].unique())[:12]:
        ep = H.exec_pos(int(p))
        ok = ep >= 0
        bad += int((TD[ep[ok], np.nonzero(ok)[0]] <= DD[int(p)].to_datetime64().astype("datetime64[D]").astype(np.int64)).sum())
    d["entries_not_after_decision"] = bad
    # SND bonds never on the grid (pre-2026)
    fs = sorted(glob.glob(str(H.HIST / "snd_trades_20*.csv.gz")))
    s = pd.concat([pd.read_csv(f, parse_dates=["date"]) for f in fs], ignore_index=True)
    s = s[s["date"] < H.HOLDOUT]
    s["vol"] = s["qty"] * s["pu_avg"]
    grid = set(CODES)
    on = s["codigo"].isin(grid)
    off = s[~on]
    pcn = off.groupby("codigo")["pct_curve"].apply(lambda x: x.isna().mean() > 0.5)
    mn = s[s["pct_curve"].notna()].groupby(["codigo"])["pct_curve"].min()
    d.update({"snd_codes": int(s["codigo"].nunique()), "snd_codes_off_grid": int(off["codigo"].nunique()),
              "off_grid_volume_share": float(off["vol"].sum() / s["vol"].sum()),
              "off_grid_codes_without_pct_curve": int(pcn.sum()),
              "min_pct_curve_below70_share_on_grid": float((mn[mn.index.isin(grid)] < 70).mean()),
              "min_pct_curve_below70_share_off_grid": float((mn[~mn.index.isin(grid)] < 70).mean())})
    return d


# =====================================================================================================================
# 2. survivorship: stopped bonds and a point-in-time distress list (information <= 2025-12-31 only)
# =====================================================================================================================
NAMED = {"00776574": "Americanas", "60444437": "Light SESA", "05303439": "Unigel", "33041260": "Casas Bahia (Via)",
         "01599101": "Sequoia", "13270520": "Kora Saude", "12528708": "Aeris", "10209063": "Rech Agricola",
         "12648266": "Ambipar", "09527023": "Environmental ESG (Ambipar)", "42150391": "Braskem",
         "12104241": "Oncoclinicas", "12420164": "CM Hospitalar", "04368865": "Ligga Telecom",
         "07714104": "Giga Mais Fibra", "31748174": "Vero", "09053134": "Elfa", "47508411": "GPA (CBD)",
         "09305994": "Azul", "61486650": "Dasa", "35764708": "Brasil Tecnologia e Part.", "33042730": "CSN",
         "21314559": "Movida", "07415333": "Simpar", "41570356": "Vrental", "08070508": "Raizen Energia",
         "33453598": "Raizen SA", "10760260": "CVC"}


NAMED_EVENTS = {  # public event dates (fato relevante / RJ / RE filing), approximate to the week
    "00776574": "2023-01-11",   # Americanas: accounting inconsistencies, RJ 2023-01-19
    "60444437": "2023-05-12",   # Light SESA: RJ
    "05303439": "2023-11-01",   # Unigel: standstill / restructuring
    "33041260": "2024-04-28",   # Casas Bahia: extrajudicial reorganisation
    "12648266": "2025-09-26",   # Ambipar: court protection
}


def distress_lists():
    g = lm[lm["day"] < H.HOLDOUT]
    # credit event = a DI+spread floater (little rate sensitivity) marked below 85% of par, or a named public event.
    # (A first version also used cdi_bps > 1000, which flags IPCA conversion noise at 77 issuers incl. Sabesp /
    # Petrobras and was rejected, see README.)
    di_low = g[(g["kind"] == "DI_SPREAD") & (g["ratio"] < 0.85)]
    ev = di_low.groupby("cnpj8")["day"].min()
    for cn, d in NAMED_EVENTS.items():
        d = pd.Timestamp(d)
        ev.loc[cn] = min(ev.get(cn, d), d)
    last = g.groupby("codigo").agg(last_day=("day", "max"), cnpj8=("cnpj8", "last"))
    lr = g.sort_values("day").groupby("codigo")["ratio"].last()
    last["last_ratio"] = lr
    ref = pd.read_pickle(H.HIST / "nightly" / "event_driven" / "bonds_ref.pkl").drop_duplicates("codigo").set_index("codigo")
    last["maturity"] = ref["maturity"].reindex(last.index)
    last["exit_reason"] = ref["exit_reason"].reindex(last.index)
    stopped = last[(last["last_day"] <= pd.Timestamp("2025-09-30"))
                   & (last["maturity"].isna() | (last["maturity"] > last["last_day"] + pd.Timedelta(days=60)))]
    # the bond went silent shortly before / during its issuer's credit event (Americanas-type hole: last print near
    # par, then the default is never marked because the bond never trades again). Bonds that stop near par long
    # before any event are treated as early redemptions (the SND table has no reliable early-redemption flag).
    evd = stopped["cnpj8"].map(ev)
    near = evd.notna() & (evd <= stopped["last_day"] + pd.Timedelta(days=365)) & \
        (evd >= stopped["last_day"] - pd.Timedelta(days=180))
    stopped = stopped.assign(dist_issuer=near | (stopped["cnpj8"].isin(ev.index) & (stopped["last_ratio"] < 0.97)))
    return ev, stopped


def R_recovery(R, codes_ratio: pd.Series, rec=0.40):
    """Bonds in codes_ratio (index codigo -> last ratio) jump to `rec` of par at their last grid row."""
    G, _ = pd.read_pickle(H.HIST / "sellab_returns.pkl")
    lastpos = G.groupby("b")["pos"].max()
    R2 = R.copy()
    n = 0
    for c, lr in codes_ratio.items():
        if c not in CODES or not (lr > rec):
            continue
        b = CODES.get_loc(c)
        p = int(lastpos.get(b, -1))
        if p < 0 or p >= HP:
            continue
        R2[p, b] = (1 + R2[p, b]) * (rec / lr) - 1
        n += 1
    return R2, n


def default_case_table(ev, rows_p4q, rows_p4, rows_u):
    """For each named / data-driven distressed issuer: was it held by P4 / P4+Q at its event, how was it marked."""
    g = lm[lm["day"] < H.HOLDOUT]
    out = []
    iss_b = pd.Series(H._issuer_of_b())
    for cn, d0 in ev.items():
        bonds = g[g["cnpj8"] == cn]
        codes = bonds["codigo"].unique()
        bidx = [CODES.get_loc(c) for c in codes if c in CODES]
        p0 = int(DD.searchsorted(d0))
        def held(rows):
            r = rows[rows["b"].isin(bidx) & (rows["k"] >= 0)]
            live = r[(r["k"] <= p0) & (r["e"] > p0)]
            ever = r[(r["k"] > p0 - 252) & (r["k"] <= p0)]
            return float(live["w"].sum()), int(len(ever))
        w4q, n4q = held(rows_p4q)
        w4, n4 = held(rows_p4)
        pre = bonds[(bonds["day"] < d0) & (bonds["day"] >= d0 - pd.Timedelta(days=90))]
        post = bonds[(bonds["day"] >= d0)]
        # issuer loss realised in R on the universe of its bonds from 63 bdays before to 126 after the event
        cols = np.array(bidx)
        seg = R0[max(p0 - 63, 0):min(p0 + 126, HP)][:, cols] if len(cols) else np.zeros((1, 1))
        cum = float(np.expm1(np.log1p(np.clip(seg, -0.99, None)).sum(axis=0)).mean()) if len(cols) else np.nan
        out.append({"cnpj8": cn, "name": NAMED.get(cn, ""), "event": str(d0.date()), "n_bonds": int(len(codes)),
                    "pre_ratio_med": float(pre["ratio"].median()) if len(pre) else np.nan,
                    "post_ratio_min": float(post["ratio"].min()) if len(post) else np.nan,
                    "P4Q_tranche_weight_live_at_event": w4q, "P4Q_entries_252d_before": n4q,
                    "P4_tranche_weight_live_at_event": w4, "P4_entries_252d_before": n4,
                    "mean_bond_R_-63_+126_%": cum * 100})
    return pd.DataFrame(out).sort_values("event")


# =====================================================================================================================
# 3. costs: ANBIMA bid/ask (Sep 2026 cross-section), SND intraday range, marks_liquidity institutional table
# =====================================================================================================================
def bond_liquidity():
    fs = sorted(glob.glob(str(H.HIST / "snd_trades_20*.csv.gz")))
    s = pd.concat([pd.read_csv(f, parse_dates=["date"]) for f in fs], ignore_index=True)
    s = s[s["date"] < H.HOLDOUT]
    s["vol"] = s["qty"] * s["pu_avg"]
    life = lm[lm["day"] < H.HOLDOUT].groupby("codigo")["day"].agg(["min", "max"])
    bd = np.busday_count(life["min"].to_numpy().astype("datetime64[D]"), life["max"].to_numpy().astype("datetime64[D]")) + 1
    adv = s.groupby("codigo")["vol"].sum().reindex(life.index) / bd
    kind = lm.drop_duplicates("codigo").set_index("codigo")["kind"]
    df = pd.DataFrame({"adv": adv, "kind": kind.reindex(life.index)})
    df["q"] = np.ceil(df["adv"].rank(pct=True) * 5).clip(1, 5).fillna(1).astype(int)
    df["grp"] = np.where(df["kind"] == "DI_SPREAD", "DI", "IPCA_PRE")
    return df


def anbima_costs(liq):
    rows = []
    for f in sorted(glob.glob(str(H.DATA_DIR / "cache" / "anbima_db_2026*.txt"))):
        if Path(f).stat().st_size == 0:
            continue
        from datetime import datetime
        from rfmonitor.sources import anbima_public as A
        d = datetime.strptime(Path(f).stem[-8:], "%Y%m%d").date()
        x = A.debentures(d)
        if x is not None:
            rows.append(x)
    if not rows:
        return None, {}
    a = pd.concat(rows, ignore_index=True)
    a = a[a["taxa_compra"].notna() & a["taxa_venda"].notna() & (a["duration_du"] > 0)]
    a["ba_price_bps"] = (a["taxa_compra"] - a["taxa_venda"]) * a["duration_du"] / 252 * 100   # rate % x dur -> bps
    a = a[(a["ba_price_bps"] > 0) & (a["ba_price_bps"] < 2000)]
    a = a.merge(liq[["q", "grp"]], left_on="codigo", right_index=True, how="inner")
    tab = a.groupby(["grp", "q"])["ba_price_bps"].median().unstack()
    return a, tab


def cost_vectors(liq, anb_tab, rng_tab):
    ml = {"DI": {1: 65, 2: 60, 3: 55, 4: 50, 5: 45}, "IPCA_PRE": {1: 169, 2: 161, 3: 152, 4: 144, 5: 136}}
    out = {}
    L = liq.reindex(CODES)
    grp = L["grp"].fillna("IPCA_PRE").to_numpy()
    q = L["q"].fillna(1).astype(int).to_numpy()
    out["liq_inst"] = np.array([ml[g][k] for g, k in zip(grp, q)], dtype=float)
    if anb_tab is not None:
        out["anbima_ba"] = np.array([anb_tab.loc[g, k] if (g in anb_tab.index and k in anb_tab.columns) else np.nan
                                     for g, k in zip(grp, q)], dtype=float)
        out["anbima_ba"] = np.where(np.isfinite(out["anbima_ba"]), out["anbima_ba"], np.nanmedian(out["anbima_ba"]))
    out["snd_range"] = np.array([rng_tab.loc[g, k] for g, k in zip(grp, q)], dtype=float)
    return out


def range_table(liq):
    s = B.snd_range()
    s = s[(s["trades"] >= 2) & (s["date"] < pd.Timestamp("2024-01-01"))]
    s = s.merge(liq[["q", "grp"]], left_on="codigo", right_index=True, how="inner")
    s["rt_bps"] = (s["hb"] + s["hs"]) * 1e4          # buy at max + sell at min = full range, round trip
    return s.groupby(["grp", "q"])["rt_bps"].median().unstack()


# =====================================================================================================================
# 4. fundamentals timing variants for the Q filter
# =====================================================================================================================
def worstq_variant(panel, lag_days=0, strict=True, drop_brapi=False, q=0.2):
    fu = H._fundamentals(strict=strict).copy()
    if drop_brapi:
        fu = fu[fu["f_source"].astype(str).str.lower() != "brapi"]
    fu["avail"] = pd.to_datetime(fu["avail"]) + pd.Timedelta(days=lag_days)
    fu = fu.sort_values("avail")
    g = panel[["day", "cnpj8", "univ"]].copy()
    g["cnpj8"] = g["cnpj8"].astype(str)
    m = pd.merge_asof(g.reset_index().sort_values("day"), fu, left_on="day", right_on="avail", by="cnpj8",
                      direction="backward").set_index("index").reindex(g.index)
    stale = (m["day"] - m["period_end"]).dt.days > H.FUND_STALE_DAYS
    cov = m["f_lev"].notna() & m["f_cov"].notna() & ~stale
    U = g["univ"] & cov
    rk = pd.DataFrame({c: m.loc[U, c].groupby(g.loc[U, "day"]).rank(pct=True) * s for c, s in H.QSIGN.items()})
    fq = rk.mean(axis=1, skipna=True).reindex(g.index)
    thr = fq.groupby(g["day"]).quantile(q)
    return (cov & g["univ"] & (fq <= g["day"].map(thr))).fillna(False).to_numpy()


# =====================================================================================================================
# 5. de-smoothing (Getmansky-Lo-Makarov MA(2))
# =====================================================================================================================
def glm(daily):
    from statsmodels.tsa.arima.model import ARIMA
    m = H.monthly(daily)
    m = m[m.index < H.HOLDOUT]
    try:
        fit = ARIMA(m.to_numpy(), order=(0, 0, 2)).fit()
        psi = fit.maparams
        th = np.r_[1.0, psi] / (1 + psi.sum())
        f = float(np.sqrt((th ** 2).sum()))
    except Exception:
        th, f = np.array([1.0, 0, 0]), 1.0
    sr = m.mean() / m.std() * np.sqrt(12)
    return {"theta0": float(th[0]), "sharpe_obs": float(sr), "sharpe_glm": float(sr * f),
            "vol_obs_%": float(m.std() * np.sqrt(12) * 100), "vol_true_%": float(m.std() * np.sqrt(12) * 100 / f),
            "t_nw6": float(H.nw_t(m.to_numpy(), 6)), "t_nw12": float(H.nw_t(m.to_numpy(), 12))}


# =====================================================================================================================
# main
# =====================================================================================================================
def main():
    log("diagnostics")
    dh, Rh = diag_holiday()
    RES["holiday"] = dh
    log("holiday", dh)
    du = diag_universe()
    RES["universe_pit"] = du
    log("universe", du)

    ev, stopped = distress_lists()
    RES["survivorship"] = {"n_stopped_before_maturity_pre2026": int(len(stopped)),
                           "n_stopped_distressed_issuer": int(stopped["dist_issuer"].sum()),
                           "n_stopped_last_ratio_lt_0.98": int((stopped["last_ratio"] < 0.98).sum()),
                           "n_stopped_last_ratio_lt_0.90": int((stopped["last_ratio"] < 0.90).sum()),
                           "n_distress_issuers_pre2026": int(len(ev))}
    pd.DataFrame({"cnpj8": ev.index, "event_date": ev.values}).assign(
        name=lambda z: z["cnpj8"].map(NAMED).fillna("")).to_csv(OUTD / "distress_events_pit.csv", index=False)
    Rd, nd = R_recovery(R0, stopped.loc[stopped["dist_issuer"], "last_ratio"])
    Rharsh, nh = R_recovery(R0, stopped.loc[stopped["last_ratio"] < 0.98, "last_ratio"])
    RES["survivorship"].update({"rec_dist_bonds_hit": nd, "rec_harsh_bonds_hit": nh})
    log("survivorship", RES["survivorship"])

    liq = bond_liquidity()
    anb, anb_tab = anbima_costs(liq)
    rtab = range_table(liq)
    COST = cost_vectors(liq, anb_tab, rtab)
    RES["costs"] = {"anbima_bidask_rt_bps_by_grp_q": None if anb_tab is None else anb_tab.round(1).to_dict(),
                    "anbima_n_quotes": 0 if anb is None else int(len(anb)),
                    "snd_intraday_range_rt_bps_by_grp_q_pre2024": rtab.round(1).to_dict(),
                    "marks_liquidity_institutional_rt_bps": {"DI": [65, 60, 55, 50, 45], "IPCA_PRE": [169, 161, 152, 144, 136]}}
    log("costs", RES["costs"])

    # fundamentals variants
    Pv = P.copy()
    Pv["wq_lag90"] = worstq_variant(P, lag_days=90)
    Pv["wq_nobrapi"] = worstq_variant(P, drop_brapi=True)
    Pv["wq_lag90_nobrapi"] = worstq_variant(P, lag_days=90, drop_brapi=True)
    Pv["wq_nonstrict"] = worstq_variant(P, strict=False)
    Pv["wq_repl"] = worstq_variant(P)
    RES["fund_replication_agreement"] = float((Pv["wq_repl"] == Pv["worstQ"].fillna(False).to_numpy())[Pv["univ"]].mean())
    for c in ["wq_lag90", "wq_nobrapi", "wq_lag90_nobrapi", "wq_nonstrict"]:
        Pv["p4q_" + c] = Pv["p4"] & ~Pv[c]
    SIG.update({"P4Q_lag90": lambda x: x["p4q_wq_lag90"].to_numpy(), "P4Q_nobrapi": lambda x: x["p4q_wq_nobrapi"].to_numpy(),
                "P4Q_lag90_nobrapi": lambda x: x["p4q_wq_lag90_nobrapi"].to_numpy(),
                "P4Q_nonstrict": lambda x: x["p4q_wq_nonstrict"].to_numpy()})

    def T(name):
        return T_of(name, panel=Pv)

    base = {n: E.tranche_book(T(n)) for n in ["U", "P4", "P4Q"]}
    RES["n_variants_tried"] += 0

    # ------------------------------------------------------------------ individual biases
    SC = {
        "v4_base": dict(),
        "holiday_accrual_fix": dict(R=Rh),
        "survivorship_rec40_harness": dict(R=H._Rmat("rec40")),
        "survivorship_distress_issuers_rec40": dict(R=Rd),
        "survivorship_harsh_all_stopped_lt98_rec40": dict(R=Rharsh),
        "raw_prints_no_spike_cleaning_at_execution": dict(pen_buy=pen(EX["SB"], 1), pen_sell=pen(EX["SB"], -1)),
        "exit_only_at_real_trade": dict(exit_fresh=True),
        "entry_latency_+1d": dict(entry_delay=1),
        "entry_latency_+5d": dict(entry_delay=5),
        "cost_flat_50": dict(cost_bps=50),
        "cost_liq_institutional": dict(cost_b=COST["liq_inst"]),
        "cost_anbima_bidask": dict(cost_b=COST.get("anbima_ba")),
        "cost_snd_intraday_range": dict(cost_b=COST["snd_range"]),
        "exec_worst_of_day_buy_max_sell_min": dict(pen_buy=pen(EX["RB"], 1), pen_sell=pen(EX["RS"], 1)),
    }
    if COST.get("anbima_ba") is None:
        SC.pop("cost_anbima_bidask")
    ind = {}
    books = {}
    for sc, kw in SC.items():
        r = {n: E.tranche_book(T(n), **kw) for n in ["U", "P4", "P4Q"]}
        books[sc] = r
        row = {}
        for n in ["U", "P4", "P4Q"]:
            row[n] = H.stats(r[n]["daily"])["ann_excess_%"]
        d1 = H.stats(r["P4Q"]["daily"], bench=r["U"]["daily"])
        d2 = H.stats(r["P4Q"]["daily"], bench=r["P4"]["daily"])
        d3 = H.stats(r["P4"]["daily"], bench=r["U"]["daily"])
        row.update({"P4Q-U": d1["diff_ann_%"], "t(P4Q-U)": d1["diff_t_nw"], "P4-U": d3["diff_ann_%"],
                    "P4Q-P4": d2["diff_ann_%"], "t(P4Q-P4)": d2["diff_t_nw"],
                    "P4Q_cost_%/yr": r["P4Q"]["cost_ann_%"]})
        ind[sc] = row
        RES["n_variants_tried"] += 1
        log(sc, {k: round(v, 3) for k, v in row.items()})
    # fundamentals timing (P4Q only)
    for n in ["P4Q_lag90", "P4Q_nobrapi", "P4Q_lag90_nobrapi", "P4Q_nonstrict"]:
        r = E.tranche_book(T(n))
        books["fund_" + n] = {"P4Q": r, "U": base["U"], "P4": base["P4"]}
        d1 = H.stats(r["daily"], bench=base["U"]["daily"])
        d2 = H.stats(r["daily"], bench=base["P4"]["daily"])
        ind["fundamentals_" + n] = {"U": ind["v4_base"]["U"], "P4": ind["v4_base"]["P4"],
                                    "P4Q": H.stats(r["daily"])["ann_excess_%"], "P4Q-U": d1["diff_ann_%"],
                                    "t(P4Q-U)": d1["diff_t_nw"], "P4-U": ind["v4_base"]["P4-U"],
                                    "P4Q-P4": d2["diff_ann_%"], "t(P4Q-P4)": d2["diff_t_nw"],
                                    "P4Q_cost_%/yr": r["cost_ann_%"]}
        RES["n_variants_tried"] += 1
        log(n, {k: round(v, 3) for k, v in ind["fundamentals_" + n].items()})
    IND = pd.DataFrame(ind).T
    base_row = IND.loc["v4_base"]
    for c in ["U", "P4", "P4Q", "P4Q-U", "P4-U", "P4Q-P4"]:
        IND["Δ" + c] = IND[c] - base_row[c]
    RES["individual"] = IND.round(4).to_dict(orient="index")
    IND.round(3).to_csv(OUTD / "haircut_individual.csv")

    # ------------------------------------------------------------------ cumulative 'honest' build-up
    Rcum = B.R_hol(Rd)          # holiday fix on top of distress-issuer recovery
    steps = [
        ("v4_base", dict(), "P4Q"),
        ("+ holiday accrual fix", dict(R=B.R_hol(R0)), "P4Q"),
        ("+ distress-issuer recovery 40%", dict(R=Rcum), "P4Q"),
        ("+ raw prints at execution", dict(R=Rcum, pen_buy=pen(EX["SB"], 1), pen_sell=pen(EX["SB"], -1)), "P4Q"),
        ("+ exit only at a real trade", dict(R=Rcum, pen_buy=pen(EX["SB"], 1), pen_sell=pen(EX["SB"], -1), exit_fresh=True), "P4Q"),
        ("+ liquidity-bucket costs (institutional)", dict(R=Rcum, pen_buy=pen(EX["SB"], 1), pen_sell=pen(EX["SB"], -1), exit_fresh=True,
                                                     cost_b=COST["liq_inst"]), "P4Q"),
        ("+ fundamentals lag 90d (honest)", dict(R=Rcum, pen_buy=pen(EX["SB"], 1), pen_sell=pen(EX["SB"], -1), exit_fresh=True,
                                                 cost_b=COST["liq_inst"]), "P4Q_lag90"),
    ]
    cum = {}
    for lab, kw, q in steps:
        r = {"U": E.tranche_book(T("U"), **kw), "P4": E.tranche_book(T("P4"), **kw), "P4Q": E.tranche_book(T(q), **kw)}
        books["cum:" + lab] = r
        d1 = H.stats(r["P4Q"]["daily"], bench=r["U"]["daily"])
        d2 = H.stats(r["P4Q"]["daily"], bench=r["P4"]["daily"])
        d3 = H.stats(r["P4"]["daily"], bench=r["U"]["daily"])
        cum[lab] = {"U": H.stats(r["U"]["daily"])["ann_excess_%"], "P4": H.stats(r["P4"]["daily"])["ann_excess_%"],
                    "P4Q": H.stats(r["P4Q"]["daily"])["ann_excess_%"], "P4Q-U": d1["diff_ann_%"],
                    "t(P4Q-U)": d1["diff_t_nw"], "P4-U": d3["diff_ann_%"], "t(P4-U)": d3["diff_t_nw"],
                    "P4Q-P4": d2["diff_ann_%"], "t(P4Q-P4)": d2["diff_t_nw"],
                    "P4Q_h1": d1["h1_2022_23_%"], "P4Q_h2": d1["h2_2024_25_%"],
                    "P4Q-U_h1": d1["diff_h1_%"], "P4Q-U_h2": d1["diff_h2_%"]}
        log("cum", lab, {k: round(v, 3) for k, v in cum[lab].items()})
    CUM = pd.DataFrame(cum).T
    RES["cumulative"] = CUM.round(4).to_dict(orient="index")
    CUM.round(3).to_csv(OUTD / "haircut_cumulative.csv")
    honest_kw = steps[-1][1]
    honest = books["cum:" + steps[-1][0]]

    # stress layer on top of honest: worst-of-day execution + harsh survivorship + 5d latency
    stress_kw = dict(R=B.R_hol(Rharsh), pen_buy=lambda k, b: EX["SB"][k, b] + EX["RB"][k, b],
                     pen_sell=lambda k, b: -EX["SB"][k, b] + EX["RS"][k, b], exit_fresh=True, cost_b=COST["liq_inst"])
    stress = {"U": E.tranche_book(T("U"), **stress_kw), "P4": E.tranche_book(T("P4"), **stress_kw),
              "P4Q": E.tranche_book(T("P4Q_lag90"), **stress_kw)}
    s1 = H.stats(stress["P4Q"]["daily"], bench=stress["U"]["daily"])
    RES["stress"] = {"U": H.stats(stress["U"]["daily"])["ann_excess_%"], "P4": H.stats(stress["P4"]["daily"])["ann_excess_%"],
                     "P4Q": H.stats(stress["P4Q"]["daily"])["ann_excess_%"], "P4Q-U": s1["diff_ann_%"],
                     "t(P4Q-U)": s1["diff_t_nw"]}
    log("stress", RES["stress"])

    # ------------------------------------------------------------------ paired tests vs base P4+Q (Holm across all tried)
    cmp = {"honest_P4Q": honest["P4Q"], "honest_P4": honest["P4"], "honest_U": honest["U"],
           "stress_P4Q": stress["P4Q"]}
    for sc in SC:
        if sc != "v4_base":
            cmp["ind_" + sc + "_P4Q"] = books[sc]["P4Q"]
    for n in ["P4Q_lag90", "P4Q_nobrapi", "P4Q_lag90_nobrapi", "P4Q_nonstrict"]:
        cmp["fund_" + n] = books["fund_" + n]["P4Q"]
    tab = H.compare(cmp, bench=base["P4Q"]["daily"], universe=base["U"]["daily"])
    tab.round(4).to_csv(OUTD / "compare_vs_base_P4Q.csv")
    RES["compare_vs_base_P4Q"] = tab.round(4).to_dict(orient="index")

    # ------------------------------------------------------------------ parameter-selection (snooping) haircut
    log("parameter neighbourhood")
    grid = []
    for cut in [0.2, 0.3, 0.4]:
        for rich in [None, -1.0, -1.5, -2.0]:
            for press in [True, False]:
                for q in [0.1, 0.2, 0.3]:
                    grid.append((cut, rich, press, q))
    wq = {q: worstq_variant(P, q=q) for q in [0.1, 0.2, 0.3]}
    nz = lambda s, v: s.fillna(v)
    neigh = []
    for cut, rich, press, q in grid:
        f = Pv["univ"] & (nz(Pv["cdi_pct"], 1) <= cut)
        if rich is not None:
            f &= ~(nz(Pv["resid_z"], 0) <= rich)
        if press:
            f &= nz(Pv["press_neg_30d"], 0) < 1
        f4 = f.to_numpy()
        fq = f4 & ~wq[q]
        Pv["_s4"] = f4
        Pv["_sq"] = fq
        T4 = E.targets(lambda x: x["_s4"].to_numpy(), Pv)
        TQ = E.targets(lambda x: x["_sq"].to_numpy(), Pv)
        r4 = H.stats(E.tranche_book(T4)["daily"], bench=base["U"]["daily"])["diff_ann_%"]
        rq = H.stats(E.tranche_book(TQ)["daily"], bench=base["U"]["daily"])["diff_ann_%"]
        neigh.append({"cut": cut, "rich": rich, "press": press, "q": q, "P4-U": r4, "P4Q-U": rq})
        RES["n_variants_tried"] += 2
    NB_ = pd.DataFrame(neigh)
    NB_.to_csv(OUTD / "param_neighbourhood.csv", index=False)
    chosen = NB_[(NB_.cut == 0.3) & (NB_.rich == -1.5) & (NB_.press) & (NB_.q == 0.2)].iloc[0]
    p4n = NB_.drop_duplicates(["cut", "rich", "press"])
    RES["snooping"] = {"P4Q-U_chosen": float(chosen["P4Q-U"]), "P4Q-U_neigh_median": float(NB_["P4Q-U"].median()),
                       "P4Q-U_neigh_p10": float(NB_["P4Q-U"].quantile(0.1)), "P4Q-U_neigh_p90": float(NB_["P4Q-U"].quantile(0.9)),
                       "P4Q_chosen_pct_rank": float((NB_["P4Q-U"] < chosen["P4Q-U"]).mean()),
                       "P4-U_chosen": float(chosen["P4-U"]), "P4-U_neigh_median": float(p4n["P4-U"].median()),
                       "P4_chosen_pct_rank": float((p4n["P4-U"] < chosen["P4-U"]).mean()),
                       "n_neighbours": int(len(NB_))}
    log("snooping", RES["snooping"])

    # ------------------------------------------------------------------ de-smoothing
    RES["glm"] = {"base_P4Q": glm(base["P4Q"]["daily"]), "honest_P4Q": glm(honest["P4Q"]["daily"]),
                  "honest_U": glm(honest["U"]["daily"]),
                  "honest_P4Q_minus_U": glm(honest["P4Q"]["daily"] - honest["U"]["daily"])}
    log("glm", RES["glm"])

    # ------------------------------------------------------------------ placebo on the honest engine
    rng = np.random.default_rng(0)
    TQ = T("P4Q_lag90")
    TU = T("U")
    Ub = {p: np.array(P.loc[(P["dpos"] == p) & P["univ"], "b"]) for p in TQ}
    iss = H._issuer_of_b()
    pl = []
    for i in range(10):
        Tp = {}
        for p, w in TQ.items():
            if not len(w):
                Tp[p] = w
                continue
            pick = rng.choice(Ub[p], size=min(len(w), len(Ub[p])), replace=False)
            ww = H.cap_weights(iss[pick], H.ISSUER_CAP)
            Tp[p] = pd.Series(ww, index=pick).groupby(level=0).sum()
        pl.append(H.stats(E.tranche_book(Tp, **honest_kw)["daily"], bench=honest["U"]["daily"])["diff_ann_%"])
    RES["placebo_honest"] = {"P4Q-U_actual": cum[steps[-1][0]]["P4Q-U"], "random_mean": float(np.mean(pl)),
                             "random_p95": float(np.percentile(pl, 95)), "n": 10}
    log("placebo", RES["placebo_honest"])

    # ------------------------------------------------------------------ default case table
    dct = default_case_table(ev, base["P4Q"]["rows"], base["P4"]["rows"], base["U"]["rows"])
    dct.to_csv(OUTD / "default_cases.csv", index=False)
    RES["default_cases_top"] = dct.sort_values("P4Q_tranche_weight_live_at_event", ascending=False).head(15).round(4).to_dict(orient="records")
    RES["default_cases_summary"] = {"n_issuers": int(len(dct)), "held_by_P4Q_at_event": int((dct["P4Q_tranche_weight_live_at_event"] > 0).sum()),
                                    "held_by_P4_at_event": int((dct["P4_tranche_weight_live_at_event"] > 0).sum())}

    # ------------------------------------------------------------------ charts
    plot(base, honest, stress)
    # save honest daily series for combiners
    pd.DataFrame({k: v["daily"] for k, v in {"honest_U": honest["U"], "honest_P4": honest["P4"], "honest_P4Q": honest["P4Q"],
                                             "base_U": base["U"], "base_P4": base["P4"], "base_P4Q": base["P4Q"]}.items()}
                 ).to_pickle(CACHE / "honest_daily.pkl")

    # ------------------------------------------------------------------ sealed holdout, once, frozen spec
    if "--holdout" in sys.argv:
        RES["holdout"] = holdout_eval(honest_kw, Rcum)
    RES["runtime_s"] = time.time() - T0
    json.dump(RES, open(OUTD / "results.json", "w"), indent=1, default=float)
    log("done")


def plot(base, honest, stress):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cols = {"U": "#8c8c8c", "P4": "#4c78a8", "P4Q": "#e45756"}
    cdi = H.cdi_daily()
    fig, ax = plt.subplots(figsize=(11, 6))
    idx = base["U"]["daily"].index
    cd = (1 + cdi.reindex(idx).fillna(0)).cumprod() * 100
    ax.plot(cd.index, cd, color="black", lw=1.2, label="CDI")
    for n in ["U", "P4", "P4Q"]:
        tr = H.total_return_curve(base[n]["daily"])
        ax.plot(tr.index, tr, color=cols[n], lw=1.0, ls="--", label=f"{n} v4 harness")
        tr = H.total_return_curve(honest[n]["daily"])
        ax.plot(tr.index, tr, color=cols[n], lw=2.0, label=f"{n} honest")
    tr = H.total_return_curve(stress["P4Q"]["daily"])
    ax.plot(tr.index, tr, color=cols["P4Q"], lw=1.0, ls=":", label="P4Q stress")
    for nm, c in [("IDADI", "#54a24b"), ("IBOV", "#b279a2")]:
        try:
            x = H.index_excess(nm).reindex(idx).fillna(0)
            tr = H.total_return_curve(x)
            ax.plot(tr.index, tr, color=c, lw=1.0, alpha=0.8, label=nm)
        except Exception as e:
            log("index", nm, e)
    ax.set_yscale("log")
    ax.set_title("Total return (CDI x (1+excess)), 2022-2025, pre-holdout: v4 harness vs bias-adjusted 'honest'")
    ax.legend(ncol=3, fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTD / "equity_total_return.png", dpi=130)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(11, 5))
    for lab, bk, ls in [("v4 harness", base, "--"), ("honest", honest, "-"), ("stress", stress, ":")]:
        for n in ["P4", "P4Q"]:
            d = (bk[n]["daily"] - bk["U"]["daily"]).cumsum() * 100
            ax.plot(d.index, d, color=cols[n], ls=ls, lw=2 if lab == "honest" else 1, label=f"{n} - U ({lab})")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("cumulative excess vs universe, %")
    ax.set_title("Cumulative excess vs the universe (same assumptions on both sides)")
    ax.legend(ncol=3, fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTD / "cum_excess.png", dpi=130)
    plt.close(fig)


def holdout_eval(honest_kw, Rcum):
    PH = H.load_panel("M", holdout=True)
    PH = PH.copy()
    PH["wq_lag90"] = worstq_variant(PH, lag_days=90)
    PH["p4q_wq_lag90"] = PH["p4"] & ~PH["wq_lag90"]
    kw = {k: v for k, v in honest_kw.items() if k not in ("pen_buy", "pen_sell")}
    kw["pen_buy"] = pen(EX["SB"], 1)
    kw["pen_sell"] = pen(EX["SB"], -1)
    out = {}
    for lab, sig in [("U", SIG["U"]), ("P4", SIG["P4"]), ("P4Q", SIG["P4Q"]), ("P4Q_honest", lambda x: x["p4q_wq_lag90"].to_numpy())]:
        Tt = E.targets(sig, PH, t_end=ND)
        rb = E.tranche_book(Tt, t_end=ND)
        rh = E.tranche_book(Tt, t_end=ND, **kw)
        out[lab] = {"base": rb["daily"], "honest": rh["daily"]}
    res = {}
    for lab in out:
        res[lab + "_base"] = H.stats(out[lab]["base"], holdout="only")["ann_excess_%"]
        res[lab + "_honest"] = H.stats(out[lab]["honest"], holdout="only")["ann_excess_%"]
    d = H.stats(out["P4Q_honest"]["honest"], bench=out["U"]["honest"], holdout="only")
    res["honest_P4Q-U"] = d["diff_ann_%"]
    res["honest_P4Q-U_t"] = d["diff_t_nw"]
    d = H.stats(out["P4Q_honest"]["honest"], bench=out["P4Q"]["base"], holdout="only")
    res["honest_P4Q_vs_base_P4Q"] = d["diff_ann_%"]
    log("holdout", res)
    return res


if __name__ == "__main__":
    main()
