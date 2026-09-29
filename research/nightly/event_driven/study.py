"""Event studies on hedged bond returns (harness v4 labels), pre-2026 only.

For every event we take the first WEEKLY harness decision strictly after the event date, and the event bonds'
forward hedged excess returns fwd_H (entry at the first trade after the decision). Abnormal returns:
  AR_U   = fwd_H - mean fwd_H of the universe that date
  AR_C   = fwd_H - carry/duration-matched expectation (per-date OLS of fwd_H on cdi_bps, dur over the universe)
Pre-event drift: realised bond return over [p-63, p) minus the universe mean (same grid window).
Stats: equal-weight events within a date, then dates; Newey-West t with lag = H/5 weeks.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from research.nightly import harness as H
from research.nightly.event_driven.events import build

HS = (21, 63, 126)


def ar_panel(P: pd.DataFrame) -> pd.DataFrame:
    """Adds ar_u_H and ar_c_H to every row of the (weekly) panel."""
    P = P.copy()
    P["cb"] = P["cdi_bps"].clip(-100, 1000)
    P["du"] = P["dur"].clip(0, 15)
    for h in HS:
        y = f"fwd_{h}"
        P[f"ar_u_{h}"] = np.nan
        P[f"ar_c_{h}"] = np.nan
    for p, idx in P.groupby("dpos").groups.items():
        x = P.loc[idx]
        u = x["univ"].to_numpy() & x["cb"].notna().to_numpy()
        X = np.c_[np.ones(len(x)), x["cb"].fillna(x["cb"][u].median() if u.any() else 0).to_numpy(),
                  x["du"].fillna(3).to_numpy(), (x["kind"] == "IPCA").to_numpy(float),
                  (x["kind"] == "PRE").to_numpy(float), x["incent"].fillna(0).to_numpy(float)]
        for h in HS:
            y = x[f"fwd_{h}"].to_numpy()
            ok = u & np.isfinite(y)
            if ok.sum() < 30:
                continue
            yw = np.clip(y, -0.5, 0.5)
            P.loc[idx, f"ar_u_{h}"] = y - yw[ok].mean()
            beta = np.linalg.lstsq(X[ok], yw[ok], rcond=None)[0]
            P.loc[idx, f"ar_c_{h}"] = y - X @ beta
    # pre-event realised drift [p-63, p)
    LC = H._LC()
    pre = np.full(len(P), np.nan)
    pos = P["dpos"].to_numpy()
    b = P["b"].to_numpy()
    ok = pos >= 63
    pre[ok] = np.expm1(LC[pos[ok], b[ok]] - LC[pos[ok] - 63, b[ok]])
    P["pre63"] = pre
    P["pre63_ar"] = P["pre63"] - P["pre63"].where(P["univ"]).groupby(P["dpos"]).transform("mean")
    return P


def first_decision_after(dates: pd.Series, P: pd.DataFrame) -> np.ndarray:
    dd = np.sort(P["day"].unique())
    i = np.searchsorted(dd, dates.to_numpy().astype("datetime64[ns]"), side="right")   # strictly after
    out = np.full(len(dates), np.datetime64("NaT"), dtype="datetime64[ns]")
    ok = i < len(dd)
    out[ok] = dd[i[ok]]
    return out


def summarize(rows: pd.DataFrame, label: str, univ_only: bool = False) -> dict:
    r = rows[rows["univ"]] if univ_only else rows
    out = {"event": label, "n_events": int(r[["key", "day"]].drop_duplicates().shape[0]) if "key" in r else int(len(r)),
           "n_bond_obs": int(len(r)), "n_dates": int(r["day"].nunique())}
    for h in HS:
        for k in ("u", "c"):
            c = f"ar_{k}_{h}"
            s = r.dropna(subset=[c])
            s = s[s["day"] < H.HOLDOUT]
            if len(s) < 5:
                continue
            per = s.groupby("day")[c].mean().sort_index()
            out[f"ar_{k}_{h}_%"] = round(float(per.mean() * 100), 3)
            out[f"t_{k}_{h}"] = round(H.nw_t(per.to_numpy(), max(1, h // 5)), 2)
    s = r.dropna(subset=["pre63_ar"])
    if len(s) >= 5:
        per = s.groupby("day")["pre63_ar"].mean()
        out["pre63_ar_%"] = round(float(per.mean() * 100), 3)
        out["t_pre63"] = round(H.nw_t(per.to_numpy(), 12), 2)
    out["mean_ratio"] = round(float(r["ratio"].mean()), 4)
    out["mean_resid_z"] = round(float(r["resid_z"].mean()), 3)
    out["share_univ"] = round(float(r["univ"].mean()), 3)
    out["share_p4q"] = round(float(r["p4q"].mean()), 3)
    return out


def issuer_event_rows(E: pd.DataFrame, P: pd.DataFrame, etype: str, dedupe_days: int = 90) -> pd.DataFrame:
    e = E[E["etype"] == etype].sort_values(["cnpj8", "date"]).copy()
    # first event of this type per issuer within `dedupe_days` (filings come in bunches: edital, ata, ...)
    prev = e.groupby("cnpj8")["date"].shift(1)
    e = e[prev.isna() | ((e["date"] - prev).dt.days > dedupe_days)]
    e["day"] = first_decision_after(e["date"], P)
    e = e.dropna(subset=["day"])
    rows = P.merge(e[["cnpj8", "day", "date"]].rename(columns={"date": "edate"}), on=["cnpj8", "day"], how="inner")
    rows["key"] = rows["cnpj8"]
    return rows


def run(P=None, E=None, ref=None):
    if E is None:
        ref, E = build()
    if P is None:
        P = ar_panel(H.load_panel("W"))
    res = []
    for et in ["agd", "agd_waiver", "resgate", "deb_buyback", "equity_raise", "ma", "rj", "rating_up", "rating_down",
               "ipe_new_deb", "supply_new_issue"]:
        rows = issuer_event_rows(E, P, et)
        if et == "supply_new_issue":   # the issuer's EXISTING bonds (not the new bonds themselves)
            nd = ref.set_index("codigo")["dist_start"]
            ds = rows["codigo"].map(nd)
            rows = rows[ds.isna() | (ds < rows["edate"] - pd.Timedelta(days=30))]
        res.append(summarize(rows, et))
        res.append(summarize(rows, et + " [univ]", univ_only=True))
    # ---- bond-level: new issues (first appearance on the grid within 120 days of distribution start)
    nd = ref.set_index("codigo")["dist_start"]
    P["dist_start"] = P["codigo"].map(nd)
    P["months_since_issue"] = (P["day"] - P["dist_start"]).dt.days / 30.44
    first = P.sort_values("day").drop_duplicates("codigo")
    ni = first[(first["months_since_issue"] >= 0) & (first["months_since_issue"] <= 4)]
    ni_rows = ni.copy()
    ni_rows["key"] = ni_rows["codigo"]
    res.append(summarize(ni_rows, "new_issue_first_week"))
    res.append(summarize(ni_rows, "new_issue_first_week [univ]", univ_only=True))
    for flag, lab in [("incent", "incent"), ("icvm476", "476")]:
        m = ni_rows["codigo"].map(ref.set_index("codigo")[flag]).fillna(False).astype(bool)
        res.append(summarize(ni_rows[m], f"new_issue [{lab}]"))
        res.append(summarize(ni_rows[~m], f"new_issue [not {lab}]"))
    for k in ["DI_SPREAD", "IPCA"]:
        res.append(summarize(ni_rows[ni_rows["kind"] == k], f"new_issue [{k}]"))
    # ---- seasoning curve: every universe row by months since distribution start (monthly spacing of the weekly panel)
    Pm = P[P["univ"] & P["day"].isin(H.load_panel("M")["day"].unique())]
    bins = [-1, 3, 6, 12, 24, 48, 1000]
    Pm = Pm.assign(bucket=pd.cut(Pm["months_since_issue"], bins, labels=["0-3m", "3-6m", "6-12m", "12-24m", "24-48m", "48m+"]))
    season = []
    for bk, g in Pm.groupby("bucket", observed=True):
        d = summarize(g.assign(key=g["codigo"]), f"season {bk}")
        d["resid_bps_mean"] = round(float(g["resid_bps"].mean()), 1)
        d["cdi_bps_mean"] = round(float(g["cdi_bps"].mean()), 1)
        season.append(d)
    # ---- redemptions actually executed (registry reason: RESGATE TOTAL ANTECIPADO), ex-post
    rd = ref[ref["exit_reason"].fillna("").str.contains("RESGATE|VENCIMENTO ANTECIPADO|AQUISI") &
             (ref["exit_date"] >= "2021-06-01") & (ref["exit_date"] < H.HOLDOUT)]
    dd = H.days()
    LC = H._LC()
    codes = H._core()["codes"]
    red = []
    for _, r in rd.iterrows():
        if r["codigo"] not in codes:
            continue
        bb = codes.get_loc(r["codigo"])
        pe = int(dd.searchsorted(r["exit_date"]))
        if pe < 130 or pe >= len(dd):
            continue
        rows = P[(P["codigo"] == r["codigo"]) & (P["dpos"] < pe)]
        last_ratio = float(rows["ratio"].iloc[-1]) if len(rows) else np.nan
        red.append({"codigo": r["codigo"], "exit": r["exit_date"], "reason": r["exit_reason"],
                    "ret_126_pre": float(np.expm1(LC[pe, bb] - LC[pe - 126, bb])),
                    "ret_21_pre": float(np.expm1(LC[pe, bb] - LC[pe - 21, bb])),
                    "last_ratio": last_ratio, "callable": bool(r["callable"])})
    red = pd.DataFrame(red)
    return P, pd.DataFrame(res), pd.DataFrame(season), red


if __name__ == "__main__":
    P, tab, season, red = run()
    pd.set_option("display.width", 250, "display.max_columns", 40)
    print(tab[["event", "n_events", "n_bond_obs", "n_dates", "ar_u_63_%", "t_u_63", "ar_c_63_%", "t_c_63", "ar_c_126_%",
               "t_c_126", "pre63_ar_%", "t_pre63", "mean_ratio", "share_p4q"]])
    print(season)
    P.to_pickle(H.HIST / "nightly" / "event_driven" / "panelW_ar.pkl")
