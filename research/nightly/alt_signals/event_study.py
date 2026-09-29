"""Event study: what happens to an issuer's EXISTING debentures around a new debenture offer (CVM registry).

Day 0 = availability date of the offer (legacy: start + 7d; RCVM 160: request + 1d). Daily excess of the issuer's
bonds that already had a grid mark before day 0, equal-weighted, minus the equal-weight mean of all bonds with a
mark that day (harness patched returns, gap moves booked when realised). Pre-2026 only (window truncated at the
holdout). Events of the same issuer within 60 days of a previous one are dropped (first kept).
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.nightly import harness as H
from research.nightly.alt_signals import signals as S

OUT = Path(__file__).resolve().parent
PRE, POST = 63, 126


def run(Q: pd.DataFrame) -> dict:
    C = H._core()
    R = H._Rmat("base")
    pres = C["TD"] >= 0
    days = C["days"]
    hpos = H._hpos()
    iss = H._issuer_of_b()
    um = np.where(pres, R, np.nan)
    umean = np.nanmean(um[:, :], axis=1)
    X = np.where(pres, R - umean[:, None], np.nan)

    o = S.load_offers().sort_values(["cnpj8", "avail"])
    o = o[(o["avail"] >= "2021-06-01") & (o["avail"] < H.HOLDOUT)]
    keep = o.groupby("cnpj8")["avail"].diff().dt.days.fillna(9999) > 60
    o = o[keep]
    # P4Q membership of the issuer at the last monthly decision before the event
    Qm = Q[Q["univ"]][["day", "cnpj8", "p4q", "cdi_pct"]]
    rows, paths = [], []
    for _, e in o.iterrows():
        p0 = int(days.searchsorted(e["avail"]))
        if p0 - PRE < 0 or p0 >= hpos:
            continue
        bs = np.where((iss == e["cnpj8"]) & pres[p0 - 1])[0]
        if not len(bs):
            continue
        hi = min(p0 + POST, hpos)
        path = np.full(PRE + POST, np.nan)
        seg = np.nanmean(X[p0 - PRE:hi][:, bs], axis=1)
        path[:len(seg)] = seg
        q = Qm[(Qm["cnpj8"] == e["cnpj8"]) & (Qm["day"] < e["avail"])]
        q = q[q["day"] == q["day"].max()] if len(q) else q
        rows.append({"cnpj8": e["cnpj8"], "avail": e["avail"], "n_bonds": len(bs), "incent": bool(e["incent"]),
                     "p4q": bool(q["p4q"].any()) if len(q) else False,
                     "hicarry": bool((q["cdi_pct"] <= 0.3).any()) if len(q) else False})
        paths.append(path)
    ev = pd.DataFrame(rows)
    Pm = np.vstack(paths)
    rel = np.arange(-PRE, POST)
    out = {"n_events": int(len(ev))}

    def car(mask, lo, hi):
        a = np.where((rel >= lo) & (rel < hi))[0]
        c = np.nansum(Pm[mask][:, a], axis=1)
        m = ev.loc[mask, "avail"].dt.to_period("M")
        # cluster by event month
        cm = pd.Series(c).groupby(m.values).mean()
        se = cm.std(ddof=1) / np.sqrt(len(cm)) if len(cm) > 2 else np.nan
        return {"car_%": round(float(np.mean(c)) * 100, 3), "t_cluster": round(float(cm.mean() / se), 2) if se else None,
                "n": int(mask.sum())}

    groups = {"all": np.ones(len(ev), bool), "p4q_issuer": ev["p4q"].to_numpy(), "hicarry": ev["hicarry"].to_numpy(),
              "not_hicarry": ~ev["hicarry"].to_numpy(), "incent": ev["incent"].to_numpy()}
    wins = {"pre[-63,-1]": (-63, 0), "post[0,21)": (0, 21), "post[0,63)": (0, 63), "post[0,126)": (0, 126),
            "post[21,126)": (21, 126)}
    for g, m in groups.items():
        out[g] = {w: car(m, *lh) for w, lh in wins.items()}
    # plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for g in ("all", "hicarry", "not_hicarry", "p4q_issuer"):
        m = groups[g]
        cp = np.nancumsum(np.nanmean(Pm[m], axis=0)) * 100
        cp -= cp[PRE - 1]
        ax.plot(rel, cp, label=f"{g} (n={m.sum()})", lw=2 if g == "p4q_issuer" else 1.2)
    ax.axvline(0, color="k", lw=0.8)
    ax.axhline(0, color="grey", lw=0.6)
    ax.set_xlabel("business days from offer availability (CVM registry)")
    ax.set_ylabel("cum. excess vs universe, % (rebased at day -1)")
    ax.set_title("Existing bonds of an issuer launching a new debenture offer (pre-2026)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "supply_event_study.png", dpi=120)
    plt.close(fig)
    return out


if __name__ == "__main__":
    Q = pd.read_pickle(S.CACHE / "panel_alt_M.pkl")
    r = run(Q)
    print(json.dumps(r, indent=1))
    (OUT / "event_study.json").write_text(json.dumps(r, indent=1))
