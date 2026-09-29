"""Brazil USD corporate eurobonds: carry screen, LONG basket only (control), monthly.

Selection = the long leg of the eurobonds study's S2 carry signal (research/ibkr_lab/eurobonds/rv.py: premium vs the
fitted Brazil sovereign curve, 'healthy' bonds only: price >= 70, not suspect), as written in eurobonds/results.json
backtests.current_picks.S2_carry_LS.long (re-run eurobonds/collect.py + rv.py before the monthly rebalance to refresh;
the pick date is stored in each fill note). Top N=5 by signal that have a quote in the latest cross_section.csv.
No shorts (eurobonds are effectively not borrowable) and no sovereign hedge: this is the unhedged long carry leg, so it
carries Brazil/HY beta (the study's S2 was L/S; its long leg alone is closer to the EW healthy-corp benchmark).
Size: face value per bond = floor(48k / (price% x 10) ) in USD 1,000 steps (IBKR bond size = face value in USD;
valuation = face x price / 100). NB some lines have larger minimum pieces at IBKR; a real paper order may be rejected.
"""
from __future__ import annotations

import json
import math
from datetime import date

import pandas as pd

from research.ibkr_lab import safe
from research.ibkr_lab.runner.strategies import _common as C

NAME = "eurobond_carry"
DESCRIPTION = "Brazil USD corporates, top-5 carry (premium vs sovereign curve) long-only, ~48k each, unhedged"
REBALANCE = "monthly"
N = 5
EXPECTED = {"ann_return": 1.9, "vol": 4.0, "sharpe": 0.48, "status": "control",
            "note": "eurobonds: S2 carry L/S (sov-hedged) net 21.6%/yr, 9.6% vol, SR 2.26 but t 1.45 on 5 months, "
                    "2nd half SR -0.65, corr 0.59 with HY beta. Long-only unhedged is closer to the EW healthy corp "
                    "benchmark: 1.9%/yr, 4.0% vol, SR 0.48 (t 0.3) -- used as EXPECTED. Bid/ask 54 bp median."}


def targets(ib, asof: date):
    from ib_async import Contract
    from research.ibkr_lab.eurobonds import rv
    res = json.loads((rv.HERE / "results.json").read_text(encoding="utf-8"))
    picks = res["backtests"]["current_picks"]["S2_carry_LS"]
    xs = pd.read_csv(safe.DATA / "eurobonds" / "cross_section.csv").set_index("conId")
    longs = [p for p in sorted(picks["long"], key=lambda p: -p["signal"])
             if p["conId"] in xs.index and pd.notna(xs.loc[p["conId"], "price"])
             and not bool(xs.loc[p["conId"], "suspect_quote"])][:N]
    out = []
    for p in longs:
        c = Contract(conId=int(p["conId"]), secType="BOND")
        q = ib.qualifyContracts(c)
        if not q or not c.conId:
            continue
        px = C.quote(ib, c)
        price = px["ask"] or px["mid"] or float(xs.loc[p["conId"], "price"])
        face = math.floor(C.POS_CAP / (price / 100) / 1000) * 1000
        if face < 1000:
            continue
        out.append({"contract": c, "key": f"BOND:{p['desc']}", "qty": face, "side": "LONG",
                    "target_notional_usd": face * price / 100,
                    "note": f"carry prem {p['signal']:.0f}bp picks={picks.get('date')}"})
    return out
