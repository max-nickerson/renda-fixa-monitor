"""Retire runner strategies: close their PAPER positions (marketable limit / market for futures, only what fills is
booked) and move the plugin file to strategies/_retired/ so the runner stops loading it.
    python research/ibkr_lab/runner/retire.py name1 name2 ...
"""
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from ib_async import Contract  # noqa: E402

from research.ibkr_lab import safe  # noqa: E402
from research.ibkr_lab.runner import forward as F  # noqa: E402

STRAT = Path(__file__).parent / "strategies"


def close_positions(ib, name: str) -> list:
    out = []
    with F.db() as con:
        rows = [dict(r) for r in con.execute("SELECT * FROM positions WHERE strategy=?", (name,))]
    for r in rows:
        c = Contract(conId=r["conid"])
        ib.qualifyContracts(c)
        q = F._price(ib, c)
        qty = abs(r["qty"])
        action = "SELL" if r["qty"] > 0 else "BUY"
        otype = "LMT" if c.secType in ("STK", "BOND") else "MKT"
        lim = None
        if otype == "LMT":
            ref = (q["bid"] if action == "SELL" else q["ask"]) or q["last"]
            lim = round(ref * (0.997 if action == "SELL" else 1.003), 2)
        st = safe.place_paper_order(ib, c, action, qty, otype, lim, strategy=f"retire:{name}",
                                    est_notional_usd=None, wait=3, wait_fill=60, cancel_unfilled=True)
        filled = float(st.get("filled") or 0)
        with F.db() as con:
            if filled > 0:
                left = r["qty"] - (filled if r["qty"] > 0 else -filled)
                con.execute("INSERT INTO fills VALUES (?,?,?,?,?,?,?,?,?,?)",
                            (datetime.now(timezone.utc).isoformat(timespec="seconds"), name, r["key"], action, filled,
                             st.get("avg_price"), q["bid"], q["ask"], f"paper:{st.get('status')}", "retire"))
                if abs(left) < 1e-9:
                    con.execute("DELETE FROM positions WHERE strategy=? AND key=?", (name, r["key"]))
                else:
                    con.execute("UPDATE positions SET qty=? WHERE strategy=? AND key=?", (left, name, r["key"]))
        out.append((r["key"], action, qty, st.get("status"), filled))
    return out


if __name__ == "__main__":
    names = sys.argv[1:]
    ib = safe.connect_paper(client_id=95, timeout=40)
    try:
        for n in names:
            res = close_positions(ib, n)
            with F.db() as con:
                left = con.execute("SELECT count(*) FROM positions WHERE strategy=?", (n,)).fetchone()[0]
                if left == 0:
                    con.execute("UPDATE strategies SET active=0 WHERE name=?", (n,))
            src = STRAT / f"{n}.py"
            if left == 0 and src.exists():
                (STRAT / "_retired").mkdir(exist_ok=True)
                shutil.move(str(src), str(STRAT / "_retired" / src.name))
            print(json.dumps({"strategy": n, "orders": res, "positions_left": left}))
    finally:
        ib.disconnect()
