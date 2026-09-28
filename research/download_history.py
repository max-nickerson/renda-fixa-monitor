"""Download / refresh the research datasets (cached under data/history/). Safe to re-run."""
import sys as _sys
from pathlib import Path as _P
_sys.path.insert(0, str(_P(__file__).resolve().parent.parent))  # run from anywhere

import logging
import sys
import time
from datetime import date, timedelta

from rfmonitor import history as h

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
START = date(int(sys.argv[1]) if len(sys.argv) > 1 else 2021, 1, 1)

# SND trades: one request per month.
for m in h._month_starts(START, date.today()):
    try:
        logging.info("SND %s: %d rows", m, len(h.snd_trades_month(m)))
    except Exception as e:
        logging.warning("SND %s failed: %s", m, e)

# B3 TaxaSwap curves: one file per business day.
days = [START + timedelta(days=i) for i in range((date.today() - START).days + 1)]
days = [d for d in days if d.weekday() < 5]
missing = [d for d in days if not (h.HIST / "b3" / f"{d:%Y%m%d}.csv").exists()]
logging.info("B3 curves: %d cached, %d to fetch", len(days) - len(missing), len(missing))
for i, d in enumerate(missing, 1):  # sequential + pause: B3 answers 429 to bursts
    h.b3_curves_day(d)
    time.sleep(0.8)
    if i % 100 == 0:
        logging.info("B3 curves %d/%d", i, len(missing))

for n in ("IDADI", "IDAIPCA", "IDAGERAL", "IMAB", "IDALIQDI", "IDALIQIPCA"):
    logging.info("%s %s", n, h.ida(n).shape)
logging.info("done")
