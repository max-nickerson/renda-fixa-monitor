"""Rebuild every research dataset and study in order (after data/method changes)."""
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}
steps = [
    ("lab dataset", [PY, "-W", "ignore", "research/build_lab.py"], {}),
    ("monthly selection study", [PY, "-W", "ignore", "research/run_selection.py", "--rebuild"], {}),
    ("lab (hedged, CVM+press)", [PY, "-W", "ignore", "research/run_lab2.py", "data/history/press_weekly.pkl"],
     {"RET": "ret_hedged"}),
    ("final charts", [PY, "-W", "ignore", "research/plot_lab_final.py"], {}),
]
for name, cmd, extra in steps:
    t = time.time()
    print(f"=== {name} ===", flush=True)
    r = subprocess.run(cmd, cwd=ROOT, env={**env, **extra})
    print(f"=== {name}: exit {r.returncode} ({time.time() - t:.0f}s) ===", flush=True)
    if r.returncode:
        sys.exit(r.returncode)
print("ALL DONE", flush=True)
