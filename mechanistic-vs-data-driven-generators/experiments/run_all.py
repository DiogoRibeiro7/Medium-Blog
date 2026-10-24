"""Run every experiment in order, then write results/summary.txt.

    python run_all.py            # full sizes, about half an hour
    python run_all.py --quick    # tiny sizes, to check the code runs
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
STEPS = ["e1_fidelity.py", "e2_surge.py", "e3_intervention.py",
         "e4_error_budget.py", "e5_calibration.py", "e6_privacy_cost.py",
         "e7_decision.py", "report.py"]


def main() -> None:
    extra = ["--quick"] if "--quick" in sys.argv else []
    for step in STEPS:
        start = time.perf_counter()
        args = [sys.executable, str(HERE / step)]
        if step != "report.py":
            args += extra
        subprocess.run(args, check=True, cwd=HERE)
        print(f"== {step} finished in {time.perf_counter() - start:.0f}s",
              flush=True)


if __name__ == "__main__":
    main()
