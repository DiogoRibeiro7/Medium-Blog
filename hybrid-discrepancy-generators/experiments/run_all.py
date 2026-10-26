"""Run every experiment in order, then write results/summary.txt.

    python run_all.py            # full sizes, about an hour on two cores
    python run_all.py --quick    # tiny sizes, to check the code runs
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
STEPS = ["h1_fit.py", "h2_surge.py", "h3_redials.py", "h4_confounding.py",
         "h5_intervention.py", "h6_decision.py", "report.py"]


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
