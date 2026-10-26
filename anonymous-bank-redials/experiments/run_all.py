"""Run every step in order and write results/summary.txt.

    python run_all.py            # full sizes, about six minutes
    python run_all.py --quick    # small sizes, to check the code runs

R1 and R2 need the raw data (``python fetch.py``); without it they are
skipped and the later steps use the committed table ``data/halfhour.csv.gz``.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import raw_source  # noqa: E402

STEPS = ["r1_data.py", "r2_ids.py", "r3_slopes.py", "r4_simcheck.py",
         "r5_estimate.py", "report.py"]


def main() -> None:
    extra = ["--quick"] if "--quick" in sys.argv else []
    have_raw = raw_source() is not None
    for step in STEPS:
        if step in ("r1_data.py", "r2_ids.py") and not have_raw:
            print(f"== {step} skipped (no raw data)", flush=True)
            continue
        start = time.perf_counter()
        args = [sys.executable, str(HERE / step)]
        if step != "report.py":
            args += extra
        subprocess.run(args, check=True, cwd=HERE)
        print(f"== {step} finished in {time.perf_counter() - start:.0f}s",
              flush=True)


if __name__ == "__main__":
    main()
