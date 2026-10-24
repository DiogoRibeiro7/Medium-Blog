"""Rerun everything except the neural-network training (see deep.py).

    python run_all.py            # about 20 minutes on two cores
    python run_all.py --figures  # redraw the figures and summary only
"""

from __future__ import annotations

import sys
import time

import coverage_adult
import coverage_mechanism
import evaluate_axes
import figures
import summarize


def main() -> None:
    steps = []
    if "--figures" not in sys.argv:
        steps += [
            ("fidelity, utility and privacy", evaluate_axes.main),
            ("coverage under a known mechanism", coverage_mechanism.main),
            ("coverage with Adult as the population", coverage_adult.main),
        ]
    steps += [("summary", summarize.main), ("figures", figures.main)]
    for label, step in steps:
        start = time.time()
        step()
        print(f"{label}: {time.time() - start:.0f} s", flush=True)


if __name__ == "__main__":
    main()
