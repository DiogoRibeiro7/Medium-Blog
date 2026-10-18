"""Command line: python -m prescribing {fetch|analyse|all} [source ...]

fetch    download the public data and write the tidy extracts to data/
analyse  run the analyses on data/ and write results/ and figures/
all      both
Sources for fetch: nhanes, trials, faers, medicare (default: all four).
A failing source is reported and skipped, so the others still run.
"""

from __future__ import annotations

import sys
import traceback

from . import faers, figures, medicare, nhanes, report, trials
from .paths import ensure

SOURCES = {"nhanes": nhanes.build, "trials": trials.build, "faers": faers.build, "medicare": medicare.build}


def fetch(names: list[str]) -> int:
    failed = 0
    for n in names or list(SOURCES):
        print(f"--- fetch {n}")
        try:
            SOURCES[n]()
        except Exception:
            failed += 1
            traceback.print_exc()
            print(f"--- fetch {n} FAILED")
    return failed


def analyse() -> None:
    out = report.compute()
    print(report.write(out))
    figures.draw(out)


def main(argv: list[str]) -> int:
    ensure()
    if not argv or argv[0] not in ("fetch", "analyse", "all"):
        print(__doc__)
        return 2
    cmd, rest = argv[0], argv[1:]
    failed = 0
    if cmd in ("fetch", "all"):
        failed = fetch(rest)
    if cmd in ("analyse", "all"):
        analyse()
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
