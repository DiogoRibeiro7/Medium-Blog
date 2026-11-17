"""Download the public raw inputs into data/raw/.

Sources (all public):
  * MIT Election Data and Science Lab, U.S. House 1976-2022 (CC0), via the
    TidyTuesday mirror of 7 Nov 2023 (house.csv, MEDSL version 20230706).
  * ICPSR 6311 / Gary King's JudgeIt package, House elections 1896-1992
    (two-party Democratic share by district); used here for 1972 and 1974 only.
  * MIT Election Data and Science Lab, 2024 precinct-level returns (CC0),
    GitHub repository MEDSL/2024-elections-official, aggregated here to
    congressional districts.

Run once: python fetch.py
"""
from __future__ import annotations

import subprocess
import sys
import urllib.request
from pathlib import Path

RAW = Path(__file__).resolve().parent / "data" / "raw"

MEDSL_1976_2022 = (
    "https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/"
    "data/2023/2023-11-07/house.csv"
)
JUDGEIT_REPO = "https://github.com/cran/JudgeIt"
MEDSL_2024_REPO = "https://github.com/MEDSL/2024-elections-official"


def _run(cmd: list[str], cwd: Path | None = None) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=cwd)


def main() -> int:
    RAW.mkdir(parents=True, exist_ok=True)
    target = RAW / "medsl_house_1976_2022.csv"
    if not target.exists():
        print("downloading", MEDSL_1976_2022)
        urllib.request.urlretrieve(MEDSL_1976_2022, target)

    judgeit = RAW / "JudgeIt"
    if not judgeit.exists():
        _run(["git", "clone", "--depth", "1", JUDGEIT_REPO, str(judgeit)])

    medsl24 = RAW / "medsl2024"
    if not medsl24.exists():
        _run(["git", "clone", "--depth", "1", MEDSL_2024_REPO, str(medsl24)])
    return 0


if __name__ == "__main__":
    sys.exit(main())
