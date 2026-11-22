"""Download the OECD satisfaction-with-public-services table, pinned and checked, into .cache/.

Source: OECD, Government at a Glance 2025, dataflow OECD.GOV.GIP,DSD_GOV_INT@DF_GOV_SPS_2025
("Satisfaction with public services"), measures CS_HC (healthcare system), CS_ES (education system)
and TRUST_S_AS (administrative services), in percent of respondents satisfied. The 2023 values come
from the OECD Survey on Drivers of Trust in Public Institutions; 2021 values exist for some countries.

The response is checked against a SHA-256 digest, so the input cannot drift; a revision by the OECD
makes this script fail rather than change the results. build_oecd() turns it into a compact table.

    python fetch_data.py            # download if missing, then verify
    python fetch_data.py --check    # verify only; exit 1 if the file is missing or altered
"""

import hashlib
import sys
import urllib.request
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
CACHE = HERE / ".cache"

OECD_URL = (
    "https://sdmx.oecd.org/public/rest/data/OECD.GOV.GIP,DSD_GOV_INT@DF_GOV_SPS_2025,"
    "/A..CS_HC+CS_ES+TRUST_S_AS.....?format=csvfile"
)
OECD_FILE = "oecd_gov_sps_2025.csv"
OECD_SHA256 = "99f57c65c66a5ac474c62fc9876f260c6ed9ec8b2bd3a7efbde8d7578b3673f7"
OECD_YEAR = 2023
# The OECD endpoint refuses urllib's default agent string.
USER_AGENT = "Mozilla/5.0 (compatible; Medium-Blog ranking-extremes reproduction)"
MEASURES = {"CS_HC": "healthcare", "CS_ES": "education", "TRUST_S_AS": "administration"}


def sha256(path):
    """Hex SHA-256 of a file."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def download(url, target, attempts=4):
    """Fetch one file, retrying a cut transfer a few times."""
    for attempt in range(1, attempts + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "text/csv"})
            with urllib.request.urlopen(req, timeout=300) as r:
                target.write_bytes(r.read())
            return
        except OSError as e:
            if attempt == attempts:
                raise
            print(f"  retry {attempt}: {type(e).__name__}", file=sys.stderr)


def build_oecd(path=None):
    """Percent satisfied in 2023, one row per country, with the three measures and their mean."""
    g = pd.read_csv(path or CACHE / OECD_FILE)
    g = g[(g["TIME_PERIOD"] == OECD_YEAR) & (g["MEASURE"].isin(MEASURES)) & (g["REF_AREA"] != "OECD_REP")]
    wide = g.pivot(index="REF_AREA", columns="MEASURE", values="OBS_VALUE").rename(columns=MEASURES)
    wide = wide.dropna()
    wide["composite"] = wide[list(MEASURES.values())].mean(axis=1)
    wide.index.name = "iso3"
    return wide.reset_index()


def main(check_only=False):
    CACHE.mkdir(exist_ok=True)
    target = CACHE / OECD_FILE
    if not target.exists() and not check_only:
        print(f"downloading {OECD_FILE}")
        download(OECD_URL, target)
    if not target.exists():
        print(f"{OECD_FILE}: missing", file=sys.stderr)
        sys.exit(1)
    if sha256(target) != OECD_SHA256:
        print(f"{OECD_FILE}: digest mismatch (the OECD may have revised the table)", file=sys.stderr)
        sys.exit(1)
    print(f"{OECD_FILE} present and verified")


if __name__ == "__main__":
    main(check_only="--check" in sys.argv[1:])
