"""Download the six OECD and Eurostat tables, pinned and checked, into .cache/.

Every response is checked against a SHA-256 digest taken on 24 November 2026, so the inputs cannot
drift: a revision by the OECD or Eurostat makes this script fail rather than change the results.
Nothing is committed; the files are downloaded again on every run.

    python fetch_data.py            # download what is missing, then verify all
    python fetch_data.py --check    # verify only; exit 1 if a file is missing or altered
"""

import hashlib
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
CACHE = HERE / ".cache"
OECD = "https://sdmx.oecd.org/public/rest/data/"
EUROSTAT = "https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/data/"
# The OECD endpoint refuses urllib's default agent string.
USER_AGENT = "Mozilla/5.0 (compatible; Medium-Blog uk-innovation-ownership reproduction)"

SOURCES = {
    # Foreign ownership of domestic inventions: patent applications with at least one inventor in the
    # country (W), those owned by applicants abroad (WXD) and by US applicants (USA); PCT and EPO,
    # priority date, all reporting countries.
    "patents_foreign.csv": (
        OECD + "OECD.STI.PIE,DSD_PATENTS@DF_PATENTS_FOREIGN,1.0/9P50_1+6F0.A.AP.PATN.PRIORITY..W+WXD+USA....",
        "18bfd5d3990bae6ae861b3d504bea33b58b66deea39d4b162de70f6b4af90c66",
    ),
    # The same table with every owner country, for six inventor countries (PCT).
    "patents_owners.csv": (
        OECD + "OECD.STI.PIE,DSD_PATENTS@DF_PATENTS_FOREIGN,1.0/9P50_1.A.AP.PATN.PRIORITY.GBR+DEU+FRA+ITA+NLD+SWE.....",
        "68739f108ef6d8a4d8677c3d8a666be68e5daac3da6a0311939855095cd45d1a",
    ),
    # Venture capital investment by stage (market statistics: by location of the company).
    "venture_capital.csv": (
        OECD + "OECD.SDD.TPS,DSD_VC@DF_VC_INV,1.0/all",
        "7e9ec1ba4dd91db54b72b2ba752c1b7cd8efc268587b29c7fea066c2be3810d9",
    ),
    # Medium- and high-growth employer enterprises (10% a year in employment over three years), as a
    # percentage of enterprises with 10 or more employees: ICT, programming, information services, R&D.
    "high_growth.csv": (
        OECD + "OECD.SDD.TPS,DSD_SDBSBD_ISIC4@DF_BD_EMP_HG,1.0/A..ENT.J+J62+J63+M72.S_GE10._T.ENT_EMP.HGE10.PT_ENT10",
        "58e557f648f4176ed46a720f1b1db4e271e75942edece4c414ae8525cc160ec4",
    ),
    # Asset-backed pensions: investment as % of GDP (1000) and private equity funds as % of assets (1286).
    "pensions.csv": (
        OECD + "OECD.DAF.CM,DSD_FP@DF_FPS,1.0/.A.1000+1286.PT_B1GQ+PT_AS._T._T._T+PF",
        "0dc315320409fa873c18d7b097e4ee1b305650274d930dab083c3f17763dc01d",
    ),
    # Inward FATS: persons employed in enterprises controlled from abroad (WRL_X_REP), from the US, and
    # in all enterprises (WORLD); ICT and R&D industries, 2008-2020, EU countries, the UK and others.
    "fats_ict.csv": (
        EUROSTAT + "fats_g1a_08/A.J+J62+J63+M72.V16110.WORLD+WRL_X_REP+US.",
        "26f8940d01e881a02b528ff230e1186421ebe7c1a57366ffcd9ab96784d66926",
    ),
}


def url_for(name):
    """Full request URL for one source (OECD csvfile or Eurostat SDMX-CSV)."""
    base, _ = SOURCES[name]
    return base + ("?format=SDMX-CSV" if base.startswith(EUROSTAT) else "?format=csvfile")


def sha256(path):
    """Hex SHA-256 of a file."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def download(url, target, attempts=4):
    """Fetch one file, retrying a cut transfer a few times."""
    for attempt in range(1, attempts + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=600) as r:
                target.write_bytes(r.read())
            return
        except OSError as e:
            if attempt == attempts:
                raise
            print(f"  retry {attempt}: {type(e).__name__}", file=sys.stderr)


def main(check_only=False):
    CACHE.mkdir(exist_ok=True)
    failed = False
    for name, (_, digest) in SOURCES.items():
        target = CACHE / name
        if not target.exists() and not check_only:
            print(f"downloading {name}")
            download(url_for(name), target)
        if not target.exists():
            print(f"{name}: missing", file=sys.stderr)
            failed = True
        elif sha256(target) != digest:
            print(f"{name}: digest mismatch (the source may have been revised)", file=sys.stderr)
            failed = True
        else:
            print(f"{name} present and verified")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main(check_only="--check" in sys.argv[1:])
