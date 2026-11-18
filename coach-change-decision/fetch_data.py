"""Download the source tables, pinned and checked, into .cache/.

Two public sources:

* transfermarkt-datasets (CC0): every league match with both clubs' managers, and player market
  valuations. The project stopped updating on 6 July 2026; the files below are that final state.
* datasets/football-datasets (PDDL), a mirror of football-data.co.uk: shots and shots on target per
  match, read from one fixed commit.

Every file is checked against a SHA-256 digest, so the inputs cannot drift. build_panel.py turns them
into the compact tables in data/, which are committed; the analysis reads only data/.

    python fetch_data.py            # download what is missing, then verify everything
    python fetch_data.py --check    # verify only; exit 1 if a file is missing or altered
"""

import hashlib
import sys
import urllib.request
from pathlib import Path

CACHE = Path(__file__).resolve().parent / ".cache"

TM = "https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/data/"
TM_FILES = {
    "games.csv.gz": "142561989017d379bcf9e72bad0b87e0996bae0e7b710ebb3def9255f464c741",
    "player_valuations.csv.gz": "608e32e5ad2231074f3505c469ad727be66d7e6efda6af918afd85379439631d",
}

FD_COMMIT = "436a51f15258e7d0c6042c231227202b70a5a0ae"
FD = f"https://raw.githubusercontent.com/datasets/football-datasets/{FD_COMMIT}/datasets/"
FD_LEAGUES = {
    "GB1": "premier-league",
    "ES1": "la-liga",
    "IT1": "serie-a",
    "L1": "bundesliga",
    "FR1": "ligue-1",
}
SEASONS = range(2012, 2026)  # 2012/13 to 2025/26


def fd_name(season):
    """football-datasets file name for the season that starts in `season`: 2012 -> season-1213.csv."""
    return f"season-{season % 100:02d}{(season + 1) % 100:02d}.csv"


def fd_files():
    """(cache name, URL) for every league-season shot file."""
    for code, folder in FD_LEAGUES.items():
        for s in SEASONS:
            yield f"fd_{code}_{s}.csv", FD + f"{folder}/{fd_name(s)}"


def sha256(path):
    """Hex SHA-256 of a file, read in chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url, target, attempts=4):
    """Fetch one file, retrying a cut transfer a few times."""
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(url, timeout=600) as r:
                target.write_bytes(r.read())
            return
        except OSError as e:
            if attempt == attempts:
                raise
            print(
                f"  retry {attempt} for {target.name}: {type(e).__name__}",
                file=sys.stderr,
            )


def expected():
    """Every cached file with its pinned digest (shot files are pinned in fd_digests.txt)."""
    digests = dict(TM_FILES)
    pinned = Path(__file__).resolve().parent / "fd_digests.txt"
    for line in pinned.read_text().splitlines():
        name, digest = line.split()
        digests[name] = digest
    return digests


def main(check_only=False):
    CACHE.mkdir(exist_ok=True)
    urls = {name: TM + name for name in TM_FILES} | dict(fd_files())
    bad = []
    for name, digest in expected().items():
        target = CACHE / name
        if not target.exists():
            if check_only:
                bad.append(f"missing {name}")
                continue
            print(f"downloading {name}", file=sys.stderr)
            download(urls[name], target)
        if sha256(target) != digest:
            bad.append(f"digest mismatch {name}")
    if bad:
        print("\n".join(bad), file=sys.stderr)
        sys.exit(1)
    print(f"{len(urls)} files verified in {CACHE}")


if __name__ == "__main__":
    main(check_only="--check" in sys.argv)
