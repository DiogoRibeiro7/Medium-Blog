"""Download the four national YRBS waves from CDC and check them byte for byte.

For each of 2017, 2019, 2021 and 2023 the script fetches the ASCII data file and the SAS input program
that gives its column layout, from https://www.cdc.gov/yrbs/files/, and compares every file with the
SHA-256 recorded in digests.txt. Nothing is committed: the files go to data/ (ignored by git).

    python fetch_data.py            # download what is missing, then verify everything
    python fetch_data.py --check    # verify only; exit 1 if a file is missing or altered
    python fetch_data.py --pin      # record the digests of the files now in data/ (first download only)
"""

import hashlib
import sys
import urllib.request

import yrbs_core as Y

DIGESTS = Y.HERE / "digests.txt"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pinned():
    if not DIGESTS.exists():
        return {}
    rows = (line.split() for line in DIGESTS.read_text().splitlines() if line.strip() and not line.startswith("#"))
    return {name: digest for name, digest in rows}


def targets():
    for wave, (dat, sas) in Y.FILES.items():
        for rel in (dat, sas):
            yield wave, rel, Y.DATA / rel.split("/")[1]


def download(url, dest):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (research reproduction)"})
    with urllib.request.urlopen(req, timeout=300) as r:
        dest.write_bytes(r.read())


def main(argv):
    check_only = "--check" in argv
    pin = "--pin" in argv
    Y.DATA.mkdir(exist_ok=True)
    digests = pinned()
    bad = []
    for _wave, rel, dest in targets():
        if not dest.exists():
            if check_only:
                bad.append(f"missing {dest.name}")
                continue
            download(f"{Y.BASE}/{rel}", dest)
        got = sha256(dest)
        if pin:
            digests[dest.name] = got
        elif digests.get(dest.name) != got:
            bad.append(f"digest mismatch {dest.name}: {got}")
    if pin:
        lines = ["# SHA-256 of the CDC files used, as downloaded from https://www.cdc.gov/yrbs/files/"]
        lines += [f"{name} {digests[name]}" for name in sorted(digests)]
        DIGESTS.write_text("\n".join(lines) + "\n")
        print(f"pinned {len(digests)} files in {DIGESTS.name}")
        return 0
    for b in bad:
        print(b, file=sys.stderr)
    if not bad:
        print(f"all {sum(1 for _ in targets())} files match digests.txt")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
