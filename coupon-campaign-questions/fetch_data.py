"""Download the Complete Journey tables this analysis needs, pinned to one commit, into .cache/.

The data are the 84.51 degrees "Complete Journey" study (2,469 households, one year of grocery
receipts, 27 coupon campaigns) as packaged by the R package completejourney (CC0). The files are
read from the package's GitHub repository at a fixed commit and checked against SHA-256 digests,
so the inputs cannot drift. They are then converted once to Parquet for fast loading.

    python fetch_data.py            # download (if needed), verify, convert
    python fetch_data.py --check    # verify the cached files only; exit 1 if any is missing or altered
"""
import hashlib
import sys
import urllib.request
from pathlib import Path

COMMIT = "5b5d06192b9856edd04e4d405787af2f2e4a1fef"
BASE = f"https://raw.githubusercontent.com/bradleyboehmke/completejourney/{COMMIT}/data/"
FILES = {
    "campaign_descriptions.rda": "601bd17ea1a6de92cc288f393164813fbaf293b997d44054dcf1f3ebbe8dacee",
    "campaigns.rda": "33c7b3dc1bc722d465f97416fb1327c6a9a640190e464c73e354bce8c001a772",
    "coupon_redemptions.rda": "4b68148175ed19300bb2615d68d1a0300a5f971db3240819ab5d92e5919e1598",
    "coupons.rda": "4d87effce5b8813c0934b581a83292707347577743320ca746878de89cc16f34",
    "transactions.rds": "1fa0700033f1e5d9bb6b09e2be063d8d68474d346e95c50f2833e09d083e0007",
}
CACHE = Path(__file__).resolve().parent / ".cache"


def sha256(path):
    """Hex SHA-256 of a file, read in chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(name, attempts=4):
    """Fetch one file from the pinned commit, retrying a cut transfer a few times."""
    target = CACHE / name
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(BASE + name, timeout=600) as r:
                target.write_bytes(r.read())
            return
        except OSError as e:
            if attempt == attempts:
                raise
            print(f"  retry {attempt} for {name}: {type(e).__name__}", file=sys.stderr)


def convert():
    """Write one Parquet file per table, so the analysis does not parse R files on every run."""
    import pyreadr

    for name in FILES:
        stem = name.split(".")[0]
        out = CACHE / f"{stem}.parquet"
        if out.exists():
            continue
        frame = next(iter(pyreadr.read_r(CACHE / name).values()))
        frame.to_parquet(out, index=False)


def main(argv):
    CACHE.mkdir(exist_ok=True)
    check_only = "--check" in argv
    bad = []
    for name, digest in FILES.items():
        path = CACHE / name
        if not path.exists() or sha256(path) != digest:
            if check_only:
                bad.append(name)
                continue
            print(f"downloading {name}", file=sys.stderr)
            download(name)
            if sha256(path) != digest:
                bad.append(name)
    if bad:
        print("missing or altered: " + ", ".join(bad), file=sys.stderr)
        return 1
    if not check_only:
        convert()
    print(f"all {len(FILES)} files verified against commit {COMMIT[:7]}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
