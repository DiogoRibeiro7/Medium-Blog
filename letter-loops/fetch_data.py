"""Get the two inputs at pinned versions and check them byte for byte.

1. Word frequencies: wordfreq 3.1.1 (Robyn Speer), installed from PyPI by requirements.txt. Its word
   lists ship inside the package, so nothing is downloaded; this script checks the SHA-256 of every
   list used here against digests.txt.
2. Alphabets: the standard ("exemplar") letters of each language from the Unicode Common Locale Data
   Repository, CLDR 49, as published on npm in cldr-misc-full 49.0.0. The tarball is downloaded,
   checked against digests.txt, and the characters.json of each locale used is written to data/cldr/.

    python fetch_data.py            # download what is missing, then verify everything
    python fetch_data.py --check    # verify only; exit 1 if a file is missing or altered
"""

import hashlib
import io
import json
import sys
import tarfile
import urllib.request
from pathlib import Path

import letters_core as L

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
CLDR_TARBALL = "cldr-misc-full-49.0.0.tgz"
CLDR_URL = f"https://registry.npmjs.org/cldr-misc-full/-/{CLDR_TARBALL}"


def sha256_bytes(b):
    """Hex SHA-256 of a byte string."""
    return hashlib.sha256(b).hexdigest()


def pinned():
    """File name -> pinned digest, from digests.txt."""
    rows = (line.split() for line in (HERE / "digests.txt").read_text().splitlines() if line.strip())
    return dict(rows)


def check_wordfreq(digests):
    """Check the installed wordfreq version and the digest of every word list used."""
    bad = []
    if L.wordfreq_version() != L.WORDFREQ_VERSION:
        bad.append(f"wordfreq {L.wordfreq_version()} installed, {L.WORDFREQ_VERSION} pinned")
    for code in L.LANGUAGES:
        path = L.wordlist_path(code)
        key = f"wordfreq/{path.name}"
        if digests.get(key) != sha256_bytes(path.read_bytes()):
            bad.append(f"digest mismatch {key}")
    return bad


def fetch_cldr(digests, check_only):
    """Download the CLDR tarball if needed, verify it and extract one characters.json per locale."""
    bad = []
    target = DATA / CLDR_TARBALL
    if not target.exists():
        if check_only:
            return [f"missing {CLDR_TARBALL}"]
        DATA.mkdir(exist_ok=True)
        with urllib.request.urlopen(CLDR_URL, timeout=120) as r:
            target.write_bytes(r.read())
    blob = target.read_bytes()
    if sha256_bytes(blob) != digests[CLDR_TARBALL]:
        return [f"digest mismatch {CLDR_TARBALL}"]
    out = DATA / "cldr"
    out.mkdir(parents=True, exist_ok=True)
    locales = {L.LANGUAGES[c][1] for c in L.LANGUAGES}
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tar:
        for loc in sorted(locales):
            member = tar.extractfile(f"package/main/{loc}/characters.json")
            if member is None:
                bad.append(f"no characters.json for {loc} in {CLDR_TARBALL}")
                continue
            payload = json.loads(member.read().decode("utf-8"))
            (out / f"{loc}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    return bad


def main(check_only=False):
    digests = pinned()
    bad = check_wordfreq(digests) + fetch_cldr(digests, check_only)
    if bad:
        print("\n".join(bad), file=sys.stderr)
        sys.exit(1)
    print(f"wordfreq {L.WORDFREQ_VERSION}: {len(L.LANGUAGES)} word lists verified")
    print(f"CLDR: {CLDR_TARBALL} verified, {len({v[1] for v in L.LANGUAGES.values()})} alphabets in {DATA / 'cldr'}")


if __name__ == "__main__":
    main(check_only="--check" in sys.argv)
