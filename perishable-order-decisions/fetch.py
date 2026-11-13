"""Download FreshRetailNet-50K at a recorded revision and check the files (PLAN.md, "Data").

python fetch.py --data data --sources sources

The first run records the dataset revision in sources/REVISION and the SHA-256 of each file in sources/FILES; later
runs download that revision and stop if a file does not match.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

from feed.data import FILES, REPO


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--sources", type=Path, default=Path("sources"))
    a = ap.parse_args()
    a.data.mkdir(parents=True, exist_ok=True)
    a.sources.mkdir(parents=True, exist_ok=True)
    rev_file = a.sources / "REVISION"
    if rev_file.exists():
        revision = rev_file.read_text(encoding="utf-8").strip()
    else:
        revision = HfApi().dataset_info(REPO).sha
        rev_file.write_text(revision + "\n", encoding="utf-8")
    print(f"{REPO} at revision {revision}")
    files_file = a.sources / "FILES"
    known = {}
    if files_file.exists():
        for line in files_file.read_text(encoding="utf-8").splitlines():
            name, size, digest = line.split()
            known[name] = (int(size), digest)
    lines = []
    for split in FILES:
        target = a.data / f"{split}.parquet"
        if not target.exists():
            path = hf_hub_download(REPO, f"data/{split}.parquet", repo_type="dataset", revision=revision)
            shutil.copyfile(path, target)
        size, digest = target.stat().st_size, sha256(target)
        if split in known and known[split] != (size, digest):
            raise SystemExit(f"{split}.parquet does not match sources/FILES: {size} {digest}")
        lines.append(f"{split} {size} {digest}")
        print(lines[-1])
    files_file.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
