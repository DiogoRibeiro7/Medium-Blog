"""Bundle the per-month profile files into two gzipped JSON-lines files, and read them back.

    python -m feed.bundle profiles      # profiles/real/*.json, profiles/fault/<f>/*.json -> *.jsonl.gz
"""

from __future__ import annotations

import gzip
import json
import shutil
import sys
from pathlib import Path


def write(root: Path) -> None:
    with gzip.GzipFile(root / "real.jsonl.gz", "wb", mtime=0) as f:
        for p in sorted((root / "real").glob("*.json")):
            f.write((json.dumps({"month": p.stem, "profile": json.loads(p.read_text())}) + "\n").encode())
    with gzip.GzipFile(root / "fault.jsonl.gz", "wb", mtime=0) as f:
        for p in sorted((root / "fault").glob("*/*.json")):
            rec = {"fault": p.parent.name, "month": p.stem, "profile": json.loads(p.read_text())}
            f.write((json.dumps(rec) + "\n").encode())
    shutil.rmtree(root / "real")
    shutil.rmtree(root / "fault", ignore_errors=True)


def read(root: Path) -> tuple[dict[str, dict], dict[str, dict[str, dict]]]:
    real: dict[str, dict] = {}
    faults: dict[str, dict[str, dict]] = {}
    with gzip.open(root / "real.jsonl.gz", "rt") as f:
        for line in f:
            r = json.loads(line)
            real[r["month"]] = r["profile"]
    if (root / "fault.jsonl.gz").exists():
        with gzip.open(root / "fault.jsonl.gz", "rt") as f:
            for line in f:
                r = json.loads(line)
                faults.setdefault(r["fault"], {})[r["month"]] = r["profile"]
    return real, faults


if __name__ == "__main__":
    write(Path(sys.argv[1]))
