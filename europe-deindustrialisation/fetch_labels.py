"""Download the names of HS 6-digit products (Comext DS-045409 category labels).

    python fetch_labels.py --out data

One request for EU-27 imports from outside the EU in 2024, all products; the JSON-stat response
carries the label of every product code. Writes ``data/hs6_labels.csv``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from fetch_comext import get
from fetch_hs6 import url


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    args = ap.parse_args(argv)
    out = Path(args.out)
    for reporter in ("EU27_2020", "EU"):
        status, body = get(url([("reporter", reporter), ("partner", "EXT_EU27_2020"), ("flow", "1")], (2024,)), timeout=300)
        if status == 200:
            break
    if status != 200:
        print(f"labels: HTTP {status} {body[:300]!r}")
        return 1
    labels = json.loads(body)["dimension"]["product"]["category"]["label"]
    d = pd.DataFrame(sorted(labels.items()), columns=["product", "label"])
    d = d[d["product"].str.fullmatch(r"\d{2}|\d{4}|\d{6}")]
    d.to_csv(out / "hs6_labels.csv", index=False)
    with open(out / "fetch_log.txt", "a", encoding="utf-8") as f:
        f.write(f"labels\tROWS\t{len(d)}\n")
    print(f"labels: {len(d)} product names")
    return 0


if __name__ == "__main__":
    sys.exit(main())
