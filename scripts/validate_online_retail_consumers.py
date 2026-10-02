#!/usr/bin/env python3
"""Validate the pinned Online Retail II consumer migration."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data_sources" / "online-retail-ii.json"

OLD_DATA_PATHS = [
    "customer_life_value/Year 2009-2010.csv",
    "customer_life_value/Year 2010-2011.csv",
    "customer_life_value/data.csv",
    "customer_life_value/online_retail_II.xlsx",
    "customer_segmentation/customer_segmentation.csv",
    "R_code/customer_segmentation/data.csv",
]

R_CONSUMERS = [
    "R_code/customer_segmentation/customer_purchase_patterns.R",
    "R_code/customer_segmentation/customer_segmentation.R",
    "R_code/customer_segmentation/data_analysis.R",
    "R_code/customer_segmentation/product_insights.R",
    "R_code/customer_segmentation/sales.R",
    "R_code/customer_segmentation/sales_perfomance.R",
]

NOTEBOOK_CONSUMERS = [
    "customer_life_value/Untitled.ipynb",
    "customer_segmentation/customer segmentation.ipynb",
]


def fail(message: str) -> None:
    raise AssertionError(message)


def main() -> int:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

    commit = manifest["commit"]
    checksum = manifest["sha256"]
    if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        fail("consumer manifest must pin a full 40-character commit SHA")
    if re.fullmatch(r"[0-9a-f]{64}", checksum) is None:
        fail("consumer manifest must pin a SHA-256 checksum")

    fetcher = (ROOT / "scripts" / "fetch_online_retail_ii.py").read_text(encoding="utf-8")
    for expected in (commit, manifest["path"], checksum):
        if expected not in fetcher:
            fail(f"fetcher is not pinned to manifest value: {expected}")

    declared = sorted(manifest["consumers"])
    expected_consumers = sorted(R_CONSUMERS + NOTEBOOK_CONSUMERS)
    if declared != expected_consumers:
        fail("consumer manifest does not list the complete audited consumer set")

    for relative in R_CONSUMERS:
        path = ROOT / relative
        content = path.read_text(encoding="utf-8")
        if 'source("load_online_retail_ii.R")' not in content:
            fail(f"{relative} does not use the canonical R loader")
        if "read.csv(" in content:
            fail(f"{relative} still contains a direct CSV read")

    for relative in NOTEBOOK_CONSUMERS:
        path = ROOT / relative
        notebook = json.loads(path.read_text(encoding="utf-8"))
        source_text = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook.get("cells", [])
            if cell.get("cell_type") == "code"
        )
        if "fetch_online_retail_ii.py" not in source_text:
            fail(f"{relative} does not call the pinned fetcher")

    for relative in OLD_DATA_PATHS:
        if (ROOT / relative).exists():
            fail(f"obsolete retail data copy still exists: {relative}")

    print("Online Retail II consumer migration is structurally valid.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AssertionError, KeyError, json.JSONDecodeError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
