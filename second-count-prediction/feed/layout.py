"""Which CSVs of a Citi Bike archive make up a month: the rule registered in PLAN.md, and the corrected one.

The yearly archives hold some months twice: once as one file (``201808-citibike-tripdata.csv``) and once as
numbered parts in a month folder (``8_August/201808-citibike-tripdata_1.csv``, ``_2.csv``). Monthly archives
split a month into parts of a million rows (``_1`` ... ``_5``) or into ``-part1`` ... ``-part4``.

- Registered rule (PLAN.md): in the order of the archive, a CSV is left out if its base name was already
  counted, or if it has the same rows in every month as a CSV already counted. The second clause also drops
  legitimate parts of equal size (two parts of exactly 1,000,000 rows), and neither clause recognises a month
  held both as one file and as parts. Both faults were found after the registered run (PLAN.md).
- Corrected rule (post hoc): files with the same SHA-256 are counted once. The remaining files are grouped by
  their stem (the name without ``_N`` or ``-partN``); within a stem, the single file and each folder's set of
  parts are copies of the same data. The first copy is counted; a later copy with exactly the same rows in
  every month is left out; a later copy that differs is counted and flagged.
"""

from __future__ import annotations

import re
from collections import defaultdict

PART = re.compile(r"^(?P<stem>.+?)(?:[_-](?:part)?(?P<n>\d{1,2}))?\.csv$", re.I)


def stem(base: str) -> str:
    m = PART.match(base)
    return (m["stem"] if m else base).lower()


def add(into: dict[str, int], months: dict[str, int]) -> None:
    for m, n in months.items():
        into[m] = into.get(m, 0) + n


def registered(files: list[dict]) -> tuple[dict[str, int], list[str]]:
    months: dict[str, int] = {}
    names, sigs, dropped = set(), set(), []
    for f in files:
        base = f["path"].rsplit("/", 1)[-1]
        sig = (f["rows"], tuple(sorted(f["months"].items())))
        if base in names or (f["rows"] > 0 and sig in sigs):
            dropped.append(f["path"])
            continue
        names.add(base)
        sigs.add(sig)
        add(months, f["months"])
    return months, dropped


def corrected(files: list[dict]) -> tuple[dict[str, int], list[str], list[str]]:
    seen_sha, unique = set(), []
    dropped, flagged = [], []
    for f in files:
        if f["sha256"] in seen_sha:
            dropped.append(f["path"])
            continue
        seen_sha.add(f["sha256"])
        unique.append(f)
    copies: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for f in unique:
        folder, _, base = f["path"].rpartition("/")
        is_part = bool(PART.match(base) and PART.match(base)["n"])
        copies[stem(base)][f"{folder}/{'parts' if is_part else base}"].append(f)
    months: dict[str, int] = {}
    for groups in copies.values():
        kept: dict[str, int] | None = None
        for fs in groups.values():
            m: dict[str, int] = {}
            for f in fs:
                add(m, f["months"])
            if kept is None:
                kept = m
                add(months, m)
            elif m == kept:
                dropped.extend(f["path"] for f in fs)
            else:
                flagged.extend(f["path"] for f in fs)
                add(months, m)
    return months, dropped, flagged
