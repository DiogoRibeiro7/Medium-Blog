"""Read a Eurostat JSON-stat 2.0 response into a long pandas DataFrame.

Eurostat stores observations in a flat dictionary keyed by the row-major position over the
dimensions listed in "id", with sizes in "size". Missing observations are absent from "value".
Status flags ("p" provisional, "e" estimated, "b" break, ...) are kept in a column.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd


def to_number(value):
    """Numbers stay numbers; time-use durations written as "hh:mm" become minutes."""
    if isinstance(value, str) and ":" in value:
        hours, minutes = value.split(":")
        return 60.0 * int(hours) + int(minutes)
    return float(value)


def read_jsonstat(path):
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    dims = d["id"]
    sizes = d["size"]
    codes = []
    for k in dims:
        index = d["dimension"][k]["category"]["index"]
        order = sorted(index, key=index.get)
        codes.append(order)
    strides = np.cumprod([1] + sizes[::-1][:-1])[::-1]
    status = d.get("status", {})
    rows = []
    for key, value in d["value"].items():
        pos = int(key)
        row = {}
        for k, c, s in zip(dims, codes, strides, strict=True):
            row[k] = c[pos // s]
            pos %= s
        row["value"] = to_number(value)
        row["flag"] = status.get(key, "")
        rows.append(row)
    df = pd.DataFrame(rows, columns=[*dims, "value", "flag"])
    if "time" in df:
        df["time"] = df["time"].astype(int)
    return df.drop(columns=[c for c in ("freq",) if c in df])
