"""Record the column schema of NYC Open Data's 311 Call Center Inquiry dataset (wewp-mm3p), no data values.

python probe.py --out sources/schema.json
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

URL = "https://data.cityofnewyork.us/api/views/wewp-mm3p.json"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=Path("sources/schema.json"))
    a = ap.parse_args()
    with urllib.request.urlopen(URL, timeout=120) as r:
        meta = json.load(r)
    keep = {
        "id": meta.get("id"),
        "name": meta.get("name"),
        "description": meta.get("description"),
        "rowsUpdatedAt": meta.get("rowsUpdatedAt"),
        "attribution": meta.get("attribution"),
        "license": (meta.get("license") or {}).get("name"),
        "columns": [
            {k: c.get(k) for k in ("fieldName", "name", "dataTypeName", "description")} for c in meta.get("columns", [])
        ],
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(keep, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(keep, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
