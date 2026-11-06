"""Download EU production by product for 2025 (Prodcom) and the CN-Prodcom correspondence.

    python fetch_prodcom.py --out data

1. Eurostat Prodcom dataset DS-059367 (CPA 2.2, reference year 2025) through the Comext API:
   sold production, exports and imports in euros for the EU-27, every Prodcom code. For the EU
   total, exports and imports are trade with countries outside the EU.
2. The correspondence between Combined Nomenclature (CN 2025) codes and Prodcom 2025 codes,
   published as a spreadsheet by Statistics Netherlands (CBS).

Writes ``data/prodcom_2025.csv``, ``data/cn_prodcom_2025.csv`` and ``data/fetch_log.txt``.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

API = "https://ec.europa.eu/eurostat/api/comext/dissemination/statistics/1.0/data/{ds}"
CBS = "https://www.cbs.nl/-/media/_excel/2026/04/prodcom_cn_correspondance_table_2025_en.xlsx"
AGENT = "dependency-screen-design/1.0"


def get(url: str, tries: int = 3, timeout: int = 300) -> tuple[int, bytes]:
    last: Exception | None = None
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.status, r.read()
        except urllib.error.HTTPError as e:
            if 400 <= e.code < 500:
                return e.code, e.read()
            last = e
        except (urllib.error.URLError, TimeoutError) as e:
            last = e
        time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"{url}: {last}")


def jsonstat_to_long(doc: dict) -> pd.DataFrame:
    """Flatten a JSON-stat 2.0 dataset into one row per cell."""
    ids, sizes = doc["id"], doc["size"]
    cats = []
    for d in ids:
        index = doc["dimension"][d]["category"]["index"]
        cats.append(index if isinstance(index, list) else [c for c, _ in sorted(index.items(), key=lambda kv: kv[1])])
    values = doc.get("value", {})
    if isinstance(values, list):
        values = {str(i): v for i, v in enumerate(values) if v is not None}
    strides = [1] * len(sizes)
    for i in range(len(sizes) - 2, -1, -1):
        strides[i] = strides[i + 1] * sizes[i + 1]
    rows = []
    for key, v in values.items():
        k = int(key)
        row = {d: codes[(k // s) % n] for d, codes, s, n in zip(ids, cats, strides, sizes)}
        row["value"] = v
        rows.append(row)
    return pd.DataFrame(rows)


def prodcom(log: list[str]) -> pd.DataFrame | None:
    attempts = [
        ("DS-059367", [("reporter", "EU27_2020")]),
        ("DS-059367", [("decl", "EU27_2020")]),
        ("DS-059367", []),
    ]
    for ds, extra in attempts:
        params = [("format", "JSON"), ("lang", "en"), ("freq", "A"), ("time", "2025")] + extra + [("indicators", i) for i in ("PRODVAL", "EXPVAL", "IMPVAL")]
        url = API.format(ds=ds) + "?" + urllib.parse.urlencode(params)
        try:
            status, body = get(url)
        except RuntimeError as e:
            log.append(f"prodcom\tERROR\t{e}")
            continue
        log.append(f"prodcom\t{status}\t{hashlib.sha256(body).hexdigest()}\t{len(body)} bytes\t{url}")
        if status != 200:
            log.append(f"prodcom\tBODY\t{body[:400]!r}")
            continue
        try:
            doc = json.loads(body)
        except json.JSONDecodeError:
            log.append(f"prodcom\tNOT JSON\t{body[:300]!r}")
            continue
        log.append(f"prodcom\tDIMENSIONS\t{doc.get('id')}\t{doc.get('size')}")
        if not doc.get("value"):
            log.append(f"prodcom\tEMPTY\t{json.dumps(doc)[:400]}")
            continue
        d = jsonstat_to_long(doc)
        labels = doc["dimension"].get("product", {}).get("category", {}).get("label", {})
        if labels:
            d["label"] = d["product"].map(labels)
        return d
    return None


def correspondence(log: list[str]) -> pd.DataFrame | None:
    status, body = get(CBS)
    log.append(f"cbs\t{status}\t{hashlib.sha256(body).hexdigest()}\t{len(body)} bytes\t{CBS}")
    if status != 200:
        return None
    sheets = pd.read_excel(io.BytesIO(body), sheet_name=None, dtype=str, header=None)
    for name, s in sheets.items():
        log.append(f"cbs\tSHEET\t{name}\t{s.shape}\t{s.head(8).fillna('').values.tolist()}")
    name, s = max(sheets.items(), key=lambda kv: kv[1].shape[0])
    return s


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = [f"# fetch_prodcom.py run {datetime.now(timezone.utc).isoformat(timespec='seconds')}"]
    bad = 0
    p = prodcom(log)
    if p is None:
        bad += 1
    else:
        p.to_csv(out / "prodcom_2025.csv", index=False)
        log.append(f"prodcom\tROWS\t{len(p)}")
    c = correspondence(log)
    if c is None:
        bad += 1
    else:
        c.to_csv(out / "cn_prodcom_2025_raw.csv", index=False, header=False)
        log.append(f"cbs\tROWS\t{len(c)}")
    (out / "fetch_log.txt").write_text("\n".join(log) + "\n", encoding="utf-8")
    print("\n".join(log))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
