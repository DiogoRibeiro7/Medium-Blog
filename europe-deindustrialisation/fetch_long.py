"""Download long sectoral series (AMECO, from 1960) through the DBnomics mirror.

    python fetch_long.py --out data

Step 1 lists every AMECO dataset on DBnomics and keeps those whose name is sectoral
employment ("Employment, persons: ..."), sectoral gross value added at constant or current
prices, the current-account balance or net exports of goods.
Step 2 downloads every series of those datasets with their observations.

Writes ``data/ameco_datasets.csv`` (the discovery list), ``data/ameco_long.csv`` (one row per
series and year) and appends to ``data/fetch_log.txt``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

API = "https://api.db.nomics.world/v22"
PROVIDER = "AMECO"


def get_json(url: str, tries: int = 4) -> tuple[dict, str]:
    last: Exception | None = None
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "europe-deindustrialisation/1.0"})
            with urllib.request.urlopen(req, timeout=180) as r:
                body = r.read()
                return json.loads(body), hashlib.sha256(body).hexdigest()
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            last = e
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"{url}: {last}")


def keep(name: str) -> bool:
    n = name.lower()
    if n.startswith("employment, persons:") or "employment, persons: " in n:
        return True
    if "gross value added" in n and ":" in n and ("prices" in n):
        return True
    if n.startswith("balance on current transactions with the rest of the world"):
        return True
    if n.startswith("net exports of goods"):
        return True
    return False


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = [f"# fetch_long.py run {datetime.now(timezone.utc).isoformat(timespec='seconds')}"]
    # 1. discovery
    datasets = []
    offset = 0
    while True:
        url = f"{API}/datasets/{PROVIDER}?limit=500&offset={offset}"
        doc, digest = get_json(url)
        docs = doc["datasets"]["docs"]
        datasets.extend({"code": d["code"], "name": d.get("name", "")} for d in docs)
        log.append(f"datasets\toffset {offset}\t{len(docs)}\t{digest}\t{url}")
        offset += len(docs)
        if not docs or offset >= doc["datasets"]["num_found"]:
            break
    ds = pd.DataFrame(datasets)
    ds["kept"] = ds.name.map(keep)
    ds.to_csv(out / "ameco_datasets.csv", index=False)
    kept = ds[ds.kept]
    log.append(f"datasets found {len(ds)}, kept {len(kept)}: {', '.join(kept.code)}")
    # 2. series
    rows = []
    failures = 0
    for code, name in zip(kept.code, kept.name):
        offset = 0
        while True:
            url = f"{API}/series/{PROVIDER}/{urllib.parse.quote(code)}?observations=1&limit=1000&offset={offset}"
            try:
                doc, digest = get_json(url)
            except RuntimeError as e:
                log.append(f"series\t{code}\tERROR\t{e}")
                failures += 1
                break
            docs = doc["series"]["docs"]
            log.append(f"series\t{code}\toffset {offset}\t{len(docs)}\t{digest}\t{url}")
            for s in docs:
                for p, v in zip(s.get("period", []), s.get("value", [])):
                    if v is None or v == "NA":
                        continue
                    rows.append(
                        {
                            "dataset": code,
                            "dataset_name": name,
                            "series_code": s["series_code"],
                            "series_name": s.get("series_name", ""),
                            "dimensions": json.dumps(s.get("dimensions", {}), sort_keys=True),
                            "period": p,
                            "value": v,
                        }
                    )
            offset += len(docs)
            if not docs or offset >= doc["series"]["num_found"]:
                break
    pd.DataFrame(rows).to_csv(out / "ameco_long.csv", index=False)
    log.append(f"rows {len(rows)}")
    with open(out / "fetch_log.txt", "a", encoding="utf-8") as fh:
        fh.write("\n".join(log) + "\n")
    print("\n".join(log))
    return 1 if failures or not rows else 0


if __name__ == "__main__":
    sys.exit(main())
