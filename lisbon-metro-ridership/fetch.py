"""Download the INE and Eurostat responses listed in sources.py and check them against the pinned contracts.

    python fetch.py                    # download what is missing, stop if any content changed
    python fetch.py --accept-revision  # keep going after a revision and print the new hashes
    python fetch.py --record           # (maintainer) write the two contracts owned by this folder

Contracts, all in ../data_sources/:

* ine-portugal-metro-ridership.json: INE stamps every response with its extraction time and with the date
  of the indicator's last update. The metro and airport series are updated every month, so that date moves
  even when the months requested here do not. The pin is therefore the SHA-256 of the indicator code and
  the data block only; a mismatch means INE revised one of the requested months.
* ine-portugal-municipal-infrastructure.json (shared with portugal-infrastructure-saturation): content hash
  of the indicator code, INE's last-update date and the data block, as recorded there.
* eurostat-regional-employment-pt.json: Eurostat serves identical bytes for identical data, so the pin is the
  SHA-256 of the file.

Nothing is committed here: data/ is rebuilt at each run and ignored by git.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from sources import EUROSTAT, INE_NEW, INE_SHARED, eurostat_url, ine_url

HERE = Path(__file__).resolve().parent
SOURCES = HERE.parent / "data_sources"
INE_CONTRACT = SOURCES / "ine-portugal-metro-ridership.json"
SHARED_CONTRACT = SOURCES / "ine-portugal-municipal-infrastructure.json"
EU_CONTRACT = SOURCES / "eurostat-regional-employment-pt.json"
RAW = HERE / "data" / "raw"
UA = "Mozilla/5.0 (X11; Linux x86_64) research-download (github.com/DiogoRibeiro7/Medium-Blog)"


def ine_data_hash(payload: list) -> str:
    """Pin for this folder's INE responses: indicator code and data block only."""
    head = payload[0]
    core = {"IndicadorCod": head.get("IndicadorCod"), "Dados": head.get("Dados")}
    blob = json.dumps(core, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def ine_content_hash(payload: list) -> str:
    """Pin used by the shared contract: indicator code, last-update date and data block."""
    head = payload[0]
    core = {"IndicadorCod": head.get("IndicadorCod"), "DataUltimoAtualizacao": head.get("DataUltimoAtualizacao"), "Dados": head.get("Dados")}
    blob = json.dumps(core, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def download(address: str, ine: bool, tries: int = 6) -> bytes:
    for attempt in range(tries):
        try:
            req = urllib.request.Request(address, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=240) as r:
                body = r.read()
            payload = json.loads(body)
            if ine and "Dados" not in payload[0]:
                raise RuntimeError(f"INE refused the query: {str(payload[0])[:200]}")
            if not ine and "value" not in payload:
                raise RuntimeError(f"Eurostat returned no values: {str(payload)[:200]}")
            return body
        except (urllib.error.URLError, TimeoutError, ConnectionError, json.JSONDecodeError) as e:
            if attempt == tries - 1:
                raise RuntimeError(f"{address[:120]}: {e}") from e
            time.sleep(6 * (attempt + 1))
    raise RuntimeError(address)


def plan() -> list[dict]:
    """One entry per file: where it goes, where it comes from, how it is hashed, what it should hash to."""
    shared = {f["path"]: f for f in json.loads(SHARED_CONTRACT.read_text())["files"]}
    ine = {f["path"]: f for f in json.loads(INE_CONTRACT.read_text())["files"]} if INE_CONTRACT.exists() else {}
    eu = {f["path"]: f for f in json.loads(EU_CONTRACT.read_text())["files"]} if EU_CONTRACT.exists() else {}
    out = []
    for label, code, dims, about in INE_NEW:
        rel = f"data/raw/{label}.json"
        out.append(
            {
                "rel": rel,
                "url": ine_url(code, dims),
                "kind": "ine",
                "expected": ine.get(rel, {}).get("data_sha256"),
                "meta": {"indicator": code, "about": about},
            }
        )
    for label in INE_SHARED:
        rel = f"data/raw/{label}.json"
        f = shared[rel]
        out.append({"rel": rel, "url": f["url"], "kind": "ine_shared", "expected": f["content_sha256"], "meta": None})
    for label, dataset, query, about in EUROSTAT:
        rel = f"data/raw/{label}.json"
        out.append(
            {
                "rel": rel,
                "url": eurostat_url(dataset, query),
                "kind": "eurostat",
                "expected": eu.get(rel, {}).get("sha256"),
                "meta": {"dataset": dataset, "query": query, "about": about},
            }
        )
    return out


def file_hash(path: Path, kind: str) -> str:
    body = path.read_bytes()
    if kind == "ine":
        return ine_data_hash(json.loads(body))
    if kind == "ine_shared":
        return ine_content_hash(json.loads(body))
    return hashlib.sha256(body).hexdigest()


def record() -> int:
    ine_files, eu_files = [], []
    for it in plan():
        path = HERE / it["rel"]
        if it["meta"] is None:
            continue
        if it["kind"] == "ine":
            payload = json.loads(path.read_bytes())
            ine_files.append(
                {
                    "path": it["rel"],
                    **it["meta"],
                    "url": it["url"],
                    "ine_last_updated": payload[0].get("DataUltimoAtualizacao"),
                    "periods": len(payload[0]["Dados"]),
                    "data_sha256": ine_data_hash(payload),
                }
            )
        else:
            body = path.read_bytes()
            eu_files.append(
                {
                    "path": it["rel"],
                    **it["meta"],
                    "url": it["url"],
                    "eurostat_updated": json.loads(body).get("updated"),
                    "bytes": len(body),
                    "sha256": hashlib.sha256(body).hexdigest(),
                }
            )
    for contract, files in [(INE_CONTRACT, ine_files), (EU_CONTRACT, eu_files)]:
        c = json.loads(contract.read_text()) if contract.exists() else {}
        c["files"] = files
        contract.write_text(json.dumps(c, indent=2, ensure_ascii=False) + "\n")
        print(f"recorded {len(files)} files in {contract.name}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--accept-revision", action="store_true")
    ap.add_argument("--record", action="store_true")
    args = ap.parse_args()
    if args.record:
        return record()
    RAW.mkdir(parents=True, exist_ok=True)
    changed = 0
    items = plan()
    for it in items:
        path = HERE / it["rel"]
        if path.exists() and it["expected"] and file_hash(path, it["kind"]) == it["expected"]:
            print(f"ok       {it['rel']}")
            continue
        body = download(it["url"], it["kind"] != "eurostat")
        tmp = path.with_suffix(".part")
        tmp.write_bytes(body)
        tmp.replace(path)
        got = file_hash(path, it["kind"])
        if it["expected"] and got != it["expected"]:
            changed += 1
            print(f"REVISED  {it['rel']}: sha256 {got} (pinned {it['expected']})", file=sys.stderr)
        else:
            print(f"fetched  {it['rel']} ({len(body):,} bytes)")
        time.sleep(1.5)
    if changed and not args.accept_revision:
        print(f"{changed} response(s) differ from the pinned vintage; rerun with --accept-revision to use them", file=sys.stderr)
        return 1
    pinned = sum(1 for it in items if it["expected"])
    print(f"{pinned - changed} of {len(items)} responses match the pinned hashes; {len(items) - pinned} have no pin yet")
    return 0


if __name__ == "__main__":
    sys.exit(main())
