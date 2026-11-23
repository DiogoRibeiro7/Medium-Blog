"""Download the INE responses listed in sources.py and check them against the pinned contract.

    python fetch.py                    # download what is missing, stop if any content changed
    python fetch.py --accept-revision  # keep going after a revision and print the new hashes
    python fetch.py --record           # (maintainer) write the contract from the files in data/raw

The responses are an external source in the data registry (DiogoRibeiro7/data,
``external/ine-portugal-municipal-infrastructure``). The registry commit, URLs and hashes are
copied in ``../data_sources/ine-portugal-municipal-infrastructure.json``. Nothing is committed here:
``data/`` is rebuilt at each run and ignored by git.

The INE API stamps every response with the time of extraction (``DataExtracao``), so the bytes
differ between two downloads of the same data. The contract therefore pins a content hash: the
SHA-256 of the indicator code, INE's last-update date and the data block, serialised with sorted
keys. Two downloads of the same vintage give the same content hash; a revision does not.
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

from sources import QUERIES, url

HERE = Path(__file__).resolve().parent
CONTRACT = HERE.parent / "data_sources" / "ine-portugal-municipal-infrastructure.json"
RAW = HERE / "data" / "raw"
UA = "Mozilla/5.0 (X11; Linux x86_64) research-download (dfr@esmad.ipp.pt)"


def content_hash(payload: list) -> str:
    """SHA-256 of the parts of an INE response that carry data, not the extraction time."""
    head = payload[0]
    core = {
        "IndicadorCod": head.get("IndicadorCod"),
        "DataUltimoAtualizacao": head.get("DataUltimoAtualizacao"),
        "Dados": head.get("Dados"),
    }
    blob = json.dumps(core, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def download(address: str, tries: int = 6) -> bytes:
    for attempt in range(tries):
        try:
            req = urllib.request.Request(address, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=240) as r:
                body = r.read()
            payload = json.loads(body)
            if "Dados" not in payload[0]:
                raise RuntimeError(f"INE refused the query: {payload[0]}")
            return body
        except (urllib.error.URLError, TimeoutError, ConnectionError, json.JSONDecodeError) as e:
            if attempt == tries - 1:
                raise RuntimeError(f"{address}: {e}") from e
            time.sleep(6 * (attempt + 1))
    raise RuntimeError(address)


def record() -> int:
    files = []
    for label, code, dims, about in QUERIES:
        path = RAW / f"{label}.json"
        payload = json.loads(path.read_bytes())
        files.append(
            {
                "path": f"data/raw/{label}.json",
                "indicator": code,
                "about": about,
                "url": url(code, dims),
                "ine_last_updated": payload[0].get("DataUltimoAtualizacao"),
                "content_sha256": content_hash(payload),
            }
        )
    contract = json.loads(CONTRACT.read_text()) if CONTRACT.exists() else {}
    contract["files"] = files
    CONTRACT.write_text(json.dumps(contract, indent=2, ensure_ascii=False) + "\n")
    print(f"recorded {len(files)} files in {CONTRACT.name}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--accept-revision", action="store_true")
    ap.add_argument("--record", action="store_true")
    args = ap.parse_args()
    if args.record:
        return record()

    pinned = {}
    if CONTRACT.exists():
        pinned = {f["path"]: f for f in json.loads(CONTRACT.read_text())["files"]}
    RAW.mkdir(parents=True, exist_ok=True)
    changed = 0
    for label, code, dims, _ in QUERIES:
        rel = f"data/raw/{label}.json"
        path = HERE / rel
        expected = pinned.get(rel, {}).get("content_sha256")
        if path.exists() and expected and content_hash(json.loads(path.read_bytes())) == expected:
            print(f"ok       {rel}")
            continue
        body = download(url(code, dims))
        got = content_hash(json.loads(body))
        tmp = path.with_suffix(".part")
        tmp.write_bytes(body)
        tmp.replace(path)
        if expected and got != expected:
            changed += 1
            print(f"REVISED  {rel}: content sha256 {got} (pinned {expected})", file=sys.stderr)
        else:
            print(f"fetched  {rel} ({len(body):,} bytes)")
        time.sleep(1.5)
    if changed and not args.accept_revision:
        print(f"{changed} response(s) differ from the pinned vintage; rerun with --accept-revision to use them", file=sys.stderr)
        return 1
    print(f"{len(QUERIES) - changed} of {len(QUERIES)} responses match the pinned content hashes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
