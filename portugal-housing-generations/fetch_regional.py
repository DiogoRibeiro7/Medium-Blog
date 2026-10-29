"""Download and pin the INE series used by the regional panel.

For each indicator in ``INDICATORS`` the script requests INE's metadata and then
the data, as many years per request as INE's 40,000-value limit allows, and writes every response body
unchanged into a gzipped JSON-lines container
``data/ine_<code>_<YYYY-MM-DD>.jsonl.gz``. Each line records the URL, retrieval
time, HTTP status, text encoding, byte count and SHA-256 of the raw body next
to the body itself, so ``regional.py`` can check every response it reads. The
first line is a header with the number of responses that follow.

Nothing is parsed or transformed here. Run on a GitHub runner (the workflow job
``fetch-regional``); standard library only.

Run:  python fetch_regional.py --out data
"""

from __future__ import annotations

import argparse
import datetime as dt
import gzip
import hashlib
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

META_URL = "https://www.ine.pt/ine/json_indicador/pindicaMeta.jsp"
DATA_URL = "https://www.ine.pt/ine/json_indicador/pindica.jsp"

# varcd -> (short name, years, fixed filters on non-time, non-geographic dimensions)
INDICATORS: dict[str, tuple[str, list[int], dict[str, str]]] = {
    # Resident population, annual estimates (INE), totals for both sexes and all ages.
    "0012917": ("population_nuts2024", list(range(2021, 2026)), {"Dim3": "T", "Dim4": "T"}),
    "0008273": ("population_nuts2013", list(range(2011, 2024)), {"Dim3": "T", "Dim4": "T"}),
    # Dwellings in new construction for family housing, all typologies.
    "0013248": ("completions_nuts2024", list(range(2011, 2026)), {"Dim3": "T"}),
    "0013235": ("permits_nuts2024", list(range(2011, 2026)), {"Dim3": "T"}),
    # Census private households.
    "0014406": ("households_census_series", [2011, 2021], {}),
    "0014627": ("households_census2021_nuts2024", [2021], {"Dim3": "T", "Dim4": "T", "Dim5": "T"}),
}
PAUSE_SECONDS = 10
MAX_CELLS = 39_000  # INE serves at most 40,000 values per request


def request(url: str, timeout: float = 600.0, attempts: int = 6) -> tuple[bytes, int, str]:
    """GET ``url``; on HTTP 429 or a network error wait (honouring Retry-After) and retry."""
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Medium-Blog reproducibility fetch"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (fixed https host)
                charset = resp.headers.get_content_charset() or "utf-8"
                return resp.read(), resp.status, charset
        except urllib.error.HTTPError as exc:
            if exc.code != 429:
                raise
            last_error = exc
            time.sleep(min(int(exc.headers.get("Retry-After") or 60 * (attempt + 1)), 300))
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            last_error = exc
            time.sleep(60 * (attempt + 1))
    assert last_error is not None
    raise last_error


def record(url: str, body: bytes, status: int, charset: str, kind: str) -> dict[str, object]:
    """One container line: provenance fields plus the body decoded losslessly."""
    text = body.decode(charset)
    if text.encode(charset) != body:
        raise ValueError(f"Body of {url} does not round-trip through {charset}")
    return {
        "kind": kind,
        "url": url,
        "retrieved_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "http_status": status,
        "encoding": charset,
        "bytes": len(body),
        "sha256": hashlib.sha256(body).hexdigest(),
        "body": text,
    }


def geography_count(metadata: bytes, charset: str) -> int:
    """Number of geographic categories (every level) listed in the metadata."""
    item = json.loads(metadata.decode(charset))[0]
    return sum(1 for key in item["Dimensoes"]["Categoria_Dim"][0] if key.startswith("Dim_Num2_"))


def returned_years(body: bytes, charset: str) -> set[str]:
    """Years present in a data response (empty if INE reports a failure)."""
    item = json.loads(body.decode(charset))[0]
    if "Verdadeiro" not in item.get("Sucesso", {}):
        return set()
    return {year for year, rows in item.get("Dados", {}).items() if rows}


def fetch_indicator(code: str, years: list[int], filters: dict[str, str]) -> tuple[list[dict[str, object]], str]:
    """Metadata plus the data, several years per request within INE's cell limit."""
    lines = []
    meta_url = f"{META_URL}?{urllib.parse.urlencode({'varcd': code, 'lang': 'PT'})}"
    meta_body, meta_status, meta_charset = request(meta_url)
    lines.append(record(meta_url, meta_body, meta_status, meta_charset, kind="metadata"))
    per_request = max(1, MAX_CELLS // geography_count(meta_body, meta_charset))
    batches = [years[i:i + per_request] for i in range(0, len(years), per_request)]
    for batch in batches:
        query = {"op": "2", "varcd": code, "lang": "PT", "Dim1": ",".join(f"S7A{y}" for y in batch), **filters}
        url = f"{DATA_URL}?{urllib.parse.urlencode(query)}"
        body, status, charset = request(url)
        if returned_years(body, charset) != {str(y) for y in batch}:
            raise ValueError(f"{code}: response for {batch} has years {sorted(returned_years(body, charset))}")
        lines.append(record(url, body, status, charset, kind="data"))
        time.sleep(PAUSE_SECONDS)
    return lines, f"all geographic levels; up to {per_request} years per request"


def main() -> int:
    """Fetch every indicator and write one container per indicator."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "data")
    parser.add_argument("--only", nargs="*", default=None, help="indicator codes to fetch (default: all)")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    today = dt.datetime.now(dt.timezone.utc).date().isoformat()
    failures = 0
    for code, (name, years, filters) in INDICATORS.items():
        if args.only and code not in args.only:
            continue
        try:
            lines, mode = fetch_indicator(code, years, filters)
        except Exception as exc:  # noqa: BLE001 - report and continue with the others
            print(f"{code} {name}: FAILED {exc!r}", file=sys.stderr)
            failures += 1
            continue
        header = {"kind": "header", "indicator": code, "name": name, "filters": filters, "geography": mode,
                  "years": years, "responses": len(lines)}
        path = args.out / f"ine_{code}_{today}.jsonl.gz"
        with gzip.open(path, "wt", encoding="utf-8", compresslevel=9) as fh:
            for line in (header, *lines):
                fh.write(json.dumps(line, ensure_ascii=False) + "\n")
        print(f"{code} {name}: {len(lines)} responses, {mode} -> {path.name}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
