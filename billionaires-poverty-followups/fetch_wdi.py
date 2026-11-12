"""World Development Indicators for the within-country panel (Part 3).

Downloads a small set of WDI series for every economy and year from the
World Bank API (v2, JSON, paged) and writes:

``data/wdi_series.csv.gz``
    iso3, indicator, year, value.

``data/wdi_manifest.json``
    Indicators, URLs, retrieval time, SHA-256 of each raw page and row counts.

Indicators:

* NY.GDP.PCAP.PP.KD  GDP per person, PPP, constant 2021 international dollars
* NY.GNP.PCAP.PP.KD  GNI per person, PPP, constant 2021 international dollars
* NE.CON.PRVT.PP.KD  household final consumption, PPP, constant 2021 international dollars
* SP.POP.TOTL        population
* PA.NUS.PPP         PPP conversion factor, GDP (local currency per international dollar)
* PA.NUS.FCRF        official exchange rate (local currency per US dollar); the ratio
                     PA.NUS.PPP / PA.NUS.FCRF is the price level ratio
* CM.MKT.LCAP.GD.ZS  market capitalisation of listed domestic companies, % of GDP (optional)

Optional indicators that the API reports as archived are recorded in the
manifest and skipped; required ones fail the run.

Run in GitHub Actions::

    python fetch_wdi.py --output-dir data
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "billionaires-poverty-audit"))
from resilient_fetch import fetch_bytes  # noqa: E402

API = "https://api.worldbank.org/v2/country/all/indicator/{ind}?format=json&per_page=20000&date={first}:{last}&page={page}"
INDICATORS = ("NY.GDP.PCAP.PP.KD", "NY.GNP.PCAP.PP.KD", "NE.CON.PRVT.PP.KD", "SP.POP.TOTL",
              "PA.NUS.PPP", "PA.NUS.FCRF", "CM.MKT.LCAP.GD.ZS")
OPTIONAL = {"CM.MKT.LCAP.GD.ZS"}


class IndicatorMissing(ValueError):
    """The API says the indicator does not exist or was archived."""


def parse_page(raw: bytes, indicator: str) -> tuple[list[dict[str, Any]], int]:
    """Rows and total page count from one API page; fails on unexpected shapes."""
    doc = json.loads(raw)
    if isinstance(doc, list) and len(doc) == 1 and isinstance(doc[0], dict) and "message" in doc[0]:
        raise IndicatorMissing(f"{indicator}: {doc[0]['message']}")
    if not isinstance(doc, list) or len(doc) != 2 or not isinstance(doc[0], dict):
        raise ValueError(f"unexpected WDI response for {indicator}: {str(doc)[:200]}")
    meta, records = doc
    rows = []
    for r in records or []:
        if r.get("indicator", {}).get("id") != indicator:
            raise ValueError(f"row for another indicator in {indicator}")
        iso3 = r.get("countryiso3code")
        if not iso3 or r.get("value") is None:
            continue
        rows.append({"iso3": iso3, "indicator": indicator, "year": int(r["date"]), "value": float(r["value"])})
    return rows, int(meta.get("pages", 1))


def fetch(output_dir: Path, first: int = 1995, last: int = 2025,
          fetcher: Callable[..., bytes] = fetch_bytes) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {"retrieved_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                "first": first, "last": last, "pages": []}
    rows: list[dict[str, Any]] = []
    manifest["skipped"] = {}
    for ind in INDICATORS:
        page, pages = 1, 1
        while page <= pages:
            url = API.format(ind=ind, first=first, last=last, page=page)
            raw = fetcher(url, name=f"wdi_{ind}_{page}", timeout_seconds=180, max_attempts=4, wait_seconds=10)
            try:
                part, pages = parse_page(raw, ind)
            except IndicatorMissing as exc:
                if ind not in OPTIONAL:
                    raise
                manifest["skipped"][ind] = str(exc)
                break
            rows += part
            manifest["pages"].append({"indicator": ind, "page": page, "url": url, "bytes": len(raw),
                                      "sha256": hashlib.sha256(raw).hexdigest(), "rows": len(part)})
            page += 1
        print(f"{ind}: {sum(p['rows'] for p in manifest['pages'] if p['indicator'] == ind)} values", flush=True)
    table = pd.DataFrame(rows).sort_values(["indicator", "iso3", "year"]).reset_index(drop=True)
    if table.duplicated(["iso3", "indicator", "year"]).any():
        raise ValueError("duplicate WDI rows")
    data = table.to_csv(index=False, lineterminator="\n").encode("utf-8")
    (output_dir / "wdi_series.csv.gz").write_bytes(gzip.compress(data, compresslevel=9, mtime=0))
    manifest["wdi_series_sha256"] = hashlib.sha256((output_dir / "wdi_series.csv.gz").read_bytes()).hexdigest()
    manifest["n_rows"] = int(len(table))
    manifest["by_indicator"] = {k: int(v) for k, v in table.groupby("indicator").size().items()}
    (output_dir / "wdi_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    args = parser.parse_args(argv)
    manifest = fetch(args.output_dir)
    print(json.dumps({k: manifest[k] for k in ("n_rows", "by_indicator")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
