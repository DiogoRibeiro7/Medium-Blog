"""Download EU imports from outside the EU by supplier country (Eurostat Comext).

    python fetch_comext.py --out data

Dataset DS-045409 ("EU trade since 1988 by HS2-4-6 and CN8"): annual imports into the EU-27
(reporter EU27_2020) from every partner, in euros, one request per HS2 chapter (all 97) and per
selected HS4 heading of strategic interest. Writes ``data/comext_imports.csv`` (product,
partner, year, value in euros) and appends to ``data/fetch_log.txt``.
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

from fetch import jsonstat_to_long

API = "https://ec.europa.eu/eurostat/api/comext/dissemination/statistics/1.0/data/DS-045409"
AGENT = "europe-deindustrialisation/1.0"

CHAPTERS = [f"{i:02d}" for i in range(1, 98) if i != 77]  # chapter 77 is reserved
STRATEGIC = {
    "2504": "natural graphite",
    "2805": "alkali, alkaline-earth and rare-earth metals",
    "2846": "rare-earth compounds",
    "2941": "antibiotics",
    "3004": "medicaments in measured doses",
    "8471": "computers",
    "8501": "electric motors and generators",
    "8504": "transformers and converters",
    "8505": "permanent magnets",
    "8507": "electric accumulators (batteries)",
    "8517": "telephone sets and network equipment",
    "8541": "semiconductor devices, including photovoltaic cells",
    "8542": "integrated circuits",
    "8703": "passenger cars",
    "8708": "parts of motor vehicles",
}
REPORTERS = ("EU27_2020", "EU")


def get(url: str, tries: int = 3, timeout: int = 180) -> tuple[int, bytes]:
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


def url_for(product: str, reporter: str) -> str:
    params = [
        ("format", "JSON"),
        ("lang", "en"),
        ("freq", "A"),
        ("reporter", reporter),
        ("product", product),
        ("flow", "1"),
        ("indicators", "VALUE_IN_EUROS"),
    ]
    return f"{API}?{urllib.parse.urlencode(params)}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--limit", type=int, default=None, help="only the first N products (for testing)")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = [f"# fetch_comext.py run {datetime.now(timezone.utc).isoformat(timespec='seconds')}"]
    products = CHAPTERS + list(STRATEGIC)
    if args.limit:
        products = products[: args.limit]
    frames, failures, reporter_ok = [], 0, None
    for product in products:
        got = False
        for reporter in ([reporter_ok] if reporter_ok else REPORTERS):
            url = url_for(product, reporter)
            try:
                status, body = get(url)
            except RuntimeError as e:
                log.append(f"comext {product}\tERROR\t{e}")
                continue
            log.append(f"comext {product}\t{status}\t{reporter}\t{hashlib.sha256(body).hexdigest()}\t{len(body)} bytes\t{url}")
            if status != 200:
                log.append(f"comext {product}\tBODY\t{body[:300]!r}")
                continue
            doc = json.loads(body)
            if not doc.get("value"):
                log.append(f"comext {product}\tEMPTY")
                continue
            df = jsonstat_to_long(doc)
            frames.append(df)
            reporter_ok = reporter
            got = True
            break
        if not got:
            failures += 1
        time.sleep(0.5)
    if frames:
        d = pd.concat(frames, ignore_index=True)
        d.to_csv(out / "comext_imports.csv", index=False)
        log.append(f"comext\tROWS\t{len(d)}\tproducts {d['product'].nunique() if 'product' in d else '?'}")
    log.append(f"comext\tFAILED PRODUCTS\t{failures}")
    with open(out / "fetch_log.txt", "a", encoding="utf-8") as f:
        f.write("\n".join(log) + "\n")
    print("\n".join(log[-5:]))
    return 1 if failures > len(products) // 2 or not frames else 0


if __name__ == "__main__":
    sys.exit(main())
