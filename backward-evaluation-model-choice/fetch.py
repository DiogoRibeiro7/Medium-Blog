"""Download the test set named in PREREGISTRATION.md and apply its inclusion rules.

    python fetch.py --out data

Writes ``data/wikipedia_test_daily.csv.gz`` (daily page views, agent = user, all access, one column
``<list>:<title>`` per included article, days not returned set to 0), ``data/inclusion.txt`` and
``data/fetch_log.txt`` (UTC time, URL, SHA-256 of every response).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import time
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("mc_probe", HERE / "probe" / "probe.py")
probe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(probe)
gen = probe.gen

START, END, DAYS = "2015-07-01", "2026-10-31", 4141


def log(out: Path, url: str, body: bytes | None, note: str) -> None:
    sha = hashlib.sha256(body).hexdigest() if body is not None else "-"
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    with open(out / "fetch_log.txt", "a", encoding="utf-8") as f:
        f.write(f"{stamp}\t{url}\t{sha}\t{note}\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    days = pd.date_range(START, END, freq="D")
    keep, report = {}, []
    for group, block in probe.LISTS.items():
        titles = gen.split(block)
        canon = gen.resolve(titles)
        for t in titles:
            c, missing = canon[t]
            if missing or c != t:
                report.append(f"{group}\t{t}\texcluded\tcanonical title '{c}' (renamed or missing)")
                continue
            url = gen.PV_API.format(title=urllib.parse.quote(c.replace(" ", "_"), safe=""))
            body = gen.get(url)
            items = json.loads(body)["items"]
            s = pd.Series({pd.Timestamp(it["timestamp"][:8]): it["views"] for it in items}, dtype=float)
            n = int(s.index.isin(days).sum())
            ok = n >= 0.99 * DAYS
            log(out, url, body, f"{c}: {n} days")
            report.append(f"{group}\t{t}\t{'included' if ok else 'excluded'}\tdays returned {n} of {DAYS}")
            if ok:
                keep[f"{group}:{c}"] = s.reindex(days).fillna(0).astype(int)
            time.sleep(0.05)
    panel = pd.DataFrame(keep)
    panel.index.name = "date"
    panel.to_csv(out / "wikipedia_test_daily.csv.gz", compression={"method": "gzip", "mtime": 0})
    (out / "inclusion.txt").write_text("list\tarticle\tdecision\treason\n" + "\n".join(report) + "\n", encoding="utf-8")
    print(f"included {len(keep)} of {len(report)} articles")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
