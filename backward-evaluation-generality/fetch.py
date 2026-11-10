"""Download the data named in PREREGISTRATION.md and apply its inclusion rules.

    python fetch.py --out data

Writes
* ``data/electricity_daily.csv``: daily load in MWh (24 x the mean of the hours with data, days with
  at least 20 hours), one column per included country, gaps filled as pre-registered;
* ``data/wikipedia_daily.csv.gz``: daily page views (agent = user, all access), one column per
  included article, days not returned by the API set to 0;
* ``data/inclusion.txt``: every candidate series with the rule that included or excluded it;
* ``data/fetch_log.txt``: UTC time, URL, SHA-256 of every response.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import re
import time
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("probe", HERE / "probe" / "probe.py")
probe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(probe)

OPSD_VERSION = "2020-10-06"
OPSD_SHA256 = "6a7f2bc571314cbf9c321cc03437691cd4be95c3a6f075e60ff99e8035c704c8"
ELEC_START, ELEC_END = "2015-01-01", "2020-09-30"
WEB_START, WEB_END, WEB_DAYS = "2015-07-01", "2026-10-31", 4141


def log(out: Path, url: str, body: bytes | None, note: str) -> None:
    sha = hashlib.sha256(body).hexdigest() if body is not None else "-"
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    with open(out / "fetch_log.txt", "a", encoding="utf-8") as f:
        f.write(f"{stamp}\t{url}\t{sha}\t{note}\n")


def electricity(out: Path, report: list[str]) -> None:
    url = probe.OPSD_CSV.format(version=OPSD_VERSION)
    body = probe.get(url, timeout=600)
    sha = hashlib.sha256(body).hexdigest()
    log(out, url, body, f"{len(body)} bytes; matches probe: {sha == OPSD_SHA256}")
    if sha != OPSD_SHA256:
        raise RuntimeError("OPSD file differs from the one recorded by the probe")
    head = pd.read_csv(io.BytesIO(body), nrows=0).columns
    cols = [c for c in head if re.fullmatch(r"[A-Z]{2}_load_actual_entsoe_transparency", c)]
    df = pd.read_csv(io.BytesIO(body), usecols=["utc_timestamp", *cols], parse_dates=["utc_timestamp"])
    df = df.set_index("utc_timestamp")
    days = pd.date_range(ELEC_START, ELEC_END, freq="D", tz="UTC")
    keep = {}
    for c in cols:
        cc = c[:2]
        s = df[c].dropna()
        g = s.groupby(s.index.floor("D"))
        daily = (24 * g.mean()).where(g.size() >= 20).reindex(days)
        first_ok = (not s.empty) and s.index.min() <= pd.Timestamp("2015-01-31 23:00", tz="UTC")
        miss = daily.isna()
        run = int(miss.groupby((~miss).cumsum()).sum().max()) if miss.any() else 0
        share = float(miss.mean())
        ok = first_ok and share <= 0.02 and run <= 7
        report.append(f"elec\t{cc}\t{'included' if ok else 'excluded'}\tfirst hour ok {first_ok}; "
                      f"missing days {int(miss.sum())} ({100 * share:.1f}%); longest missing run {run}")
        if ok:
            filled = np.exp(np.log(daily).interpolate(limit_direction="both"))
            keep[cc] = filled.round(1)
    panel = pd.DataFrame(keep)
    panel.index = panel.index.tz_localize(None)
    panel.index.name = "date"
    panel.to_csv(out / "electricity_daily.csv")


def web(out: Path, report: list[str]) -> None:
    days = pd.date_range(WEB_START, WEB_END, freq="D")
    keep = {}
    for group, titles in (("country", probe.split(probe.COUNTRIES)), ("element", probe.split(probe.ELEMENTS))):
        canon = probe.resolve(titles)
        for t in titles:
            c, missing = canon[t]
            if missing or c != t:
                report.append(f"web\t{t}\texcluded\t{group}; canonical title '{c}' (renamed or missing)")
                continue
            url = probe.PV_API.format(title=urllib.parse.quote(c.replace(" ", "_"), safe=""))
            body = probe.get(url)
            items = json.loads(body)["items"]
            s = pd.Series({pd.Timestamp(it["timestamp"][:8]): it["views"] for it in items}, dtype=float)
            n = int(s.index.isin(days).sum())
            ok = n >= 0.99 * WEB_DAYS
            log(out, url, body, f"{c}: {n} days")
            report.append(f"web\t{t}\t{'included' if ok else 'excluded'}\t{group}; days returned {n} of {WEB_DAYS}")
            if ok:
                keep[f"{group}:{c}"] = s.reindex(days).fillna(0).astype(int)
            time.sleep(0.05)
    panel = pd.DataFrame(keep)
    panel.index.name = "date"
    panel.to_csv(out / "wikipedia_daily.csv.gz", compression={"method": "gzip", "mtime": 0})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report: list[str] = []
    electricity(out, report)
    web(out, report)
    (out / "inclusion.txt").write_text("domain\tseries\tdecision\treason\n" + "\n".join(report) + "\n", encoding="utf-8")
    inc = [r for r in report if "\tincluded\t" in r]
    print(f"included: {sum(r.startswith('elec') for r in inc)} electricity, {sum(r.startswith('web') for r in inc)} web")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
