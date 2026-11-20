"""Download the series behind the article, and compare the OECD ones with the committed snapshot.

python fetch_data.py           # download into data/raw/, check Shiller's file, report whether the snapshot matches
python fetch_data.py --write   # same, then overwrite the committed OECD snapshot in data/

- OECD DF_FINMARK, SHARE    share price indices, monthly average                 -> data/share_prices.csv
- OECD DF_FINMARK, IR3TIB   3-month interbank rates, per cent a year            -> data/rate_3m.csv
- OECD DF_FINMARK, IRSTCI   overnight interbank rates, per cent a year          -> data/rate_overnight.csv
- Shiller's monthly S&P composite with dividends, as kept by the datasets/s-and-p-500 repository at a pinned
  commit. It is downloaded into data/raw/ and checked against a SHA-256 digest; it is not committed here.
"""

import csv
import hashlib
import io
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import lump_core as lc

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RAW = DATA / "raw"
OECD = "https://sdmx.oecd.org/public/rest/data/OECD.SDD.STES,DSD_STES@DF_FINMARK,4.0/"
LAST = "2026-10"
AGENT = "lump-sum-vs-spreading/0.1 (+https://github.com/DiogoRibeiro7/Medium-Blog)"
SHILLER_COMMIT = "87a2284276f79d75041c71c8f6d9f2aa1c965e84"
SHILLER_URL = f"https://raw.githubusercontent.com/datasets/s-and-p-500/{SHILLER_COMMIT}/data/data.csv"
SHILLER_SHA256 = "b797a3c8de4586af837b86a156b05f49be11ffab99e13171dae763b2a11bdf79"
SHILLER = RAW / "shiller.csv"


def download(name, address, accept=None):
    target = RAW / name
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        headers = {"User-Agent": AGENT} | ({"Accept": accept} if accept else {})
        request = urllib.request.Request(address, headers=headers)
        for attempt in range(5):
            try:
                with urllib.request.urlopen(request, timeout=300) as r:
                    body = r.read()
                break
            except urllib.error.HTTPError as e:
                if e.code not in (429, 500, 502, 503, 504) or attempt == 4:
                    raise
                time.sleep(30 * (attempt + 1))
        target.write_bytes(body)
    return target.read_bytes()


def oecd(measure):
    key = "+".join(lc.PANEL) + f".M.{measure}......"
    text = download(
        f"{measure}.csv",
        f"{OECD}{key}?dimensionAtObservation=AllDimensions&endPeriod={LAST}",
        accept="application/vnd.sdmx.data+csv; charset=utf-8",
    ).decode("utf-8")
    rows = sorted(
        (r["REF_AREA"], r["TIME_PERIOD"], r["OBS_VALUE"]) for r in csv.DictReader(io.StringIO(text)) if r["OBS_VALUE"]
    )
    out = io.StringIO()
    w = csv.writer(out, lineterminator="\n")
    w.writerow(["country", "month", "value"])
    w.writerows(rows)
    return out.getvalue()


def shiller():
    body = download("shiller.csv", SHILLER_URL)
    digest = hashlib.sha256(body).hexdigest()
    if digest != SHILLER_SHA256:
        raise SystemExit(f"shiller.csv: digest {digest} does not match the pinned {SHILLER_SHA256}")
    print(f"shiller.csv: matches the pinned digest (datasets/s-and-p-500 at {SHILLER_COMMIT[:7]})")


def main(write):
    shiller()
    for name, measure in (("share_prices", "SHARE"), ("rate_3m", "IR3TIB"), ("rate_overnight", "IRSTCI")):
        text = oecd(measure)
        path = DATA / f"{name}.csv"
        old = path.read_text() if path.exists() else ""
        if old == text:
            print(f"{name}.csv: identical to the OECD")
            continue
        print(
            f"{name}.csv: differs from the OECD ({len(old.splitlines())} lines committed, {len(text.splitlines())} now)"
        )
        if write:
            path.write_text(text)


if __name__ == "__main__":
    main("--write" in sys.argv)
