"""Download the BOEM lease-sale bid files for the auction-entry-boem analysis.

    python fetch.py --out data

Two sources:

1. The case study of the game- and auction-theory series
   (https://gitlab.com/DiogoRibeiro7/game-auction-theory-ds, folder ``case-study``), which
   holds the BOEM files for the Gulf sales from 249 (2017) to BBG3 and the parsed table
   ``bids.csv``, checked there against BOEM's published totals. The folder is copied to
   ``data/case_study/`` and the commit it came from is written to ``data/case_study/SOURCE``.
2. Older Gulf sales, read from BOEM's own sale pages (``https://www.boem.gov/Sale-<n>/``):
   for each sale the data archive (five fixed-format ASCII files and their record layout),
   the sale statistics page and the final bid recap are saved under ``data/boem/<n>/``.
   Sales without a page are skipped.

``data/fetch_log.txt`` records the URL, HTTP status, size and SHA-256 of every response.
BOEM states that material produced by federal agencies is generally in the public domain
(https://www.boem.gov/copyright).
"""

from __future__ import annotations

import argparse
import hashlib
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

GITLAB = "https://gitlab.com/DiogoRibeiro7/game-auction-theory-ds.git"
BOEM = "https://www.boem.gov"
UA = "Mozilla/5.0 (X11; Linux x86_64) research-download (dfr@esmad.ipp.pt)"
# Gulf sale numbers before those in the case study; most numbers in this range are Gulf
# sales, the rest (Alaska, Pacific, Atlantic) have no page under this pattern or no Gulf data.
OLD_SALES = list(range(166, 249))

LOG: list[str] = []


def get(url: str, tries: int = 3) -> tuple[int, bytes, str]:
    """GET a URL with retries; return (status, body, final URL)."""
    last = (0, b"", url)
    for attempt in range(tries):
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                last = (r.status, r.read(), r.geturl())
                break
        except urllib.error.HTTPError as e:
            last = (e.code, b"", url)
            if e.code in (403, 404, 410):
                break
        except Exception as e:  # network error: retry
            last = (0, str(e).encode(), url)
        time.sleep(3 * (attempt + 1))
    status, body, final = last
    LOG.append(f"{url}\t{status}\t{len(body)}\t{hashlib.sha256(body).hexdigest()}\t{final}")
    time.sleep(0.7)
    return last


def copy_case_study(out: Path) -> bool:
    dest = out / "case_study"
    with tempfile.TemporaryDirectory() as tmp:
        r = subprocess.run(["git", "clone", "--depth", "1", GITLAB, tmp + "/repo"],
                           capture_output=True, text=True)
        LOG.append(f"git clone {GITLAB}\texit {r.returncode}\t{r.stderr.strip()[-200:]}")
        if r.returncode != 0:
            return False
        sha = subprocess.run(["git", "-C", tmp + "/repo", "rev-parse", "HEAD"],
                             capture_output=True, text=True).stdout.strip()
        src = Path(tmp) / "repo" / "case-study"
        if not src.is_dir():
            LOG.append("case-study folder not found in the series repository")
            return False
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(src, dest)
        for extra in ("README.md", "LICENSE"):
            p = Path(tmp) / "repo" / extra
            if p.exists():
                shutil.copy(p, dest / f"SERIES_{extra}")
        (dest / "SOURCE").write_text(f"{GITLAB}\ncommit {sha}\nfolder case-study\n")
        for p in sorted(dest.rglob("*")):
            if p.is_file():
                LOG.append(f"  copied {p.relative_to(out)}\t{p.stat().st_size}\t"
                           f"{hashlib.sha256(p.read_bytes()).hexdigest()}")
    return True


HREF = re.compile(r'href="([^"]+)"', re.I)


def links(html: bytes) -> list[str]:
    text = html.decode("utf-8", "replace")
    return [urllib.parse.urljoin(BOEM + "/", h) for h in HREF.findall(text)]


def fetch_old_sales(out: Path) -> int:
    got = 0
    for n in OLD_SALES:
        status, page, _ = get(f"{BOEM}/Sale-{n}/")
        if status != 200:
            continue
        d = out / "boem" / str(n)
        d.mkdir(parents=True, exist_ok=True)
        (d / "sale_page.html").write_bytes(page)
        hrefs = links(page)
        wanted = []
        for h in hrefs:
            low = h.lower()
            if (f"sale-{n}-zip-file" in low or f"{n}-zip-file" in low
                    or (low.endswith(".zip") and "ascii" in low)):
                wanted.append(("data.zip", h))
            elif "stats" in low and str(n) in low:
                wanted.append(("stats.html", h))
            elif "final-bid-recap" in low and low.endswith(".pdf"):
                wanted.append(("final_bid_recap.pdf", h))
        if not any(k == "data.zip" for k, _ in wanted):
            wanted.append(("data.zip", f"{BOEM}/oil-gas-energy/sale-{n}-zip-file"))
        seen = set()
        for name, url in wanted:
            if name in seen:
                continue
            st, body, _ = get(url)
            if st != 200 or not body:
                continue
            if name == "data.zip" and body[:2] != b"PK":
                continue
            (d / name).write_bytes(body)
            seen.add(name)
            if name == "data.zip":
                got += 1
    return got


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    LOG.append(f"# fetch.py run {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    LOG.append("# url\tstatus\tbytes\tsha256\tfinal url")
    ok = copy_case_study(out)
    n_old = fetch_old_sales(out)
    LOG.append(f"# case study copied: {ok}; older sale archives saved: {n_old}")
    (out / "fetch_log.txt").write_text("\n".join(LOG) + "\n")
    print("\n".join(LOG[-40:]))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
