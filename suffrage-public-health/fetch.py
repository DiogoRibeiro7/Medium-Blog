"""Download the data Grant Miller published with "Women's Suffrage, Political Responsiveness,
and Child Survival in American History" (Quarterly Journal of Economics 123(3), 2008).

    python fetch.py --out data

The files are linked from the paper's page on Miller's Stanford site
(https://ngmiller.people.stanford.edu/publications/historical-health-improvement/womens-suffrage-political-responsiveness-and-child):
the online appendix, and the coded Voteview, digitized municipal public finance and digitized
mortality data, raw and harmonized, stored on Google Drive. Each file is saved unchanged under
``data/raw/`` (archives are also unpacked there), and ``data/fetch_log.txt`` records the URL,
HTTP status, size and SHA-256 of every response.

The two Voteview files are not shared publicly: Google Drive answers every request for them with
its sign-in page. They are tried and logged as restricted, and the run does not fail on them.
"""

from __future__ import annotations

import argparse
import hashlib
import http.cookiejar
import io
import re
import sys
import time
import urllib.error
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

PAGE = ("https://ngmiller.people.stanford.edu/publications/historical-health-improvement/"
        "womens-suffrage-political-responsiveness-and-child")
APPENDIX = ("https://ngmiller.people.stanford.edu/sites/g/files/sbiybj24836/files/media/file/"
            "suffrage_online_appendix.pdf")
# key -> (Google Drive file id, is a Google Docs spreadsheet link)
FILES = {
    "voteview_raw": ("1SrR48FXbuiUA4qWa6rGHUj6IcRdofBUP", True),
    "voteview_harmonized": ("1dt0dtVLOy1XYUsoP7xsHYiGaKU8eSXU4", False),
    "public_finance": ("1btRcnnJuHJljMRUrGCGdRNbdLrEUFTWU", False),
    "mortality_raw": ("1Dw_KQKJQfyUz5mmVgi3jqP6rrGypAQ2N", True),
    "mortality_harmonized": ("1ubiSqKHZYhCbVkKaqS_lM7gZXERQyw5v", False),
}
# files the analysis can do without (both need a Google sign-in)
OPTIONAL = {"voteview_raw", "voteview_harmonized"}
UA = "Mozilla/5.0 (X11; Linux x86_64) research-download (dfr@esmad.ipp.pt)"
LOG: list[str] = []
JAR = http.cookiejar.CookieJar()
OPENER = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(JAR))


def get(url: str, tries: int = 3) -> tuple[int, bytes, dict]:
    last = (0, b"", {})
    for attempt in range(tries):
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        try:
            with OPENER.open(req, timeout=180) as r:
                last = (r.status, r.read(), dict(r.headers))
                break
        except urllib.error.HTTPError as e:
            last = (e.code, e.read() or b"", dict(e.headers or {}))
            if e.code in (403, 404):
                break
        except Exception as e:  # network error: retry
            last = (0, str(e).encode(), {})
        time.sleep(3 * (attempt + 1))
    status, body, _ = last
    LOG.append(f"{url}\t{status}\t{len(body)}\t{hashlib.sha256(body).hexdigest()}")
    time.sleep(1.0)
    return last


def is_html(body: bytes) -> bool:
    head = body[:600].lstrip().lower()
    return head.startswith(b"<!doctype html") or head.startswith(b"<html")


def extension(body: bytes, headers: dict) -> str:
    cd = headers.get("Content-Disposition", "") or headers.get("content-disposition", "")
    m = re.search(r'filename\*?=(?:UTF-8\'\')?"?([^";]+)"?', cd)
    if m and "." in m.group(1):
        return Path(m.group(1)).suffix.lower()
    if body[:2] == b"PK":
        return ".xlsx" if b"xl/" in body[:4000] else ".zip"
    if body[:4] == b"%PDF":
        return ".pdf"
    if body[:8] == b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1":
        return ".xls"
    if body[:11] == b"<stata_dta>" or body[:1] in (b"\x71", b"\x72", b"\x73", b"\x75"):
        return ".dta"
    return ".bin"


def original_name(headers: dict) -> str | None:
    cd = headers.get("Content-Disposition", "") or headers.get("content-disposition", "")
    m = re.search(r'filename\*?=(?:UTF-8\'\')?"?([^";]+)"?', cd)
    return m.group(1) if m else None


def drive(key: str, file_id: str, sheet: bool, raw: Path) -> bool:
    urls = [f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t",
            f"https://drive.google.com/uc?export=download&id={file_id}"]
    if sheet:
        urls.append(f"https://docs.google.com/spreadsheets/d/{file_id}/export?format=xlsx")
    for url in urls:
        status, body, headers = get(url)
        if status != 200 or not body:
            continue
        if is_html(body):
            text = body.decode("utf-8", "replace").replace("\\u003d", "=").replace("\\u0026", "&")
            if "accounts.google.com" in text[:2000]:
                LOG.append(f"  {key}: Google Drive asks for a sign-in (file not shared publicly)")
                break
            # a virus-scan or confirmation page: follow its form or a download link in it
            inputs = dict(re.findall(r'name="([a-z_]+)" value="([^"]*)"', text))
            links = re.findall(r'https://drive\.usercontent\.google\.com/download\?[^"\'\s<>]+', text)
            followed = False
            candidates = []
            if "uuid" in inputs or "confirm" in inputs:
                q = "&".join(f"{k}={v}" for k, v in inputs.items() if k in ("id", "export", "confirm", "uuid", "at"))
                candidates.append("https://drive.usercontent.google.com/download?" + (q or f"id={file_id}&export=download"))
            candidates += links[:3]
            for url2 in candidates:
                status, body, headers = get(url2.replace("&amp;", "&"))
                if status == 200 and body and not is_html(body):
                    followed = True
                    break
            if not followed:
                continue
        name = original_name(headers)
        path = raw / f"{key}{extension(body, headers)}"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
        LOG.append(f"  saved {path.name} (original name: {name})")
        if path.suffix == ".zip":
            with zipfile.ZipFile(io.BytesIO(body)) as z:
                for info in z.infolist():
                    if info.is_dir() or Path(info.filename).name.startswith("."):
                        continue
                    out = raw / key / Path(info.filename).name
                    out.parent.mkdir(parents=True, exist_ok=True)
                    out.write_bytes(z.read(info))
                    LOG.append(f"    unpacked {out.relative_to(raw)}\t{info.file_size}")
        return True
    LOG.append(f"  FAILED {key}" + (" (optional, restricted)" if key in OPTIONAL else ""))
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    args = ap.parse_args()
    out = Path(args.out)
    raw = out / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    LOG.append(f"# fetch.py run {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    LOG.append("# url\tstatus\tbytes\tsha256")
    status, body, _ = get(PAGE)
    if status == 200:
        (raw / "paper_page.html").write_bytes(body)
    status, body, _ = get(APPENDIX)
    ok = status == 200 and body[:4] == b"%PDF"
    if ok:
        (raw / "suffrage_online_appendix.pdf").write_bytes(body)
    results = {k: drive(k, fid, sheet, raw) for k, (fid, sheet) in FILES.items()}
    LOG.append(f"# appendix saved: {ok}; files saved: {sum(results.values())} of {len(results)}")
    (out / "fetch_log.txt").write_text("\n".join(LOG) + "\n")
    print("\n".join(LOG))
    needed = [k for k in FILES if k not in OPTIONAL]
    return 0 if ok and all(results[k] for k in needed) else 1


if __name__ == "__main__":
    sys.exit(main())
