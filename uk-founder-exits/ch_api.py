"""Companies House API crawl, resumable across GitHub runner jobs.

Two crawls, both stored in one SQLite file so that a job that hits its time budget can stop and
the next job (restoring the file from the Actions cache) carries on where it stopped:

1. advanced search: every company incorporated in the cohort window with a technology SIC code,
   live or dissolved, one query per SIC code and calendar month;
2. filing history: for each company found, its whole filing history (type, category and date of
   every filing), from which the build step reads allotments of shares (SH01), resolutions and
   insolvency filings.

    python ch_api.py --db crawl.sqlite search
    python ch_api.py --db crawl.sqlite filings --budget-minutes 320

The API key is read from COMPANIES_HOUSE_API_KEY. The API allows 600 requests per five minutes;
the client paces itself below that and sleeps through a 429.
"""

import argparse
import base64
import json
import os
import random
import sqlite3
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from datetime import date, timedelta

API = "https://api.company-information.service.gov.uk"
USER_AGENT = "Mozilla/5.0 (compatible; Medium-Blog uk-founder-exits)"

# Technology SIC 2007 codes: software, games, data and web, R&D, pharmaceuticals, electronics,
# medical instruments. IT consultancy (62020, 62030, 62090) is left out: it is dominated by
# one-person contracting companies.
TECH_SIC = [
    "58210", "58290", "62011", "62012", "63110", "63120", "72110", "72190", "21100", "21200", "32500",
    "26110", "26120", "26200", "26301", "26309", "26400", "26511", "26512", "26513", "26514", "26520",
    "26600", "26701", "26702", "26800",
]  # fmt: skip
COHORT_FROM = date(2016, 7, 1)  # PSCs are recorded at incorporation from 30 June 2016
COHORT_TO = date(2020, 12, 31)
PAGE = 5000  # the largest page the advanced search returns
MIN_INTERVAL = 0.52  # seconds between requests: about 577 per five minutes


def months(first, last):
    """(first day, last day) of every calendar month from first to last."""
    d = date(first.year, first.month, 1)
    while d <= last:
        nxt = date(d.year + (d.month == 12), d.month % 12 + 1, 1)
        yield max(d, first), min(nxt - timedelta(days=1), last)
        d = nxt


class Client:
    """Minimal authenticated client with pacing and retries."""

    def __init__(self, key, min_interval=MIN_INTERVAL, opener=None):
        if not key:
            raise SystemExit("COMPANIES_HOUSE_API_KEY is not set")
        self.auth = "Basic " + base64.b64encode(f"{key}:".encode()).decode()
        self.min_interval = min_interval
        self.last = 0.0
        self.calls = 0
        self.opener = opener or urllib.request.urlopen

    def get(self, path, params=None, attempts=6):
        """Return (status, parsed JSON or None). 404 is a normal answer, not an error."""
        url = API + path + ("?" + urllib.parse.urlencode(params) if params else "")
        for attempt in range(attempts):
            wait = self.min_interval - (time.monotonic() - self.last)
            if wait > 0:
                time.sleep(wait)
            self.last = time.monotonic()
            self.calls += 1
            req = urllib.request.Request(url, headers={"Authorization": self.auth, "User-Agent": USER_AGENT})
            try:
                with self.opener(req, timeout=60) as r:
                    return r.status, json.loads(r.read() or b"null")
            except urllib.error.HTTPError as e:
                if e.code in (404, 400, 416):
                    return e.code, None
                if e.code == 429:
                    reset = e.headers.get("X-Ratelimit-Reset") if e.headers else None
                    pause = max(5.0, float(reset) - time.time()) if reset and reset.isdigit() else 60.0
                    time.sleep(min(pause, 330.0))
                    continue
                if e.code >= 500:
                    time.sleep(2**attempt)
                    continue
                raise
            except (urllib.error.URLError, TimeoutError, ConnectionError):
                time.sleep(2**attempt)
        return 599, None


def connect(path):
    db = sqlite3.connect(path)
    db.execute("create table if not exists search (chunk text primary key, status int, hits int, body text)")
    db.execute("create table if not exists filings (company text, category text, status int, body text,"
               " primary key (company, category))")  # fmt: skip
    return db


def search_chunks():
    for sic in TECH_SIC:
        for first, last in months(COHORT_FROM, COHORT_TO):
            yield sic, first, last


def crawl_search(db, client, deadline):
    """Advanced search, one SIC code and month at a time, every page."""
    done = {r[0] for r in db.execute("select chunk from search where status in (200, 206)")}
    for sic, first, last in search_chunks():
        if time.monotonic() > deadline:
            return False
        start, items, hits = 0, [], None
        key = f"{sic}|{first.isoformat()}"
        if key in done:
            continue
        status = 200
        while True:
            params = {
                "sic_codes": sic,
                "company_type": "ltd",
                "incorporated_from": first.isoformat(),
                "incorporated_to": last.isoformat(),
                "size": PAGE,
                "start_index": start,
            }
            status, body = client.get("/advanced-search/companies", params)
            if status == 404:  # no company matches
                status, body = 200, {"hits": 0, "items": []}
            if status != 200 or body is None:
                break
            hits = body.get("hits", 0)
            page = body.get("items") or []
            items.extend(page)
            start += len(page)
            if not page or start >= hits:
                break
        if status == 200 and hits is not None and len(items) < hits:
            status = 206  # the API capped the result: record it, the build step reports it
        db.execute(
            "insert or replace into search values (?, ?, ?, ?)",
            (key, status, hits, json.dumps([_slim(i) for i in items])),
        )
        db.commit()
    return True


def _slim(item):
    """Keep the fields the cohort needs; drop the rest of the search hit."""
    addr = item.get("registered_office_address") or {}
    return {
        "company_number": item.get("company_number"),
        "company_name": item.get("company_name"),
        "company_status": item.get("company_status"),
        "company_type": item.get("company_type"),
        "date_of_creation": item.get("date_of_creation"),
        "date_of_cessation": item.get("date_of_cessation"),
        "sic_codes": item.get("sic_codes") or [],
        "postal_code": addr.get("postal_code"),
    }


def cohort_companies(db):
    """Unique private limited companies found by the search, with their search record."""
    seen = {}
    for (body,) in db.execute("select body from search where status in (200, 206)"):
        for item in json.loads(body):
            num = item.get("company_number")
            if num and item.get("company_type") == "ltd":
                seen.setdefault(num, item)
    return seen


def crawl_filings(db, client, deadline, seed=20161):
    """Whole filing history of every cohort company, one request per hundred filings.

    Companies are visited in a fixed random order, so a crawl stopped by the time budget has
    covered a random subset of the cohort rather than the first SIC codes in the list.
    """
    companies = sorted(cohort_companies(db))
    random.Random(seed).shuffle(companies)
    done = {r[0] for r in db.execute("select company from filings where status in (200, 404)")}
    for num in companies:
        if num in done:
            continue
        if time.monotonic() > deadline:
            db.commit()
            return False
        items, start, status = [], 0, 200
        while True:
            status, body = client.get(f"/company/{num}/filing-history", {"items_per_page": 100, "start_index": start})
            if status != 200 or body is None:
                break
            page = body.get("items") or []
            items.extend(_slim_filing(i) for i in page)
            start += len(page)
            if not page or start >= body.get("total_count", 0):
                break
        db.execute("insert or replace into filings values (?, ?, ?, ?)", (num, "all", status, json.dumps(items)))
        if client.calls % 200 == 0:
            db.commit()
    db.commit()
    return True


def _slim_filing(item):
    return {
        "type": item.get("type"),
        "category": item.get("category"),
        "subcategory": item.get("subcategory"),
        "date": item.get("date"),
        "action_date": item.get("action_date"),
        "dv_date": (item.get("description_values") or {}).get("date"),
    }


def progress(db):
    s = db.execute("select count(*), sum(status = 200), sum(status = 206), sum(hits) from search").fetchone()
    f = db.execute("select count(*), sum(status = 200), sum(status = 404) from filings").fetchone()
    cohort = cohort_companies(db)
    total_chunks = len(TECH_SIC) * len(list(months(COHORT_FROM, COHORT_TO)))
    return {
        "search_chunks": f"{s[0]} of {total_chunks} ({s[1] or 0} complete, {s[2] or 0} capped)",
        "search_hits": s[3] or 0,
        "cohort_companies": len(cohort),
        "cohort_status": dict(Counter(i.get("company_status") for i in cohort.values()).most_common()),
        "filing_histories": f"{f[1] or 0} fetched, {f[2] or 0} not found, {f[0] - (f[1] or 0) - (f[2] or 0)} failed",
        "crawl_complete": bool((s[1] or 0) + (s[2] or 0) == total_chunks and (f[1] or 0) + (f[2] or 0) == len(cohort)),
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--budget-minutes", type=float, default=320)
    ap.add_argument("--require-complete", action="store_true", help="exit 1 if the crawl is not finished")
    ap.add_argument("stage", choices=["search", "filings", "progress"])
    args = ap.parse_args(argv)
    db = connect(args.db)
    deadline = time.monotonic() + 60 * args.budget_minutes
    complete = True
    if args.stage != "progress":
        client = Client(os.environ.get("COMPANIES_HOUSE_API_KEY", "").strip())
        if args.stage == "search":
            complete = crawl_search(db, client, deadline)
        else:
            complete = crawl_filings(db, client, deadline)
        print(f"{args.stage}: {client.calls} requests, {'complete' if complete else 'stopped at the time budget'}")
    state = progress(db)
    print(json.dumps(state, indent=1))
    # An unfinished crawl is resumed by the next job, so it is an error only when asked.
    return 1 if args.require_complete and not state["crawl_complete"] else 0


if __name__ == "__main__":
    sys.exit(main())
