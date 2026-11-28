"""Minimal, polite EDGAR client: rate-limited, cached on disk, descriptive User-Agent.

SEC fair-access rules: at most 10 requests per second and a User-Agent that
identifies the requester. Set EDGAR_USER_AGENT to "Name contact@email".
"""
import hashlib
import json
import os
import time
import urllib.request
import urllib.error

MIN_INTERVAL = 0.15  # seconds between requests (about 6 per second)


class Edgar:
    def __init__(self, cache_dir="cache", user_agent=None):
        self.ua = user_agent or os.environ.get("EDGAR_USER_AGENT")
        if not self.ua or "@" not in self.ua:
            raise RuntimeError(
                "EDGAR_USER_AGENT is empty or has no email. Set the repository VARIABLE "
                "(not a secret) EDGAR_USER_AGENT to 'Name contact@email'.")
        self.cache = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self._last = 0.0

    def _path(self, url):
        return os.path.join(self.cache, hashlib.sha256(url.encode()).hexdigest())

    def get(self, url, retries=7):
        p = self._path(url)
        if os.path.exists(p):
            with open(p, "rb") as f:
                return f.read()
        for attempt in range(retries):
            wait = MIN_INTERVAL - (time.time() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.time()
            req = urllib.request.Request(url, headers={"User-Agent": self.ua, "Accept-Encoding": "identity"})
            try:
                with urllib.request.urlopen(req, timeout=60) as r:
                    data = r.read()
                with open(p, "wb") as f:
                    f.write(data)
                return data
            except urllib.error.HTTPError as e:
                if (e.code == 429 or e.code >= 500) and attempt < retries - 1:
                    time.sleep(min(60, 2 ** attempt * 2))
                    continue
                if e.code == 404:
                    return None
                if e.code == 403:
                    raise RuntimeError(
                        f"SEC returned 403 for {url}. Check that EDGAR_USER_AGENT looks like "
                        f"'Name name@example.org' (current: {self.ua!r}).") from e
                raise
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
                if attempt < retries - 1:
                    time.sleep(min(60, 2 ** attempt * 2))
                    continue
                raise RuntimeError(f"network failure for {url}: {e}") from e
        return None

    def json(self, url):
        raw = self.get(url)
        return None if raw is None else json.loads(raw)

    # ---- endpoints -------------------------------------------------------
    def ticker_map(self):
        d = self.json("https://www.sec.gov/files/company_tickers.json")
        return {v["ticker"].upper(): int(v["cik_str"]) for v in d.values()}

    def submissions(self, cik):
        """All filings for a CIK as a list of dicts (recent block plus older files)."""
        d = self.json(f"https://data.sec.gov/submissions/CIK{cik:010d}.json")
        if d is None:
            return []
        blocks = [d["filings"]["recent"]]
        for extra in d["filings"].get("files", []):
            e = self.json("https://data.sec.gov/submissions/" + extra["name"])
            if e:
                blocks.append(e)
        rows = []
        for b in blocks:
            n = len(b["accessionNumber"])
            for i in range(n):
                rows.append({
                    "accession": b["accessionNumber"][i],
                    "filing_date": b["filingDate"][i],
                    "form": b["form"][i],
                    "primary_doc": b["primaryDocument"][i],
                    "items": b.get("items", [""] * n)[i] or "",
                })
        return rows

    @staticmethod
    def doc_url(cik, accession, primary_doc):
        return f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession.replace('-', '')}/{primary_doc}"
