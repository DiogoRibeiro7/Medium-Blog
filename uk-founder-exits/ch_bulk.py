"""Extract persons-with-significant-control (PSC) records from the Companies House bulk snapshot.

The snapshot covers live and dissolved companies. Two passes: the first keeps the records of the
cohort companies; the second keeps the records of the UK companies that took control of them, so
that a UK acquirer can be traced one level up to a foreign parent. Names of individuals and
dates of birth are dropped on read; only nationality and country of residence are kept.

    python ch_bulk.py --db crawl.sqlite --work-dir /tmp/ch --out psc.json.gz
"""

import argparse
import gzip
import io
import json
import re
import sys
import urllib.request
import zipfile
from pathlib import Path

import ch_api

DOWNLOAD = "https://download.companieshouse.gov.uk/"
USER_AGENT = "Mozilla/5.0 (compatible; Medium-Blog uk-founder-exits)"
SNAPSHOT = re.compile(r"persons-with-significant-control-snapshot-\d{4}-\d{2}-\d{2}\.zip")


def get(url, timeout=3600):
    return urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": USER_AGENT}), timeout=timeout)


def latest_snapshot():
    html = get(DOWNLOAD + "en_pscdata.html", timeout=120).read().decode("utf-8", "replace")
    names = sorted(set(SNAPSHOT.findall(html)))
    if not names:
        raise RuntimeError("no PSC snapshot link on the download page")
    return names[-1]


def download(name, target):
    with get(DOWNLOAD + name) as r, open(target, "wb") as f:
        while chunk := r.read(1 << 22):
            f.write(chunk)


def slim(d):
    """The fields the cohort needs from one PSC record."""
    ident = d.get("identification") or {}
    kind = d.get("kind", "")
    out = {
        "kind": kind,
        "notified_on": d.get("notified_on"),
        "ceased_on": d.get("ceased_on"),
        "natures": d.get("natures_of_control") or [],
    }
    if kind.startswith("individual"):
        out["nationality"] = d.get("nationality")
        out["residence"] = d.get("country_of_residence")
    elif "corporate" in kind or "legal-person" in kind:
        out["name"] = d.get("name")
        out["country_registered"] = ident.get("country_registered")
        out["place_registered"] = ident.get("place_registered")
        out["legal_authority"] = ident.get("legal_authority")
        out["registration_number"] = ident.get("registration_number")
    return out


def iter_records(zpath, wanted=None):
    with zipfile.ZipFile(zpath) as z, z.open(z.namelist()[0]) as raw:
        for line in io.TextIOWrapper(raw, encoding="utf-8", errors="replace"):
            if wanted is not None:
                m = NUMBER.search(line)
                if not m or m.group(1) not in wanted:
                    continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            num = rec.get("company_number")
            if num and (wanted is None or num in wanted):
                yield num, rec.get("data") or {}


NUMBER = re.compile(r'"company_number"\s*:\s*"([^"]+)"')


def extract(zpath, wanted):
    """company_number -> list of slim PSC records, for the wanted companies only.

    A regular expression screens each line before the JSON is parsed: most of the 11 million
    companies in the snapshot are not wanted, and parsing every line would dominate the run.
    """
    out = {}
    for num, d in iter_records(zpath, wanted):
        out.setdefault(num, []).append(slim(d))
    return out


def uk_controllers(records_by_company):
    """Company numbers of the UK-registered corporate controllers in a set of PSC records."""
    return {
        n
        for recs in records_by_company.values()
        for r in recs
        if ("corporate" in r["kind"] or "legal-person" in r["kind"]) and (n := uk_registration_number(r))
    }


def uk_registration_number(rec):
    """Company number of a UK-registered corporate PSC, normalised to Companies House form."""
    num = (rec.get("registration_number") or "").strip().upper().replace(" ", "")
    if not num or not is_uk_place(rec):
        return None
    if num.isdigit():
        return num.zfill(8)
    m = re.fullmatch(r"(SC|NI|OC|SO|NC|R|FC)(\d+)", num)
    return m.group(1) + m.group(2).zfill(8 - len(m.group(1))) if m else None


UK = re.compile(
    r"england|wales|scotland|united kingdom|\buk\b|u\.k\.|northern ireland|great britain|britain|"
    r"companies house|companies act|registrar of companies for|london|cardiff|edinburgh|belfast"
)


def is_uk_place(rec):
    place = " ".join(str(rec.get(k) or "") for k in ("country_registered", "place_registered", "legal_authority"))
    return bool(UK.search(place.lower()))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--work-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    cohort = set(ch_api.cohort_companies(ch_api.connect(args.db)))
    name = latest_snapshot()
    zpath = args.work_dir / "psc.zip"
    if not zpath.exists():
        download(name, zpath)
    first = extract(zpath, cohort)
    # Corporate controllers registered in the UK, followed up to three levels up: a cohort company bought
    # by a UK holding company that is itself owned from abroad counts as a foreign acquisition.
    parents, frontier = {}, first
    for _ in range(3):
        wanted = uk_controllers(frontier) - set(first) - set(parents)
        frontier = extract(zpath, wanted) if wanted else {}
        parents.update(frontier)
    second = parents
    payload = {"snapshot": name, "cohort": first, "parents": second}
    with gzip.open(args.out, "wt", encoding="utf-8") as f:
        json.dump(payload, f)
    print(
        f"{name}: PSC records for {len(first)} of {len(cohort)} cohort companies; "
        f"records for {len(second)} UK corporate controllers up to three levels above them"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
