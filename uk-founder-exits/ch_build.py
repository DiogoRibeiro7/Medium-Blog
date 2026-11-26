"""Build the company-level cohort table from the API crawl and the PSC extract.

One row per private limited company incorporated from July 2016 to December 2020 with a
technology SIC code that allotted new shares (filed an SH01) within three years of incorporation.
The row records who the founders were (nationality of the individuals with significant control
notified at incorporation), when the company allotted shares, and the first date of each event
that can end its independence:

- foreign_acquisition: a majority corporate controller registered abroad, under a different name,
  or a UK corporate controller that is itself controlled from abroad;
- flip: a majority corporate controller registered abroad under a name like the company's own
  (a new foreign parent, often in Delaware, created by the founders themselves);
- uk_acquisition: a majority UK corporate controller under a different name, with no foreign
  controller above it;
- offshore: a majority corporate controller registered in the Crown dependencies or an offshore
  financial centre;
- closure: dissolution, or the first insolvency filing.

A UK holding company with the company's own name is a reorganisation, not an event: the holding
company's own controllers are followed instead. The analysis step picks the time origin and takes
the first event after it. No names of individuals are kept.

    python ch_build.py --db crawl.sqlite --psc psc.json.gz --out cohort.csv --report build_report.json
"""

import argparse
import csv
import gzip
import json
import re
import sys
from collections import Counter
from datetime import date, timedelta
from difflib import SequenceMatcher

import ch_api
from ch_bulk import is_uk_place, uk_registration_number

EVENTS = ["foreign_acquisition", "flip", "uk_acquisition", "offshore", "closure"]
FOUNDER_WINDOW = timedelta(days=30)
ENTRY_WINDOW = timedelta(days=3 * 365 + 1)
ROUND_WINDOW = timedelta(days=60)
MAJORITY = {
    "ownership-of-shares-50-to-75-percent",
    "ownership-of-shares-75-to-100-percent",
    "voting-rights-50-to-75-percent",
    "voting-rights-75-to-100-percent",
    "right-to-appoint-and-remove-directors",
}
OFFSHORE = re.compile(
    r"jersey|guernsey|isle of man|british virgin|\bbvi\b|cayman|bermuda|gibraltar|bahamas|anguilla|turks and caicos"
)
BRITISH = re.compile(r"british|english|scottish|welsh|northern irish|united kingdom|\buk\b|great britain|^gb$")
UK_HOME = re.compile(r"british|united kingdom|\buk\b|england|wales|scotland|northern ireland|great britain|^gb$")
EU_EEA = {
    "austrian", "belgian", "bulgarian", "croatian", "cypriot", "czech", "danish", "dutch", "estonian", "finnish",
    "french", "german", "greek", "hungarian", "irish", "italian", "latvian", "lithuanian", "luxembourger",
    "luxembourgish", "maltese", "polish", "portuguese", "romanian", "slovak", "slovakian", "slovenian", "spanish",
    "swedish", "norwegian", "icelandic", "swiss", "liechtensteiner",
}  # fmt: skip
NORTH_AMERICA = {"american", "canadian", "united states", "usa", "us citizen", "u.s. citizen", "citizen of the usa"}
INDIA = {"indian"}
CHINA = {"chinese", "hong konger", "hong kong", "taiwanese"}
LEGAL_WORDS = {
    "limited", "ltd", "plc", "inc", "incorporated", "llc", "corp", "corporation", "co", "company", "holdings",
    "holding", "holdco", "group", "topco", "bidco", "midco", "uk", "gmbh", "ag", "sa", "sas", "bv", "nv", "ab",
    "oy", "as", "pty", "the", "and", "technologies", "technology", "international",
}  # fmt: skip
SIC_GROUPS = {
    "software": {"58210", "58290", "62011", "62012"},
    "data and web": {"63110", "63120"},
    "research": {"72110", "72190"},
    "hardware and health": {"21100", "21200", "32500"} | {c for c in ch_api.TECH_SIC if c.startswith("26")},
}
LONDON = re.compile(r"^(E|EC|N|NW|SE|SW|W|WC)\d")
OXCAM = re.compile(r"^(OX|CB)\d")


def to_date(s):
    try:
        return date.fromisoformat((s or "")[:10])
    except ValueError:
        return None


def name_key(name):
    words = re.sub(r"[^a-z0-9 ]+", " ", (name or "").lower()).split()
    return " ".join(w for w in words if w not in LEGAL_WORDS)


def similar_names(a, b):
    """True when two company names differ only in legal form or generic words."""
    ka, kb = name_key(a), name_key(b)
    if not ka or not kb:
        return False
    if ka == kb or ka.startswith(kb + " ") or kb.startswith(ka + " "):
        return True
    return SequenceMatcher(None, ka, kb).ratio() >= 0.85


def nationality_group(nat):
    """Coarse group of a nationality string as typed on the register; dual nationals with British count as British."""
    n = (nat or "").strip().lower()
    if not n:
        return "unknown"
    if BRITISH.search(n):
        return "British"
    tokens = {t.strip() for t in re.split(r"[,/;&]| and ", n)}
    if tokens & EU_EEA:
        return "EU and EEA"
    if tokens & NORTH_AMERICA:
        return "North America"
    if tokens & INDIA:
        return "India"
    if tokens & CHINA:
        return "China"
    return "other"


def is_corporate(r):
    return "corporate-entity-person" in r["kind"] or "legal-person-person" in r["kind"]


def majority(r):
    """More than half the shares or votes, or the right to appoint the board (including "-as-firm" forms)."""
    return any(n.startswith(m) for n in r.get("natures") or [] for m in MAJORITY)


def at_incorporation(r, inc):
    d = to_date(r.get("notified_on"))
    return d is not None and d <= inc + FOUNDER_WINDOW


def founders(records, inc):
    """Individuals with significant control notified within 30 days of incorporation."""
    return [r for r in records if r["kind"].startswith("individual-person") and at_incorporation(r, inc)]


def place_of(r):
    place = " ".join(str(r.get(k) or "") for k in ("country_registered", "place_registered", "legal_authority")).lower()
    if OFFSHORE.search(place):
        return "offshore"
    if is_uk_place(r):
        return "uk"
    return "foreign" if place.strip() else "unknown"


def controller_events(records, inc, after=None):
    """Majority corporate controllers notified more than 30 days after incorporation, in date order."""
    out = []
    for r in records or []:
        if not (is_corporate(r) and majority(r)):
            continue
        d = to_date(r.get("notified_on"))
        if d is None or d <= inc + FOUNDER_WINDOW or (after and d < after):
            continue
        out.append((d, r))
    return sorted(out, key=lambda t: t[0])


def controlled_from_abroad(records, at, lookup, depth=0):
    """Was a UK company controlled, at date `at` or within a year after, by a company registered abroad?

    Follows UK corporate controllers upwards through the records available in `lookup`.
    """
    for r in records or []:
        if not (is_corporate(r) and majority(r)):
            continue
        start, end = to_date(r.get("notified_on")), to_date(r.get("ceased_on"))
        if not (start and start <= at + timedelta(days=365) and (end is None or end >= at)):
            continue
        place = place_of(r)
        if place in ("foreign", "offshore"):
            return True
        if place == "uk" and depth < 3:
            reg = uk_registration_number(r)
            if reg and controlled_from_abroad(lookup.get(reg), at, lookup, depth + 1):
                return True
    return False


def classify(company_name, d, r, lookup, depth=0):
    """Event type and date for one majority corporate controller of a cohort company."""
    place, same = place_of(r), similar_names(company_name, r.get("name"))
    if place == "foreign":
        return ("flip" if same else "foreign_acquisition"), d
    if place == "offshore":
        return "offshore", d
    if place == "uk":
        reg = uk_registration_number(r)
        if same:
            if depth == 0:
                # Reorganisation under a UK holding company: the holding company's controllers decide.
                for d2, r2 in controller_events(lookup.get(reg), date.min, after=d):
                    kind, when = classify(company_name, d2, r2, lookup, depth=1)
                    if kind != "reorganisation":
                        return kind, when
            return "reorganisation", d
        abroad = controlled_from_abroad(lookup.get(reg), d, lookup)
        return ("foreign_acquisition" if abroad else "uk_acquisition"), d
    return "unknown_controller", d


def region(postcode):
    pc = (postcode or "").strip().upper().replace(" ", "")
    if LONDON.match(pc):
        return "London"
    if OXCAM.match(pc):
        return "Oxford and Cambridge"
    return "rest of UK" if pc else "unknown"


def sic_group(codes):
    for g, s in SIC_GROUPS.items():
        if set(codes) & s:
            return g
    return "other"


def filing_dates(items, kind=None, category=None):
    out = []
    for i in items:
        if (kind and i.get("type") != kind) or (category and i.get("category") != category):
            continue
        d = to_date(i.get("action_date")) or to_date(i.get("dv_date")) or to_date(i.get("date"))
        if d:
            out.append(d)
    return sorted(out)


def founder_group(nationalities):
    """British (all known founders British), non-British (none), mixed; and the origin of non-British teams."""
    groups = Counter(nationality_group(n) for n in nationalities)
    groups.pop("unknown", None)
    known = sum(groups.values())
    if not known:
        return None, None
    british = groups.get("British", 0)
    if british == known:
        return "British", "British"
    if british:
        return "mixed", "mixed"
    return "non-British", next(iter(groups)) if len(groups) == 1 else "several"


def build_row(num, s, items, records, lookup, censor):
    """One cohort row, or (None, reason) when the company is left out."""
    inc = to_date(s.get("date_of_creation"))
    if inc is None:
        return None, "no incorporation date"
    allot = [d for d in filing_dates(items, kind="SH01") if inc <= d <= inc + ENTRY_WINDOW]
    if not allot:
        return None, "no allotment of shares within three years"
    if any(is_corporate(r) and majority(r) and at_incorporation(r, inc) for r in records):
        return None, "majority corporate controller at incorporation"
    f = founders(records, inc)
    if not f:
        return None, "no individual founder recorded"
    group, origin = founder_group([x.get("nationality") for x in f])
    if group is None:
        return None, "founder nationality not recorded"

    first = {}
    for d, r in controller_events(records, inc):
        kind, when = classify(s.get("company_name"), d, r, lookup)
        if kind in EVENTS:
            first[kind] = min(first.get(kind, when), when)
    resolutions = filing_dates(items, category="resolution")
    closure = [d for d in [to_date(s.get("date_of_cessation")), *filing_dates(items, category="insolvency")] if d]
    if closure:
        first["closure"] = min(closure)
    rounds = [d for d in allot if any(abs((d - x).days) <= ROUND_WINDOW.days for x in resolutions)]
    later = [d for d in filing_dates(items, kind="SH01") if d > allot[0] + timedelta(days=365)]
    abroad = any((x.get("residence") or "").strip() and not UK_HOME.search(x["residence"].lower()) for x in f)
    row = {
        "company_number": num,
        "incorporated": inc.isoformat(),
        "censor": censor.isoformat(),
        "entry": allot[0].isoformat(),
        "round_entry": rounds[0].isoformat() if rounds else "",
        **{e: first[e].isoformat() if e in first else "" for e in EVENTS},
        "status": s.get("company_status"),
        "cohort_year": inc.year,
        "sic_group": sic_group(s.get("sic_codes") or []),
        "region": region(s.get("postal_code")),
        "founders": len(f),
        "founder_group": group,
        "founder_origin": origin,
        "founder_abroad": int(abroad),
        "allotments_first_year": sum(1 for d in allot if d <= allot[0] + timedelta(days=365)),
        "next_allotment": later[0].isoformat() if later else "",
    }
    return row, None


def build(search, filings, psc, censor):
    """Rows of the cohort table and a Counter of the reasons companies were left out."""
    rows, why = [], Counter()
    lookup = {**psc["parents"], **psc["cohort"]}
    for num, s in sorted(search.items()):
        if num not in filings:
            why["filing history not fetched"] += 1
            continue
        row, reason = build_row(num, s, filings[num], psc["cohort"].get(num, []), lookup, censor)
        if row is None:
            why[reason] += 1
        else:
            rows.append(row)
    return rows, why


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--psc", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", help="also write the aggregate build report to this JSON file")
    args = ap.parse_args(argv)
    db = ch_api.connect(args.db)
    search = ch_api.cohort_companies(db)
    filings = {c: json.loads(b) for c, _, st, b in db.execute("select * from filings") if st in (200, 404)}
    with gzip.open(args.psc, "rt", encoding="utf-8") as fh:
        psc = json.load(fh)
    censor = to_date(re.search(r"\d{4}-\d{2}-\d{2}", psc["snapshot"]).group(0))
    rows, why = build(search, filings, psc, censor)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    capped = db.execute("select count(*) from search where status = 206").fetchone()[0]
    report = {
        "snapshot": psc["snapshot"],
        "censor_date": censor.isoformat(),
        "searched_companies": len(search),
        "capped_search_chunks": capped,
        "cohort_rows": len(rows),
        "excluded": dict(why.most_common()),
        "founder_groups": dict(Counter(r["founder_group"] for r in rows).most_common()),
        "first_events_recorded": {e: sum(1 for r in rows if r[e]) for e in EVENTS},
    }
    print(json.dumps(report, indent=1))
    if args.report:
        with open(args.report, "w") as fh:
            json.dump(report, fh, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
