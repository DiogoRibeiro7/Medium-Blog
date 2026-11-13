"""Where large arrivals on the Forbes list come from: families, co-owners and returns (Part 7).

Part 4 conjectured that most arrivals far above the $1 billion line are new
valuations of fortunes that already existed (listings, funding rounds) or
divisions of existing fortunes among heirs, rather than fortunes created within
the year. This script requests the same annual lists as Parts 4 to 6, links
them in memory with Part 4's rules, and for every entrant and every departure
records which kinds of link to other people on the lists exist. Only counts by
cell are written; no name, identifier or company is stored.

Family links are computed in memory under four rules of decreasing strictness,
because Forbes labels many old family fortunes with generic words ("diversified",
"investments") and common surnames link unrelated people by chance:

* strict: same country of citizenship, a shared surname token (Forbes's
  ``lastName``; particles such as "de" or "van" ignored) and the same specific
  company label (Forbes's ``organization``, else its ``source``, normalised,
  and not a sector word or a label carried by many unrelated surnames);
* label: as strict, but the label may be generic;
* rare: same country and a shared surname token that at most ``RARE`` people
  in that country carry over all the lists;
* loose: same country and a shared surname token.

A company link is the same specific label, any surname or country. A record of
the same person (a shared linking key or the same normalised name) is never
counted as a relative. Forbes's ``familyList`` turned out to be a yes/no field,
so it links no one.

Entrants on list t+1 who were not on list t (``year`` = t, as in Part 4's
flows). For rule k = 0..3 in RULES, bit 4k: a family member left at this or the
previous transition; 4k+1: a family member on both lists whose fortune fell by
0.1 or more in log10; 4k+2: a family member on both lists; 4k+3: placebo, a
family member who leaves two or three transitions later. Then E_WINDOW (the
placebo window exists), E_CO_ENTRY (another entrant with the same company
label), E_CO_RISE (a co-owner on both lists rose by 0.1 or more), E_CO_INC (a
co-owner on both lists), E_GENERIC (own label empty or generic), E_RELISTED
(someone with the same name or key left at this transition: the same person
relisted under a key Part 4's linking did not match).

People on list t who are not on list t+1. For rule k, bit 4k: a family member
enters at this or the next transition; 4k+1: a family member on both lists rose
by 0.1 or more; 4k+2: a family member on both lists; 4k+3: placebo, a family
member entered three or four transitions earlier. Then X_WINDOW, X_NEXT (the
next transition exists), X_RETURNS (the person appears on a later list),
X_RELISTED.

Output ``forbes_arrivals.csv.gz`` with columns year, table, band_lo_musd, tenure,
self_made, age, region, links (the bits above), n. Tables: ``entry`` (band and
age on list t+1), ``exit`` (band and age on list t) and ``stock`` (everyone on
list t; bit k = another family member on the same list under rule k). Manifest
``forbes_arrivals_manifest.json``: raw-response hashes, whether each is
byte-identical to the response Part 6 used, and aggregate diagnostics of the
fields used (coverage of ``lastName``, ``organization`` and ``source``, the type
of ``familyList``, the number of generic labels and of rare surnames).

Run in GitHub Actions::

    python fetch_forbes_arrivals.py --first 2010 --last 2025 --output-dir data
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

from fetch_forbes_flows import FLOW_EDGES, URL, Person, _keys, _write, band, extract_people, fetch_bytes, link

COLUMNS = ["year", "table", "band_lo_musd", "tenure", "self_made", "age", "region", "links", "n"]
# entrant bits: for rule k in RULES (strict, label, rare, loose), bit 4k = a family member left at this or the
# previous transition, 4k+1 = a family incumbent's fortune fell by 0.1 or more, 4k+2 = a family incumbent, 4k+3 =
# placebo (a family member leaves two or three transitions later)
E_WINDOW, E_CO_ENTRY, E_CO_RISE, E_CO_INC, E_GENERIC, E_RELISTED = 1 << 16, 1 << 17, 1 << 18, 1 << 19, 1 << 20, 1 << 21
# departure bits: for rule k, 4k = a family member enters at this or the next transition, 4k+1 = a family incumbent
# rose by 0.1 or more, 4k+2 = a family incumbent, 4k+3 = placebo (a family member entered three or four transitions
# earlier)
X_WINDOW, X_NEXT, X_RETURNS, X_RELISTED = 1 << 16, 1 << 17, 1 << 18, 1 << 19
CHANGE = 0.1
RARE = 5                      # a surname carried by at most this many people in a country over all lists is "rare"
GENERIC_SURNAMES = 8          # a label carried by this many distinct surnames on one list is a sector, not a company
PARTICLES = {"de", "da", "do", "dos", "das", "del", "della", "di", "du", "van", "von", "der", "den", "ter", "la", "le",
             "al", "el", "bin", "ibn", "binti", "family", "and", "jr", "sr", "ii", "iii", "iv", "mc", "st"}
SUFFIXES = {"inc", "ltd", "llc", "plc", "corp", "corporation", "co", "company", "group", "holdings", "holding", "sa",
            "ag", "nv", "spa", "gmbh", "se", "limited", "the", "and"}
SECTOR_WORDS = {
    "real estate", "investments", "investment", "diversified", "diversified investments", "hedge funds", "hedge fund",
    "private equity", "manufacturing", "construction", "banking", "finance", "retail", "retailing",
    "pharmaceuticals", "pharmaceutical", "software", "media", "mining", "oil", "oil gas", "gas", "shipping",
    "chemicals", "hotels", "casinos", "beverages", "food", "fashion", "textiles", "consumer goods", "logistics",
    "telecom", "telecoms", "technology", "internet", "e commerce", "ecommerce", "healthcare", "health care",
    "medical devices", "electronics", "steel", "coal", "metals", "energy", "agriculture", "agribusiness",
    "insurance", "venture capital", "money management", "asset management", "entertainment", "sports",
    "sports team", "sports teams", "gaming", "video games", "semiconductors", "cosmetics", "jewelry", "restaurants",
    "auto parts", "automobiles", "cars", "airlines", "packaging", "paper", "cement", "fintech", "cryptocurrency",
    "stock brokerage", "trading", "commodities", "supermarkets", "consumer products", "building materials",
    "infrastructure", "engineering", "education", "biotech", "biotechnology", "pharmacies", "fashion retail",
    "batteries", "solar", "lithium", "beer", "wine", "spirits", "tobacco", "dairy", "poultry", "meat", "seafood",
    "furniture", "appliances", "machinery", "aluminum", "copper", "gold", "iron ore", "nickel", "glass", "plastics",
    "fertilizer", "fertilizers", "power", "utilities", "electricity", "airports", "ports", "toys", "apparel", "shoes",
    "footwear", "retail stores", "department stores", "convenience stores", "e-commerce", "online retail",
    "medical equipment", "hospitals", "pharmacy", "investing", "inheritance", "self made", "philanthropy",
}
EUROPE = {"Austria", "Belgium", "Bulgaria", "Croatia", "Cyprus", "Czech Republic", "Czechia", "Denmark", "Estonia",
          "Finland", "France", "Germany", "Greece", "Hungary", "Ireland", "Italy", "Latvia", "Lithuania", "Luxembourg",
          "Malta", "Netherlands", "Poland", "Portugal", "Romania", "Slovakia", "Slovenia", "Spain", "Sweden",
          "United Kingdom", "Switzerland", "Norway", "Iceland", "Liechtenstein", "Monaco"}


def _ascii(text: str) -> str:
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()


def _words(text: str) -> list[str]:
    return re.sub(r"[^a-z0-9 ]", " ", _ascii(text).replace("&", " and ")).split()


def surname_tokens(record: dict[str, Any]) -> frozenset[str]:
    last = record.get("lastName")
    if not (isinstance(last, str) and last.strip()):
        name = record.get("personName") or record.get("name")
        if not (isinstance(name, str) and name.strip()):
            return frozenset()
        cleaned = re.sub(r"&\s*family|\bfamily\b|\(.*?\)", " ", _ascii(name))
        words = [w for w in re.sub(r"[^a-z ]", " ", cleaned).split() if w not in PARTICLES]
        last = words[-1] if words else ""
    return frozenset(w for w in re.sub(r"[^a-z ]", " ", _ascii(last)).split() if len(w) >= 2 and w not in PARTICLES)


def company_label(record: dict[str, Any]) -> str | None:
    for field in ("organization", "source"):
        value = record.get(field)
        if isinstance(value, str) and value.strip():
            words = [w for w in _words(value) if w not in SUFFIXES]
            if words:
                return " ".join(words)
    return None


def _strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for v in value.values():
            yield from _strings(v)
    elif isinstance(value, list):
        for v in value:
            yield from _strings(v)


def _plain_name(record: dict[str, Any]) -> str | None:
    name = record.get("personName") or record.get("name")
    if not (isinstance(name, str) and name.strip()):
        return None
    cleaned = re.sub(r"&\s*family|\bfamily\b|\(.*?\)", " ", _ascii(name))
    words = re.sub(r"[^a-z ]", " ", cleaned).split()
    return " ".join(words) or None


def family_strings(record: dict[str, Any]) -> frozenset[str]:
    out = set()
    for s in _strings(record.get("familyList")):
        if not s.strip():
            continue
        out.add(s.strip().lower())
        words = re.sub(r"[^a-z ]", " ", _ascii(s)).split()
        if len(words) >= 2:
            out.add(" ".join(words))
    return frozenset(out)


def region(country: str | None) -> str:
    if country is None:
        return "unknown"
    if country == "United States":
        return "United States"
    if country in ("China", "Hong Kong", "Macau", "Macao"):
        return "China"
    if country == "India":
        return "India"
    return "Europe" if country in EUROPE else "Other"


def age_group(age: int | None) -> str:
    if age is None:
        return "unknown"
    return "<50" if age < 50 else "50-69" if age < 70 else "70-84" if age < 85 else "85+"


class Extra:
    __slots__ = ("tokens", "label", "fam", "plain", "uri")

    def __init__(self, record: dict[str, Any]):
        self.tokens = surname_tokens(record)
        self.label = company_label(record)
        self.fam = family_strings(record)
        self.plain = _plain_name(record)
        uri = record.get("uri")
        self.uri = uri.strip().lower() if isinstance(uri, str) and uri.strip() else None


def generic_labels(extras: dict[int, list[Extra]]) -> set[str]:
    out = {lab for lab in (" ".join(w for w in _words(s) if w not in SUFFIXES) for s in SECTOR_WORDS) if lab}
    for xs in extras.values():
        surnames: defaultdict[str, set[frozenset[str]]] = defaultdict(set)
        for x in xs:
            if x.label:
                surnames[x.label].add(x.tokens)
        out |= {lab for lab, s in surnames.items() if len(s) >= GENERIC_SURNAMES}
    return out


RULES = ("strict", "label", "rare", "loose")


class Index:
    """Lookups of family and company links within one list, under four matching rules of decreasing strictness."""

    def __init__(self, people: list[Person], extras: list[Extra], generic: set[str], rare: set[tuple]):
        self.loose: defaultdict[tuple, set[int]] = defaultdict(set)
        self.label: defaultdict[tuple, set[int]] = defaultdict(set)
        self.company: defaultdict[str, set[int]] = defaultdict(set)
        self.generic, self.rare = generic, rare
        for i, (p, x) in enumerate(zip(people, extras)):
            for tok in x.tokens:
                self.loose[(p.country, tok)].add(i)
                if x.label:
                    self.label[(p.country, tok, x.label)].add(i)
            if self.specific(x.label):
                self.company[x.label].add(i)

    def specific(self, label: str | None) -> bool:
        return bool(label) and label not in self.generic

    def family(self, p: Person, x: Extra, rule: str) -> set[int]:
        """strict: surname, country and a specific company label; label: surname, country and the same label even if
        generic; rare: surname and country, for surnames few people in that country carry; loose: surname and country."""
        if p.country is None:
            return set()
        out: set[int] = set()
        for tok in x.tokens:
            if rule == "strict":
                if self.specific(x.label):
                    out |= self.label.get((p.country, tok, x.label), set())
            elif rule == "label":
                if x.label:
                    out |= self.label.get((p.country, tok, x.label), set())
            elif rule == "rare":
                if (p.country, tok) in self.rare:
                    out |= self.loose.get((p.country, tok), set())
            else:
                out |= self.loose.get((p.country, tok), set())
        return out

    def same_company(self, x: Extra) -> set[int]:
        return set(self.company.get(x.label, set())) if self.specific(x.label) else set()


def same_person(p: Person, x: Extra, q: Person, z: Extra) -> bool:
    """Two records of the same person on different lists: a shared linking key or the same normalised name."""
    return bool(_keys(p) & _keys(q)) or (x.plain is not None and x.plain == z.plain)


def _change(a: Person, b: Person) -> float | None:
    if a.worth and b.worth and a.worth > 0 and b.worth > 0:
        return math.log10(b.worth / a.worth)
    return None


def rare_surnames(lists: dict[int, list[Person]], extras: dict[int, list[Extra]]) -> set[tuple]:
    """(country, surname token) pairs carried by at most RARE distinct people over all the lists."""
    people: defaultdict[tuple, set[str]] = defaultdict(set)
    for y in lists:
        for p, x in zip(lists[y], extras[y]):
            ident = x.uri or x.plain or p.name
            for tok in x.tokens:
                if p.country and ident:
                    people[(p.country, tok)].add(ident)
    return {k for k, v in people.items() if len(v) <= RARE}


def aggregate(lists: dict[int, list[Person]], extras: dict[int, list[Extra]]) -> tuple[list[dict[str, Any]], dict]:
    """Entry, exit and stock tables from linked lists (pure function, used by the tests)."""
    years = sorted(lists)
    generic = generic_labels(extras)
    rare = rare_surnames(lists, extras)
    idx = {y: Index(lists[y], extras[y], generic, rare) for y in years}
    links = {y: link(lists[y], lists[y + 1])[0] for y in years if y + 1 in lists}
    inverse = {y + 1: {j: i for i, j in m.items()} for y, m in links.items()}
    entrants = {y: {j for j in range(len(lists[y + 1])) if j not in inverse[y + 1]} for y in links}
    leavers = {y: {i for i in range(len(lists[y])) if i not in links[y]} for y in links}
    stayers_next = {y: set(inverse[y + 1]) for y in links}              # indices on list y+1 also on y
    change_next = {y: {j: _change(lists[y][i], lists[y + 1][j]) for j, i in inverse[y + 1].items()} for y in links}
    keys_by_year = {y: set().union(*(_keys(p) for p in lists[y])) if lists[y] else set() for y in years}
    counts: Counter[tuple] = Counter()

    def fam_on(y_list: int, members: set[int], p: Person, x: Extra, rule: str) -> set[int]:
        if y_list not in idx:
            return set()
        found = idx[y_list].family(p, x, rule) & members
        return {k for k in found if not same_person(p, x, lists[y_list][k], extras[y_list][k])}

    def relisted(y_list: int, members: set[int], p: Person, x: Extra) -> bool:
        return any(same_person(p, x, lists[y_list][k], extras[y_list][k]) for k in members
                   if lists[y_list][k].country == p.country) if y_list in idx else False

    for y in sorted(links):
        nxt = y + 1
        earlier = set().union(*(keys_by_year[s] for s in years if s < y)) if any(s < y for s in years) else set()
        window = y + 2 in leavers and y + 3 in leavers
        for j in sorted(entrants[y]):
            p, x = lists[nxt][j], extras[nxt][j]
            if p.worth is None:
                continue
            f = 0
            for k, rule in enumerate(RULES):
                base = 4 * k
                left = fam_on(y, leavers[y], p, x, rule)
                if y - 1 in leavers:
                    left |= fam_on(y - 1, leavers[y - 1], p, x, rule)
                if left:
                    f |= 1 << base
                inc = fam_on(nxt, stayers_next[y], p, x, rule)
                if any((change_next[y].get(m) or 0) <= -CHANGE for m in inc):
                    f |= 1 << (base + 1)
                if inc:
                    f |= 1 << (base + 2)
                if window and (fam_on(y + 2, leavers[y + 2], p, x, rule) or fam_on(y + 3, leavers[y + 3], p, x, rule)):
                    f |= 1 << (base + 3)
            if window:
                f |= E_WINDOW
            co = idx[nxt].same_company(x)
            if (co & entrants[y]) - {j}:
                f |= E_CO_ENTRY
            if any((change_next[y].get(m) or 0) >= CHANGE for m in co & stayers_next[y]):
                f |= E_CO_RISE
            if co & stayers_next[y]:
                f |= E_CO_INC
            if not idx[nxt].specific(x.label):
                f |= E_GENERIC
            if relisted(y, leavers[y], p, x):
                f |= E_RELISTED
            tenure = "returning" if _keys(p) & earlier else "new"
            counts[(y, "entry", band(p.worth, FLOW_EDGES), tenure, p.self_made, age_group(p.age), region(p.country),
                    f)] += 1
        later = set().union(*(keys_by_year[s] for s in years if s > nxt)) if any(s > nxt for s in years) else set()
        back = y - 3 in entrants and y - 4 in entrants
        for i in sorted(leavers[y]):
            p, x = lists[y][i], extras[y][i]
            if p.worth is None:
                continue
            f = 0
            for k, rule in enumerate(RULES):
                base = 4 * k
                if fam_on(nxt, entrants[y], p, x, rule) or (nxt in entrants and fam_on(nxt + 1, entrants[nxt], p, x, rule)):
                    f |= 1 << base
                inc = fam_on(nxt, stayers_next[y], p, x, rule)
                if any((change_next[y].get(m) or 0) >= CHANGE for m in inc):
                    f |= 1 << (base + 1)
                if inc:
                    f |= 1 << (base + 2)
                if back and (fam_on(y - 2, entrants[y - 3], p, x, rule) or fam_on(y - 3, entrants[y - 4], p, x, rule)):
                    f |= 1 << (base + 3)
            if back:
                f |= X_WINDOW
            if nxt in entrants:
                f |= X_NEXT
            if _keys(p) & later:
                f |= X_RETURNS
            if relisted(nxt, entrants[y], p, x):
                f |= X_RELISTED
            counts[(y, "exit", band(p.worth, FLOW_EDGES), "exit", p.self_made, age_group(p.age), region(p.country),
                    f)] += 1
    # stock: how often a person shares a family with someone else on the same list, by rule
    for y in years:
        everyone = set(range(len(lists[y])))
        for i, (p, x) in enumerate(zip(lists[y], extras[y])):
            if p.worth is None:
                continue
            f = sum(1 << k for k, rule in enumerate(RULES) if fam_on(y, everyone - {i}, p, x, rule))
            counts[(y, "stock", band(p.worth, FLOW_EDGES), "all", p.self_made, age_group(p.age), region(p.country), f)] += 1
    rows = [dict(zip(COLUMNS, k + (n,))) for k, n in sorted(counts.items(), key=lambda kv: tuple(str(v) for v in kv[0]))
            if n > 0]
    return rows, {"generic_labels": len(generic), "rare_surname_pairs": len(rare)}


def diagnostics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate description of the fields used; no values are kept except familyList's dict keys."""
    n = len(records)
    fam = [r.get("familyList") for r in records]
    nonempty = [v for v in fam if v not in (None, "", [], {})]
    kinds = Counter(type(v).__name__ for v in nonempty)
    elem = Counter(type(e).__name__ for v in nonempty if isinstance(v, list) for e in v)
    keys = Counter(k for v in nonempty if isinstance(v, list) for e in v if isinstance(e, dict) for k in e)
    filled = lambda f: sum(1 for r in records if isinstance(r.get(f), str) and r.get(f).strip()) / n if n else 0  # noqa: E731
    return {"n": n, "familyList_nonempty": len(nonempty) / n if n else 0, "familyList_types": dict(kinds),
            "familyList_element_types": dict(elem), "familyList_dict_keys": dict(keys.most_common(12)),
            "lastName_filled": filled("lastName"), "organization_filled": filled("organization"),
            "source_filled": filled("source"), "family_entries": sum(1 for r in records if "family" in str(
                r.get("personName") or "").lower()) / n if n else 0}


def fetch(first: int, last: int, output_dir: Path, fetcher: Callable[..., bytes] = fetch_bytes,
          pause: float = 3.0) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    previous = {}
    old = output_dir / "forbes_persistence_manifest.json"
    if old.exists():
        previous = {y: e.get("sha256") for y, e in json.loads(old.read_text(encoding="utf-8"))["years"].items()}
    manifest: dict[str, Any] = {"retrieved_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                "url_pattern": URL, "flow_edges_musd": FLOW_EDGES, "change": CHANGE,
                                "generic_surnames": GENERIC_SURNAMES, "years": {}}
    lists: dict[int, list[Person]] = {}
    extras: dict[int, list[Extra]] = {}
    for year in range(first, last + 1):
        url = URL.format(year=year)
        entry: dict[str, Any] = {"url": url}
        try:
            raw = fetcher(url, name=f"forbes_{year}", timeout_seconds=180, max_attempts=3, wait_seconds=15)
            sha = hashlib.sha256(raw).hexdigest()
            entry.update(bytes=len(raw), sha256=sha, same_as_part6=previous.get(str(year)) == sha)
            records, _ = extract_people(json.loads(raw))
            lists[year] = [Person(r, year) for r in records]
            extras[year] = [Extra(r) for r in records]
            entry["n_persons"] = len(lists[year])
            entry["fields"] = diagnostics(records)
        except Exception as exc:  # one failed year should not stop the others
            entry["error"] = repr(exc)[:500]
        manifest["years"][str(year)] = entry
        print(f"{year}: {entry.get('n_persons', 'ERROR')} people, same as Part 6: {entry.get('same_as_part6')}",
              flush=True)
        time.sleep(pause)
    rows, info = aggregate(lists, extras)
    manifest.update(info)
    manifest["arrivals_sha256"] = _write(rows, COLUMNS, output_dir / "forbes_arrivals.csv.gz")
    manifest["n_years_ok"] = len(lists)
    (output_dir / "forbes_arrivals_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--first", type=int, default=2010)
    parser.add_argument("--last", type=int, default=2025)
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    args = parser.parse_args(argv)
    manifest = fetch(args.first, args.last, args.output_dir)
    return 0 if manifest["n_years_ok"] >= 5 else 1


if __name__ == "__main__":
    raise SystemExit(main())
