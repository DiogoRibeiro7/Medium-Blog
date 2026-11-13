"""Linked Forbes lists: who enters, who leaves, and how far from $1 billion (Part 4).

Part 3 conjectured that most year-to-year changes in a country's billionaire
count are fortunes near the $1 billion threshold crossing it. Testing that
needs each year's list linked to the next, which the country totals of
``fetch_forbes_history.py`` cannot provide. This script requests the same
annual-list endpoint for every year, links people across years in memory and
writes only aggregate counts. No person-level record is written.

People are linked by ``uri``; when the ``uri`` changes, by ``naturalId``; when
both change, by a normalized name together with the birth date. The manifest
records how many links each rule made.

Outputs (all under ``--output-dir``):

``forbes_bands.csv.gz``
    year, country_label (citizenship), band_lo_musd, n, worth_musd. Bands of
    $100 million from $1 to $3 billion, then wider (``BAND_EDGES``).

``forbes_flows.csv.gz``
    year (the earlier list of the pair), country_label, flow, band_lo_musd, n,
    n_linked_elsewhere. ``flow`` is ``stay`` (on both lists with the same
    citizenship, banded at year t), ``exit`` (on list t but not t+1, banded at
    t), ``entry`` (on list t+1 but not t, banded at t+1), ``switch_out`` and
    ``switch_in`` (on both lists with a different citizenship label). For
    ``entry`` the last column counts people who had been on an earlier list;
    for ``exit`` it counts people who reappear on a later list.

``forbes_global.csv.gz``
    year, table, key, band_lo_musd, n. Global tables without countries:
    ``logchange`` (stayers' change in log10 net worth, bins of 0.02, by band at
    t), ``exit_age`` and ``stay_age`` (age at t in bands, by band at t), and
    ``entry_selfmade`` (entrants by the ``selfMade`` flag, by band at t+1).

``forbes_flows_manifest.json``
    Per year: URL, bytes, SHA-256 of the raw response, people, missing fields,
    values of ``status``; per pair: links by rule and counts by flow.

Run in GitHub Actions (the analysis workspace cannot reach forbes.com)::

    python fetch_forbes_flows.py --first 2010 --last 2026 --output-dir data
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import re
import sys
import time
import unicodedata
from bisect import bisect_right
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "billionaires-poverty-audit"))
from forbes_pages import extract_people  # noqa: E402
from resilient_fetch import fetch_bytes  # noqa: E402

URL = "https://www.forbes.com/forbesapi/person/billionaires/{year}/position/true.json?limit=5000"

BAND_EDGES = [0] + list(range(1000, 3000, 100)) + [3000, 3500, 4000, 5000, 6000, 8000, 10000, 15000,
                                                   20000, 30000, 50000, 100000]
FLOW_EDGES = [0, 1000, 1100, 1200, 1300, 1400, 1500, 1750, 2000, 2500, 3000, 4000, 5000, 7500, 10000, 20000]
LOG_BIN = 0.02
LOG_RANGE = 1.0  # changes beyond +/-1 in log10 go to the end bins
AGE_EDGES = [0, 50, 60, 70, 80, 85, 90]


def band(value: float, edges: list[int]) -> int:
    return edges[max(bisect_right(edges, value) - 1, 0)]


def _label(value: Any) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _worth(person: dict[str, Any]) -> float | None:
    value = person.get("finalWorth")
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        return float(value)
    return None


def _name_key(person: dict[str, Any]) -> str | None:
    name = person.get("personName") or person.get("name")
    if not isinstance(name, str) or not name.strip():
        return None
    text = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()
    text = re.sub(r"&\s*family|\bfamily\b|\(.*?\)", " ", text)
    text = re.sub(r"[^a-z ]", " ", text)
    birth = person.get("birthDate")
    birth_key = str(int(birth)) if isinstance(birth, (int, float)) and not isinstance(birth, bool) else ""
    return " ".join(text.split()) + "|" + birth_key


def _age(person: dict[str, Any], year: int) -> int | None:
    age = person.get("age")
    if isinstance(age, (int, float)) and not isinstance(age, bool) and 0 < age < 120:
        return int(age)
    birth = person.get("birthDate")  # milliseconds since the epoch
    if isinstance(birth, (int, float)) and not isinstance(birth, bool):
        try:
            born = datetime.fromtimestamp(birth / 1000, tz=timezone.utc).year
        except (OverflowError, OSError, ValueError):
            return None
        if 0 < year - born < 120:
            return year - born
    return None


def _self_made(person: dict[str, Any]) -> str:
    value = person.get("selfMade")
    return "yes" if value is True else "no" if value is False else "unknown"


class Person:
    __slots__ = ("uri", "natural", "name", "worth", "country", "age", "self_made")

    def __init__(self, record: dict[str, Any], year: int):
        self.uri = _label(record.get("uri"))
        self.natural = _label(record.get("naturalId"))
        self.name = _name_key(record)
        self.worth = _worth(record)
        self.country = _label(record.get("countryOfCitizenship"))
        self.age = _age(record, year)
        self.self_made = _self_made(record)


def link(before: list[Person], after: list[Person]) -> tuple[dict[int, int], Counter[str]]:
    """Map indices of ``before`` to indices of ``after``; one-to-one, by rule order."""
    links: dict[int, int] = {}
    used: set[int] = set()
    rules: Counter[str] = Counter()
    for rule in ("uri", "natural", "name"):
        index: dict[str, list[int]] = {}
        for j, person in enumerate(after):
            key = getattr(person, rule)
            if key and j not in used:
                index.setdefault(key, []).append(j)
        seen = Counter(getattr(p, rule) for i, p in enumerate(before) if i not in links and getattr(p, rule))
        for i, person in enumerate(before):
            key = getattr(person, rule)
            if i in links or not key or len(index.get(key, [])) != 1 or seen[key] != 1:
                continue
            j = index[key][0]
            if j in used:
                continue
            links[i] = j
            used.add(j)
            rules[rule] += 1
    return links, rules


def _ever_keys(people: Iterable[Person]) -> set[str]:
    keys = set()
    for person in people:
        for rule in ("uri", "natural", "name"):
            value = getattr(person, rule)
            if value:
                keys.add(f"{rule}:{value}")
    return keys


def _keys(person: Person) -> set[str]:
    return {f"{rule}:{getattr(person, rule)}" for rule in ("uri", "natural", "name") if getattr(person, rule)}


def aggregate(lists: dict[int, list[Person]]) -> dict[str, Any]:
    """All aggregate tables from linked yearly lists (pure function, used by the tests)."""
    years = sorted(lists)
    bands, flows, glob = [], [], []
    pairs: dict[str, Any] = {}
    for year in years:
        cells: Counter[tuple[str, int]] = Counter()
        worth: Counter[tuple[str, int]] = Counter()
        for person in lists[year]:
            if person.country is None or person.worth is None:
                continue
            key = (person.country, band(person.worth, BAND_EDGES))
            cells[key] += 1
            worth[key] += person.worth
        bands += [{"year": year, "country_label": c, "band_lo_musd": b, "n": k, "worth_musd": worth[(c, b)]}
                  for (c, b), k in sorted(cells.items())]
    for year, nxt in zip(years, years[1:]):
        if nxt != year + 1:
            continue
        before, after = lists[year], lists[nxt]
        links, rules = link(before, after)
        earlier_than_t = _ever_keys(p for y in years if y < year for p in lists[y])
        later = _ever_keys(p for y in years if y > nxt for p in lists[y])
        flow: Counter[tuple[str, str, int]] = Counter()
        elsewhere: Counter[tuple[str, str, int]] = Counter()
        logchange: Counter[tuple[str, int, int]] = Counter()
        age_tab: Counter[tuple[str, int, int]] = Counter()
        selfmade: Counter[tuple[str, int]] = Counter()
        linked_after = set(links.values())
        for i, person in enumerate(before):
            if person.worth is None:
                continue
            b_flow = band(person.worth, FLOW_EDGES)
            age_key = band(person.age, AGE_EDGES) if person.age is not None else -1
            if i in links:
                other = after[links[i]]
                if person.country == other.country and person.country is not None:
                    flow[(person.country, "stay", b_flow)] += 1
                else:
                    if person.country is not None:
                        flow[(person.country, "switch_out", b_flow)] += 1
                    if other.country is not None and other.worth is not None:
                        flow[(other.country, "switch_in", band(other.worth, FLOW_EDGES))] += 1
                age_tab[("stay_age", age_key, b_flow)] += 1
                if other.worth is not None and other.worth > 0 and person.worth > 0:
                    change = math.log10(other.worth / person.worth)
                    clipped = max(-LOG_RANGE, min(LOG_RANGE, change))
                    lo = math.floor(round(clipped / LOG_BIN, 9)) * LOG_BIN
                    logchange[("logchange", int(round(lo * 1000)), b_flow)] += 1
            else:
                if person.country is not None:
                    flow[(person.country, "exit", b_flow)] += 1
                    if _keys(person) & later:
                        elsewhere[(person.country, "exit", b_flow)] += 1
                age_tab[("exit_age", age_key, b_flow)] += 1
        for j, person in enumerate(after):
            if j in linked_after or person.worth is None:
                continue
            b_flow = band(person.worth, FLOW_EDGES)
            if person.country is not None:
                flow[(person.country, "entry", b_flow)] += 1
                if _keys(person) & earlier_than_t:
                    elsewhere[(person.country, "entry", b_flow)] += 1
            selfmade[(person.self_made, b_flow)] += 1
        flows += [{"year": year, "country_label": c, "flow": f, "band_lo_musd": b, "n": k,
                   "n_linked_elsewhere": elsewhere.get((c, f, b), 0)}
                  for (c, f, b), k in sorted(flow.items())]
        glob += [{"year": year, "table": t, "key": k, "band_lo_musd": b, "n": n}
                 for (t, k, b), n in sorted(logchange.items()) + sorted(age_tab.items())]
        glob += [{"year": year, "table": "entry_selfmade", "key": k, "band_lo_musd": b, "n": n}
                 for (k, b), n in sorted(selfmade.items())]
        totals: Counter[str] = Counter()
        for (_, f, _), k in flow.items():
            totals[f] += k
        pairs[f"{year}-{nxt}"] = {"links_by_rule": dict(rules), "n_before": len(before), "n_after": len(after),
                                  "flows": dict(totals)}
    return {"bands": bands, "flows": flows, "global": glob, "pairs": pairs}


def _write(table: list[dict[str, Any]], columns: list[str], path: Path) -> str:
    frame = pd.DataFrame(table, columns=columns)
    data = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
    path.write_bytes(gzip.compress(data, compresslevel=9, mtime=0))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch(first: int, last: int, output_dir: Path, fetcher: Callable[..., bytes] = fetch_bytes,
          pause: float = 3.0) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {"retrieved_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                "url_pattern": URL, "band_edges_musd": BAND_EDGES, "flow_edges_musd": FLOW_EDGES,
                                "logchange_bin": LOG_BIN, "age_edges": AGE_EDGES, "years": {}}
    lists: dict[int, list[Person]] = {}
    for year in range(first, last + 1):
        url = URL.format(year=year)
        entry: dict[str, Any] = {"url": url}
        try:
            raw = fetcher(url, name=f"forbes_{year}", timeout_seconds=180, max_attempts=3, wait_seconds=15)
            entry.update(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
            records, path = extract_people(json.loads(raw))
            entry["json_path"] = path
            people = [Person(r, year) for r in records]
            years_seen = Counter(r.get("year") for r in records)
            entry.update(
                n_persons=len(people),
                year_field_values={str(k): v for k, v in years_seen.most_common(3)},
                missing_uri=sum(p.uri is None for p in people),
                missing_natural=sum(p.natural is None for p in people),
                missing_worth=sum(p.worth is None for p in people),
                missing_citizenship=sum(p.country is None for p in people),
                missing_age=sum(p.age is None for p in people),
                below_1000=sum(p.worth is not None and p.worth < 1000 for p in people),
                status_values=dict(Counter(str(r.get("status")) for r in records).most_common(8)),
            )
            lists[year] = people
        except Exception as exc:  # one failed year should not stop the others
            entry["error"] = repr(exc)[:500]
        manifest["years"][str(year)] = entry
        print(f"{year}: {entry.get('n_persons', 'ERROR')} people"
              f"{'' if 'error' not in entry else ' ' + entry['error'][:160]}", flush=True)
        time.sleep(pause)
    tables = aggregate(lists)
    manifest["pairs"] = tables["pairs"]
    manifest["bands_sha256"] = _write(tables["bands"], ["year", "country_label", "band_lo_musd", "n", "worth_musd"],
                                      output_dir / "forbes_bands.csv.gz")
    manifest["flows_sha256"] = _write(tables["flows"], ["year", "country_label", "flow", "band_lo_musd", "n",
                                                        "n_linked_elsewhere"], output_dir / "forbes_flows.csv.gz")
    manifest["global_sha256"] = _write(tables["global"], ["year", "table", "key", "band_lo_musd", "n"],
                                       output_dir / "forbes_global.csv.gz")
    manifest["n_years_ok"] = len(lists)
    (output_dir / "forbes_flows_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--first", type=int, default=2010)
    parser.add_argument("--last", type=int, default=2026)
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    args = parser.parse_args(argv)
    manifest = fetch(args.first, args.last, args.output_dir)
    print(json.dumps({"years": {y: e.get("n_persons", e.get("error")) for y, e in manifest["years"].items()},
                      "pairs": manifest["pairs"]}, indent=1))
    return 0 if manifest["n_years_ok"] >= 2 else 1


if __name__ == "__main__":
    raise SystemExit(main())
