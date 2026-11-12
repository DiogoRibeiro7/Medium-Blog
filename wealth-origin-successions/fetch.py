"""Detect founder-to-heir successions in the Forbes billionaire lists, aggregated by country and year.

    python fetch.py --out data

Every Forbes annual list from 2011 to 2026 is downloaded from the Forbes person-list API and
held in memory only. Between consecutive lists, a *succession* is a self-made billionaire who
leaves the list and, in the same country, at least one new inherited billionaire whose source of
wealth (Forbes' ``source`` field, normalised) is the same. Only country x year aggregates and
per-year matching diagnostics are written; no name, identifier or person-level row is saved.

Matching rules (all applied in memory):
- a person is identified across lists by whichever of Forbes' ``uri``, ``naturalId`` or the name
  keeps the most people from one list to the next (``data/forbes_coverage.csv`` reports the
  persistence of each); people whose identifier is missing or shared on a list are left out;
- an exit is a self-made person on list t who is absent from list t+1; exits who are back on
  list t+2 are discarded as temporary drop-offs;
- a new heir is an inherited person on list t+1 who was not on list t;
- sources are compared after lower-casing and removing punctuation and corporate suffixes; a
  source used by people of three or more citizenships, or by more than ``GENERIC_MAX`` people across
  the lists compared ("real estate", "investments", "hedge funds"), names a sector rather than a
  fortune and is not used for matching;
- ``strict`` matches heirs who appear on list t+1; ``window`` also accepts heirs who first appear
  on list t+2 (Forbes sometimes lists heirs a year after a founder's death).

The workspace where the analysis is written cannot reach forbes.com, so
``.github/workflows/wealth-origin-successions-data.yml`` runs this script on GitHub's runners and
commits the aggregates; ``data/fetch_log.txt`` records URL, status, size and SHA-256 of every
response. Growth, GDP and the composition table are copied from earlier snapshots
(data/PROVENANCE.txt).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

UA = {"User-Agent": "wealth-origin-successions/1.0 (research companion; dfr@esmad.ipp.pt)"}
FORBES = "https://www.forbes.com/forbesapi/person/billionaires/{year}/position/true.json?limit=5000"
YEARS = range(2011, 2027)
GENERIC_MAX = 30  # people sharing a source across the lists compared
GENERIC_COUNTRIES = 3  # a source used by people of three or more citizenships is a sector, not a fortune
OLD_AGE = 70

LOG: list[str] = []


def log(url: str, status: int | str, payload: bytes | None = None) -> None:
    size = sha = None
    if payload is not None:
        size, sha = len(payload), hashlib.sha256(payload).hexdigest()
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    LOG.append(f"{stamp}\t{status}\t{size}\t{sha}\t{url}")
    print(LOG[-1], flush=True)


def get(url: str, tries: int = 6) -> tuple[int, bytes]:
    last: Exception | None = None
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=300) as r:
                body = r.read()
                log(url, r.status, body)
                return r.status, body
        except urllib.error.HTTPError as e:
            if e.code == 429 or e.code >= 500:
                last = e
            else:
                body = e.read()
                log(url, e.code, body)
                return e.code, body
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            last = e
        time.sleep(15 * (attempt + 1))
    log(url, f"failed: {last}")
    raise RuntimeError(f"{url}: {last}")


def _persons(doc: dict) -> list[dict]:
    node = doc.get("personList", {})
    rows = node.get("personsLists") if isinstance(node, dict) else None
    if not isinstance(rows, list):
        raise ValueError("personList.personsLists not found")
    return [r for r in rows if isinstance(r, dict)]


def _bool(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, str) and v.lower() in {"true", "false"}:
        return v.lower() == "true"
    return None


def norm_source(s) -> str | None:
    if not isinstance(s, str) or not s.strip():
        return None
    s = re.sub(r"[^a-z0-9 ]+", " ", s.lower())
    s = re.sub(r"\b(inc|co|corp|corporation|ltd|llc|group|holdings?|the)\b", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s or None


def _age(p: dict, year: int) -> float | None:
    a = p.get("age")
    if isinstance(a, (int, float)) and 15 < a < 120:
        return float(a)
    b = p.get("birthDate")
    if isinstance(b, (int, float)) and b != 0:
        born = datetime.fromtimestamp(b / 1000, tz=timezone.utc).year
        if 1890 < born < year:
            return float(year - born)
    return None


KEY_FIELDS = ("uri", "naturalId", "name")


def _keys(p: dict) -> dict:
    out = {}
    for k in ("uri", "naturalId"):
        v = p.get(k)
        out[k] = v.strip() if isinstance(v, str) and v.strip() else None
    n = p.get("personName") or p.get("name")
    out["name"] = n.strip().lower() if isinstance(n, str) and n.strip() else None
    return out


def load_year(year: int) -> tuple[list, dict]:
    """Persons of one list, in memory: (keys, country, worth, self_made, source, age)."""
    url = FORBES.format(year=year)
    status, body = get(url)
    if status != 200:
        return [], {"year": year, "status": status}
    out = []
    for p in _persons(json.loads(body)):
        worth = p.get("finalWorth")
        if not isinstance(worth, (int, float)):
            continue
        out.append((_keys(p), str(p.get("countryOfCitizenship") or "").strip(), float(worth), _bool(p.get("selfMade")),
                    norm_source(p.get("source")), _age(p, year)))
    n = len(out)
    diag = {"year": year, "status": status, "n_persons": n,
            **{f"share_key_{k}": sum(r[0][k] is not None for r in out) / n if n else 0.0 for k in KEY_FIELDS},
            "share_source": sum(r[4] is not None for r in out) / n if n else 0.0,
            "share_age": sum(r[5] is not None for r in out) / n if n else 0.0,
            "share_selfmade_flag": sum(r[3] is not None for r in out) / n if n else 0.0}
    return out, diag


def persistence(raw: dict[int, list]) -> dict[str, float]:
    """Share of people on list t found again on list t+1, for each candidate identifier."""
    res = {}
    for k in KEY_FIELDS:
        shares = []
        for t in sorted(raw):
            if t + 1 in raw:
                a = {r[0][k] for r in raw[t] if r[0][k]}
                b = {r[0][k] for r in raw[t + 1] if r[0][k]}
                if a:
                    shares.append(len(a & b) / len(a))
        res[k] = float(np.mean(shares)) if shares else 0.0
    return res


def keyed(raw: dict[int, list], key: str) -> dict[int, dict]:
    """key -> (country, worth, self_made, source, age); people without the key or sharing it are dropped."""
    out = {}
    for t, rows in raw.items():
        seen, d = defaultdict(int), {}
        for r in rows:
            if r[0][key]:
                seen[r[0][key]] += 1
        for r in rows:
            k = r[0][key]
            if k and seen[k] == 1:
                d[k] = r[1:]
        out[t] = d
    return out


def detect(lists: dict[int, dict]) -> tuple[list[dict], list[dict]]:
    """Aggregate successions, unmatched exits and unmatched new heirs by citizenship x event year."""
    agg: dict[tuple[str, int], dict] = defaultdict(lambda: defaultdict(float))
    diags = []
    years = sorted(lists)
    carried: set = set()  # heirs already matched, through the window, to an exit on the previous pair
    for t in years:
        t1, t2 = t + 1, t + 2
        if t1 not in lists:
            continue
        L0, L1, L2 = lists[t], lists[t1], lists.get(t2)
        counts, countries = defaultdict(int), defaultdict(set)
        for L in (L0, L1) + ((L2,) if L2 else ()):
            for v in L.values():
                if v[3]:
                    counts[v[3]] += 1
                    countries[v[3]].add(v[0])
        generic = {s for s in counts if len(countries[s]) >= GENERIC_COUNTRIES or counts[s] > GENERIC_MAX}
        new_heirs = {k: v for k, v in L1.items() if v[2] is False and k not in L0}
        new_heirs2 = ({k: v for k, v in L2.items() if v[2] is False and k not in L1 and k not in L0}
                      if L2 else {})
        by_src1, by_src2 = defaultdict(list), defaultdict(list)
        for k, v in new_heirs.items():
            if v[3] and v[3] not in generic:
                by_src1[(v[0], v[3])].append(k)
        for k, v in new_heirs2.items():
            if v[3] and v[3] not in generic:
                by_src2[(v[0], v[3])].append(k)
        exits = {k: v for k, v in L0.items() if v[2] is True and k not in L1 and not (L2 and k in L2)}
        matched_heirs1, matched_heirs2 = set(), set()
        d = {"from_list": t, "to_list": t1, "exits": len(exits), "new_heirs": len(new_heirs),
             "generic_sources": len(generic), "window_available": bool(L2)}
        for k, (c, w, _, src, age) in exits.items():
            old = age is not None and age >= OLD_AGE
            h1 = by_src1.get((c, src), []) if src and src not in generic else []
            h2 = by_src2.get((c, src), []) if src and src not in generic else []
            a = agg[(c, t1)]
            if h1:
                a["n_succ_strict"] += 1
                a["worth_succ_strict"] += w
                a["heirs_worth_strict"] += sum(L1[h][1] for h in h1)
                a["n_heirs_strict"] += len(h1)
                a["n_succ_strict_old"] += old
                a["worth_succ_strict_old"] += w * old
                matched_heirs1.update(h1)
                counts_key = "matched_strict"
            elif h2:
                counts_key = "matched_window_only"
            else:
                a["n_exit_unmatched"] += 1
                a["worth_exit_unmatched"] += w
                a["n_exit_unmatched_old"] += old
                a["worth_exit_unmatched_old"] += w * old
                counts_key = "unmatched"
            if h1 or h2:
                a["n_succ_window"] += 1
                a["worth_succ_window"] += w
                matched_heirs2.update(h2)
            d[counts_key] = d.get(counts_key, 0) + 1
            d[counts_key + "_old"] = d.get(counts_key + "_old", 0) + old
            d["exits_old"] = d.get("exits_old", 0) + old
            d["exits_age_known"] = d.get("exits_age_known", 0) + (age is not None)
        for k, (c, w, _, src, age) in new_heirs.items():
            if k not in matched_heirs1 and k not in carried:
                a = agg[(c, t1)]
                a["n_heir_unmatched"] += 1
                a["worth_heir_unmatched"] += w
        d["new_heirs_matched_strict"] = len(matched_heirs1)
        d["new_heirs_next_list_matched"] = len(matched_heirs2)
        carried = matched_heirs2
        d["new_heirs_with_source"] = sum(1 for v in new_heirs.values() if v[3])
        d["new_heirs_generic_source"] = sum(1 for v in new_heirs.values() if v[3] in generic)
        diags.append(d)
    rows = [{"citizenship": c, "year": y, **{k: float(v) for k, v in vals.items()}} for (c, y), vals in agg.items()]
    return rows, diags


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    raw, cov = {}, []
    for y in YEARS:
        try:
            rows, diag = load_year(y)
        except Exception as e:  # keep going so the log shows every failure
            print(f"{y} failed: {e}", file=sys.stderr, flush=True)
            cov.append({"year": y, "status": f"failed: {e}"})
            continue
        cov.append(diag)
        if rows:
            raw[y] = rows
        time.sleep(3)
    pers = persistence(raw)
    key = max(pers, key=pers.get)  # the identifier under which people persist from list to list
    rows, diags = detect(keyed(raw, key))
    cov_df = pd.DataFrame(cov)
    for k, v in pers.items():
        cov_df[f"persistence_{k}"] = v
    cov_df["key_used"] = key
    cov_df.to_csv(args.out / "forbes_coverage.csv", index=False)
    pd.DataFrame(diags).to_csv(args.out / "succession_diagnostics.csv", index=False)
    pd.DataFrame(rows).sort_values(["year", "citizenship"]).to_csv(
        args.out / "succession_country_year.csv", index=False, float_format="%.3f")
    with open(args.out / "fetch_log.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(LOG) + "\n")
    missing = [y for y in YEARS if y not in raw]
    if [y for y in missing if y <= 2024]:
        print(f"lists missing: {missing}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
