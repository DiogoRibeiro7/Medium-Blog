"""Fill in what Part 9 needs beyond Part 7's extracts.

Part 7 summed the rows of the Johns Hopkins global time series by country. Eight countries have more than
one row: Australia, Canada and China are split into states, provinces and regions, and Denmark, France, the
Netherlands, New Zealand and the United Kingdom have a row for the country itself and rows for overseas
territories. A sum of rows is unchanged only when every row is, so for these eight the sum is not one
producer's file. This script reads, from the same 1,081 file versions Part 7 read (listed in
part7/data/jhu_versions.csv.gz), the rows of those eight countries one by one, and writes:

  jhu_global_rows.csv.gz  one row per date (24 March 2020 to 9 March 2023) and row of the eight countries:
                          from the first version of the global time series that contains the date, the
                          cumulative count of that date, of the day before and of seven days before, and the
                          count of the date in the final version.
  calendars.csv.gz        legal holidays of 2020 to 2023 from python-holidays for the calendars that Part 7
                          did not write: the provinces and territories of Canada, the states and territories
                          of Australia, and Hong Kong.
  weekends.csv            the weekend days python-holidays gives each calendar used in Part 9, by year
                          (0 = Monday ... 6 = Sunday).
  fetch_log.txt           the source, the pinned commit and the counts.

The repository is cloned without file contents and only the 1,082 versions are fetched (about 70 MB).

Usage: pip install holidays==0.108; python fetch_global.py [--workdir DIR]
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import io
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
PART7 = HERE.parent / "part7"
sys.path.insert(0, str(PART7))
import fetch_jhu as f7  # noqa: E402  (Part 7's clone and blob helpers, and its pinned commit)

SPLIT = ["Australia", "Canada", "China", "Denmark", "France", "Netherlands", "New Zealand", "United Kingdom"]
YEARS = range(2020, 2024)

CANADA = {"Alberta": "AB", "British Columbia": "BC", "Manitoba": "MB", "New Brunswick": "NB",
          "Newfoundland and Labrador": "NL", "Northwest Territories": "NT", "Nova Scotia": "NS", "Nunavut": "NU",
          "Ontario": "ON", "Prince Edward Island": "PE", "Quebec": "QC", "Saskatchewan": "SK", "Yukon": "YT"}
AUSTRALIA = {"Australian Capital Territory": "ACT", "New South Wales": "NSW", "Northern Territory": "NT",
             "Queensland": "QLD", "South Australia": "SA", "Tasmania": "TAS", "Victoria": "VIC",
             "Western Australia": "WA"}


def rows_by_version(blobs: f7.Blobs, oid: str) -> tuple[dict, dict]:
    reader = csv.reader(io.StringIO(blobs.read(oid)))
    head = next(reader)
    cols = {}
    for i, c in enumerate(head[4:], start=4):
        try:
            cols[dt.datetime.strptime(c, "%m/%d/%y").date()] = i
        except ValueError:
            pass
    out = {}
    for r in reader:
        if r[1] in SPLIT:
            out[(r[1], r[0])] = r
    return cols, out


def value(cols: dict, rows: dict, key, d: dt.date):
    i = cols.get(d)
    r = rows.get(key)
    if i is None or r is None or len(r) <= i or r[i] == "":
        return None
    return int(float(r[i]))


def global_rows(repo: Path) -> pd.DataFrame:
    v = pd.read_csv(PART7 / "data" / "jhu_versions.csv.gz")
    first = v[v.kind == "global_first"].set_index("date").blob.to_dict()
    final_oid = v[v.kind == "global_final"].blob.iloc[0]
    f7.fetch(repo, sorted(set(first.values())) + [final_oid])
    blobs = f7.Blobs(repo)
    fcols, frows = rows_by_version(blobs, final_oid)
    out, cache = [], {}
    for ds, oid in sorted(first.items()):
        d = dt.date.fromisoformat(ds)
        if oid not in cache:
            cache.clear()
            cache[oid] = rows_by_version(blobs, oid)
        cols, rows = cache[oid]
        for key in sorted(rows):
            cur = value(cols, rows, key, d)
            if cur is None:
                continue
            out.append((d, key[0], key[1], cur, value(cols, rows, key, d - dt.timedelta(days=1)),
                        value(cols, rows, key, d - dt.timedelta(days=7)), value(fcols, frows, key, d)))
    df = pd.DataFrame(out, columns=["date", "country", "province", "first", "prev_in_first", "week_in_first",
                                    "final"])
    for c in ("prev_in_first", "week_in_first", "final"):
        df[c] = df[c].astype("Int64")
    return df


def calendars() -> pd.DataFrame:
    import holidays

    rows = []
    for name, code in CANADA.items():
        rows += [(f"CA-{code}", d, n) for d, n in holidays.CA(subdiv=code, years=YEARS).items()]
    for name, code in AUSTRALIA.items():
        rows += [(f"AU-{code}", d, n) for d, n in holidays.AU(subdiv=code, years=YEARS).items()]
    rows += [("HK", d, n) for d, n in holidays.HK(years=YEARS).items()]
    return pd.DataFrame(rows, columns=["calendar", "date", "name"]).sort_values(["calendar", "date"])


def weekends(repo: Path) -> pd.DataFrame:
    """Weekend days for every calendar Part 9 uses: US states, countries (by Part 7's ISO lookup), and the
    calendars written above."""
    import holidays

    lookup = pd.read_csv(io.StringIO(f7.git(repo, "show", f"{f7.COMMIT}:{f7.LOOKUP}")),
                         keep_default_na=False, na_values=[""])
    iso = lookup[lookup.Province_State.isna()].set_index("Country_Region").iso2.to_dict()
    part7_cals = pd.read_csv(PART7 / "data" / "holidays.csv.gz").calendar.unique()
    rows = []
    for cal in part7_cals:
        for y in YEARS:
            if cal == "US" or cal.startswith("US-"):
                wk = holidays.US(years=y).weekend
            else:
                wk = holidays.country_holidays(iso[cal], years=y).weekend
            rows.append((cal, y, "".join(str(x) for x in sorted(wk))))
    for code in CANADA.values():
        rows += [(f"CA-{code}", y, "".join(str(x) for x in sorted(holidays.CA(subdiv=code, years=y).weekend)))
                 for y in YEARS]
    for code in AUSTRALIA.values():
        rows += [(f"AU-{code}", y, "".join(str(x) for x in sorted(holidays.AU(subdiv=code, years=y).weekend)))
                 for y in YEARS]
    rows += [("HK", y, "".join(str(x) for x in sorted(holidays.HK(years=y).weekend))) for y in YEARS]
    return pd.DataFrame(rows, columns=["calendar", "year", "weekend"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", default="/tmp/jhu-covid", type=Path)
    args = ap.parse_args()
    import holidays

    DATA.mkdir(exist_ok=True)
    repo = f7.clone(args.workdir)
    g = global_rows(repo)
    g.to_csv(DATA / "jhu_global_rows.csv.gz", index=False)
    cal = calendars()
    cal.to_csv(DATA / "calendars.csv.gz", index=False)
    wk = weekends(repo)
    wk.to_csv(DATA / "weekends.csv", index=False)
    with open(DATA / "fetch_log.txt", "w") as fh:
        fh.write(f"source {f7.REPO_URL} at commit {f7.COMMIT}\n")
        fh.write(f"global time series rows of {', '.join(SPLIT)}: {g[['country', 'province']].drop_duplicates().shape[0]} "
                 f"rows, {g.date.nunique()} dates, {len(g)} row-dates\n")
        fh.write(f"calendars: python-holidays {holidays.__version__}; {cal.calendar.nunique()} added "
                 f"(Canadian provinces and territories, Australian states and territories, Hong Kong); "
                 f"weekends for {wk.calendar.nunique()} calendars\n")


if __name__ == "__main__":
    main()
