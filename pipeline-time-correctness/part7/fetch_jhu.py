"""Read every version of two files of the Johns Hopkins CSSE COVID-19 repository that matter for Part 7,
and keep, for each morning, what a pipeline landing that morning's file would have seen:

  jhu_us_states.csv.gz   one row per daily US report (12 April 2020 to 9 March 2023) and state (50 states
                         and the District of Columbia): the confirmed count in the file as first published,
                         the count of the day before and of seven days before as they stood when the file
                         landed, the count in the final version of the file, the file's landing time (its
                         commit time), the number of versions the file went through, and the Last_Update
                         timestamp of the state's row in the new file and in the file of the day before.
  jhu_countries.csv.gz   one row per date (24 March 2020 to 9 March 2023) and country: from the first
                         version of the global time series that contains the date, the cumulative count of
                         that date, of the day before and of seven days before; the count of the date in
                         the final version; and the landing time of that first version.
  holidays.csv.gz        the public holidays of 2020 to 2023 used in the analysis, from the python-holidays
                         package: US federal holidays, the legal holidays of each state, and the public
                         holidays of each country in jhu_countries.csv.gz that the package covers.
  jhu_versions.csv.gz    the git blob id and commit time of every file version read, for audit.
  jhu_fetch_log.txt      the source, the pinned commit and the counts.

The repository is cloned without file contents (a blobless partial clone), and only the versions needed are
fetched, about 6,000 blobs and 70 MB. Run time is a few minutes.

Usage: pip install holidays==0.108; python fetch_jhu.py [--workdir DIR]
"""
from __future__ import annotations

import argparse
import bisect
import collections
import csv
import datetime as dt
import io
import re
import subprocess
from pathlib import Path

import pandas as pd

REPO_URL = "https://github.com/CSSEGISandData/COVID-19.git"
COMMIT = "4360e50239b4eb6b22f3a1759323748f36752177"  # last commit, 10 March 2023; the repository is archived
US_DIR = "csse_covid_19_data/csse_covid_19_daily_reports_us"
GLOBAL_DIR = "csse_covid_19_data/csse_covid_19_daily_reports"
GLOBAL_TS = "csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv"
LOOKUP = "csse_covid_19_data/UID_ISO_FIPS_LookUp_Table.csv"
GLOBAL_FIRST, GLOBAL_LAST = dt.date(2020, 3, 24), dt.date(2023, 3, 9)
YEARS = range(2020, 2024)
DATA = Path(__file__).resolve().parent / "data"

STATES = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas", "CA": "California", "CO": "Colorado",
    "CT": "Connecticut", "DE": "Delaware", "DC": "District of Columbia", "FL": "Florida", "GA": "Georgia",
    "HI": "Hawaii", "ID": "Idaho", "IL": "Illinois", "IN": "Indiana", "IA": "Iowa", "KS": "Kansas",
    "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine", "MD": "Maryland", "MA": "Massachusetts",
    "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi", "MO": "Missouri", "MT": "Montana",
    "NE": "Nebraska", "NV": "Nevada", "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico",
    "NY": "New York", "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma",
    "OR": "Oregon", "PA": "Pennsylvania", "RI": "Rhode Island", "SC": "South Carolina", "SD": "South Dakota",
    "TN": "Tennessee", "TX": "Texas", "UT": "Utah", "VT": "Vermont", "VA": "Virginia", "WA": "Washington",
    "WV": "West Virginia", "WI": "Wisconsin", "WY": "Wyoming",
}


def git(repo: Path, *args: str, check: bool = True) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=check).stdout


def clone(workdir: Path) -> Path:
    repo = workdir / "jhu-covid-19"
    if not (repo / ".git").exists():
        workdir.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "-q", "--filter=blob:none", "--no-checkout", REPO_URL, str(repo)], check=True)
    for key, value in (("gc.auto", "0"), ("diff.renames", "false")):
        git(repo, "config", key, value)
    git(repo, "cat-file", "-e", COMMIT)
    return repo


def fetch(repo: Path, oids: list[str]) -> None:
    """Fetch blobs in batches; a blobless clone would otherwise fetch them one at a time."""
    missing = [o for o in sorted(set(oids))
               if subprocess.run(["git", "-C", str(repo), "cat-file", "-e", o], capture_output=True,
                                 env={"GIT_NO_LAZY_FETCH": "1", "PATH": "/usr/bin:/bin:/usr/local/bin"}).returncode]
    for i in range(0, len(missing), 50):
        subprocess.run(["git", "-C", str(repo), "-c", "fetch.negotiationAlgorithm=noop", "fetch", "-q", "--no-tags",
                        "--no-write-fetch-head", "--filter=blob:none", "origin", *missing[i:i + 50]], check=True)


class Blobs:
    def __init__(self, repo: Path):
        self.p = subprocess.Popen(["git", "-C", str(repo), "cat-file", "--batch"], stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE)

    def read(self, oid: str) -> str:
        self.p.stdin.write((oid + "\n").encode())
        self.p.stdin.flush()
        size = int(self.p.stdout.readline().split()[2])
        data = self.p.stdout.read(size)
        self.p.stdout.read(1)
        return data.decode("utf-8-sig")


def history(repo: Path, path: str, diff_filter: str = "AM") -> list[tuple[int, str, str, str]]:
    """(commit time, commit, file name, blob) for every added or modified file under path, oldest first."""
    out, t, h = [], 0, ""
    log = git(repo, "log", COMMIT, "--no-renames", "--raw", "--no-abbrev", f"--diff-filter={diff_filter}",
              "--format=C %H %ct", "--", path)
    for line in log.splitlines():
        if line.startswith("C "):
            _, h, t = line.split()
        elif line.startswith(":"):
            meta, p = line.split("\t")
            out.append((int(t), h, p.rsplit("/", 1)[-1], meta.split()[3]))
    return sorted(out)


def file_date(name: str) -> dt.date | None:
    return dt.datetime.strptime(name[:10], "%m-%d-%Y").date() if re.fullmatch(r"\d{2}-\d{2}-\d{4}\.csv", name) else None


def us_states(repo: Path, blobs: Blobs, audit: list) -> pd.DataFrame:
    versions = collections.defaultdict(list)
    for t, h, name, oid in history(repo, US_DIR):
        d = file_date(name)
        if d:
            versions[d].append((t, oid))

    def at(d: dt.date, t: int):
        v = versions.get(d, [])
        i = bisect.bisect_right([x[0] for x in v], t) - 1
        return v[i] if i >= 0 else (None, None)

    plan = {}
    for d, v in versions.items():
        landed = v[0][0]
        plan[d] = dict(landed=landed, first=v[0][1], prev=at(d - dt.timedelta(days=1), landed)[1],
                       week=at(d - dt.timedelta(days=7), landed)[1], final=v[-1][1], n=len(v))
        for kind in ("first", "prev", "week", "final"):
            if plan[d][kind]:
                audit.append(("us_" + kind, d, plan[d][kind]))
    fetch(repo, [p[k] for p in plan.values() for k in ("first", "prev", "week", "final") if p[k]])

    cache: dict[str, dict] = {}

    def parse(oid):
        if oid is None:
            return {}
        if oid not in cache:
            rows = {}
            for r in csv.DictReader(io.StringIO(blobs.read(oid))):
                s, c = r.get("Province_State"), r.get("Confirmed")
                if s in STATES.values() and c not in (None, ""):
                    rows[s] = (int(float(c)), r.get("Last_Update"))
            cache[oid] = rows
        return cache[oid]

    rows = []
    for d in sorted(plan):
        p = plan[d]
        first, prev, week, final = parse(p["first"]), parse(p["prev"]), parse(p["week"]), parse(p["final"])
        for s in STATES.values():
            rows.append((d, s, first.get(s, (None,))[0], prev.get(s, (None,))[0], week.get(s, (None,))[0],
                         final.get(s, (None,))[0],
                         dt.datetime.fromtimestamp(p["landed"], dt.timezone.utc).strftime("%Y-%m-%d %H:%M"),
                         p["n"], first.get(s, (None, None))[1], prev.get(s, (None, None))[1]))
    df = pd.DataFrame(rows, columns=["date", "state", "first", "prev_at_landing", "week_at_landing", "final",
                                     "landed_utc", "n_versions", "last_update", "last_update_prev"])
    for c in ("first", "prev_at_landing", "week_at_landing", "final"):
        df[c] = df[c].astype("Int64")
    return df


def countries(repo: Path, blobs: Blobs, audit: list) -> pd.DataFrame:
    versions = [(t, oid) for t, h, name, oid in history(repo, GLOBAL_TS)]
    times = [t for t, _ in versions]
    added = {file_date(name): t for t, h, name, oid in history(repo, GLOBAL_DIR, diff_filter="A") if file_date(name)}
    candidates = set()
    for t in added.values():
        i = bisect.bisect_left(times, t - 3600)
        candidates.update(o for _, o in versions[max(i - 1, 0):i + 2])
    fetch(repo, sorted(candidates) + [versions[-1][1]])

    cache: dict[str, tuple] = {}

    def parse(oid):
        if oid not in cache:
            reader = csv.reader(io.StringIO(blobs.read(oid)))
            head = next(reader)
            cols = {}
            for i, c in enumerate(head[4:], start=4):
                try:
                    cols[dt.datetime.strptime(c, "%m/%d/%y").date()] = i
                except ValueError:
                    pass
            if len(cache) > 8:
                cache.pop(next(iter(cache)))
            cache[oid] = (cols, list(reader))
        return cache[oid]

    def by_country(oid, d):
        cols, rows = parse(oid)
        i = cols.get(d)
        if i is None:
            return {}
        out = collections.defaultdict(int)
        for r in rows:
            if len(r) > i and r[i] != "":
                out[r[1]] += int(float(r[i]))
        return out

    final_oid = versions[-1][1]
    rows, d = [], GLOBAL_FIRST
    while d <= GLOBAL_LAST:
        i = bisect.bisect_left(times, added.get(d, 0) - 3600)
        while i < len(versions) and d not in parse(versions[i][1])[0]:
            i += 1
        while i > 0 and d in parse(versions[i - 1][1])[0]:
            i -= 1
        t, oid = versions[i]
        audit.append(("global_first", d, oid))
        cur, prev = by_country(oid, d), by_country(oid, d - dt.timedelta(days=1))
        week, final = by_country(oid, d - dt.timedelta(days=7)), by_country(final_oid, d)
        landed = dt.datetime.fromtimestamp(t, dt.timezone.utc).strftime("%Y-%m-%d %H:%M")
        for c in sorted(cur):
            rows.append((d, c, cur[c], prev.get(c), week.get(c), final.get(c), landed))
        d += dt.timedelta(days=1)
    audit.append(("global_final", GLOBAL_LAST, final_oid))
    df = pd.DataFrame(rows, columns=["date", "country", "first", "prev_in_first", "week_in_first", "final",
                                     "landed_utc"])
    for c in ("prev_in_first", "week_in_first", "final"):
        df[c] = df[c].astype("Int64")
    return df


def holiday_table(repo: Path, country_names: list[str]) -> tuple[pd.DataFrame, int]:
    import holidays

    rows = [("US", d, n) for d, n in holidays.US(years=YEARS).items()]
    for code in STATES:
        rows += [(f"US-{code}", d, n) for d, n in holidays.US(subdiv=code, years=YEARS).items()]
    lookup = pd.read_csv(io.StringIO(git(repo, "show", f"{COMMIT}:{LOOKUP}")))
    iso = lookup[lookup.Province_State.isna()].set_index("Country_Region").iso2.to_dict()
    covered = 0
    for name in country_names:
        code = iso.get(name)
        if not isinstance(code, str) or code == "US":
            continue
        try:
            cal = holidays.country_holidays(code, years=YEARS)
        except NotImplementedError:
            continue
        covered += 1
        rows += [(name, d, n) for d, n in cal.items()]
    out = pd.DataFrame(rows, columns=["calendar", "date", "name"]).sort_values(["calendar", "date"])
    return out, covered


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", default="/tmp/jhu-covid", type=Path)
    args = ap.parse_args()
    import holidays

    repo = clone(args.workdir)
    blobs = Blobs(repo)
    audit: list = []
    us = us_states(repo, blobs, audit)
    us.to_csv(DATA / "jhu_us_states.csv.gz", index=False)
    glob = countries(repo, blobs, audit)
    glob.to_csv(DATA / "jhu_countries.csv.gz", index=False)
    hol, covered = holiday_table(repo, sorted(glob.country.unique()))
    hol.to_csv(DATA / "holidays.csv.gz", index=False)
    pd.DataFrame(audit, columns=["kind", "date", "blob"]).drop_duplicates().to_csv(
        DATA / "jhu_versions.csv.gz", index=False)
    with open(DATA / "jhu_fetch_log.txt", "w") as f:
        f.write(f"source {REPO_URL} at commit {COMMIT}\n")
        f.write(f"US daily reports: {us.date.nunique()} files, {us.date.min()} to {us.date.max()}, "
                f"{len(us)} state rows\n")
        f.write(f"global time series: {glob.date.nunique()} dates, {glob.date.min()} to {glob.date.max()}, "
                f"{glob.country.nunique()} countries, {len(glob)} rows\n")
        f.write(f"holidays: python-holidays {holidays.__version__}; US federal, {len(STATES)} states, "
                f"{covered} countries\n")
        f.write(f"blob versions read: {len(set(a[2] for a in audit))}\n")


if __name__ == "__main__":
    main()
