"""Download the RKI's daily COVID-19 case files and keep, for each publication, the counts by date.

Source: robert-koch-institut/SARS-CoV-2-Infektionen_in_Deutschland_Archiv on GitHub, pinned to
ARCHIVE_COMMIT. Each file `Archiv/YYYY-MM-DD_Deutschland_SarsCov2_Infektionen.csv.xz` is the full
case data set as published on that day (the "Datenstand"). Every row is a group of cases with a
reporting date at the local health office (Meldedatum), a reference date (Refdatum, the onset of
symptoms when IstErkrankungsbeginn = 1), and a flag NeuerFall: 0 = in this and the previous
publication, 1 = new in this publication, -1 = in the previous publication but no longer in this
one (AnzahlFall is then negative).

For each publication the script keeps three aggregates and writes them, gzipped, to data/:
  rki_melde.csv.gz  publication, Meldedatum, cases (NeuerFall 0 or 1), new (1), removed (-1)
                    for Meldedatum within MAX_DELAY days of the publication
  rki_onset.csv.gz  the same by onset date (rows with IstErkrankungsbeginn = 1)
  rki_state.csv.gz  publication, federal state, Meldedatum, cases, for Meldedatum within
                    STATE_DELAY days of the publication
and, from a monthly snapshot and the latest file, the counts for every Meldedatum:
  rki_snapshots.csv.gz  publication, Meldedatum, cases
The git blob id and size of every file read are written to data/rki_fetch_log.txt.

Daily publications from DAILY_FIRST to DAILY_LAST are read; after that, the first publication of
each month and the latest one. Run time is about an hour (some 1,000 files of 4-23 MB).

Usage: python fetch_rki.py [--workdir DIR]
"""
from __future__ import annotations

import argparse
import datetime as dt
import gzip
import os
import subprocess
import sys
import threading
from pathlib import Path

import duckdb

ARCHIVE_URL = "https://github.com/robert-koch-institut/SARS-CoV-2-Infektionen_in_Deutschland_Archiv"
ARCHIVE_COMMIT = "3375bddec5f30a547d62dfb1daf24612c75789d6"
DAILY_FIRST = dt.date(2021, 4, 2)
DAILY_LAST = dt.date(2023, 12, 31)
MAX_DELAY = 120
STATE_DELAY = 28
BATCH = 12

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

COLUMNS = ("{'IdLandkreis':'INTEGER','Altersgruppe':'VARCHAR','Geschlecht':'VARCHAR','Meldedatum':'DATE',"
           "'Refdatum':'DATE','IstErkrankungsbeginn':'INTEGER','NeuerFall':'INTEGER','NeuerTodesfall':'INTEGER',"
           "'NeuGenesen':'INTEGER','AnzahlFall':'INTEGER','AnzahlTodesfall':'INTEGER','AnzahlGenesen':'INTEGER'}")

AGG_SQL = f"""
with t as (
    select Meldedatum as melde,
           case when IstErkrankungsbeginn = 1 then Refdatum end as onset,
           IdLandkreis // 1000 as state, NeuerFall as nf, AnzahlFall as n
    from read_csv(?, header = true, columns = {COLUMNS})
)
select grouping(melde) as g_melde, grouping(onset) as g_onset, grouping(state) as g_state,
       melde, onset, state, nf, sum(n) as n
from t
group by grouping sets ((melde, nf), (onset, nf), (state, melde, nf))
"""


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True).stdout


def clone(workdir: Path) -> Path:
    repo = workdir / "rki-archive"
    if not (repo / ".git").exists():
        env = os.environ | {"GIT_LFS_SKIP_SMUDGE": "1"}
        subprocess.run(["git", "init", "-q", str(repo)], check=True)
        git(repo, "remote", "add", "origin", ARCHIVE_URL)
        git(repo, "config", "gc.auto", "0")
        subprocess.run(["git", "-C", str(repo), "fetch", "-q", "--depth", "1", "--filter=blob:none", "origin",
                        ARCHIVE_COMMIT], check=True, env=env)
    return repo


def plan(repo: Path) -> list[tuple[dt.date, str, str]]:
    """(publication date, path, blob id) for every file to read."""
    files = []
    for line in git(repo, "ls-tree", "-r", ARCHIVE_COMMIT, "Archiv").splitlines():
        meta, path = line.split("\t")
        blob = meta.split()[2]
        files.append((dt.date.fromisoformat(Path(path).name[:10]), path, blob))
    files.sort()
    chosen, months = [], set()
    for d, path, blob in files:
        if DAILY_FIRST <= d <= DAILY_LAST:
            chosen.append((d, path, blob))
        elif d > DAILY_LAST and (d.year, d.month) not in months:
            months.add((d.year, d.month))
            chosen.append((d, path, blob))
    if chosen[-1] != files[-1]:
        chosen.append(files[-1])
    return chosen


def checkout(repo: Path, paths: list[str]) -> None:
    subprocess.run(["git", "-C", str(repo), "checkout", ARCHIVE_COMMIT, "--", *paths], check=True,
                   capture_output=True, env=os.environ | {"GIT_LFS_SKIP_SMUDGE": "1"})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", default=str(Path.home() / "rki-work"))
    ap.add_argument("--out", default=str(DATA))
    ap.add_argument("--limit", type=int, default=0, help="read only the first N publications (for testing)")
    args = ap.parse_args()
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    data = Path(args.out)
    data.mkdir(parents=True, exist_ok=True)
    repo = clone(workdir)
    files = plan(repo)
    if args.limit:
        files = files[:args.limit]
    print(f"{len(files)} publications to read", file=sys.stderr)

    con = duckdb.connect()
    con.execute("set threads = 2")
    out = {k: gzip.open(data / f"rki_{k}.csv.gz", "wt", compresslevel=6) for k in ("melde", "onset", "state", "snapshots")}
    out["melde"].write("publication,date,cases,new,removed\n")
    out["onset"].write("publication,date,cases,new,removed\n")
    out["state"].write("publication,state,date,cases\n")
    out["snapshots"].write("publication,date,cases\n")
    log = open(data / "rki_fetch_log.txt", "w")
    log.write(f"source {ARCHIVE_URL} at commit {ARCHIVE_COMMIT}\n")

    batches = [files[i:i + BATCH] for i in range(0, len(files), BATCH)]
    prefetch: threading.Thread | None = None

    def start(b):
        th = threading.Thread(target=checkout, args=(repo, [p for _, p, _ in b]))
        th.start()
        return th

    prefetch = start(batches[0])
    for bi, batch in enumerate(batches):
        prefetch.join()
        if bi + 1 < len(batches):
            prefetch = start(batches[bi + 1])
        for d, path, blob in batch:
            xz = repo / path
            csv = workdir / "current.csv"
            with open(csv, "wb") as fh:
                subprocess.run(["xz", "-dc", str(xz)], check=True, stdout=fh)
            df = con.execute(AGG_SQL, [str(csv)]).df()
            xz.unlink()
            size = git(repo, "cat-file", "-s", blob).strip()
            log.write(f"{d.isoformat()} {path} blob {blob} bytes {size}\n")
            snapshot = d.day == 1 or d == DAILY_FIRST or d > DAILY_LAST
            for kind, key in (("melde", "melde"), ("onset", "onset")):
                g = df[(df[f"g_{kind}"] == 0) & (df["g_state"] == 1) & (df[f"g_{'onset' if kind == 'melde' else 'melde'}"] == 1)]
                g = g.pivot_table(index=key, columns="nf", values="n", aggfunc="sum", fill_value=0)
                for c in (-1, 0, 1):
                    if c not in g.columns:
                        g[c] = 0
                g = g.sort_index()
                age = (d - g.index.to_series().dt.date).apply(lambda x: x.days)
                if kind == "melde" and snapshot:
                    for date, row in g.iterrows():
                        out["snapshots"].write(f"{d},{date.date()},{int(row[0] + row[1])}\n")
                recent = g[(age >= 0) & (age <= MAX_DELAY)]
                for date, row in recent.iterrows():
                    out[kind].write(f"{d},{date.date()},{int(row[0] + row[1])},{int(row[1])},{int(row[-1])}\n")
            s = df[(df["g_state"] == 0) & (df["g_melde"] == 0)]
            s = s[s.nf >= 0].groupby(["state", "melde"])["n"].sum()
            for (state, date), n in s.items():
                if 0 <= (d - date.date()).days <= STATE_DELAY:
                    out["state"].write(f"{d},{int(state)},{date.date()},{int(n)}\n")
            csv.unlink()
        print(f"batch {bi + 1}/{len(batches)} done ({batch[-1][0]})", file=sys.stderr, flush=True)
    for fh in out.values():
        fh.close()
    log.close()


if __name__ == "__main__":
    main()
