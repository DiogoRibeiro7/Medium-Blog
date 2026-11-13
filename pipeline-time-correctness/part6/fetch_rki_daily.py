"""Download the RKI's daily COVID-19 case files from 1 January 2024 to 10 November 2026 and keep, for
each publication, the same aggregates Part 3 kept for 2021-2023:

  rki_state_2024_2026.csv.gz  publication, federal state, Meldedatum, cases, for Meldedatum within
                              28 days of the publication (the input of the pipeline)
  rki_melde_2024_2026.csv.gz  publication, Meldedatum, cases, new, removed, for Meldedatum within
                              120 days of the publication (national, for checks)

The source, the pinned commit, the parsing and the aggregation are those of part3/fetch_rki.py, imported
from it. The git blob id and size of every file read are written to data/rki_fetch_log_2024_2026.txt.
Run time is about an hour (some 1,050 files of about 22 MB). Fetched blobs are deleted from the local
clone after each batch, so the disk never holds more than a batch.

Usage: python fetch_rki_daily.py [--workdir DIR]
"""
from __future__ import annotations

import argparse
import datetime as dt
import gzip
import subprocess
import sys
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "part3"))
import fetch_rki as F  # noqa: E402

FIRST, LAST = dt.date(2024, 1, 1), dt.date(2026, 11, 10)
DATA = Path(__file__).resolve().parent / "data"


def plan(repo: Path) -> list[tuple[dt.date, str, str]]:
    files = []
    for line in F.git(repo, "ls-tree", "-r", F.ARCHIVE_COMMIT, "Archiv").splitlines():
        meta, path = line.split("\t")
        d = dt.date.fromisoformat(Path(path).name[:10])
        if FIRST <= d <= LAST:
            files.append((d, path, meta.split()[2]))
    return sorted(files)


def drop_new_packs(repo: Path, keep: set[str]) -> None:
    pack = repo / ".git" / "objects" / "pack"
    for f in pack.iterdir():
        if f.name not in keep:
            f.unlink()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", default=str(Path.home() / "rki-work"))
    args = ap.parse_args()
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    repo = F.clone(workdir)
    keep = {f.name for f in (repo / ".git" / "objects" / "pack").iterdir()}
    files = plan(repo)
    print(f"{len(files)} publications to read, {files[0][0]} to {files[-1][0]}", file=sys.stderr)

    con = duckdb.connect()
    con.execute("set threads = 2")
    state = gzip.open(DATA / "rki_state_2024_2026.csv.gz", "wt", compresslevel=6)
    melde = gzip.open(DATA / "rki_melde_2024_2026.csv.gz", "wt", compresslevel=6)
    state.write("publication,state,date,cases\n")
    melde.write("publication,date,cases,new,removed\n")
    log = open(DATA / "rki_fetch_log_2024_2026.txt", "w")
    log.write(f"source {F.ARCHIVE_URL} at commit {F.ARCHIVE_COMMIT}\n")

    batches = [files[i:i + F.BATCH] for i in range(0, len(files), F.BATCH)]
    for bi, batch in enumerate(batches):
        F.checkout(repo, [p for _, p, _ in batch])
        for d, path, blob in batch:
            xz = repo / path
            csv = workdir / "current.csv"
            with open(csv, "wb") as fh:
                subprocess.run(["xz", "-dc", str(xz)], check=True, stdout=fh)
            df = con.execute(F.AGG_SQL, [str(csv)]).df()
            xz.unlink()
            csv.unlink()
            size = F.git(repo, "cat-file", "-s", blob).strip()
            log.write(f"{d.isoformat()} {path} blob {blob} bytes {size}\n")
            g = df[(df.g_melde == 0) & (df.g_state == 1) & (df.g_onset == 1)]
            g = g.pivot_table(index="melde", columns="nf", values="n", aggfunc="sum", fill_value=0)
            for c in (-1, 0, 1):
                if c not in g.columns:
                    g[c] = 0
            for date, row in g.sort_index().iterrows():
                if 0 <= (d - date.date()).days <= F.MAX_DELAY:
                    melde.write(f"{d},{date.date()},{int(row[0] + row[1])},{int(row[1])},{int(row[-1])}\n")
            s = df[(df.g_state == 0) & (df.g_melde == 0)]
            s = s[s.nf >= 0].groupby(["state", "melde"])["n"].sum()
            for (st, date), n in s.items():
                if 0 <= (d - date.date()).days <= F.STATE_DELAY:
                    state.write(f"{d},{int(st)},{date.date()},{int(n)}\n")
        drop_new_packs(repo, keep)
        print(f"batch {bi + 1}/{len(batches)} done ({batch[-1][0]})", file=sys.stderr, flush=True)
    state.close()
    melde.close()
    log.close()


if __name__ == "__main__":
    main()
