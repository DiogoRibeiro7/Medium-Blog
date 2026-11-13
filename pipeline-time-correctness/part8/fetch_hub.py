"""Read the weekly case forecasts for Germany and its 16 states from the German and Polish COVID-19 Forecast Hub,
the truth data the hub scored them against, and the RKI's final case counts by state, reporting date and
reference date, and write the extracts Part 8 uses:

  hub_forecasts.csv.gz   every forecast of weekly incident cases for Germany (GM) and the states (GM01 to GM16),
                         1 to 4 weeks ahead: model, forecast date, target week (ending Saturday), location,
                         horizon, and the median and the 2.5%, 10%, 25%, 75%, 90% and 97.5% quantiles (the point
                         forecast where a model gave no median).
  hub_truth.csv.gz       the hub's daily truth for cases by location, dated by the day the RKI published the
                         counts (the differences of the RKI's cumulative counts from one day to the next), as the
                         file stood at the hub's last commit.
  rki_dates.csv.gz       the RKI's final case counts, from its file of 10 November 2026: cases by state,
                         reporting date (Meldedatum, the day the local health office was notified), reference
                         date (Refdatum: the onset of symptoms where known, otherwise the reporting date) and
                         whether the onset is known.
  holidays_de.csv.gz     the public holidays of 2020 to 2023 in Germany and in each state, from python-holidays.
  hub_files.csv.gz       the path and git blob id of every forecast file read.
  hub_fetch_log.txt      sources, pinned commits, the RKI file's blob id and counts.

Both repositories are cloned without file contents and only the files needed are fetched: about 2,500
forecast files (50 MB) and one RKI file (23 MB compressed, 414 MB as text). Run time is a few minutes.

The forecasts are the work of the teams that submitted them, under licences that differ from team to team,
so hub_forecasts.csv.gz is not committed: `python fetch_hub.py --forecasts-only` rebuilds it from the pinned
commit (the reproduce workflow does this before the tests). The hub's truth before April 2020 came partly from
IHME under a non-commercial licence and is left out; the analysis starts in May 2020.

Usage: pip install holidays==0.108; python fetch_hub.py [--workdir DIR] [--forecasts-only]
"""
from __future__ import annotations

import argparse
import io
import lzma
import subprocess
from pathlib import Path

import duckdb
import pandas as pd

HUB_URL = "https://github.com/KITmetricslab/covid19-forecast-hub-de.git"
HUB_COMMIT = "991c7a054246faee4abace7f5b705f044b0ffd01"  # 31 May 2023, the hub's last commit
HUB_TRUTH = "data-truth/RKI/truth_RKI-Incident Cases_Germany.csv"
RKI_URL = "https://github.com/robert-koch-institut/SARS-CoV-2-Infektionen_in_Deutschland_Archiv"
RKI_COMMIT = "3375bddec5f30a547d62dfb1daf24612c75789d6"  # the archive commit of Parts 3 and 6
RKI_FILE = "Archiv/2026-11-10_Deutschland_SarsCov2_Infektionen.csv.xz"
QUANTILES = (0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975)
DATA = Path(__file__).resolve().parent / "data"
SUBDIVISIONS = ("BW", "BY", "BE", "BB", "HB", "HH", "HE", "MV", "NI", "NW", "RP", "SL", "SN", "ST", "SH", "TH")


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True).stdout


def blobless(url: str, path: Path, commit: str, shallow: bool = False) -> Path:
    if not (path / ".git").exists():
        path.mkdir(parents=True, exist_ok=True)
        git(path, "init", "-q")
        git(path, "remote", "add", "origin", url)
        git(path, "config", "gc.auto", "0")
        args = ["fetch", "-q", "--filter=blob:none", "origin", commit] + (["--depth", "1"] if shallow else [])
        subprocess.run(["git", "-C", str(path), *args], check=True,
                       env={"GIT_LFS_SKIP_SMUDGE": "1", "PATH": "/usr/bin:/bin:/usr/local/bin"})
    return path


def fetch(repo: Path, oids: list[str]) -> None:
    for i in range(0, len(oids), 100):
        subprocess.run(["git", "-C", str(repo), "-c", "fetch.negotiationAlgorithm=noop", "fetch", "-q", "--no-tags",
                        "--no-write-fetch-head", "--filter=blob:none", "origin", *oids[i:i + 100]], check=True)


def read_blobs(repo: Path, oids: list[str]):
    p = subprocess.Popen(["git", "-C", str(repo), "cat-file", "--batch"], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    for oid in oids:
        p.stdin.write((oid + "\n").encode())
        p.stdin.flush()
        size = int(p.stdout.readline().split()[2])
        data = p.stdout.read(size)
        p.stdout.read(1)
        yield oid, data
    p.stdin.close()
    p.wait()


def forecasts(repo: Path, read: list[tuple[str, str]]) -> pd.DataFrame:
    files = []
    for line in git(repo, "ls-tree", "-r", HUB_COMMIT, "data-processed").splitlines():
        meta, path = line.split("\t")
        if "-Germany-" in path and path.endswith("-case.csv"):
            files.append((meta.split()[2], path))
    fetch(repo, [o for o, _ in files])
    frames = []
    for (oid, data), (_, path) in zip(read_blobs(repo, [o for o, _ in files]), files):
        df = pd.read_csv(io.BytesIO(data), dtype={"location": str})
        df = df[df.target.str.contains("wk ahead inc case", na=False)].copy()
        df["horizon"] = df.target.str.extract(r"^(\d+) wk")[0].astype(float)
        df = df[df.horizon.between(1, 4) & df.location.str.match(r"^GM(\d\d)?$")]
        df["model"] = path.split("/")[1]
        frames.append(df)
        read.append((path, oid))
    f = pd.concat(frames, ignore_index=True)
    key = ["model", "forecast_date", "target_end_date", "location", "horizon"]
    q = f[(f.type == "quantile") & f["quantile"].round(3).isin(QUANTILES)].copy()
    q["col"] = "q" + (q["quantile"].round(3) * 1000).astype(int).astype(str).str.zfill(4)
    wide = q.pivot_table(index=key, columns="col", values="value", aggfunc="first").reset_index()
    point = f[f.type == "point"].groupby(key).value.first().rename("point").reset_index()
    out = wide.merge(point, on=key, how="outer")
    out["median"] = out["q0500"].fillna(out["point"])
    cols = key + ["median"] + [c for c in ("q0025", "q0100", "q0250", "q0750", "q0900", "q0975") if c in out]
    return out[cols].dropna(subset=["median"]).sort_values(key)


def rki_dates(workdir: Path, log: list[str]) -> pd.DataFrame:
    repo = blobless(RKI_URL, workdir / "rki-archive", RKI_COMMIT, shallow=True)
    oid = git(repo, "rev-parse", f"{RKI_COMMIT}:{RKI_FILE}").strip()
    fetch(repo, [oid])
    text = workdir / "rki_final.csv"
    with open(text, "wb") as out:
        out.write(lzma.decompress(next(read_blobs(repo, [oid]))[1]))
    log.append(f"rki {RKI_FILE} blob {oid}")
    con = duckdb.connect()
    df = con.execute(f"""
        select IdLandkreis // 1000 as state, Meldedatum as reporting_date, Refdatum as reference_date,
               IstErkrankungsbeginn as onset_known, sum(AnzahlFall) as cases
        from read_csv('{text}', header = true, types = {{'Meldedatum': 'DATE', 'Refdatum': 'DATE'}})
        where NeuerFall in (0, 1)
        group by all order by all""").df()
    text.unlink()
    return df


def holiday_table() -> pd.DataFrame:
    import holidays

    years = range(2020, 2024)
    rows = [("DE", d, n) for d, n in holidays.DE(years=years).items()]
    for s in SUBDIVISIONS:
        rows += [(f"DE-{s}", d, n) for d, n in holidays.DE(subdiv=s, years=years).items()]
    return pd.DataFrame(rows, columns=["calendar", "date", "name"]).sort_values(["calendar", "date"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", default="/tmp/hub-work", type=Path)
    ap.add_argument("--forecasts-only", action="store_true", help="only rebuild hub_forecasts.csv.gz")
    args = ap.parse_args()
    hub = blobless(HUB_URL, args.workdir / "hub-de", HUB_COMMIT)
    read: list[tuple[str, str]] = []
    f = forecasts(hub, read)
    f.to_csv(DATA / "hub_forecasts.csv.gz", index=False)
    if args.forecasts_only:
        return
    import holidays

    log = [f"hub {HUB_URL} at commit {HUB_COMMIT}", f"rki {RKI_URL} at commit {RKI_COMMIT}"]
    pd.DataFrame(read, columns=["path", "blob"]).to_csv(DATA / "hub_files.csv.gz", index=False)
    truth = pd.read_csv(io.StringIO(git(hub, "show", f"{HUB_COMMIT}:{HUB_TRUTH}")))
    truth = truth[truth.date >= "2020-04-01"]
    truth.to_csv(DATA / "hub_truth.csv.gz", index=False)
    log.append(f"truth {HUB_TRUTH}")
    rki_dates(args.workdir, log).to_csv(DATA / "rki_dates.csv.gz", index=False)
    holiday_table().to_csv(DATA / "holidays_de.csv.gz", index=False)
    log.append(f"holidays python-holidays {holidays.__version__}")
    log.append(f"forecasts {len(f)} rows, {f.model.nunique()} models, {f.forecast_date.min()} to {f.forecast_date.max()}")
    log.append(f"forecast files read: {len(read)}")
    (DATA / "hub_fetch_log.txt").write_text("\n".join(log) + "\n")


if __name__ == "__main__":
    main()
