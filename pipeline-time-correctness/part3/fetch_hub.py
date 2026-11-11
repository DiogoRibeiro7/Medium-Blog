"""Download the German COVID-19 Hospitalization Nowcast Hub's reporting triangle and its nowcasts.

Source: KITmetricslab/hospitalization-nowcast-hub on GitHub, pinned to HUB_COMMIT.

- data-truth/COVID-19/COVID-19_hospitalizations.csv: for each Meldedatum (the date the case was
  reported to the health office), location and age group, the number of hospitalised cases first
  published for it (value_0d) and the change at each later publication (value_1d .. value_79d,
  then value_>79d for everything later). The hub builds it from the RKI's daily archive of the
  7-day hospitalisation incidence. Changes can be negative.
  Kept: all ages ("00+"), Germany and the 16 states -> data/hub_triangle.csv.gz
- data-processed/<model>/<date>-<model>.csv: each team's nowcast published on that date of the
  7-day hospitalisation count for the Meldedatum `target_end_date` (0 to 28 days before), as a
  mean and seven quantiles. Kept: Germany, all ages, forecast dates FIRST to LAST, every model
  listed in MODELS -> data/hub_nowcasts.csv.gz

The git blob id of every file read is written to data/hub_fetch_log.txt.

Usage: python fetch_hub.py [--workdir DIR]
"""
from __future__ import annotations

import argparse
import datetime as dt
import gzip
import io
import os
import subprocess
from pathlib import Path

import pandas as pd

HUB_URL = "https://github.com/KITmetricslab/hospitalization-nowcast-hub"
HUB_COMMIT = "7314dc60985981fdd144453e4fcf820e95a32b99"
FIRST = dt.date(2021, 11, 22)
LAST = dt.date(2022, 4, 29)
MODELS = ("Epiforecasts-independent", "ILM-prop", "KIT-simple_nowcast", "LMU_StaBLab-GAM_nowcast",
          "NowcastHub-MeanEnsemble", "NowcastHub-MedianEnsemble", "RIVM-KEW", "RKI-weekly_report",
          "SU-hier_bayes", "SZ-hosp_nowcast")
TRUTH = "data-truth/COVID-19/COVID-19_hospitalizations.csv"
BATCH = 60

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
ENV = os.environ | {"GIT_LFS_SKIP_SMUDGE": "1"}


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True,
                          env=ENV).stdout


def clone(workdir: Path) -> Path:
    repo = workdir / "nowcast-hub"
    if not (repo / ".git").exists():
        subprocess.run(["git", "init", "-q", str(repo)], check=True)
        git(repo, "remote", "add", "origin", HUB_URL)
        git(repo, "config", "gc.auto", "0")
        git(repo, "fetch", "-q", "--depth", "1", "--filter=blob:none", "origin", HUB_COMMIT)
    return repo


def blobs(repo: Path, prefix: str) -> dict[str, str]:
    out = {}
    for line in git(repo, "ls-tree", "-r", HUB_COMMIT, prefix).splitlines():
        meta, path = line.split("\t")
        out[path] = meta.split()[2]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", default=str(Path.home() / "hub-work"))
    args = ap.parse_args()
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(exist_ok=True)
    repo = clone(workdir)
    log = [f"source {HUB_URL} at commit {HUB_COMMIT}"]

    truth_blob = blobs(repo, TRUTH)[TRUTH]
    git(repo, "checkout", HUB_COMMIT, "--", TRUTH)
    tri = pd.read_csv(repo / TRUTH)
    tri = tri[tri.age_group == "00+"].drop(columns="age_group")
    with gzip.open(DATA / "hub_triangle.csv.gz", "wt") as fh:
        tri.to_csv(fh, index=False)
    log.append(f"{TRUTH} blob {truth_blob}")

    wanted = []
    for model in MODELS:
        for path, blob in sorted(blobs(repo, f"data-processed/{model}").items()):
            name = Path(path).name
            if not name.endswith(f"-{model}.csv"):
                continue
            d = dt.date.fromisoformat(name[:10])
            if FIRST <= d <= LAST:
                wanted.append((model, d, path, blob))
    frames = []
    for i in range(0, len(wanted), BATCH):
        batch = wanted[i:i + BATCH]
        git(repo, "checkout", HUB_COMMIT, "--", *[p for _, _, p, _ in batch])
        for model, d, path, blob in batch:
            text = (repo / path).read_text()
            df = pd.read_csv(io.StringIO(text))
            df = df[(df.location == "DE") & (df.age_group == "00+")]
            df.insert(0, "model", model)
            frames.append(df)
            (repo / path).unlink()
            log.append(f"{path} blob {blob}")
    now = pd.concat(frames, ignore_index=True)
    now = now.drop(columns=[c for c in ("location", "age_group", "pathogen") if c in now.columns])
    with gzip.open(DATA / "hub_nowcasts.csv.gz", "wt") as fh:
        now.to_csv(fh, index=False)
    (DATA / "hub_fetch_log.txt").write_text("\n".join(log) + "\n")
    print(f"triangle rows {len(tri)}, nowcast files {len(wanted)}, nowcast rows {len(now)}")


if __name__ == "__main__":
    main()
