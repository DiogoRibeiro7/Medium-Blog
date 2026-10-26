"""Shared helpers: data download/caching and figure style."""

from __future__ import annotations

import urllib.request
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA, FIG, RES = HERE / "data", HERE / "figures", HERE / "results"
for _d in (DATA, FIG, RES):
    _d.mkdir(exist_ok=True)

RDATASETS = "https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv"
NHANES_RAW = "https://raw.githubusercontent.com/ProjectMOSAIC/NHANES/master/data/NHANESraw.rda"

BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100",
    "#e87ba4", "#008300", "#6250d6", "#e34948",
)
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def _download(url: str, path: Path) -> Path:
    if not path.exists():
        urllib.request.urlretrieve(url, path)
    return path


def load_cps() -> pd.DataFrame:
    """March 1988 CPS, 28,155 men (AER::CPS1988 via Rdatasets). No missing values."""
    df = pd.read_csv(_download(f"{RDATASETS}/AER/CPS1988.csv", DATA / "AER_CPS1988.csv"))
    out = pd.DataFrame({
        "logwage": np.log(df.wage),
        "educ": df.education.astype(float),
        "exper": df.experience.astype(float),
        "afam": (df.ethnicity == "afam").astype(float),
        "smsa": (df.smsa == "yes").astype(float),
        "parttime": (df.parttime == "yes").astype(float),
    })
    for r in ("northeast", "midwest", "west"):
        out[r] = (df.region == r).astype(float)
    out["exper2"] = out.exper ** 2 / 100.0
    return out


def load_nhanes() -> pd.DataFrame:
    """NHANES 2009-2012 participants aged 20-79 (NHANES::NHANESraw, unweighted)."""
    import pyreadr

    path = _download(NHANES_RAW, DATA / "NHANESraw.rda")
    raw = next(iter(pyreadr.read_r(str(path)).values()))
    a = raw[(raw.Age >= 20) & (raw.Age < 80)].copy()
    edu_map = {"8th Grade": 1, "9 - 11th Grade": 2, "High School": 3,
               "Some College": 4, "College Grad": 5}
    out = pd.DataFrame({
        "sbp": a.BPSysAve.astype(float),
        "age": a.Age.astype(float),
        "female": (a.Gender.astype(str) == "female").astype(float),
        "bmi": a.BMI.astype(float),
        "poverty": a.Poverty.astype(float),
        "chol": a.TotChol.astype(float),
        "black": (a.Race1.astype(str) == "Black").astype(float),
        "hispanic": a.Race1.astype(str).isin(["Hispanic", "Mexican"]).astype(float),
        "otherrace": (a.Race1.astype(str) == "Other").astype(float),
        "smoker": a.Smoke100.astype(str).map({"Yes": 1.0, "No": 0.0}),
        "educ": a.Education.astype(str).map(edu_map),
        "dbp": a.BPDiaAve.astype(float),
        "pulse": a.Pulse.astype(float),
        "hdl": a.DirectChol.astype(float),
        "weight": a.Weight.astype(float),
        "income": a.HHIncomeMid.astype(float) / 10000,
        "homeown": (a.HomeOwn.astype(str) == "Own").astype(float)
                   .where(a.HomeOwn.notna()),
        "year2011": (a.SurveyYr.astype(str) == "2011_12").astype(float),
    })
    return out.reset_index(drop=True)


def style() -> None:
    mpl.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight",
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.size": 10.5, "axes.titlesize": 11.5, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "axes.labelcolor": INK2,
        "axes.edgecolor": INK2, "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7,
        "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
        "legend.frameon": False, "lines.linewidth": 2.0,
    })


def write(name: str, text: str) -> None:
    (RES / name).write_text(text + "\n")
    print(text)
