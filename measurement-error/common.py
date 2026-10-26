"""Shared helpers: data download/caching, NHANES loader and figure style."""

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

BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100",
    "#e87ba4", "#008300", "#6250d6", "#e34948",
)
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def fetch(package: str, name: str) -> pd.DataFrame:
    path = DATA / f"{package}_{name}.csv"
    if not path.exists():
        urllib.request.urlretrieve(f"{RDATASETS}/{package}/{name}.csv", path)
    return pd.read_csv(path)


NHANES_RAW = "https://raw.githubusercontent.com/ProjectMOSAIC/NHANES/master/data/NHANESraw.rda"


def load_nhanes() -> pd.DataFrame:
    """NHANES 2009-2012 participants aged 20-79 (NHANES::NHANESraw, unweighted).

    Systolic and diastolic pressure come as three readings taken in one
    session (BPSys1-3, BPDia1-3); NHANES's own BPSysAve is the mean of readings
    2 and 3."""
    import pyreadr

    path = DATA / "NHANESraw.rda"
    if not path.exists():
        urllib.request.urlretrieve(NHANES_RAW, path)
    raw = next(iter(pyreadr.read_r(str(path)).values()))
    a = raw[(raw.Age >= 20) & (raw.Age < 80)].copy()
    num = lambda c: pd.to_numeric(a[c].astype(str), errors="coerce")
    out = pd.DataFrame({
        "age": num("Age"), "bmi": num("BMI"),
        "female": (a.Gender.astype(str) == "female").astype(float),
        "sbp1": num("BPSys1"), "sbp2": num("BPSys2"), "sbp3": num("BPSys3"),
        "dbp1": num("BPDia1"), "dbp2": num("BPDia2"), "dbp3": num("BPDia3"),
        "sbp_ave": num("BPSysAve"),
        "diabetes": a.Diabetes.astype(str).map({"Yes": 1.0, "No": 0.0}),
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
