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
