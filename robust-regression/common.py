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


CUT = {"Fair": 1, "Good": 2, "Very Good": 3, "Premium": 4, "Ideal": 5}
COLOR = {c: i for i, c in enumerate("JIHGFED", 1)}
CLARITY = {c: i for i, c in enumerate(["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"], 1)}


def load_diamonds() -> pd.DataFrame:
    """ggplot2::diamonds with log price, log carat, log volume and numeric grades.

    `density` is carat per cubic millimetre of the recorded dimensions; for a
    real round diamond it is close to 0.0061, so values far from it mark rows
    whose dimensions cannot be right."""
    d = fetch("ggplot2", "diamonds")
    d["lp"] = np.log(d.price)
    d["lc"] = np.log(d.carat)
    vol = d.x * d.y * d.z
    d["lv"] = np.log(vol.where(vol > 0))
    d["density"] = d.carat / vol.where(vol > 0)
    d["cut_n"] = d.cut.map(CUT).astype(float)
    d["col_n"] = d.color.map(COLOR).astype(float)
    d["cla_n"] = d.clarity.map(CLARITY).astype(float)
    return d


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
