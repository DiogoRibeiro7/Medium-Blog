"""Shared helpers: data download/caching and figure style."""

from __future__ import annotations

import urllib.request
from pathlib import Path

import matplotlib as mpl
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
FIG = HERE / "figures"
RES = HERE / "results"
for _d in (DATA, FIG, RES):
    _d.mkdir(exist_ok=True)

RDATASETS = "https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv"

# Colours (validated categorical order, light surface)
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100",
    "#e87ba4", "#008300", "#6250d6", "#e34948",
)
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def fetch(package: str, name: str) -> pd.DataFrame:
    """Download an Rdatasets CSV once and cache it under data/."""
    path = DATA / f"{package}_{name}.csv"
    if not path.exists():
        url = f"{RDATASETS}/{package}/{name}.csv"
        urllib.request.urlretrieve(url, path)
    return pd.read_csv(path)


def load_radon() -> pd.DataFrame:
    """Minnesota radon (Gelman & Hill 2007), via the HLMdiag package.

    The column called `basement` is 0 = basement, 1 = first floor, so it is
    renamed `floor` to match Gelman & Hill's notation.
    """
    df = fetch("HLMdiag", "radon")
    return df.rename(columns={"log.radon": "y", "basement": "floor",
                              "uranium": "u"})[["y", "floor", "u", "county"]]


def load_hsb() -> pd.DataFrame:
    """High School and Beyond 1982 (Raudenbush & Bryk 2002), via mlmRev."""
    df = fetch("mlmRev", "Hsb82")
    df["catholic"] = (df["sector"] == "Catholic").astype(int)
    return df


def load_contraception() -> pd.DataFrame:
    """1988 Bangladesh Fertility Survey subset, via mlmRev."""
    df = fetch("mlmRev", "Contraception")
    df["use"] = (df["use"] == "Y").astype(int)
    df["urban"] = (df["urban"] == "Y").astype(int)
    for k in ("1", "2", "3+"):
        df[f"ch{k.replace('+', 'p')}"] = (df["livch"] == k).astype(int)
    return df


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
    (RES / name).write_text(text)
    print(text)
