"""Shared loaders for the German public-investment analysis."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RESULTS = HERE / "results"
FIGURES = HERE / "figures"

PEERS = ("NL", "AT", "FI", "DK", "SE")
CONTEXT = ("FR", "IT")
NAMES = {
    "DE": "Germany", "NL": "Netherlands", "AT": "Austria", "FI": "Finland", "DK": "Denmark", "SE": "Sweden",
    "FR": "France", "IT": "Italy", "BE": "Belgium", "ES": "Spain", "PT": "Portugal", "IE": "Ireland", "EL": "Greece",
}
SUBSECTORS = {"S1311": "Federal (central)", "S1312": "Länder (state)", "S1313": "Municipalities (local)", "S1314": "Social security"}
# Periods used throughout
PERIODS = {
    "before the brake, 1995-2008": (1995, 2008),
    "brake in force, 2011-2019": (2011, 2019),
    "escape clause and after, 2020-2024": (2020, 2024),
}


def gdp(path: Path = DATA, unit: str = "CP_MEUR") -> pd.DataFrame:
    y = pd.read_csv(path / "gdp.csv")
    return y[y["unit"] == unit].pivot_table(index="time", columns="geo", values="value")


def gov(item: str, sector: str = "S13", path: Path = DATA) -> pd.DataFrame:
    """A gov_10a_main item for one sector, EUR million, years by countries."""
    g = pd.read_csv(path / "gov_main.csv")
    return g[(g["na_item"] == item) & (g["sector"] == sector)].pivot_table(index="time", columns="geo", values="value")


def share(x: pd.DataFrame, y: pd.DataFrame | None = None) -> pd.DataFrame:
    """Percent of GDP."""
    y = gdp() if y is None else y
    return 100 * x / y


def debt(path: Path = DATA) -> pd.DataFrame:
    d = pd.read_csv(path / "debt.csv")
    return d.pivot_table(index="time", columns="geo", values="value")


def fixed_assets(asset: str = "N11N", path: Path = DATA) -> pd.DataFrame:
    """Net fixed assets of general government at current prices, EUR million."""
    f = pd.read_csv(path / "fixed_assets.csv")
    return f[(f["unit"] == "CP_MEUR") & (f["asset10"] == asset)].pivot_table(index="time", columns="geo", values="value")


def cofog(code: str, path: Path = DATA) -> pd.DataFrame:
    """Gross fixed capital formation of general government by function, EUR million."""
    c = pd.read_csv(path / "cofog_investment.csv")
    return c[c["cofog99"] == code].pivot_table(index="time", columns="geo", values="value")


def quarterly(path: Path = DATA) -> pd.DataFrame:
    q = pd.read_csv(path / "gov_quarterly.csv")
    q = q[q["sector"] == "S13"].pivot_table(index=["geo", "time"], columns="na_item", values="value")
    y = pd.read_csv(path / "gdp_quarterly.csv").pivot_table(index=["geo", "time"], values="value")["value"].rename("gdp")
    return q.join(y).reset_index().rename(columns={"time": "quarter"})


def peer_mean(x: pd.DataFrame, peers: tuple[str, ...] = PEERS) -> pd.Series:
    """Unweighted mean across peers (each country counts once)."""
    return x[list(peers)].mean(axis=1)


def write(name: str, lines: list[str]) -> str:
    RESULTS.mkdir(exist_ok=True)
    text = "\n".join(lines) + "\n"
    (RESULTS / name).write_text(text, encoding="utf-8")
    return text
