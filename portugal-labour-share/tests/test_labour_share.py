"""Checks on the data snapshot, the measures and the inference code."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import fetch  # noqa: E402
import labour_share as ls  # noqa: E402


@pytest.fixture(scope="module")
def portugal() -> pd.DataFrame:
    return ls.load_portugal()


@pytest.fixture(scope="module")
def parts(portugal: pd.DataFrame) -> ls.Components:
    return ls.components(portugal)


def test_gdp_identity_closes_for_every_country() -> None:
    panel = ls.load_gdp_panel()
    gap = panel["B1GQ"] - (panel["D1"] + panel["B2A3G"] + panel["D2X3"])
    assert gap.abs().max() < 0.2
    assert set(panel["geo"]) == set(ls.PEERS)


def test_sources_agree_on_operating_surplus(portugal: pd.DataFrame) -> None:
    assert (portugal["B2A3G"] - portugal["B2A3G_S1"]).abs().max() <= 1.0
    assert (portugal["B2A3G_S1"] - portugal["B2G_S1"] - portugal["B3G_S1"]).abs().max() <= 1.0


def test_sample_excludes_provisional_year(portugal: pd.DataFrame) -> None:
    assert portugal["year"].min() == ls.FIRST_YEAR
    assert portugal["year"].max() == ls.LAST_YEAR == 2024


@pytest.mark.parametrize("den", ls.DENOMINATORS)
def test_shares_are_linear_in_the_labour_fraction(parts: ls.Components, den: str) -> None:
    raw = ls.labour_share(parts, "raw", den)
    full = ls.labour_share(parts, "mixed_all", den)
    two_thirds = ls.labour_share(parts, "mixed_two_thirds", den)
    np.testing.assert_allclose(two_thirds, raw + (2 / 3) * (full - raw), rtol=0, atol=1e-10)
    np.testing.assert_allclose(full - raw, ls.mixed_income_share(parts, den), rtol=0, atol=1e-10)


def test_adjusted_shares_exceed_the_raw_share(parts: ls.Components) -> None:
    for den in ls.DENOMINATORS:
        raw = ls.labour_share(parts, "raw", den)
        for num in ls.NUMERATORS[1:]:
            assert (ls.labour_share(parts, num, den) > raw).all()


def test_shift_share_adds_up(portugal: pd.DataFrame, parts: ls.Components) -> None:
    result = ls.government_shift_share(portugal, parts, 2010, 2012)
    fc = ls.labour_share(parts, "raw", "factor_cost")
    years = list(portugal["year"])
    expected = fc.iloc[years.index(2012)] - fc.iloc[years.index(2010)]
    assert result["total"] == pytest.approx(expected, abs=1e-10)


def test_break_scan_finds_a_planted_break() -> None:
    years = np.arange(1995, 2025)
    rng = np.random.default_rng(1)
    y = 50 + 0.05 * np.arange(30) - 3.0 * (years >= 2012) + rng.normal(0, 0.3, 30)
    res = ls.break_test(y, years, "level", "synthetic")
    assert res["break_year"] == 2012
    assert res["shift"] == pytest.approx(-3.0, abs=0.5)
    assert res["p_boot_alt"] < 0.01


def test_tests_are_reproducible() -> None:
    years = np.arange(1995, 2025)
    y = np.sin(np.arange(30) / 4.0) + 0.02 * np.arange(30)
    first = ls.break_test(y, years, "level_slope", "repeat")
    second = ls.break_test(y, years, "level_slope", "repeat")
    assert first == second


def test_parse_tsv_handles_flags_and_missing_values() -> None:
    text = (
        "freq,unit,na_item,geo\\TIME_PERIOD\t1995 \t1996 \t1997 \n"
        "A,CP_MEUR,D1,PT\t42592.0 \t45839.5 p\t: \n"
        "A,CP_MEUR,B1GQ,PT\t91015.8 e\t: c\t103306.6 \n"
    )
    frame = fetch.parse_tsv(text)
    assert list(frame.columns) == ["freq", "unit", "na_item", "geo", "year", "value"]
    d1 = frame[frame["na_item"] == "D1"].set_index("year")["value"]
    assert d1[1995] == 42592.0 and d1[1996] == 45839.5 and np.isnan(d1[1997])
    gdp = frame[frame["na_item"] == "B1GQ"].set_index("year")["value"]
    assert gdp[1995] == 91015.8 and np.isnan(gdp[1996]) and gdp[1997] == 103306.6


def _as_downloaded() -> dict[str, pd.DataFrame]:
    """Rebuild, from the snapshot, the long frames ``fetch.download`` would return."""
    data = ls.DATA
    gdp = pd.read_csv(data / "nama_10_gdp.csv").melt(id_vars=["geo", "year"], var_name="na_item")
    gdp["freq"], gdp["unit"] = "A", "CP_MEUR"
    emp = pd.read_csv(data / "nama_10_a10_e_PT.csv").melt(id_vars="year", var_name="column")
    emp["na_item"] = emp["column"].str[:6]
    emp["unit"] = emp["column"].str[7:]
    emp["geo"], emp["nace_r2"] = "PT", "TOTAL"
    gov = pd.read_csv(data / "gov_10a_main_PT.csv").melt(id_vars="year", var_name="na_item")
    gov["geo"], gov["sector"], gov["unit"] = "PT", "S13", "MIO_EUR"
    nasa = pd.read_csv(data / "nasa_10_nf_tr_PT.csv").melt(id_vars="year", var_name="column")
    nasa["na_item"] = nasa["column"].str.split("_").str[0]
    nasa["sector"] = nasa["column"].str.split("_", n=1).str[1]
    nasa["geo"], nasa["direct"], nasa["unit"] = "PT", "PAID", "CP_MEUR"
    return {
        "nama_10_gdp": gdp,
        "nama_10_a10_e": emp.drop(columns="column"),
        "gov_10a_main": gov,
        "nasa_10_nf_tr": nasa.drop(columns="column"),
    }


def test_fetch_reshaping_rebuilds_the_snapshot() -> None:
    files = fetch.build_files(_as_downloaded())
    for name, frame in files.items():
        snapshot = pd.read_csv(ls.DATA / name)
        pd.testing.assert_frame_equal(
            frame.reset_index(drop=True), snapshot, check_dtype=False, check_names=False
        )
