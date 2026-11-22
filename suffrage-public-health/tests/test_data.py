import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import dates  # noqa: E402

RAW = HERE / "data" / "raw"
needs_data = pytest.mark.skipif(not (RAW / "mortality_harmonized.dta").exists(),
                                reason="run fetch.py first")


def test_coding_counts():
    g = {s: dates.suffrage_year(s) for s in dates.STATES}
    assert len(g) == 48
    assert sum(v < 1920 for v in g.values()) == 29
    assert dates.suffrage_year("TX", "full_only") == 1920
    assert dates.suffrage_year("CA", "lag") == 1912
    assert dates.suffrage_year("WI", "lag") == 1920


@needs_data
def test_dates_reproduce_miller_appendix_table_2():
    table = json.loads((Path(__file__).parent / "appendix_table2.json").read_text())["columns"]
    m = pd.read_stata(RAW / "mortality_harmonized.dta", convert_categoricals=False)
    m["e"] = m.year.astype(int) - m.state.map(dates.suffrage_year)
    for e, states in table.items():
        assert sorted(m.loc[m.e == int(e), "state"].unique()) == states, e


@needs_data
def test_id_column_is_a_rank_and_fips_identifies_cities():
    pf = pd.read_stata(RAW / "public_finance.dta", convert_categoricals=False)
    assert pf.duplicated(["fips_state", "fips_place", "year"]).sum() == 1  # Quincy, MA, 1923
    pf = pf.drop_duplicates(["fips_state", "fips_place", "year"])
    for _, d in pf.groupby("year"):
        assert sorted(d.id.astype(int)) == list(range(1, len(d) + 1))
    assert (pf.groupby("id").state.nunique() > 1).sum() > 200
    fips = pf.fips_state.astype(str) + "-" + pf.fips_place.astype(str)
    assert pf.groupby(fips).state.nunique().max() == 1
    assert fips.nunique() == 319


@needs_data
def test_1905_total_spending_is_not_comparable():
    pf = pd.read_stata(RAW / "public_finance.dta", convert_categoricals=False)
    share = (pf.e_health_sanit / pf.e_total).groupby(pf.year).median()
    assert share[1905] > 0.9
    assert share.drop(1905).between(0.07, 0.12).all()


@needs_data
def test_mortality_panel_shape():
    m = pd.read_stata(RAW / "mortality_harmonized.dta", convert_categoricals=False)
    assert len(m) == 1109
    assert m.state.nunique() == 48
    single = m[["all_1_2", "all_2_3", "all_3_4", "all_4_5"]].sum(axis=1, min_count=4)
    assert np.nanmax(np.abs(m.all_1_4 - single)) == 0
