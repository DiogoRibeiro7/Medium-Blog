import csv
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import build as B  # noqa: E402


def _row(i, price, date, pc, ptype="T", new="N", dur="F", paon="1", saon="", street="HIGH ST",
         county="LEEDS", cat="A"):
    return [f"{{T{i:04d}}}", str(price), f"{date} 00:00", pc, ptype, new, dur, paon, saon, street,
            "", "TOWN", "DISTRICT", county, cat, "A"]


@pytest.fixture()
def built(tmp_path):
    rows = [
        # property 1, Leeds: bought 2010, resold 2015 at exactly the same price
        _row(1, 150000, "2010-03-15", "LS1 1AA"), _row(2, 150000, "2015-06-01", "LS1 1AA"),
        # property 2, London flat: loss, then gain
        _row(3, 400000, "2016-01-10", "SW1A 1AA", ptype="F", dur="L", paon="10", saon="FLAT 2",
             county="GREATER LONDON"),
        _row(4, 380000, "2019-02-01", "SW1A 1AA", ptype="F", dur="L", paon="10", saon="FLAT 2",
             county="GREATER LONDON"),
        _row(5, 420000, "2023-05-01", "SW1A 1AA", ptype="F", dur="L", paon="10", saon="FLAT 2",
             county="GREATER LONDON"),
        # property 3, Cardiff: first sale new build (kept apart)
        _row(6, 200000, "2012-01-01", "CF10 1AA", new="Y", county="CARDIFF"),
        _row(7, 190000, "2014-01-01", "CF10 1AA", county="CARDIFF"),
        # property 4: resold after 3 months (dropped), category B sale (dropped)
        _row(8, 100000, "2018-01-01", "OX1 1AA", county="OXFORDSHIRE"),
        _row(9, 120000, "2018-04-01", "OX1 1AA", county="OXFORDSHIRE"),
        _row(10, 90000, "2020-01-01", "OX1 1AA", county="OXFORDSHIRE", cat="B"),
        # property 5: type changed between sales (dropped)
        _row(11, 100000, "2005-01-01", "M1 1AA", ptype="T", county="GREATER MANCHESTER"),
        _row(12, 130000, "2009-01-01", "M1 1AA", ptype="D", county="GREATER MANCHESTER"),
    ]
    ppd = tmp_path / "pp.csv"
    with ppd.open("w", newline="") as f:
        csv.writer(f, quoting=csv.QUOTE_ALL).writerows(rows)
    months = pd.date_range("2004-01-01", "2024-12-01", freq="MS")
    cpi = pd.DataFrame({"month": months.strftime("%Y-%m-%d"),
                        "cpi": [100 * 1.002 ** k for k in range(len(months))]})
    cpi_path = tmp_path / "cpi.csv"
    cpi.to_csv(cpi_path, index=False)
    out = tmp_path / "out"
    stats = B.build(str(ppd), str(cpi_path), out, memory_limit="1GB", threads=1)
    return stats, out


def test_pairs_and_filters(built):
    stats, out = built
    assert stats["standard_sales"] == 11          # the category-B sale is dropped
    cells = pd.read_csv(out / "pairs_cells.csv")
    # kept: Leeds 2010->2015, London 2016->2019, London 2019->2023, Cardiff new-build pair
    assert stats["clean_pairs"] == 4
    assert cells.n.sum() == 4
    assert cells[cells.first_sale_new == "Y"].n.sum() == 1
    assert set(cells.region) == {"North and Midlands", "London", "Wales"}


def test_exact_price_and_losses(built):
    _, out = built
    cells = pd.read_csv(out / "pairs_cells.csv")
    main = cells[cells.first_sale_new == "N"]
    assert main.exact_old_price.sum() == 1
    assert main.nominal_loss.sum() == 1             # London 2016 -> 2019
    # the Leeds pair is a real loss with no nominal loss (CPI rose): hidden
    leeds = main[main.region == "North and Midlands"].iloc[0]
    assert leeds.real_loss == 1 and leeds.hidden_loss == 1
    diffs = pd.read_csv(out / "pairs_diffs.csv")
    assert diffs["d0"].sum() == 1


def test_histogram_bins(built):
    _, out = built
    h = pd.read_csv(out / "pairs_hist.csv")
    nom = h[h.kind == "nominal"]
    assert nom.n.sum() == 3                          # new-build pair excluded
    assert int(nom.loc[nom.region == "North and Midlands", "bin"].iloc[0]) == 0
    lon = nom[(nom.region == "London") & (nom.y == 2019)]
    assert int(lon.bin.iloc[0]) == int(__import__("math").floor(__import__("math").log(380 / 400) / B.BIN))
