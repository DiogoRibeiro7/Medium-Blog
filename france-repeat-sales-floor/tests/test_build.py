import math
import sys
from pathlib import Path

import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import build as B  # noqa: E402
import synth  # noqa: E402


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("dvf")
    files = synth.write_dvf(tmp / "dvf")
    cpi = synth.write_cpi(tmp / "cpi.csv")
    out = tmp / "out"
    stats = B.build(files, cpi, out, memory_limit="1GB", threads=1, temp=str(tmp / "duck"))
    return stats, out


def test_single_dwelling_mutations_and_keys(built):
    stats, _ = built
    # 2 deeds are not single dwellings (two houses; a flat with a shop): 22 deeds - 2
    assert stats["mutations"] == 22
    assert stats["single_dwelling_mutations"] == 20
    assert stats["keyed_sales"] == 20
    assert stats["same_day_duplicates_dropped"] == 2


def test_pairs_and_filters(built):
    stats, out = built
    cells = pd.read_csv(out / "pairs_cells.csv")
    main = cells[(cells.first_sale == "sale") & (cells.second_sale == "sale") & cells.same_struct]
    # main: Bordeaux house, Paris flat x2, second Paris flat
    assert stats["main_pairs"] == 4
    assert main.n.sum() == 4
    assert cells[cells.first_sale == "off-plan"].n.sum() == 1
    assert cells[cells.second_sale == "auction"].n.sum() == 1
    assert cells[~cells.same_struct].n.sum() == 1               # the extended house
    assert set(main.region) == {"Big cities", "Paris"}
    # the three-month resale is dropped; the parcel of the two-house deed and the flat of
    # the deed with a shop have no earlier single-dwelling sale, so no pair
    assert stats["pairs_after_price_and_time_filters"] == 7


def test_lot_numbers_are_normalised_and_flats_kept_apart(built):
    _, out = built
    cells = pd.read_csv(out / "pairs_cells.csv")
    paris = cells[(cells.region == "Paris") & (cells.first_sale == "sale")]
    assert paris.n.sum() == 3
    assert paris.nominal_loss.sum() == 1


def test_exact_price_hist_and_no_identifiers(built):
    _, out = built
    diffs = pd.read_csv(out / "pairs_diffs.csv")
    assert diffs["d0"].sum() == 1
    assert diffs["d10000"].sum() == 1                             # the second Paris flat
    h = pd.read_csv(out / "pairs_hist.csv")
    nom = h[h.kind == "nominal"]
    assert nom.n.sum() == 4
    assert int(nom.loc[nom.region == "Big cities", "bin"].iloc[0]) == 0
    loss = nom[(nom.region == "Paris") & (nom.y == 2023)]
    assert int(loss.bin.iloc[0]) == math.floor(math.log(380 / 400) / B.BIN)
    for f in out.glob("*.csv"):
        text = f.read_text()
        assert "000CD0002" not in text and "000AB0001" not in text, f.name


def test_real_far_kind_needs_cumulative_inflation(built):
    _, out = built
    h = pd.read_csv(out / "pairs_hist.csv")
    # CPI rises 0.3% a month, so only holds of 32 months or more reach 10%; the one such main
    # pair (the Bordeaux house, 39 months) is an exact resale, excluded from this kind
    far = h[h.kind == "real_far_excl_exact"]
    assert far.n.sum() == 0
    qq = pd.read_csv(out / "pairs_qq.csv")
    assert qq.n.sum() == 4
