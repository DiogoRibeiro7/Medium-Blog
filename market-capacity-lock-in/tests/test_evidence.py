import hashlib
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

import evidence  # noqa: E402


def test_archived_checksums():
    arch = HERE / "data" / "archived"
    for line in (arch / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        assert hashlib.sha256((arch / name.strip("* ")).read_bytes()).hexdigest() == digest, name


@pytest.fixture(scope="module")
def ev():
    return evidence.load_all()


def test_farm_age_portugal_2023(ev):
    r = ev["farm_age"].loc[("PT", 2023)]
    assert r.ge65_share == pytest.approx(51.9, abs=0.05)
    assert r.TOTAL == pytest.approx(261500)


def test_labour_input_falls_while_output_rises(ev):
    al, ao = ev["agri_labour"], ev["agri_output"]
    assert al.loc[2025, "total"] < 0.5 * al.loc[1995, "total"]
    assert ao.loc[2025, "output_volume"] > ao.loc[1995, "output_volume"]


def test_housing_series(ev):
    h = ev["housing_obs"]
    assert h.loc[2025, "completions_long"] < h.loc[1975, "completions_long"]
    assert int(h.completions.idxmax()) == 2002
    assert h.loc[2025, "price_to_wage"] == pytest.approx(164.6, abs=0.1)


def test_construction_asymmetry(ev):
    ind = ev["industry"]
    down = evidence.lograte(ind.loc[2008, "emp_constr"], ind.loc[2013, "emp_constr"], 5)
    up = evidence.lograte(ind.loc[2014, "emp_constr"], ind.loc[2025, "emp_constr"], 11)
    assert down == pytest.approx(-0.1118, abs=1e-4)
    assert up == pytest.approx(0.0365, abs=1e-4)
