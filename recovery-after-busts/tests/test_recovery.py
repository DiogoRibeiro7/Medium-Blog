import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cure  # noqa: E402
from episodes import find_episodes, relative  # noqa: E402
from stages import stage, transition  # noqa: E402


def test_one_episode_with_recovery():
    years = np.arange(2000, 2020)
    v = np.r_[np.linspace(0, 0.1, 5), [0.0, -0.2, -0.3, -0.25, -0.1, 0.05, 0.12], np.linspace(0.13, 0.2, 8)]
    eps = find_episodes(pd.Series(v, index=years), theta=0.2)
    assert len(eps) == 1
    e = eps[0]
    assert e["peak_year"] == 2004 and e["trough_year"] == 2007
    assert e["regained_year"] == 2011
    assert np.isclose(e["peak_log"] - e["trough_log"], 0.4)


def test_no_episode_below_threshold():
    years = np.arange(1990, 2010)
    v = 0.1 * np.sin(np.arange(20))
    assert find_episodes(pd.Series(v, index=years), theta=0.2) == []


def test_unrecovered_episode_has_no_regain_year():
    years = np.arange(1990, 2010)
    v = np.r_[np.linspace(0, 0.3, 10), np.linspace(0.0, 0.1, 10)]
    eps = find_episodes(pd.Series(v, index=years), theta=0.2)
    assert len(eps) == 1 and eps[0]["regained_year"] is None


def test_stages():
    assert stage("PT", 1984) == "pre-accession"
    assert stage("PT", 1990) == "customs union"
    assert stage("PT", 1995) == "single market"
    assert stage("PT", 2009) == "euro"
    assert stage("FR", 1965) == "pre-accession"
    assert stage("FR", 1975) == "customs union"
    assert stage("AT", 1994) == "single market"
    assert stage("AT", 1993) == "pre-accession"
    assert stage("DK", 2010) == "single market"
    assert stage("UK", 2015) == "single market"
    assert stage("UK", 2022) == "outside"
    assert stage("US", 1980) == "outside"
    assert stage("EL", 2000) == "single market" and stage("EL", 2001) == "euro"
    assert transition("PL", 2000) and not transition("PL", 2009) and not transition("PT", 1980)


def test_relative_removes_common_trend():
    years = np.arange(2000, 2021)
    common = -0.03 * (years - 2000)
    p = pd.DataFrame({c: common for c in "ABCDE"}, index=years)
    p["E"] = common + np.where(years >= 2010, -0.4, 0.0)
    r = relative(p)
    # countries that only follow the common trend stay flat apart from E's shock spread over the peers
    assert np.allclose(r["A"].diff().dropna().loc[2011:], 0.0)
    assert np.isclose(r["E"].loc[2020] - r["E"].loc[2009], -0.4)


def test_cure_model_recovers_parameters():
    cure._check()


def test_kaplan_meier_simple():
    s = cure.km(np.array([1, 2, 3, 4]), np.array([1, 1, 0, 1]), np.array([0.5, 1, 2, 3, 4]))
    assert np.allclose(s, [1.0, 0.75, 0.5, 0.5, 0.0])
