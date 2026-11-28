import itertools

import pytest

from airline_ops.regulation.ftl import fdp_is_legal, max_fdp_minutes, min_rest_minutes


@pytest.mark.parametrize(
    ("minute", "sectors", "expected"),
    [
        (6 * 60, 1, 780),
        (6 * 60, 2, 780),
        (13 * 60 + 29, 3, 750),
        (13 * 60 + 30, 2, 765),
        (17 * 60, 2, 660),
        (2 * 60, 2, 660),
        (5 * 60, 2, 720),
        (5 * 60 + 45, 2, 765),
        (6 * 60, 10, 540),
        (17 * 60, 6, 540),
    ],
)
def test_table_2_values(minute: int, sectors: int, expected: int) -> None:
    assert max_fdp_minutes(minute, sectors) == expected


def test_table_2_covers_every_minute_and_never_increases_with_sectors() -> None:
    for minute in range(1440):
        limits = [max_fdp_minutes(minute, s) for s in range(1, 11)]
        assert all(a >= b for a, b in itertools.pairwise(limits))
        assert min(limits) >= 540


def test_sectors_out_of_range() -> None:
    with pytest.raises(ValueError, match="1-10"):
        max_fdp_minutes(600, 11)
    with pytest.raises(ValueError, match="1-10"):
        max_fdp_minutes(600, 0)


def test_rest_rules() -> None:
    assert min_rest_minutes(300, at_home_base=False) == 600
    assert min_rest_minutes(300, at_home_base=True) == 720
    assert min_rest_minutes(800, at_home_base=False) == 800
    with pytest.raises(ValueError, match="negative"):
        min_rest_minutes(-1, at_home_base=True)


def test_discretion() -> None:
    assert not fdp_is_legal(20 * 60, 700, 2)
    assert fdp_is_legal(20 * 60, 700, 2, discretion_min=60)
    with pytest.raises(ValueError, match="120"):
        fdp_is_legal(20 * 60, 700, 2, discretion_min=121)
