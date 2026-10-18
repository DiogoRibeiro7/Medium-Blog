import pytest

from prescribing.cvrisk import pce_risk


@pytest.mark.parametrize("female,black,expected", [(True, False, 0.021), (True, True, 0.030),
                                                   (False, False, 0.053), (False, True, 0.061)])
def test_guideline_worked_example(female, black, expected):
    """Values printed in Goff et al. (2014); the guideline rounds its intermediate sums,
    so agreement is to within 0.1 percentage point."""
    r = pce_risk([55], [female], [black], [213], [50], [120], [False], [0], [0])[0]
    assert r == pytest.approx(expected, abs=0.001)


def test_risk_rises_with_systolic_pressure():
    lo = pce_risk([65], [True], [False], [200], [50], [125], [False], [0], [0])[0]
    hi = pce_risk([65], [True], [False], [200], [50], [135], [False], [0], [0])[0]
    assert hi > lo
