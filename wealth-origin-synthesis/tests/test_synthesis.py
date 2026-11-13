import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import figures as F  # noqa: E402
import synthesis as S  # noqa: E402


# ---------------------------------------------------------------- verdict rules

def test_estimate_rule_branches():
    assert S.verdict_estimate(-1.0, -1.8, -0.2) == S.PASSED
    assert S.verdict_estimate(-0.3, -1.0, 0.4, reference=-1.2) == S.FAILED  # keeps a quarter
    assert S.verdict_estimate(0.5, -2.0, 3.0, reference=-2.8) == S.FAILED  # sign change
    assert S.verdict_estimate(-0.7, -1.4, 0.05, reference=-1.0) == S.OPEN  # keeps 70%
    assert S.verdict_estimate(0.2, -0.5, 0.9) == S.FAILED  # no reference, wrong sign
    assert S.verdict_estimate(-0.4, -1.3, 0.5) == S.OPEN


def test_p_rule_thresholds():
    assert S.verdict_p(0.049) == S.PASSED
    assert S.verdict_p(0.05) == S.OPEN
    assert S.verdict_p(0.099) == S.OPEN
    assert S.verdict_p(0.10) == S.FAILED


def test_reverse_timing_rule():
    assert S.verdict_reverse_timing(diff_hi=-0.1, back_hi=0.5) == S.PASSED
    assert S.verdict_reverse_timing(diff_hi=0.3, back_hi=-0.1) == S.FAILED
    assert S.verdict_reverse_timing(diff_hi=0.3, back_hi=0.4) == S.OPEN


def test_pretrend_rule_is_an_equivalence_test():
    # a precise lead close to zero passes; a wide one that includes zero does not
    assert S.verdict_pretrend(0.1, -0.4, 0.6, post=-2.0) == S.PASSED
    assert S.verdict_pretrend(0.1, -1.5, 1.7, post=-2.0) == S.OPEN
    assert S.verdict_pretrend(1.5, 0.2, 2.8, post=-2.0) == S.FAILED


def test_placebo_event_rule():
    assert S.verdict_placebo_event(0.8, -1.4, 3.1, main=-2.8) == S.PASSED
    assert S.verdict_placebo_event(-0.5, -2.0, 1.0, main=-2.8) == S.PASSED
    assert S.verdict_placebo_event(-2.0, -3.5, 0.2, main=-2.8) == S.OPEN
    assert S.verdict_placebo_event(-2.0, -3.5, -0.4, main=-2.8) == S.FAILED


def test_leave_one_out_and_ratio_rules():
    assert S.verdict_leave_one_out(1.0, True) == S.PASSED
    assert S.verdict_leave_one_out(0.5, True) == S.OPEN
    assert S.verdict_leave_one_out(1.0, False) == S.FAILED
    assert S.verdict_ratio(1.6) == S.PASSED and S.verdict_ratio(0.6) == S.PASSED
    assert S.verdict_ratio(2.5) == S.OPEN and S.verdict_ratio(0.4) == S.OPEN
    assert S.verdict_ratio(3.8) == S.FAILED and S.verdict_ratio(-1.0) == S.FAILED


def test_sign_test_rule():
    assert S.verdict_sign_test(0.13, 3, 17) == S.FAILED
    assert S.verdict_sign_test(-0.2, 40, 2) == S.PASSED
    assert S.verdict_sign_test(-0.1, 2, 2) == S.OPEN


# ---------------------------------------------------------------- power, type S and M

def test_type_s_m_matches_simulation():
    rng = np.random.default_rng(7)
    theta, se = -0.63, 1.11
    x = rng.normal(theta, se, 2_000_000)
    sig = np.abs(x) > S.Z95 * se
    tm = S.type_s_m(theta, se)
    assert tm["power"] == pytest.approx(sig.mean(), abs=0.002)
    assert tm["type_s"] == pytest.approx((x[sig] > 0).mean(), abs=0.005)
    assert tm["exaggeration"] == pytest.approx(np.abs(x[sig]).mean() / abs(theta), rel=0.01)


def test_type_s_m_limits():
    big = S.type_s_m(-10.0, 1.0)
    assert big["power"] > 0.999 and big["type_s"] < 1e-6
    assert big["exaggeration"] == pytest.approx(1.0, abs=0.01)
    assert S.type_s_m(0.5, 1.0)["type_s"] == pytest.approx(S.type_s_m(-0.5, 1.0)["type_s"])


def test_events_needed_scale_with_inverse_square_of_se():
    p = S.succession_power()
    assert p["multiple_of_events"] == pytest.approx((p["se_post"] / p["se_needed"]) ** 2)
    assert p["se_needed"] == pytest.approx(abs(p["panel_implied_post"]) / (S.Z95 + S.Z80))


# ---------------------------------------------------------------- parsing

def test_cell_text_forms():
    assert S.cell_text("range -1.35 to -0.89; interval excludes zero in 56 of 56", "leave one out") == "56 of 56 drops"
    assert S.cell_text("median -0.064, Stouffer p=0.235", "joint test") == "p = 0.235"
    assert S.cell_text("share of draws at or below -2.79: 0.083", "permutation") == "p = 0.083"
    assert S.cell_text("backward -0.00; forward minus backward -1.16 [-3.44, +0.22]", "reverse timing") \
        == "-1.16 [-3.44, +0.22]"
    assert S.cell_text("-0.131 vs -0.035 pp a year (ratio 3.8)", "ratio") == "3.8 times"


# ---------------------------------------------------------------- the committed inputs

@pytest.fixture(scope="module")
def est():
    return S.estimates_frame()


@pytest.fixture(scope="module")
def led(est):
    return S.ledger_frame(est, S.succession_power())


def test_headline_is_the_within_country_estimate(est):
    specs = S.read("panel", "specs.csv")
    r = S.one(specs, horizon=3, outcome="tfp", transform="log1p", sample="billionaire_countries",
              controls="income", estimator="fe")
    e = est.set_index("key").loc["panel"]
    assert e.b == pytest.approx(r["diff"]) and e.lo == pytest.approx(r["lo_diff"], abs=1e-5)


def test_estimates_are_unique_and_on_one_footing(est):
    assert est["key"].is_unique
    full = est[(est["panel"] == "composition") & ~est["without_2020"]]
    assert (full["b"] < 0).all() and (full["hi"] < 0).all()
    assert set(est["study"]) <= set(S.STUDIES)


def test_ledger_is_complete_and_consistent(led):
    assert led[["objection", "study"]].duplicated().sum() == 0
    assert set(led["verdict"]) <= {S.PASSED, S.OPEN, S.FAILED}
    grouped = {it for _, items in F.GROUPS for it in items}
    assert set(led["objection"]) == grouped
    assert set(led["study"]) == set(S.STUDIES)


def test_one_returns_a_single_row_or_raises():
    df = S.read("panel", "joint_test.csv")
    assert S.one(df, subset="tfp")["n_specs"] == 72
    with pytest.raises(ValueError):
        S.one(df, subset="no such subset")


def test_rule_sensitivity_restores_defaults_and_matches_ledger(est, led):
    power = S.succession_power()
    sens = S.rule_sensitivity(est, power)
    assert S.RULES == S.DEFAULT_RULES
    merged = led.merge(sens, on=["objection", "study"])
    assert len(merged) == len(led)
    assert (merged["verdict"] == merged["default"]).all()
    # the lenient variant never turns a default pass into a failure, and the strict one never
    # turns a default failure into a pass
    assert not ((merged["default"] == S.PASSED) & (merged["lenient"] == S.FAILED)).any()
    assert not ((merged["default"] == S.FAILED) & (merged["strict"] == S.PASSED)).any()
