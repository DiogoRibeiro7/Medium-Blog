"""Selection audit, policy pairing and leakage checks."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analysis import simulate_data
from cost_selection import (
    DEFAULT_COSTS,
    paired_policy_regret,
    run_cost_selection_experiment,
    score_validation,
    select_policies,
    validate_design,
    validation_audit,
)
from stress_analysis import apply_stress


def test_score_components_have_correct_simple_values() -> None:
    observed = np.array([0.0, 0.7, 1.3], dtype=np.float64)
    predicted = np.array([0.0, 0.9, 1.4], dtype=np.float64)
    mse, nominal, averaged = score_validation(
        observed, predicted, costs=(0.5, 1.0), nominal_cost=0.5
    )
    assert mse == pytest.approx((0.04 + 0.01) / 3)
    assert nominal == pytest.approx((0.2 + 0.8) / 3)
    assert averaged == pytest.approx(
        0.5 * ((0.2 + 0.8) / 3 + 0.3 / 3)
    )


@pytest.mark.parametrize(
    "costs",
    [(0.6, 0.3), (0.3, 0.3), (-0.2, 0.5), (0.2, float("inf")), (0.4,)],
)
def test_reject_invalid_cost_grids(costs: tuple[float, ...]) -> None:
    with pytest.raises(ValueError):
        validate_design(costs, nominal_cost=0.6, split_count=2)


def test_reject_invalid_split_and_nominal_cost() -> None:
    with pytest.raises(ValueError):
        validate_design(DEFAULT_COSTS, nominal_cost=-1.0, split_count=2)
    with pytest.raises(ValueError):
        validate_design(DEFAULT_COSTS, nominal_cost=0.6, split_count=1)
    with pytest.raises(TypeError):
        validate_design(DEFAULT_COSTS, nominal_cost=0.6, split_count=True)


def test_selection_is_deterministic_and_uses_only_training_data() -> None:
    original = simulate_data(seed=34, n_train=80, n_test=40)
    observed = apply_stress(original, stress="mar_outliers", seed=54)
    audit_one = validation_audit(
        observed.x_train, observed.y_train, split_count=2, seed=81
    )
    # Modify the entire held-out test set. This cannot influence audit scores,
    # because the function has no interface accepting the test data.
    altered_test_y = observed.y_test + 10000.0
    assert not np.array_equal(altered_test_y, observed.y_test)
    audit_two = validation_audit(
        observed.x_train.copy(), observed.y_train.copy(),
        split_count=2, seed=81,
    )
    pd.testing.assert_frame_equal(audit_one, audit_two)
    selected = select_policies(audit_one, split_count=2)
    assert len(selected) == 6
    assert set(selected["number_of_splits"]) == {1, 2}
    stable = selected.loc[selected["protocol"] == "Repeated splits"]
    assert stable["split_agreement"].between(0.0, 1.0).all()
    assert selected.loc[
        selected["protocol"] == "Single split", "split_agreement"
    ].isna().all()


def test_selection_chooses_expected_criterion_winners() -> None:
    audit = pd.DataFrame(
        [
            {"split": j, "model": name, "validation_mse": mse,
             "validation_nominal_utility": nominal,
             "validation_grid_utility": grid}
            for j in range(2)
            for name, mse, nominal, grid in (
                ("A", 1.0, -3.0, 4.0),
                ("B", 2.0, 6.0, 1.0),
            )
        ]
    )
    selected = select_policies(audit, split_count=2)
    got = dict(zip(selected["policy"], selected["selected_model"]))
    assert got["Repeated splits / RMSE"] == "A"
    assert got["Repeated splits / Nominal utility"] == "B"
    assert got["Repeated splits / Cost-grid utility"] == "A"
    assert all(value == 1.0 for value in
               selected.loc[selected["protocol"] == "Repeated splits",
                            "split_agreement"])


def test_end_to_end_runs_independent_paired_evaluation() -> None:
    evaluations, audit = run_cost_selection_experiment(
        repeats=1,
        n_train_values=(80,),
        n_test=60,
        split_count=2,
        scenarios=("low_variance_signal",),
        stresses=("clean", "mar_outliers"),
        seed=13,
    )
    assert len(evaluations) == 2 * 6 * len(DEFAULT_COSTS)
    assert len(audit) == 2 * 2 * 6
    assert len(evaluations["selected_model"].unique()) >= 1
    assert not evaluations.duplicated(
        ["scenario", "stress", "n_train", "replicate", "cost", "policy"]
    ).any()
    assert np.isfinite(
        evaluations[["regret", "test_rmse", "expected_profit"]]
    ).all().all()
    assert (evaluations["regret"] >= -1e-9).all()
    # The oracle's full-information expected payoff is paired across stresses.
    for cost in DEFAULT_COSTS:
        rows = evaluations.loc[evaluations["cost"] == cost]
        assert rows["oracle_profit"].max() == pytest.approx(
            rows["oracle_profit"].min()
        )


def test_paired_regret_difference_has_correct_sign() -> None:
    records: list[dict[str, object]] = []
    for rep, baseline, new in ((0, 0.4, 0.2), (1, 0.3, 0.4)):
        for policy, regret in (("baseline", baseline), ("new", new)):
            records.append({
                "scenario": "low_variance_signal", "stress": "clean",
                "n_train": 80, "replicate": rep,
                "cost": 0.6, "policy": policy, "regret": regret,
            })
    report = paired_policy_regret(
        pd.DataFrame(records), "baseline", "new",
        bootstrap_draws=100, seed=4
    )
    assert report.iloc[0]["mean_regret_change"] == pytest.approx(-0.05)
    assert report.iloc[0]["bootstrap_lower_95"] >= -0.2 - 1e-12
    assert report.iloc[0]["bootstrap_upper_95"] <= 0.1 + 1e-12
    with pytest.raises(ValueError):
        paired_policy_regret(
            pd.DataFrame(records).iloc[:1], "baseline", "new"
        )
