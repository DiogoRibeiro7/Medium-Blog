"""The test suite of the reviewed pipeline: every layer, pointed at Variant()."""

import datetime as dt

import pytest

from mlp import checks, model
from mlp.features import Variant

REVIEWED = Variant()


def test_feature_table_passes_the_schema(con, frames):
    result = checks.check_data_validation(con, frames, REVIEWED)
    assert result.passed, result.detail


@pytest.mark.parametrize("case", [c[0] for c in checks.unit_cases()])
def test_feature_unit_cases(ddl, case):
    results = {r.name: r for r in checks.check_unit_tests(ddl, REVIEWED)}
    assert results[case].passed


def test_features_do_not_see_the_future(con, frames):
    cutoffs = [dt.datetime(2013, 10, 3, 8, 0), dt.datetime(2013, 12, 12, 12, 0)]
    result = checks.check_future_perturbation(con, frames, REVIEWED, cutoffs)
    assert result.passed, result.detail


def test_permuted_labels_leave_nothing_to_learn(con):
    result, _ = checks.check_null_label(con, REVIEWED, perms=2)
    assert result.passed, result.detail


def test_split_and_protocol(con, frames):
    for result in checks.check_split_audit(con, frames, REVIEWED):
        assert result.passed, result.name


def test_batch_and_serving_paths_agree(con, frames):
    result = checks.check_parity(con, frames, REVIEWED, n=200)
    assert result.passed, result.detail


def test_reported_auc_floor(con, frames):
    assert checks.check_metric_gate(model.run(con, frames, REVIEWED)).passed


def test_features_ignore_any_outcome_of_the_flight(ddl):
    result = checks.check_outcome_property(ddl, REVIEWED)
    assert result.passed, result.detail
