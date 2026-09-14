"""Focused tests for refresh-sensitive influence-significance semantics."""

from __future__ import annotations

import unittest

from blood_pressure_missingness.validation.current_influence import (
    _validate_significant_deletions,
)


class CurrentInfluenceValidationTests(unittest.TestCase):
    """Allow stronger robustness without weakening scientific-drift detection."""

    def test_partial_significance_remains_valid(self) -> None:
        _validate_significant_deletions([0, 2], 4)

    def test_all_deletions_significant_is_valid_strengthening(self) -> None:
        _validate_significant_deletions([0, 1, 2, 3], 4)

    def test_zero_significant_deletions_is_rejected(self) -> None:
        with self.assertRaisesRegex(
            RuntimeError,
            "no leave-one-day-out within-episode slope remains significantly negative",
        ):
            _validate_significant_deletions([], 4)

    def test_impossible_significant_count_is_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "exceeds the number of observed days"):
            _validate_significant_deletions([0, 1, 2, 3, 4], 4)

    def test_non_list_payload_is_rejected(self) -> None:
        with self.assertRaises(TypeError):
            _validate_significant_deletions((0, 1), 4)


if __name__ == "__main__":
    unittest.main()
