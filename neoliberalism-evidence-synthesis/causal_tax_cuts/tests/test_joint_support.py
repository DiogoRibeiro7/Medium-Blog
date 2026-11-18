"""Synthetic geometry and cohort checks for pre-treatment joint donor support."""

from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from causal_tax_cuts.analysis import Key, Observation
from causal_tax_cuts.joint_support import (
    RELAXED_TOLERANCE,
    STRICT_TOLERANCE,
    _joint_witness,
    audit_joint_support,
    save_joint_support,
)
from causal_tax_cuts.macro_matching import MacroValues


def fixture() -> tuple[dict[Key, Observation], dict[Key, MacroValues]]:
    """Reforms in 2010, 2011 and 2013; three donors never reform."""
    levels = {
        "AAA": (1.0, 2.0, 3.0),
        "BBB": (1.0, 2.0, 3.0),
        "CCC": (3.0, 3.0, 6.0),
        "DDD": (4.0, 5.0, 8.0),
        "EEE": (0.0, 1.0, 2.0),
        "FFF": (7.0, 8.0, 9.0),
    }
    panel: dict[Key, Observation] = {}
    macro: dict[Key, MacroValues] = {}
    for index, (country, values) in enumerate(levels.items()):
        for year in range(2000, 2019):
            event = int(
                (country, year) in {("AAA", 2010), ("EEE", 2011), ("BBB", 2013)}
            )
            panel[country, year] = Observation(
                country, year, event, 10.0 + index + (year - 2000) * 0.1, ()
            )
            macro[country, year] = values
    return panel, macro


class JointGeometryTests(unittest.TestCase):
    """Check strict geometry, degenerate simplices and tolerance guards."""

    def test_interior_tetrahedron_uses_at_most_four_nonnegative_weights(self) -> None:
        profiles = {
            "A": np.array([0.0, 0.0, 0.0]),
            "B": np.array([1.0, 0.0, 0.0]),
            "C": np.array([0.0, 1.0, 0.0]),
            "D": np.array([0.0, 0.0, 1.0]),
        }
        target = np.array([0.25, 0.25, 0.25])
        certificate = _joint_witness(target, profiles, np.ones(3), STRICT_TOLERANCE)
        self.assertIsNotNone(certificate)
        assert certificate is not None
        self.assertEqual(len(certificate.donors), 4)
        self.assertTrue(all(weight >= 0 for weight in certificate.weights))
        self.assertAlmostEqual(sum(certificate.weights), 1.0)
        reconstructed = np.array(certificate.weights) @ np.vstack(
            [profiles[donor] for donor in certificate.donors]
        )
        np.testing.assert_allclose(reconstructed, target, atol=STRICT_TOLERANCE)

    def test_all_marginals_can_pass_outside_joint_tetrahedron(self) -> None:
        donors = {
            "origin": np.zeros(3),
            "x": np.array([1.0, 0.0, 0.0]),
            "y": np.array([0.0, 1.0, 0.0]),
            "z": np.array([0.0, 0.0, 1.0]),
        }
        target = np.array([0.8, 0.8, 0.8])
        self.assertTrue(
            np.all(target >= np.min(list(donors.values()), axis=0))
            and np.all(target <= np.max(list(donors.values()), axis=0))
        )
        self.assertIsNone(_joint_witness(target, donors, np.ones(3), RELAXED_TOLERANCE))

    def test_degenerate_line_uses_two_donors(self) -> None:
        donors = {"left": np.zeros(3), "right": np.full(3, 4.0)}
        certificate = _joint_witness(
            np.full(3, 1.0), donors, np.ones(3), STRICT_TOLERANCE
        )
        self.assertIsNotNone(certificate)
        assert certificate is not None
        self.assertEqual(certificate.donors, ("left", "right"))
        np.testing.assert_allclose(certificate.weights, [0.75, 0.25])

    def test_tolerance_sensitive_boundary_is_not_strict_support(self) -> None:
        target = np.array([1e-7, 0.0, 0.0])
        donors = {"same": np.zeros(3)}
        self.assertIsNone(_joint_witness(target, donors, np.ones(3), STRICT_TOLERANCE))
        self.assertIsNotNone(
            _joint_witness(target, donors, np.ones(3), RELAXED_TOLERANCE)
        )

    def test_missing_donors_and_invalid_scales(self) -> None:
        target = np.zeros(3)
        self.assertIsNone(_joint_witness(target, {}, np.ones(3), STRICT_TOLERANCE))
        with self.assertRaisesRegex(ValueError, "fixed pre-censor"):
            _joint_witness(target, {"A": np.zeros(3)}, np.zeros(3), STRICT_TOLERANCE)
        with self.assertRaisesRegex(ValueError, "finite nonempty"):
            _joint_witness(
                target,
                {"A": np.array([float("nan"), 0.0, 0.0])},
                np.ones(3),
                STRICT_TOLERANCE,
            )
        with self.assertRaisesRegex(ValueError, "tolerance"):
            _joint_witness(target, {"A": np.zeros(3)}, np.ones(3), 0.0)

    def test_witness_is_invariant_to_input_order_and_coordinate_scale(self) -> None:
        donors = {
            "C": np.array([0.0, 1.0, 0.0]),
            "A": np.zeros(3),
            "B": np.array([1.0, 0.0, 0.0]),
            "D": np.array([0.0, 0.0, 1.0]),
        }
        target = np.full(3, 0.2)
        first = _joint_witness(target, donors, np.ones(3), STRICT_TOLERANCE)
        reversed_order = _joint_witness(
            target,
            dict(reversed(list(donors.items()))),
            np.array([2.0, 4.0, 0.5]),
            STRICT_TOLERANCE,
        )
        self.assertIsNotNone(first)
        self.assertIsNotNone(reversed_order)
        assert first is not None and reversed_order is not None
        self.assertEqual(first.donors, reversed_order.donors)
        np.testing.assert_allclose(first.weights, reversed_order.weights, atol=1e-12)


class JointCohortTests(unittest.TestCase):
    """Keep source, exposure and marginal risk-set diagnostics unchanged."""

    def test_donor_censoring_is_nested_and_joint_support_never_reappears(self) -> None:
        panel, controls = fixture()
        rows, report = audit_joint_support(panel, controls)
        aaa = {
            row.horizon: row
            for row in rows
            if (row.country, row.reform_year) == ("AAA", 2010)
        }
        self.assertEqual(
            [aaa[h].n_surviving_donors for h in range(-1, 6)],
            [5, 5, 4, 4, 3, 3, 3],
        )
        self.assertTrue(aaa[-1].strict_joint_hull_support)
        self.assertFalse(aaa[5].strict_joint_hull_support)
        for event in {(row.country, row.reform_year) for row in rows}:
            event_rows = sorted(
                (row for row in rows if (row.country, row.reform_year) == event),
                key=lambda row: row.horizon,
            )
            for previous, current in zip(event_rows, event_rows[1:], strict=False):
                self.assertGreaterEqual(
                    previous.n_surviving_donors, current.n_surviving_donors
                )
                self.assertFalse(
                    current.strict_joint_hull_support is True
                    and previous.strict_joint_hull_support is not True
                )
        self.assertTrue(report["reconciled_to_prior_marginal_donor_audit"])
        self.assertEqual(len(report["by_horizon"]), 7)

    def test_only_pre_reform_macro_data_changes_support(self) -> None:
        panel, controls = fixture()
        original, _ = audit_joint_support(panel, controls)
        for key in list(controls):
            if key[1] >= 2010:
                controls[key] = (None, None, None)
        modified, _ = audit_joint_support(panel, controls)
        self.assertEqual(
            [row for row in original if row.country == "AAA"],
            [row for row in modified if row.country == "AAA"],
        )

    def test_reject_nonidentical_join_and_empty_events(self) -> None:
        panel, controls = fixture()
        controls.pop(("AAA", 2008))
        with self.assertRaisesRegex(ValueError, "identical"):
            audit_joint_support(panel, controls)
        panel, controls = fixture()
        for key, row in list(panel.items()):
            if row.treated:
                panel[key] = Observation(row.country, row.year, 0, row.outcome, ())
        with self.assertRaisesRegex(ValueError, "No matchable"):
            audit_joint_support(panel, controls)

    def test_frozen_source_reconciles_and_rejects_changed_prior_counts(self) -> None:
        panel, controls = fixture()
        rows, report = audit_joint_support(panel, controls)
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            source = output / "source.csv"
            source.write_text("synthetic\n", encoding="utf-8")
            prior = output / "prior.json"
            original = {
                "joined_panel_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "by_horizon": [
                    {
                        "horizon": item["horizon"],
                        "n_donor_reform_pairs_at_risk": item[
                            "n_surviving_donor_reform_pairs"
                        ],
                        "n_reforms_with_all_three_marginal_ranges": item[
                            "n_with_marginal_support"
                        ],
                    }
                    for item in report["by_horizon"]
                ],
            }
            prior.write_text(json.dumps(original), encoding="utf-8")
            save_joint_support(
                output, rows, report, source_panel=source, frozen_reference=prior
            )
            saved = json.loads((output / "joint_support_audit.json").read_text())
            self.assertEqual(saved["frozen_reference_status"], "same_source_validated")
            with (output / "joint_support_event_horizons.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), len(rows))
            original["by_horizon"][-1]["n_donor_reform_pairs_at_risk"] += 1
            prior.write_text(json.dumps(original), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Same-vintage"):
                save_joint_support(
                    output, rows, report, source_panel=source, frozen_reference=prior
                )
            source.write_text("new vintage\n", encoding="utf-8")
            save_joint_support(
                output, rows, report, source_panel=source, frozen_reference=prior
            )
            revised = json.loads((output / "joint_support_audit.json").read_text())
            self.assertEqual(
                revised["frozen_reference_status"], "source_vintage_changed"
            )


if __name__ == "__main__":
    unittest.main()
