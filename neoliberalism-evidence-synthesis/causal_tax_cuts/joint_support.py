"""Audit joint three-dimensional WDI donor support without causal estimation.

The earlier donor audit checked each marginal macro range separately.
This module asks whether each treated pre-reform macro vector belongs to
the finite donors' convex hull. By Carathéodory's theorem in R^3, any
feasible point has a representation using at most four donors. Enumerating
those small simplices avoids a new optimisation dependency.

Witness weights only certify geometric interpolation. They must not be
interpreted as estimated counterfactual weights or causal effects.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from causal_tax_cuts.analysis import Key, Observation, load_panel
from causal_tax_cuts.donor_support import (
    MAX_HORIZON,
    CandidateExposure,
    audit_future_donor_exposure,
)
from causal_tax_cuts.macro_matching import (
    MacroValues,
    pre_reform_means,
    read_macro_controls,
)
from causal_tax_cuts.world_bank import CORE_COLUMNS, INDICATORS

STRICT_TOLERANCE = 1e-9
RELAXED_TOLERANCE = 1e-6
MAX_VERTICES = len(CORE_COLUMNS) + 1


@dataclass(frozen=True, slots=True)
class HullWitness:
    """Convex combination of no more than four donors, on a fixed scale."""

    donors: tuple[str, ...]
    weights: tuple[float, ...]
    max_scaled_residual: float


@dataclass(frozen=True, slots=True)
class JointSupportRow:
    """One reform's joint support among its surviving donors at a horizon."""

    country: str
    reform_year: int
    horizon: int
    n_surviving_donors: int
    within_all_three_marginal_ranges: bool | None
    strict_joint_hull_support: bool | None
    relaxed_joint_hull_support: bool | None
    support_status: str
    witness_donors: str
    witness_weights: str
    witness_max_scaled_residual: float | None


def _certify_simplex(
    standardized_donors: np.ndarray,
    donor_names: list[str],
    tolerance: float,
) -> HullWitness | None:
    """Find a nonnegative simplex certificate for standardized target zero.

    Pre-censor donor means are *centered at the treated mean* and divided
    by one fixed standard deviation per variable. Therefore the target
    is the zero vector, and a feasible certificate has nonnegative weights
    summing to one and a small scaled coordinate residual.
    """
    n_donors, n_variables = standardized_donors.shape
    if n_variables != len(CORE_COLUMNS) or len(donor_names) != n_donors:
        raise ValueError("Invalid standardized donor matrix or identifiers")
    if n_donors == 0 or not np.isfinite(standardized_donors).all():
        raise ValueError("Simplex membership requires finite nonempty donor data")
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("Convex-hull tolerance must be positive and finite")

    target = np.array([0.0] * n_variables + [1.0])
    for size in range(1, min(MAX_VERTICES, n_donors) + 1):
        for subset in itertools.combinations(range(n_donors), size):
            selected = standardized_donors[list(subset), :]
            system = np.vstack((selected.T, np.ones(size)))
            weights, _, _, _ = np.linalg.lstsq(system, target, rcond=None)
            if not np.isfinite(weights).all() or np.min(weights) < -tolerance:
                continue
            # Tiny numerical negatives are removed and the combination is
            # renormalised. The corrected *actual* residual must still pass.
            weights = np.maximum(weights, 0.0)
            total = float(np.sum(weights))
            if total <= 0:
                continue
            weights = weights / total
            residual = float(np.max(np.abs(system @ weights - target)))
            if residual <= tolerance:
                return HullWitness(
                    tuple(donor_names[index] for index in subset),
                    tuple(float(weight) for weight in weights),
                    residual,
                )
    return None


def _joint_witness(
    target: np.ndarray,
    profiles: dict[str, np.ndarray],
    fixed_scales: np.ndarray,
    tolerance: float,
) -> HullWitness | None:
    """Check convex-hull membership for finite, sorted donor profiles."""
    if not profiles:
        return None
    if (
        target.shape != (len(CORE_COLUMNS),)
        or not np.isfinite(target).all()
        or fixed_scales.shape != target.shape
        or not np.isfinite(fixed_scales).all()
        or np.any(fixed_scales <= 0)
    ):
        raise ValueError("Invalid fixed pre-censor WDI scale")
    donors = sorted(profiles)
    matrix = np.vstack([(profiles[name] - target) / fixed_scales for name in donors])
    return _certify_simplex(matrix, donors, tolerance)


def audit_joint_support(
    panel: dict[Key, Observation],
    controls: dict[Key, MacroValues],
) -> tuple[list[JointSupportRow], dict[str, object]]:
    """Recompute only geometric donor support on the same eligibility cohort.

    All event/donor eligibility and later reform censoring are taken from
    donor_support; no information after t-1 enters the macro profiles.
    """
    if not panel or set(panel) != set(controls):
        raise ValueError("WID/WDI country-year keys must be identical")
    reforms, candidates, prior_horizons, previous = audit_future_donor_exposure(
        panel, controls
    )
    matched = sorted(
        (row for row in reforms if row.status == "baseline_donors_available"),
        key=lambda item: (item.reform_year, item.country),
    )
    if not matched:
        raise ValueError("No matchable reform episodes for joint-support audit")
    donor_index: dict[Key, list[CandidateExposure]] = defaultdict(list)
    for candidate in candidates:
        donor_index[candidate.country, candidate.reform_year].append(candidate)
    prior_index = {
        (row.country, row.reform_year, row.horizon): row
        for row in prior_horizons
    }
    rows: list[JointSupportRow] = []

    for event in matched:
        key = (event.country, event.reform_year)
        treated = pre_reform_means(controls, *key, lags=5)
        if treated is None:
            raise ValueError(f"Missing previously eligible WDI profile for {key}")
        entries = sorted(donor_index[key], key=lambda item: item.donor)
        if len(entries) != event.n_pre_censor_donors:
            raise ValueError("Pre-censor donor count does not reconcile")
        profiles: dict[str, np.ndarray] = {}
        for entry in entries:
            donor_profile = pre_reform_means(
                controls, entry.donor, event.reform_year, lags=5
            )
            if donor_profile is None:
                raise ValueError("Previously eligible donor lacks pre-period WDI")
            profiles[entry.donor] = donor_profile

        # This is the same event-fixed scale used for nearest macro distances
        # in the prior donor-support audit, not a horizon-specific metric.
        full_matrix = np.vstack([treated, *profiles.values()])
        scales = np.std(full_matrix, axis=0, ddof=0)
        scales = np.where(scales < 1e-10, 1.0, scales)

        for horizon in range(-1, MAX_HORIZON + 1):
            surviving = {
                entry.donor: profiles[entry.donor]
                for entry in entries
                if entry.first_censored_horizon is None
                or entry.first_censored_horizon > horizon
            }
            prior = prior_index.get((event.country, event.reform_year, horizon))
            if prior is None or len(surviving) != prior.n_surviving_donors:
                raise ValueError(f"Prior horizon risk set differs for {key}, {horizon}")
            strict = _joint_witness(treated, surviving, scales, STRICT_TOLERANCE)
            relaxed = strict or _joint_witness(
                treated, surviving, scales, RELAXED_TOLERANCE
            )
            if not surviving:
                status = "no_donors"
            elif strict is not None:
                status = "inside_strict"
            elif relaxed is not None:
                status = "tolerance_sensitive"
            else:
                status = "outside"
            witness = strict or relaxed
            rows.append(
                JointSupportRow(
                    event.country,
                    event.reform_year,
                    horizon,
                    len(surviving),
                    prior.within_all_three_marginal_ranges,
                    strict is not None if surviving else None,
                    relaxed is not None if surviving else None,
                    status,
                    ";".join(witness.donors) if witness else "",
                    (
                        ";".join(f"{weight:.12g}" for weight in witness.weights)
                        if witness
                        else ""
                    ),
                    (
                        float(f"{witness.max_scaled_residual:.12g}")
                        if witness
                        else None
                    ),
                )
            )

    by_horizon: list[dict[str, int]] = []
    for horizon in range(-1, MAX_HORIZON + 1):
        subset = [row for row in rows if row.horizon == horizon]
        previous_horizon = next(
            item for item in previous["by_horizon"] if item["horizon"] == horizon
        )
        summary = {
            "horizon": horizon,
            "n_baseline_eligible_reforms": len(subset),
            "n_surviving_donor_reform_pairs": sum(
                row.n_surviving_donors for row in subset
            ),
            "n_with_marginal_support": sum(
                row.within_all_three_marginal_ranges is True for row in subset
            ),
            "n_with_strict_joint_hull_support": sum(
                row.strict_joint_hull_support is True for row in subset
            ),
            "n_with_relaxed_joint_hull_support": sum(
                row.relaxed_joint_hull_support is True for row in subset
            ),
            "n_tolerance_sensitive": sum(
                row.support_status == "tolerance_sensitive" for row in subset
            ),
            "n_marginal_but_not_strict_joint": sum(
                row.within_all_three_marginal_ranges is True
                and row.strict_joint_hull_support is False
                for row in subset
            ),
        }
        if (
            summary["n_surviving_donor_reform_pairs"]
            != previous_horizon["n_donor_reform_pairs_at_risk"]
            or summary["n_with_marginal_support"]
            != previous_horizon["n_reforms_with_all_three_marginal_ranges"]
        ):
            raise ValueError("Joint support failed previous donor audit reconciliation")
        if (
            summary["n_with_strict_joint_hull_support"]
            > summary["n_with_relaxed_joint_hull_support"]
        ):
            raise ValueError("Tolerance sensitivity is not nested")
        by_horizon.append(summary)

    report: dict[str, object] = {
        "status": "three_wdi_dimension_joint_convex_hull_diagnostic_only",
        "n_original_observed_reforms": previous["n_original_reform_events"],
        "n_baseline_eligible_reforms": len(matched),
        "n_pre_censor_donor_reform_pairs": len(candidates),
        "matching_reference": {"history_years": 5, "max_horizon": MAX_HORIZON},
        "support_algorithm": (
            "enumerate every 1..4-donor simplex in 3 dimensions, "
            "solve nonnegative barycentric feasibility"
        ),
        "standardization": (
            "per-reform pre-censor treated+donor population standard "
            "deviation, held fixed across horizons"
        ),
        "strict_max_absolute_scaled_coordinate_error": STRICT_TOLERANCE,
        "relaxed_max_absolute_scaled_coordinate_error": RELAXED_TOLERANCE,
        "witness_weights": (
            "nonnegative barycentric interpolation certificate only, "
            "not causal donor weights"
        ),
        "horizon_minus_one": "before same-year and future donor reform exclusions",
        "by_horizon": by_horizon,
        "reconciled_to_prior_marginal_donor_audit": True,
        "inference": (
            "none; no treatment effect, propensity, positivity estimate "
            "or uncertainty interval"
        ),
        "caveats": [
            (
                "Finite-sample geometric interpolation support is neither "
                "a positivity estimate nor a causal exchangeability test."
            ),
            (
                "A donor convex-hull certificate exists only for these "
                "three pre-period WDI means, not for omitted political or "
                "fiscal confounders."
            ),
            (
                "The eligible cohort requires future observed WID outcomes "
                "through t+5 and may suffer outcome-availability selection."
            ),
            (
                "Donors are removed after subsequent reforms; donor and "
                "treated pathways are asymmetric in the original "
                "exploratory contrast."
            ),
            (
                "Barycentric weights certify geometry and cannot be used "
                "without a justified estimation design."
            ),
            (
                "A tolerance-sensitive classification is numerically "
                "indeterminate and must not be declared robust overlap."
            ),
        ],
    }
    return rows, report


def save_joint_support(
    output_dir: Path,
    rows: list[JointSupportRow],
    report: dict[str, object],
    *,
    source_panel: Path,
    provenance: Path | None = None,
    frozen_reference: Path | None = None,
) -> None:
    """Freeze horizon-level certificates, source hashes and audit agreement."""
    if not rows:
        raise ValueError("Refusing to save empty joint-support audit")
    data = dict(report)
    data["joined_panel_sha256"] = hashlib.sha256(source_panel.read_bytes()).hexdigest()
    if provenance is not None:
        manifest = json.loads(provenance.read_text(encoding="utf-8"))
        if manifest.get("publisher") != "World Bank World Development Indicators (DS2)":
            raise ValueError("Unexpected World Bank provenance")
        series = manifest.get("series")
        if (
            not isinstance(series, list)
            or len(series) != len(INDICATORS)
            or {item.get("indicator_code") for item in series}
            != {indicator.code for indicator in INDICATORS}
        ):
            raise ValueError("Expected the same four official WDI series")
        data["source_archive_sha256"] = {
            item["indicator_code"]: item["source_sha256"] for item in series
        }
        data["source_last_updated"] = {
            item["indicator_code"]: item["source_last_updated"] for item in series
        }
    if frozen_reference is not None:
        reference = json.loads(frozen_reference.read_text(encoding="utf-8"))
        if reference.get("joined_panel_sha256") == data["joined_panel_sha256"]:
            references = {
                int(item["horizon"]): item for item in reference["by_horizon"]
            }
            for item in data["by_horizon"]:
                earlier = references[item["horizon"]]
                if (
                    item["n_surviving_donor_reform_pairs"]
                    != earlier["n_donor_reform_pairs_at_risk"]
                    or item["n_with_marginal_support"]
                    != earlier["n_reforms_with_all_three_marginal_ranges"]
                ):
                    raise ValueError("Same-vintage donor support totals mismatch")
            data["frozen_reference_status"] = "same_source_validated"
        else:
            data["frozen_reference_status"] = "source_vintage_changed"

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "joint_support_event_horizons.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(JointSupportRow.__dataclass_fields__)
        )
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)
    (output_dir / "joint_support_audit.json").write_text(
        json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    """Measure geometric support on the verified WID/WDI merged outcome."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--provenance", type=Path)
    parser.add_argument("--frozen-reference", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    panel = load_panel(args.panel, "top1_share_pct", ())
    controls = read_macro_controls(args.panel, set(panel))
    rows, report = audit_joint_support(panel, controls)
    save_joint_support(
        args.output_dir,
        rows,
        report,
        source_panel=args.panel,
        provenance=args.provenance,
        frozen_reference=args.frozen_reference,
    )
    h5 = report["by_horizon"][-1]
    print(
        f"Joint WDI hull at horizon five: "
        f"{h5['n_with_strict_joint_hull_support']} of "
        f"{report['n_baseline_eligible_reforms']} reforms"
    )
    print("CAUTION: finite-sample geometry only; no causality or positivity estimate")


if __name__ == "__main__":
    main()
