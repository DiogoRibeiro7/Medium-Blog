"""Preregistered, stream-level evidence synthesis for detector economics.

The primary experiment compares GLR and BOCPD to CUSUM in one prespecified
scenario, intervention cost, retraining cost and feedback-delay condition.

It allocates the nominal 5% family error budget across two families:
paired economic differences and independent null false-alarm bounds.
Each family is Bonferroni-adjusted over the two competing detectors.

Bootstrapped percentile bounds are approximate *Monte Carlo* intervals
conditional on the pre-calibrated detector thresholds; they are neither
finite-sample guarantees nor confidence statements about external systems.
Streams are the independent sampling unit. Multiple policy/cost/time records
within a stream never count as independent Monte Carlo observations.

No positive conclusion is emitted until the minimum number of independent
streams has actually been evaluated. Synthetic conditional means are used
only by the upstream regret evaluator, not the model or alarm threshold.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import sqrt

import numpy as np
import pandas as pd
from scipy.stats import norm


@dataclass(frozen=True)
class PrimaryProtocol:
    """Immutable primary scientific and economic comparison specification."""

    scenario: str = "low_variance_signal"
    drift_regime: str = "gradual_drift"
    null_regime: str = "stable"
    delay: int = 0
    action_cost: float = 0.6
    refit_cost: float = 0.05
    reference: str = "CUSUM"
    candidates: tuple[str, ...] = ("GLR", "BOCPD")
    meaningful_gain: float = 0.01
    max_false_alarm: float = 0.10
    family_error: float = 0.05
    minimum_streams: int = 100
    bootstrap_draws: int = 2000
    bootstrap_seed: int = 42021

    def __post_init__(self) -> None:
        """Reject post-hoc-like configuration mistakes and invalid parameters."""
        for field_name, minimum in (
            ("delay", 0),
            ("minimum_streams", 3),
            ("bootstrap_draws", 100),
            ("bootstrap_seed", 0),
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{field_name} must be an integer")
            if value < minimum:
                raise ValueError(f"{field_name} must be >= {minimum}")

        if not self.scenario or not self.drift_regime or not self.null_regime:
            raise ValueError("Scenario and regime labels must be nonempty")
        if not self.reference or not self.candidates:
            raise ValueError("Reference and candidate methods must be specified")
        if (
            self.reference in self.candidates
            or len(set(self.candidates)) != len(self.candidates)
            or any(not label for label in self.candidates)
        ):
            raise ValueError("Candidate labels must be unique and exclude the reference")

        for field_name, strictly_positive in (
            ("action_cost", False),
            ("refit_cost", False),
            ("meaningful_gain", True),
            ("max_false_alarm", True),
            ("family_error", True),
        ):
            value = getattr(self, field_name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not np.isfinite(value)
                or (value <= 0 if strictly_positive else value < 0)
            ):
                raise ValueError(f"{field_name} must be a valid nonnegative value")
        if self.max_false_alarm >= 1 or self.family_error >= 1:
            raise ValueError("Probability thresholds must be smaller than one")

    def metadata(self) -> dict[str, object]:
        """Return a serialisable protocol without untracked mutable state."""
        return asdict(self)


def _check_table(
    frame: pd.DataFrame,
    required: set[str],
    unique: list[str],
    name: str,
) -> None:
    """Check finiteness, presence, and row identity without replacing source data."""
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError(f"{name} must be a nonempty pandas DataFrame")
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{name} missing required columns: {sorted(missing)}")
    if frame.duplicated(unique).any():
        raise ValueError(f"{name} has duplicate stream, step, policy or cost rows")


def validate_evaluation(
    models: pd.DataFrame,
    economics: pd.DataFrame,
) -> None:
    """Verify the panel structure and economic accounting of upstream data."""
    common = ["scenario", "regime", "delay", "replicate", "policy", "step"]
    _check_table(
        models,
        set(common + ["alarm", "squared_error", "refit_event"]),
        common,
        "models",
    )
    economic_identity = common + ["action_cost", "refit_cost"]
    _check_table(
        economics,
        set(economic_identity + [
            "net_regret", "expected_net_profit", "oracle_profit"
        ]),
        economic_identity,
        "economics",
    )
    for name, frame, metric_fields in (
        ("models", models, ["alarm", "squared_error", "refit_event"]),
        (
            "economics",
            economics,
            ["net_regret", "expected_net_profit", "oracle_profit"],
        ),
    ):
        if not np.isfinite(frame[metric_fields].to_numpy(dtype=float)).all():
            raise ValueError(f"Nonfinite metrics in {name}")

    expected = (
        economics["oracle_profit"].to_numpy(dtype=float)
        - economics["expected_net_profit"].to_numpy(dtype=float)
    )
    if not np.allclose(
        economics["net_regret"].to_numpy(dtype=float),
        expected,
        atol=1e-9,
        rtol=1e-9,
    ):
        raise ValueError("Regret must equal oracle profit minus net profit")
    if (economics["net_regret"] < -1e-8).any():
        raise ValueError("Negative economic regret violates the declared oracle")


def _slice_primary(
    data: pd.DataFrame, protocol: PrimaryProtocol, regime: str
) -> pd.DataFrame:
    """Extract one predeclared case, with exact cost filtering if present."""
    mask = (
        data["scenario"].eq(protocol.scenario)
        & data["regime"].eq(regime)
        & data["delay"].eq(protocol.delay)
    )
    if "action_cost" in data.columns:
        mask &= np.isclose(
            data["action_cost"].to_numpy(dtype=float),
            protocol.action_cost,
            rtol=0,
            atol=1e-12,
        )
    if "refit_cost" in data.columns:
        mask &= np.isclose(
            data["refit_cost"].to_numpy(dtype=float),
            protocol.refit_cost,
            rtol=0,
            atol=1e-12,
        )
    result = data.loc[mask].copy()
    if result.empty:
        raise ValueError(f"No data for preregistered {regime} case")
    return result


def _per_stream_net_regret(
    economics: pd.DataFrame, protocol: PrimaryProtocol
) -> pd.DataFrame:
    """Reduce a complete primary case to one mean regret per policy/stream."""
    selected = _slice_primary(economics, protocol, protocol.drift_regime)
    required = (protocol.reference,) + protocol.candidates
    selected = selected.loc[selected["policy"].isin(required)]
    if selected.empty:
        raise ValueError("Primary case includes no specified detector policies")
    frame = selected.groupby(
        ["replicate", "policy"], as_index=False
    ).agg(
        net_regret=("net_regret", "mean"),
        steps=("step", "nunique"),
    )
    repetitions = frame.groupby("replicate")["policy"].nunique()
    if repetitions.ne(len(required)).any():
        raise ValueError("One or more primary streams lack a detector policy")
    if frame["steps"].nunique() != 1:
        raise ValueError("Primary streams have unequal deployment horizons")
    return frame


def _one_sided_wilson_upper(
    alarm_count: int,
    streams: int,
    upper_tail_error: float,
) -> float:
    """One-sided Wilson upper probability bound on independent null streams."""
    if streams < 1 or alarm_count < 0 or alarm_count > streams:
        raise ValueError("Invalid null-stream alarm counts")
    if not 0 < upper_tail_error < 1:
        raise ValueError("upper_tail_error must lie strictly between 0 and 1")
    z = float(norm.ppf(1.0 - upper_tail_error))
    p = alarm_count / streams
    z2 = z * z
    denominator = 1 + z2 / streams
    centre = (p + z2 / (2 * streams)) / denominator
    spread = z / denominator * sqrt(
        p * (1 - p) / streams + z2 / (4 * streams**2)
    )
    return min(1.0, centre + spread)


def _null_alarm_bounds(
    models: pd.DataFrame, protocol: PrimaryProtocol
) -> pd.DataFrame:
    """Estimate detector-specific null alarms and simultaneous upper bounds.

    Half the global family-error budget is reserved for the alarm family,
    with Bonferroni correction over the predeclared candidates.
    """
    null = _slice_primary(models, protocol, protocol.null_regime)
    null = null.loc[null["policy"].isin(protocol.candidates)]
    flags = null.groupby(
        ["replicate", "policy"], as_index=False
    ).agg(
        any_alarm=("alarm", lambda x: int((x > 0).any())),
        steps=("step", "nunique"),
    )
    if flags.empty or flags["steps"].nunique() != 1:
        raise ValueError("Null streams are absent or have unequal horizons")
    if flags.groupby("replicate")["policy"].nunique().ne(
        len(protocol.candidates)
    ).any():
        raise ValueError("Missing candidate in one or more null streams")
    one_sided_error = protocol.family_error / (
        2.0 * len(protocol.candidates)
    )
    output: list[dict[str, str | int | float]] = []
    for name, frame in flags.groupby("policy", sort=True):
        n = len(frame)
        count = int(frame["any_alarm"].sum())
        output.append(
            {
                "candidate": name,
                "null_streams": n,
                "null_streams_with_alarm": count,
                "empirical_false_alarm": count / n,
                "simultaneous_upper_false_alarm": _one_sided_wilson_upper(
                    count, n, one_sided_error
                ),
            }
        )
    return pd.DataFrame.from_records(output)


def preregistered_verdict(
    models: pd.DataFrame,
    economics: pd.DataFrame,
    protocol: PrimaryProtocol = PrimaryProtocol(),
) -> pd.DataFrame:
    """Evaluate the declared economic and null false-alarm constraints.

    Primary statistic: stream-paired mean net-regret difference to CUSUM,
    candidate minus CUSUM. A negative difference favours the candidate.
    Bootstrap confidence limits are Bonferroni-adjusted over candidates,
    taking half of the total error budget. The other half is allocated to
    the null false-alarm family, also adjusted over candidates.

    Status 'supported' requires both a strictly material economic gain and
    an adequately bounded null alarm rate, plus the declared minimum stream
    count. Unsupported/conclusive negative evidence is distinguished from
    inconclusive uncertainty; the results never declare a winner by rank.
    """
    validate_evaluation(models, economics)
    per_stream = _per_stream_net_regret(economics, protocol)
    alarm = _null_alarm_bounds(models, protocol)
    base = per_stream.loc[
        per_stream["policy"] == protocol.reference,
        ["replicate", "net_regret"],
    ].rename(columns={"net_regret": "reference_regret"})
    if base.empty:
        raise ValueError("Reference detector has no primary streams")

    # Two-sided confidence per candidate uses alpha/(2 families * K methods),
    # then splits the per-comparison error across the two tails.
    per_comparison_error = protocol.family_error / (
        2.0 * len(protocol.candidates)
    )
    tail_probability = per_comparison_error / 2.0
    rng = np.random.default_rng(protocol.bootstrap_seed)
    rows: list[dict[str, str | int | float]] = []
    for candidate in protocol.candidates:
        alternative = per_stream.loc[
            per_stream["policy"] == candidate,
            ["replicate", "net_regret"],
        ].rename(columns={"net_regret": "candidate_regret"})
        if alternative.empty:
            raise ValueError(f"No observations for candidate {candidate}")
        paired = base.merge(
            alternative,
            on="replicate",
            how="outer",
            validate="one_to_one",
        )
        if paired[["reference_regret", "candidate_regret"]].isna().any().any():
            raise ValueError("Candidate/reference primary streams do not pair")

        deltas = (
            paired["candidate_regret"] - paired["reference_regret"]
        ).to_numpy(dtype=float)
        n = deltas.size
        if n >= 3:
            indices = rng.integers(
                0, n, size=(protocol.bootstrap_draws, n)
            )
            bootstrap_means = deltas[indices].mean(axis=1)
            lower, upper = np.quantile(
                bootstrap_means,
                [tail_probability, 1.0 - tail_probability],
            )
            mc_se = float(np.std(deltas, ddof=1) / sqrt(n))
        else:
            lower = upper = mc_se = float("nan")

        matching = alarm.loc[alarm["candidate"] == candidate]
        if len(matching) != 1:
            raise ValueError(f"Null alarm summary missing candidate {candidate}")
        null = matching.iloc[0]
        false_alarm_upper = float(null["simultaneous_upper_false_alarm"])
        ready = (
            n >= protocol.minimum_streams
            and int(null["null_streams"]) >= protocol.minimum_streams
        )
        if not ready:
            status = "exploratory_insufficient_streams"
        elif upper < -protocol.meaningful_gain and (
            false_alarm_upper <= protocol.max_false_alarm
        ):
            status = "supported_in_simulation"
        elif lower > -protocol.meaningful_gain:
            status = "material_gain_not_supported"
        elif upper < -protocol.meaningful_gain:
            status = "false_alarm_constraint_unverified"
        else:
            status = "inconclusive"

        rows.append(
            {
                "candidate": candidate,
                "reference": protocol.reference,
                "independent_primary_streams": n,
                "mean_delta_regret": float(deltas.mean()),
                "mc_standard_error": mc_se,
                "bonferroni_lower": float(lower),
                "bonferroni_upper": float(upper),
                "economically_meaningful_threshold": -protocol.meaningful_gain,
                "independent_null_streams": int(null["null_streams"]),
                "null_alarm_count": int(null["null_streams_with_alarm"]),
                "empirical_false_alarm": float(null["empirical_false_alarm"]),
                "simultaneous_upper_false_alarm": false_alarm_upper,
                "false_alarm_limit": protocol.max_false_alarm,
                "status": status,
            }
        )
    return pd.DataFrame.from_records(rows)


def descriptive_context(
    models: pd.DataFrame, economics: pd.DataFrame
) -> pd.DataFrame:
    """Describe RMSE, alarms, refits and net regret per independent stream.

    No inferential comparison or winner is inferred from these summaries.
    All outcomes are averaged by stream *before* group-level aggregation.
    """
    validate_evaluation(models, economics)
    stream = ["scenario", "regime", "delay", "replicate", "policy"]
    model = models.groupby(stream, as_index=False).agg(
        rmse=("squared_error", lambda x: float(np.sqrt(np.mean(x)))),
        any_alarm=("alarm", lambda x: float((x > 0).any())),
        refits=("refit_event", "sum"),
    )
    economic = economics.groupby(
        stream + ["action_cost", "refit_cost"], as_index=False
    ).agg(net_regret=("net_regret", "mean"))
    records = model.merge(
        economic, on=stream, how="outer", validate="one_to_many"
    )
    if records[["rmse", "net_regret"]].isna().any().any():
        raise ValueError("Unpaired model and economic results")
    grouping = [
        "scenario", "regime", "delay", "policy", "action_cost", "refit_cost"
    ]
    return records.groupby(grouping, as_index=False).agg(
        independent_streams=("replicate", "nunique"),
        mean_rmse=("rmse", "mean"),
        probability_any_alarm=("any_alarm", "mean"),
        mean_refits=("refits", "mean"),
        mean_net_regret=("net_regret", "mean"),
    )
