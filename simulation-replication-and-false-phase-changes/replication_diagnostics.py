"""Reproduce Monte Carlo precision calculations used in the companion article.

The module intentionally uses only the Python standard library so that the
article's numerical claims can be checked without a project-specific runtime.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ReplicationDiagnostic:
    """Summary of Monte Carlo precision for a Bernoulli regime indicator."""

    replications: int
    probability: float
    wrong_side_probability: float
    maximum_mcse: float


def _validate_probability(value: float, *, name: str) -> None:
    """Validate that a probability lies strictly between zero and one."""

    if not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric, got {type(value).__name__}.")
    if not 0.0 < float(value) < 1.0:
        raise ValueError(f"{name} must lie strictly between 0 and 1.")


def wrong_majority_probability(replications: int, probability: float) -> float:
    """Return the probability that an odd-sample majority lands below 0.5.

    Parameters
    ----------
    replications:
        Positive odd number of independent Bernoulli replications.
    probability:
        Bernoulli success probability. This function is intended for
        ``probability > 0.5`` so that the returned value is the probability of
        classifying the process on the wrong side of a majority threshold.

    Returns
    -------
    float
        ``P(Binomial(replications, probability) <= (replications - 1) / 2)``.
    """

    if not isinstance(replications, int):
        raise TypeError("replications must be an integer.")
    if replications <= 0 or replications % 2 == 0:
        raise ValueError("replications must be a positive odd integer.")
    _validate_probability(probability, name="probability")
    if probability <= 0.5:
        raise ValueError("probability must be greater than 0.5 for this diagnostic.")

    cutoff = (replications - 1) // 2
    log_terms: list[float] = []

    for successes in range(cutoff + 1):
        log_pmf = (
            math.lgamma(replications + 1)
            - math.lgamma(successes + 1)
            - math.lgamma(replications - successes + 1)
            + successes * math.log(probability)
            + (replications - successes) * math.log1p(-probability)
        )
        log_terms.append(log_pmf)

    maximum_log_term = max(log_terms)
    scaled_sum = sum(math.exp(term - maximum_log_term) for term in log_terms)
    return math.exp(maximum_log_term) * scaled_sum


def maximum_bernoulli_mcse(replications: int) -> float:
    """Return the worst-case Monte Carlo SE for a Bernoulli proportion."""

    if not isinstance(replications, int):
        raise TypeError("replications must be an integer.")
    if replications <= 0:
        raise ValueError("replications must be positive.")

    return 0.5 / math.sqrt(replications)


def minimum_odd_replications(
    *, probability: float, target_wrong_side_probability: float
) -> int:
    """Find the smallest odd replication count meeting a wrong-side target."""

    _validate_probability(probability, name="probability")
    _validate_probability(
        target_wrong_side_probability, name="target_wrong_side_probability"
    )
    if probability <= 0.5:
        raise ValueError("probability must be greater than 0.5.")

    replications = 1
    while (
        wrong_majority_probability(replications, probability)
        >= target_wrong_side_probability
    ):
        replications += 2

    return replications


def build_diagnostic(replications: int, probability: float) -> ReplicationDiagnostic:
    """Construct a validated diagnostic record for one replication count."""

    return ReplicationDiagnostic(
        replications=replications,
        probability=probability,
        wrong_side_probability=wrong_majority_probability(replications, probability),
        maximum_mcse=maximum_bernoulli_mcse(replications),
    )


def main() -> None:
    """Print the calculations referenced by the article."""

    probability = 0.51
    for replications in (21, 101, 201, 1001):
        diagnostic = build_diagnostic(replications, probability)
        print(
            f"N={diagnostic.replications:>4d} "
            f"wrong-side={diagnostic.wrong_side_probability:.6f} "
            f"max-MCSE={diagnostic.maximum_mcse:.6f}"
        )

    required = minimum_odd_replications(
        probability=probability,
        target_wrong_side_probability=0.05,
    )
    print(f"minimum odd N for wrong-side probability < 0.05: {required}")


if __name__ == "__main__":
    main()
