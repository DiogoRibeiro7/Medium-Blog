"""Exact toy models of verifier selection, workflow reliability and review cost.

These are mathematical counterexamples and sensitivity calculations, not
measurements of language models, employment or future AI risk.
"""
from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class JointMass:
    """Probability mass for true quality crossed with verifier acceptance."""

    true_accept: float
    false_accept: float
    true_reject: float
    false_reject: float

    def __post_init__(self) -> None:
        values = (self.true_accept, self.false_accept, self.true_reject, self.false_reject)
        if any(not math.isfinite(v) or not 0 <= v <= 1 for v in values):
            raise ValueError("Masses must be finite probabilities")
        if not math.isclose(sum(values), 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("Masses must sum to one")

    @property
    def acceptance(self) -> float:
        return self.true_accept + self.false_accept

    @property
    def quality(self) -> float:
        return self.true_accept + self.true_reject


def selected_outcome(mass: JointMass, attempts: int) -> tuple[float, float]:
    """Return acceptance and quality for first accepted draw, else first draw.

    Draws are IID. A failed search returns its first rejected candidate.
    A perfect verifier and an imperfect one have different selection effects.
    """
    if not isinstance(attempts, int) or attempts < 1:
        raise ValueError("attempts must be a positive integer")
    p = mass.acceptance
    if p == 0 or p == 1:
        return p, mass.quality
    # Probability of at least one accepted candidate in the search.
    failure = (1 - p) ** attempts
    acceptance = 1 - failure
    quality = acceptance * (mass.true_accept / p) + failure * (mass.true_reject / (1 - p))
    return acceptance, quality


def serial_success(p: float, steps: int) -> float:
    """All-success probability for independent equal-reliability steps."""
    if not math.isfinite(p) or not 0 <= p <= 1:
        raise ValueError("p must be a probability")
    if not isinstance(steps, int) or steps < 1:
        raise ValueError("steps must be a positive integer")
    return p ** steps


def expected_minutes(
    generation: float, review: float, rework: float, rework_probability: float
) -> float:
    """Expected total time in a one-review, one-rework workflow."""
    if any(not math.isfinite(v) or v < 0 for v in (generation, review, rework)):
        raise ValueError("Durations must be finite and nonnegative")
    if not math.isfinite(rework_probability) or not 0 <= rework_probability <= 1:
        raise ValueError("rework_probability must be a probability")
    return generation + review + rework * rework_probability


def break_even_probability(manual: float, generation: float, review: float, rework: float) -> float:
    """Algebraic crossing; it can lie outside [0, 1] if no interior crossing."""
    if not math.isfinite(manual) or manual < 0 or not math.isfinite(rework) or rework <= 0:
        raise ValueError("Manual time must be nonnegative and rework positive")
    return (manual - expected_minutes(generation, review, rework, 0)) / rework


def write_csv(path: Path, header: tuple[str, ...], rows: Iterable[tuple[object, ...]]) -> None:
    """Write a deterministic, UTF-8 table."""
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def generate(output: Path) -> None:
    """Generate exact scenario tables and three publication figures."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    masses = {
        "perfect": JointMass(0.55, 0.0, 0.0, 0.45),
        "imperfect": JointMass(0.25, 0.35, 0.30, 0.10),
    }
    attempts = range(1, 21)
    write_csv(
        output / "verifier_selection.csv",
        ("scenario", "attempts", "accepted_probability", "quality_probability"),
        ((name, n, *selected_outcome(m, n)) for name, m in masses.items() for n in attempts),
    )
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), layout="constrained", sharey=True)
    for (name, m), ax in zip(masses.items(), axes):
        values = [selected_outcome(m, n) for n in attempts]
        ax.plot(list(attempts), [v[0] for v in values], label="Verifier acceptance", color="#285f8e", lw=2.2)
        ax.plot(list(attempts), [v[1] for v in values], label="True quality", color="#b65432", lw=2.2, ls="--")
        ax.set(title=f"{name.title()} verifier", xlabel="Candidates sampled", ylim=(0.35, 1.04), xticks=[1, 5, 10, 15, 20])
        ax.grid(alpha=0.2)
        ax.legend(loc="lower right", frameon=False, fontsize=9)
    axes[0].set_ylabel("Probability")
    fig.suptitle("More search helps only when the verifier selects the right outcome")
    fig.savefig(output / "verifier_selection.png", dpi=180)
    plt.close(fig)

    rates = (0.95, 0.99, 0.999)
    steps = range(1, 101)
    write_csv(output / "serial_reliability.csv", ("step_reliability", "steps", "all_success"),
              ((p, n, serial_success(p, n)) for p in rates for n in steps))
    fig, ax = plt.subplots(figsize=(8, 4.6), layout="constrained")
    for p, color in zip(rates, ("#b65432", "#285f8e", "#43826d")):
        ax.plot(list(steps), [serial_success(p, n) for n in steps], label=f"{p:.1%} per step", color=color, lw=2.2)
    ax.set(title="All-success reliability under independent steps", xlabel="Number of required steps", ylabel="Probability all steps succeed", ylim=(0, 1.03))
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    fig.savefig(output / "serial_reliability.png", dpi=180)
    plt.close(fig)

    probabilities = [i / 100 for i in range(101)]
    manual, generation, review, rework = 60.0, 15.0, 20.0, 90.0
    threshold = break_even_probability(manual, generation, review, rework)
    write_csv(output / "review_cost.csv", ("rework_probability", "manual_minutes", "assisted_minutes"),
              ((q, manual, expected_minutes(generation, review, rework, q)) for q in probabilities))
    fig, ax = plt.subplots(figsize=(8, 4.6), layout="constrained")
    ax.plot(probabilities, [expected_minutes(generation, review, rework, q) for q in probabilities], color="#285f8e", lw=2.2, label="Assisted workflow")
    ax.axhline(manual, color="#555555", ls="--", label="Manual workflow")
    ax.axvline(threshold, color="#b65432", ls=":", label=f"Break-even: {threshold:.1%}")
    ax.set(title="Generation speed is not end-to-end productivity", xlabel="Probability of one additional rework episode", ylabel="Expected total minutes")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.savefig(output / "review_cost.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    generate(Path(__file__).resolve().parent / "outputs")
