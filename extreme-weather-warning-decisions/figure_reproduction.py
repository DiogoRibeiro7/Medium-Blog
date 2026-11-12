"""Create publication figures for the weather-warning decision article.

The chronological figure uses time-stamped public statements from government,
IPMA, and RTVE/AEMET; it does not reconstruct unpublished operational data.
The cost–loss, vulnerability, and waiting figures use openly hypothetical
parameters. They must never be described as estimates for Portugal or Spain.

Usage:
    python3 figure_reproduction.py

All figures are saved as 200-DPI PNG images in the ``figures`` subdirectory.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT: Path = Path(__file__).resolve().parent
OUT: Path = ROOT / "figures"
OUT.mkdir(parents=True, exist_ok=True)

NAVY: str = "#17324D"
TEAL: str = "#007C83"
CORAL: str = "#D95A4F"
GOLD: str = "#C38B27"
GRAY: str = "#617388"
PALE: str = "#F2F6F8"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 13,
        "axes.labelsize": 10,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


@dataclass(frozen=True)
class TimelineEvent:
    """A publicly documented event, with its own local time convention."""

    timestamp: datetime
    label: str
    color: str


def save_figure(figure: plt.Figure, filename: str) -> None:
    """Save a tight, high-resolution figure and release its resources."""

    target: Path = OUT / filename
    figure.savefig(target, dpi=200, bbox_inches="tight", pad_inches=0.18)
    plt.close(figure)
    print(f"Wrote {target.name}")


def timeline_panel(
    ax: plt.Axes,
    events: list[TimelineEvent],
    title: str,
    timezone_label: str,
) -> None:
    """Draw one independent event timeline with readable annotations."""

    positions: list[float] = [mdates.date2num(event.timestamp) for event in events]
    ax.hlines(0.0, min(positions), max(positions), color="#B5C3CC", linewidth=2.2)
    for index, (event, position) in enumerate(zip(events, positions, strict=True)):
        above: bool = index % 2 == 0
        direction: float = 1.0 if above else -1.0
        ax.plot(position, 0, "o", color=event.color, markersize=9, zorder=3)
        ax.vlines(position, 0, direction * 0.38, color=event.color, linewidth=1.5)
        ax.text(
            position,
            direction * 0.43,
            f"{event.timestamp:%d %b, %H:%M}\n{event.label}",
            ha="center",
            va="bottom" if above else "top",
            fontsize=9.0,
            color=NAVY,
            linespacing=1.25,
        )

    # Independent axes are intentional: events span different calendar days.
    margin: float = max(max(positions) - min(positions), 0.5) * 0.15
    ax.set_xlim(min(positions) - margin, max(positions) + margin)
    ax.set_ylim(-1.0, 1.0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.set_title(f"{title}  ·  {timezone_label}", loc="left", color=NAVY, pad=18)


def make_event_timeline() -> None:
    """Create a source-grounded visual chronology of three distinct episodes."""

    rows: list[tuple[str, str, list[TimelineEvent]]] = [
        (
            "Portugal: Depression Charles, 2026",
            "local time (UTC)",
            [
                TimelineEvent(datetime(2026, 11, 10, 10, 33), "Alert announced", TEAL),
                TimelineEvent(datetime(2026, 11, 11, 0, 0), "Measures begin", GOLD),
                TimelineEvent(datetime(2026, 11, 11, 12, 30), "Track shifts south", CORAL),
                TimelineEvent(datetime(2026, 11, 11, 21, 10), "Evening assessment", NAVY),
            ],
        ),
        (
            "Spain: Valencia DANA, 2024",
            "local time (CEST)",
            [
                TimelineEvent(datetime(2024, 10, 29, 7, 36), "AEMET red warning", CORAL),
                TimelineEvent(datetime(2024, 10, 29, 20, 14), "Public mobile alert", NAVY),
            ],
        ),
        (
            "Portugal: Storm Kristin, 2026",
            "local time (UTC)",
            [
                TimelineEvent(datetime(2026, 1, 27, 21, 0), "Rapid deepening begins", TEAL),
                TimelineEvent(datetime(2026, 1, 28, 3, 0), "Six-hour phase ends", GOLD),
                TimelineEvent(datetime(2026, 1, 28, 5, 0), "Extreme wind observed", CORAL),
            ],
        ),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(12.6, 8.6))
    fig.suptitle("Time can create or close the window for protective action", x=0.06, ha="left", color=NAVY, fontsize=17, y=1.04)
    for ax, (title, timezone_label, events) in zip(axes, rows, strict=True):
        timeline_panel(ax, events, title, timezone_label)
    fig.subplots_adjust(hspace=0.68)
    fig.text(
        0.06,
        -0.02,
        "Documented milestones only; not a causal comparison. Sources: Portugal Government, IPMA, RTVE/AEMET.",
        color=GRAY,
        fontsize=9,
    )
    save_figure(fig, "01_event_timeline.png")


def make_expected_loss() -> None:
    """Plot the two-action threshold under deliberately hypothetical inputs."""

    probability: np.ndarray = np.linspace(0.0, 0.25, 251)
    closure_cost: float = 6.0
    unprotected_loss: float = 100.0
    protected_loss: float = 20.0
    threshold: float = closure_cost / (unprotected_loss - protected_loss)

    fig, ax = plt.subplots(figsize=(10.6, 5.8))
    ax.plot(100 * probability, unprotected_loss * probability, color=CORAL, linewidth=3, label="No protection")
    ax.plot(100 * probability, closure_cost + protected_loss * probability, color=TEAL, linewidth=3, label="Protect now")
    ax.axvline(100 * threshold, linestyle="--", linewidth=1.7, color=NAVY)
    ax.annotate("Action threshold: 7.5%", xy=(100 * threshold, 7.5), xytext=(11.3, 12.7),
                color=NAVY, fontsize=11, arrowprops={"arrowstyle": "->", "color": NAVY})
    ax.fill_between(100 * probability, 0, 27, where=probability >= threshold, color=TEAL, alpha=0.045)
    ax.set(xlabel="Probability of severe event (%)", ylabel="Expected social-loss units", xlim=(0, 25), ylim=(0, 27))
    ax.set_title("A warning can be rational even when the event is unlikely", loc="left", color=NAVY, pad=18)
    ax.grid(alpha=0.16)
    ax.legend(loc="upper left", frameon=False)
    fig.text(0.12, 0.0, "Hypothetical: precaution cost = 6; event loss = 100 without action, 20 with action.", color=GRAY, fontsize=9)
    save_figure(fig, "02_expected_loss.png")


def make_vulnerability_thresholds() -> None:
    """Show the effect of different avoidable losses on the action threshold."""

    labels: list[str] = ["Higher avoidable harm", "Moderate avoidable harm", "Lower avoidable harm"]
    avoidable_losses: np.ndarray = np.asarray([160.0, 80.0, 40.0])
    action_cost: float = 6.0
    thresholds: np.ndarray = 100.0 * action_cost / avoidable_losses

    fig, ax = plt.subplots(figsize=(10.6, 4.8))
    bars = ax.barh(labels[::-1], thresholds[::-1], color=[GOLD, TEAL, CORAL][::-1], height=0.54)
    for bar, value in zip(bars, thresholds[::-1], strict=True):
        ax.text(bar.get_width() + 0.4, bar.get_y() + bar.get_height() / 2, f"{value:.2f}%", va="center", color=NAVY, fontsize=11)
    ax.set(xlim=(0, 17.8), xlabel="Probability above which precaution is preferred (%)")
    ax.set_title("The same intervention can have different rational thresholds", loc="left", color=NAVY, pad=15)
    ax.grid(axis="x", alpha=0.17)
    ax.set_axisbelow(True)
    fig.text(0.20, 0.0, "Hypothetical fixed action cost = 6; avoidable losses = 160, 80 and 40 units.", color=GRAY, fontsize=9)
    save_figure(fig, "03_vulnerability_thresholds.png")


def make_waiting_value() -> None:
    """Plot a negative net waiting value despite a useful future forecast."""

    # Prior severe-event probability and the signal's stated sensitivity/specificity.
    prior: float = 0.12
    sensitivity: float = 0.8
    specificity: float = 0.8
    positive_rate: float = prior * sensitivity + (1 - prior) * (1 - specificity)
    positive_and_severe: float = prior * sensitivity
    negative_and_severe: float = prior * (1 - sensitivity)

    delay_hours: np.ndarray = np.linspace(0, 3, 301)
    immediate_expected_loss: float = 6.0 + prior * 20.0
    delayed_residual_loss: np.ndarray = 20.0 + 20.0 * delay_hours
    # In this plotted range, act after a positive signal; otherwise do not act.
    wait_expected_loss: np.ndarray = (
        positive_rate * 6.0
        + positive_and_severe * delayed_residual_loss
        + negative_and_severe * 100.0
    )
    net_information_value: np.ndarray = immediate_expected_loss - wait_expected_loss
    crossing_hours: float = float(np.interp(0.0, net_information_value[::-1], delay_hours[::-1]))

    fig, ax = plt.subplots(figsize=(10.6, 5.5))
    ax.axhline(0.0, color=NAVY, linewidth=1.1)
    ax.plot(delay_hours, net_information_value, color=TEAL, linewidth=3)
    ax.fill_between(delay_hours, 0, net_information_value, where=net_information_value >= 0, color=TEAL, alpha=0.13)
    ax.fill_between(delay_hours, 0, net_information_value, where=net_information_value < 0, color=CORAL, alpha=0.13)
    ax.axvline(crossing_hours, color=CORAL, linestyle="--", linewidth=1.6)
    ax.annotate(f"Break-even: {crossing_hours:.2f} h", xy=(crossing_hours, 0), xytext=(1.58, 1.2),
                arrowprops={"arrowstyle": "->", "color": NAVY}, color=NAVY, fontsize=11)
    ax.set(xlabel="Delay until updated forecast (hours)", ylabel="Expected-loss gain from waiting (units)", xlim=(0, 3))
    ax.set_title("Better information can arrive after its protection value is lost", loc="left", color=NAVY, pad=17)
    ax.grid(alpha=0.16)
    fig.text(0.10, -0.01, "Hypothetical 80%/80% signal; severe-event prior 12%; delayed action effectiveness declines linearly.",
             color=GRAY, fontsize=9)
    save_figure(fig, "04_value_of_waiting.png")


def make_warning_chain() -> None:
    """Diagram the social and operational chain between forecast and safety."""

    labels: list[tuple[str, str]] = [
        ("Hazard\nobservation", "Sensors and models"),
        ("Probability\nforecast", "Uncertainty and timing"),
        ("Impact\nassessment", "Exposure and vulnerability"),
        ("Public decision", "Actions and deadlines"),
        ("Protective\nbehaviour", "Access, trust and capacity"),
    ]
    fig, ax = plt.subplots(figsize=(13.0, 4.2))
    ax.set_xlim(-0.3, 12.8)
    ax.set_ylim(-0.9, 3.5)
    ax.axis("off")
    for index, (main, sub) in enumerate(labels):
        x = index * 2.62
        patch = FancyBboxPatch((x, 1.0), 2.2, 1.65, boxstyle="round,pad=0.12,rounding_size=0.2",
                               linewidth=1.2, edgecolor="#B3C5CF", facecolor=PALE)
        ax.add_patch(patch)
        ax.text(x + 1.1, 2.12, main, ha="center", va="center", fontsize=11.5, fontweight="bold", color=NAVY)
        ax.text(x + 1.1, 1.38, sub, ha="center", va="center", fontsize=8.1, color=GRAY)
        if index < len(labels) - 1:
            arr = FancyArrowPatch((x + 2.25, 1.83), (x + 2.58, 1.83), arrowstyle="-|>",
                                  mutation_scale=17, linewidth=1.5, color=TEAL)
            ax.add_patch(arr)
    ax.text(6.25, 0.27, "Lost lead time weakens feasible interventions; unclear messages weaken public response.",
            ha="center", fontsize=10, color=CORAL)
    ax.set_title("A forecast becomes life-saving only through a functioning response chain", loc="left", color=NAVY, pad=15)
    save_figure(fig, "05_warning_chain.png")


def main() -> None:
    """Generate all article assets deterministically."""

    make_event_timeline()
    make_expected_loss()
    make_vulnerability_thresholds()
    make_waiting_value()
    make_warning_chain()


if __name__ == "__main__":
    main()
