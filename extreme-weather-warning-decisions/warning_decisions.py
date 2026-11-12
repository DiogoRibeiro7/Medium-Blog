"""Decision model and figures for the extreme-weather warning article.

Everything numerical in the article comes from this script. It has two parts.

1. A documented chronology (Figure 1) of Depression Charles (Portugal,
   November 2026) and the Valencia DANA (Spain, October 2024), aligned on the
   start of the first red-warning period. Times come from public statements
   by the Portuguese Government and IPMA and from the Maldita.es chronology of
   AEMET warnings and the ES-Alert message.
2. A cost-loss model with a later forecast signal (Figures 2-5). Its
   parameters are hypothetical teaching values, not estimates for Portugal or
   Spain.

Usage:
    python warning_decisions.py > results/output.txt

The script prints every number quoted in the article and writes five PNG
figures to ``figures/``. It is deterministic and downloads nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT: Path = Path(__file__).resolve().parent
OUT: Path = ROOT / "figures"

NAVY: str = "#17324D"
TEAL: str = "#007C83"
CORAL: str = "#D95A4F"
GOLD: str = "#C38B27"
GRAY: str = "#617388"

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


# ---------------------------------------------------------------------------
# Decision model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Scenario:
    """Hypothetical inputs of the two-action cost-loss problem.

    Attributes:
        prior: Probability of the severe event given today's information.
        cost: Cost of the protective package, paid whether or not the event occurs.
        loss_unprotected: Loss if the severe event occurs without protection.
        loss_protected: Loss if the severe event occurs with timely protection.
        accuracy: Sensitivity and specificity of the later binary signal.
        degradation_per_hour: Increase of the protected loss for each hour that
            protection is delayed (people already travelling, less time to act).
    """

    prior: float = 0.12
    cost: float = 6.0
    loss_unprotected: float = 100.0
    loss_protected: float = 20.0
    accuracy: float = 0.8
    degradation_per_hour: float = 20.0

    @property
    def threshold(self) -> float:
        """Probability above which protection minimises expected loss."""
        return self.cost / (self.loss_unprotected - self.loss_protected)


def expected_loss_now(s: Scenario) -> float:
    """Best expected loss when the decision is taken immediately."""
    act = s.cost + s.prior * s.loss_protected
    no_act = s.prior * s.loss_unprotected
    return min(act, no_act)


def signal_joint(s: Scenario) -> dict[str, float]:
    """Joint probabilities of the signal and the severe event."""
    q, p = s.accuracy, s.prior
    return {
        "pos": p * q + (1 - p) * (1 - q),
        "pos_severe": p * q,
        "neg": p * (1 - q) + (1 - p) * q,
        "neg_severe": p * (1 - q),
    }


def expected_loss_wait(s: Scenario, delay_hours: float) -> float:
    """Expected loss of waiting for the signal, then choosing the best action.

    Each signal branch picks the cheaper of protecting (with a protected loss
    that grows with the delay) and not protecting.
    """
    j = signal_joint(s)
    delayed_protected = s.loss_protected + s.degradation_per_hour * delay_hours
    total = 0.0
    for prob, severe in ((j["pos"], j["pos_severe"]), (j["neg"], j["neg_severe"])):
        act = prob * s.cost + severe * delayed_protected
        no_act = severe * s.loss_unprotected
        total += min(act, no_act)
    return total


def net_value_of_waiting(s: Scenario, delay_hours: float) -> float:
    """Positive when waiting for the signal beats deciding now."""
    return expected_loss_now(s) - expected_loss_wait(s, delay_hours)


def break_even_delay(s: Scenario, horizon: float = 10.0) -> float:
    """Longest delay for which waiting still beats deciding now (0 if never)."""
    if net_value_of_waiting(s, 0.0) <= 1e-12:
        return 0.0
    lo, hi = 0.0, horizon
    if net_value_of_waiting(s, hi) > 0:
        return float("inf")
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if net_value_of_waiting(s, mid) > 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def minimum_useful_accuracy(s: Scenario) -> float:
    """Smallest accuracy at which the signal can change the decision.

    When protecting now is the default (prior above threshold), the signal is
    useful only if a negative reading can pull the posterior below the
    threshold. When not protecting is the default, a positive reading must
    push it above.
    """
    p, t = s.prior, s.threshold
    if p >= t:
        return p * (1 - t) / (p * (1 - t) + t * (1 - p))
    return t * (1 - p) / (t * (1 - p) + p * (1 - t))


def expected_loss_protect_then_review(s: Scenario, recoverable: float) -> tuple[float, bool]:
    """Protect now, then keep or lift the measures when the signal arrives.

    ``recoverable`` is the share of the protection cost still avoidable when
    the signal arrives (0: everything already paid, 1: nothing paid yet).
    Measures kept stay fully effective. If they are lifted and the severe
    event occurs, the unprotected loss applies.

    Returns the expected loss and whether lifting is optimal after a negative signal.
    """
    j = signal_joint(s)
    positive = j["pos"] * s.cost + j["pos_severe"] * s.loss_protected
    keep = j["neg"] * s.cost + j["neg_severe"] * s.loss_protected
    lift = j["neg"] * s.cost * (1 - recoverable) + j["neg_severe"] * s.loss_unprotected
    return positive + min(keep, lift), lift < keep


def break_even_recoverable(s: Scenario) -> float:
    """Recoverable share above which lifting after a negative signal pays."""
    j = signal_joint(s)
    posterior_neg = j["neg_severe"] / j["neg"]
    return posterior_neg * (s.loss_unprotected - s.loss_protected) / s.cost


# ---------------------------------------------------------------------------
# Printed results
# ---------------------------------------------------------------------------


def report() -> None:
    """Print every number used in the article."""
    s = Scenario()
    j = signal_joint(s)
    print("== Cost-loss baseline (hypothetical social-loss units)")
    print(f"threshold p* = C/(L0-L1) = {s.threshold:.4f}")
    print(f"protect now  = {s.cost + s.prior * s.loss_protected:.3f}")
    print(f"no action    = {s.prior * s.loss_unprotected:.3f}")
    for avoidable in (160.0, 80.0, 40.0):
        print(f"threshold with avoidable loss {avoidable:.0f}: {s.cost / avoidable:.4f}")

    print("\n== Later signal, accuracy 0.80")
    print(f"P(positive)          = {j['pos']:.3f}")
    print(f"P(severe | positive) = {j['pos_severe'] / j['pos']:.4f}")
    print(f"P(severe | negative) = {j['neg_severe'] / j['neg']:.4f}")
    print(f"wait, no degradation = {expected_loss_wait(s, 0.0):.3f}")
    print(f"value of information at zero delay = {net_value_of_waiting(s, 0.0):.3f}")
    print(f"wait, protected loss 50 (1.5 h) = {expected_loss_wait(s, 1.5):.3f}")
    print(f"break-even delay = {break_even_delay(s):.3f} h")
    pos_post = j["pos_severe"] / j["pos"]
    max_protected = s.loss_unprotected - s.cost / pos_post
    stop_delay = (max_protected - s.loss_protected) / s.degradation_per_hour
    print(f"delayed protection stops paying after a positive signal at {stop_delay:.3f} h")
    print(f"net value of waiting beyond that delay = {expected_loss_now(s) - s.prior * s.loss_unprotected:.3f}")

    print("\n== Forecast accuracy and the room to wait")
    for prior in (0.05, 0.12, 0.25):
        sp = replace(s, prior=prior)
        default = "protect" if prior >= sp.threshold else "do nothing"
        print(f"prior {prior:.2f} (default: {default}): minimum useful accuracy = {minimum_useful_accuracy(sp):.4f}")
    for prior in (0.12, 0.25):
        for q in (0.60, 0.70, 0.80, 0.90, 1.00):
            sq = replace(s, prior=prior, accuracy=q)
            print(
                f"prior {prior:.2f}, accuracy {q:.2f}: value at zero delay = "
                f"{net_value_of_waiting(sq, 0.0):.3f}, break-even delay = {break_even_delay(sq):.3f} h"
            )

    print("\n== Protect now, review when the signal arrives")
    print(f"break-even recoverable share = {break_even_recoverable(s):.4f}")
    for r in (0.0, 0.25, 0.5, 0.75, 1.0):
        loss, lifted = expected_loss_protect_then_review(s, r)
        print(f"recoverable {r:.2f}: expected loss = {loss:.3f} ({'lift' if lifted else 'keep'} after negative)")

    print("\n== Chronology: hours relative to the first red-warning period")
    for row in CHRONOLOGY:
        for event in row.events:
            label = event.label.replace("\n", " ")
            print(f"{row.title}: {event.local_time} {label} -> {event.hours:+.2f} h")


# ---------------------------------------------------------------------------
# Chronology
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Milestone:
    """A documented milestone, positioned in hours from the first red-warning period."""

    hours: float
    local_time: str
    label: str
    public_message: bool = False


@dataclass(frozen=True)
class Episode:
    """One row of the chronology figure."""

    title: str
    events: tuple[Milestone, ...]


def _hours(day_offset: int, hh: int, mm: int, ref_h: int, ref_m: int) -> float:
    return day_offset * 24 + (hh * 60 + mm - (ref_h * 60 + ref_m)) / 60


# Charles: reference is 11 Nov 2026 09:00 WET (red warnings begin in Faro, Setúbal, Beja).
# Valencia: reference is 29 Oct 2024 07:31 CET (first AEMET red warning in the province).
CHRONOLOGY: tuple[Episode, ...] = (
    Episode(
        "Portugal, Depression Charles, 10-11 Nov 2026 (WET = UTC)",
        (
            Milestone(_hours(-1, 10, 33, 9, 0), "10 Nov 10:33", "Alert and measures\nannounced", True),
            Milestone(_hours(0, 0, 0, 9, 0), "11 Nov 00:00", "Measures\nin force"),
            Milestone(_hours(0, 9, 0, 9, 0), "11 Nov 09:00", "Red warnings begin\n(3 southern districts)"),
            Milestone(_hours(0, 13, 31, 9, 0), "11 Nov 13:31", "Update: track\nshifts south"),
            Milestone(_hours(0, 21, 10, 9, 0), "11 Nov 21:10", "Prime minister:\nless impact than forecast"),
        ),
    ),
    Episode(
        "Spain, Valencia DANA, 29 Oct 2024 (CET = UTC+1)",
        (
            Milestone(_hours(0, 7, 31, 7, 31), "07:31", "First AEMET red\nwarning in the province"),
            Milestone(_hours(0, 9, 41, 7, 31), "09:41", "Whole province\nunder red"),
            Milestone(_hours(0, 20, 11, 7, 31), "20:11", "ES-Alert to\nmobile phones", True),
        ),
    ),
)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def save_figure(fig: plt.Figure, name: str) -> None:
    """Save a figure at publication resolution and close it."""
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)


def figure_chronology() -> None:
    """Figure 1: two episodes on one hour axis, centred on the first red warning."""
    fig, ax = plt.subplots(figsize=(12.0, 6.4))
    rows_y = (1.55, 0.0)
    ax.axvline(0, color=GRAY, linestyle="--", linewidth=1.2, zorder=1)
    ax.text(0.3, 2.3, "First red-warning\nperiod begins", color=GRAY, fontsize=9, va="top")
    for y, episode in zip(rows_y, CHRONOLOGY, strict=True):
        hours = [m.hours for m in episode.events]
        ax.plot([min(hours), max(hours)], [y, y], color="#B9C7D1", linewidth=3, zorder=2)
        ax.text(-25.5, y + 0.62, episode.title, color=NAVY, fontsize=11, fontweight="bold")
        for k, m in enumerate(episode.events):
            colour = CORAL if m.public_message else TEAL
            ax.scatter([m.hours], [y], s=120 if m.public_message else 70, color=colour, zorder=3)
            offset = -0.2 if k % 2 == 0 else 0.15
            va = "top" if offset < 0 else "bottom"
            ax.text(m.hours, y + offset, f"{m.local_time}\n{m.label}", ha="center", va=va,
                    fontsize=8.6, color=NAVY, zorder=4,
                    bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5})
    ax.set_xlim(-26, 15)
    ax.set_ylim(-0.85, 2.35)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xticks(range(-24, 16, 4))
    ax.set_xlabel("Hours relative to the start of the first red-warning period")
    ax.scatter([], [], s=120, color=CORAL, label="Public protective message")
    ax.scatter([], [], s=70, color=TEAL, label="Warning or operational milestone")
    ax.legend(loc="lower left", frameon=False, fontsize=9)
    ax.set_title("When the public message came, relative to the red warning", loc="left", color=NAVY, pad=12)
    fig.text(0.01, -0.03,
             "Sources: Portuguese Government (10-11 Nov 2026), IPMA warnings as reported by the Ministry of "
             "Internal Administration; Maldita.es chronology of AEMET warnings and ES-Alert. Different hazards; "
             "not a causal comparison.", color=GRAY, fontsize=8, wrap=True)
    save_figure(fig, "01_chronology.png")


def figure_cost_loss() -> None:
    """Figure 2: expected loss of protecting now and of doing nothing."""
    s = Scenario()
    p = np.linspace(0.0, 0.25, 251)
    fig, ax = plt.subplots(figsize=(10.0, 5.6))
    ax.plot(100 * p, s.loss_unprotected * p, color=CORAL, linewidth=3, label="No protection")
    ax.plot(100 * p, s.cost + s.loss_protected * p, color=TEAL, linewidth=3, label="Protect now")
    ax.axvline(100 * s.threshold, linestyle="--", linewidth=1.6, color=NAVY)
    ax.annotate(f"Threshold {100 * s.threshold:.1f}%", xy=(100 * s.threshold, 7.5), xytext=(1.0, 13.0),
                color=NAVY, fontsize=11, arrowprops={"arrowstyle": "->", "color": NAVY})
    ax.axvline(100 * s.prior, linestyle=":", linewidth=1.4, color=GRAY)
    ax.text(100 * s.prior + 0.3, 23.5, "Example: 12%", color=GRAY, fontsize=9)
    ax.set(xlabel="Probability of the severe event (%)", ylabel="Expected loss (social-loss units)",
           xlim=(0, 25), ylim=(0, 26))
    ax.set_title("Protection pays above the cost-loss threshold", loc="left", color=NAVY, pad=12)
    ax.grid(alpha=0.16)
    ax.legend(loc="upper left", frameon=False)
    fig.text(0.1, -0.02, "Hypothetical: protection costs 6; the severe event costs 100 without protection, 20 with it.",
             color=GRAY, fontsize=9)
    save_figure(fig, "02_cost_loss.png")


def figure_waiting_delay() -> None:
    """Figure 3: net value of waiting for an 80%-accurate signal as protection degrades."""
    s = Scenario()
    delays = np.linspace(0.0, 4.0, 401)
    value = np.array([net_value_of_waiting(s, d) for d in delays])
    be = break_even_delay(s)
    fig, ax = plt.subplots(figsize=(10.0, 5.4))
    ax.axhline(0.0, color=NAVY, linewidth=1.0)
    ax.plot(delays, value, color=TEAL, linewidth=3)
    ax.fill_between(delays, 0, value, where=value >= 0, color=TEAL, alpha=0.13)
    ax.fill_between(delays, 0, value, where=value < 0, color=CORAL, alpha=0.13)
    ax.axvline(be, color=CORAL, linestyle="--", linewidth=1.5)
    ax.annotate(f"Break-even {be:.2f} h", xy=(be, 0), xytext=(1.6, 1.4), color=NAVY, fontsize=11,
                arrowprops={"arrowstyle": "->", "color": NAVY})
    ax.annotate("Delayed protection no longer\nworth taking: waiting = doing nothing", xy=(3.6, -3.6),
                xytext=(2.75, -1.1), color=GRAY, fontsize=9, arrowprops={"arrowstyle": "->", "color": GRAY})
    ax.set(xlabel="Hours until the next forecast arrives", ylabel="Gain from waiting (social-loss units)",
           xlim=(0, 4), ylim=(-4.2, 3.0))
    ax.set_title("A better forecast can arrive after its value is gone", loc="left", color=NAVY, pad=12)
    ax.grid(alpha=0.16)
    fig.text(0.1, -0.02, "Hypothetical: prior 12%, signal 80% accurate, protected loss rises by 20 units per hour of delay.",
             color=GRAY, fontsize=9)
    save_figure(fig, "03_value_of_waiting.png")


def figure_accuracy() -> None:
    """Figure 4: longest worthwhile delay as a function of signal accuracy, for two priors."""
    base = Scenario()
    q = np.linspace(0.5, 1.0, 501)
    fig, ax = plt.subplots(figsize=(10.0, 5.4))
    for prior, colour in ((0.12, TEAL), (0.25, GOLD)):
        delays = [break_even_delay(replace(base, prior=prior, accuracy=x)) for x in q]
        q_min = minimum_useful_accuracy(replace(base, prior=prior))
        ax.plot(q, delays, color=colour, linewidth=3, label=f"Prior {prior:.0%}")
        ax.axvline(q_min, color=colour, linestyle=":", linewidth=1.3)
        ax.text(q_min - 0.004, 2.35, f"{q_min:.2f}", color=colour, fontsize=9, ha="right")
    ax.set(xlabel="Accuracy of the next forecast (sensitivity = specificity)",
           ylabel="Longest delay for which waiting pays (hours)", xlim=(0.5, 1.0), ylim=(0, 2.5))
    ax.set_title("Below a minimum accuracy, the next forecast is worth nothing", loc="left", color=NAVY, pad=12)
    ax.grid(alpha=0.16)
    ax.legend(loc="upper left", frameon=False)
    fig.text(0.1, -0.02, "Hypothetical: cost 6, losses 100 and 20, protected loss rises by 20 units per hour. "
             "Dotted lines: minimum useful accuracy.", color=GRAY, fontsize=9)
    save_figure(fig, "04_accuracy_and_delay.png")


def figure_reversibility() -> None:
    """Figure 5: protect now and review, as a function of how much of the cost is recoverable."""
    s = Scenario()
    r = np.linspace(0.0, 1.0, 201)
    review = np.array([expected_loss_protect_then_review(s, x)[0] for x in r])
    r_star = break_even_recoverable(s)
    fig, ax = plt.subplots(figsize=(10.0, 5.4))
    ax.plot(r, review, color=TEAL, linewidth=3, label="Protect now, review at the next forecast")
    ax.axhline(s.cost + s.prior * s.loss_protected, color=NAVY, linestyle="--", linewidth=1.4,
               label="Protect now, no review")
    ax.axhline(expected_loss_wait(s, 1.5), color=CORAL, linestyle="--", linewidth=1.4,
               label="Wait 1.5 h, then decide")
    ax.axhline(expected_loss_wait(s, 0.0), color=GRAY, linestyle=":", linewidth=1.4,
               label="Wait at no cost (ideal)")
    ax.axvline(r_star, color=GOLD, linestyle=":", linewidth=1.3)
    ax.text(r_star + 0.01, 9.15, f"Lifting pays above {r_star:.2f}", color=GOLD, fontsize=9)
    ax.set(xlabel="Share of the protection cost still avoidable when the forecast arrives",
           ylabel="Expected loss (social-loss units)", xlim=(0, 1), ylim=(5.5, 9.4))
    ax.set_title("Acting early keeps the value of the next forecast only if the measures can be undone",
                 loc="left", color=NAVY, pad=12, fontsize=12)
    ax.grid(alpha=0.16)
    ax.legend(loc="lower left", frameon=False, fontsize=9)
    fig.text(0.1, -0.02, "Hypothetical: prior 12%, signal 80% accurate. Lifted measures leave the full loss of 100 if the event occurs.",
             color=GRAY, fontsize=9)
    save_figure(fig, "05_reversibility.png")


def main() -> None:
    """Print the results and draw every figure."""
    report()
    figure_chronology()
    figure_cost_loss()
    figure_waiting_delay()
    figure_accuracy()
    figure_reversibility()


if __name__ == "__main__":
    main()
