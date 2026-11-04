"""Reproduce a provisional 2026 House snapshot audit without scoring the election.

The source record is a dated, archived *live* Associated Press aggregate. It is
not the official initial-election party-division outcome defined in the
pre-election lock. This module intentionally has no function that can create a
verified-outcome score: ``../score_after_election.py`` owns that future step.

Run from anywhere::

    python midterm-forecast-fit-vs-prediction/post_election/analysis.py

Only frozen inputs, the archived snapshot and fixed plotting code are used.
There is no network fetch, polling update, model refit or live scraping.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT: Path = Path(__file__).resolve().parent
STUDY_ROOT: Path = ROOT.parent
SNAPSHOT: Path = ROOT / "data" / "ap_house_live_2026-11-04.json"
LOCK: Path = STUDY_ROOT / "results" / "forecast_lock.json"
PALETTE: dict[str, str] = {
    "dem": "#3B6AA0",
    "gop": "#B65448",
    "charcoal": "#252B35",
    "muted": "#778493",
    "teal": "#267E74",
    "yellow": "#CB973D",
    "pale": "#E9EDF2",
}


@dataclass(frozen=True, slots=True)
class HouseSnapshot:
    """An unofficial national AP call/lead summary at a fixed UTC timestamp."""

    source_name: str
    source_url: str
    reported_at_utc: str
    retrieved_at_utc: str
    seats: int
    majority: int
    democratic_won: int
    republican_won: int
    democratic_leading: int
    republican_leading: int
    other: int

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> HouseSnapshot:
        """Parse and enforce the distinction between live and final outcomes."""

        if value.get("snapshot_kind") != "live_unofficial":
            raise ValueError("Only expressly unofficial snapshots are accepted")
        if value.get("official_initial_election_party_division_available") is not False:
            raise ValueError("Official results belong to the separate final scorer")
        if value.get("observed_republican_seat_change") is not None:
            raise ValueError("No observed seat-change target may enter this audit")
        if value.get("not_a_final_score") is not True:
            raise ValueError("The live-snapshot disclaimer is mandatory")
        if value.get("election_date") != "2026-11-03":
            raise ValueError("Incorrect election date")
        if value.get("source_name") != "Associated Press":
            raise ValueError("This archived snapshot must retain its publisher")
        if value.get("source_url") != "https://apnews.com/hub/elections":
            raise ValueError("The source URL differs from the audited snapshot")
        h: dict[str, Any] = value["house"]
        fields: tuple[str, ...] = (
            "total_seats", "majority_threshold", "democratic_won",
            "republican_won", "democratic_leading_uncalled",
            "republican_leading_uncalled", "other_won_or_leading",
        )
        if any(type(h.get(name)) is not int for name in fields):
            raise TypeError("All seat counts must be integers, not booleans")
        record = cls(
            source_name=value["source_name"], source_url=value["source_url"],
            reported_at_utc=value["reported_at_utc"],
            retrieved_at_utc=value["retrieved_at_utc"],
            seats=h["total_seats"], majority=h["majority_threshold"],
            democratic_won=h["democratic_won"],
            republican_won=h["republican_won"],
            democratic_leading=h["democratic_leading_uncalled"],
            republican_leading=h["republican_leading_uncalled"],
            other=h["other_won_or_leading"],
        )
        record.validate()
        return record

    def validate(self) -> None:
        """Check accounting invariants, source timestamps and called majority."""

        from datetime import datetime, timezone

        counts: tuple[int, ...] = (
            self.democratic_won, self.republican_won,
            self.democratic_leading, self.republican_leading, self.other,
        )
        if self.seats != 435 or self.majority != 218:
            raise ValueError("Unexpected House size or majority threshold")
        if min(counts) < 0 or sum(counts) != self.seats:
            raise ValueError("Live won/leading categories must sum to 435")
        if self.democratic_won < self.majority:
            raise ValueError("AP majority-call statement does not reconcile")
        reported = datetime.fromisoformat(self.reported_at_utc.replace("Z", "+00:00"))
        retrieved = datetime.fromisoformat(self.retrieved_at_utc.replace("Z", "+00:00"))
        if not all(ts.tzinfo is not None for ts in (reported, retrieved)):
            raise ValueError("All source timestamps must have explicit UTC offsets")
        if not (datetime(2026, 11, 3, tzinfo=timezone.utc) <= reported <= retrieved):
            raise ValueError("Snapshot timestamps are out of sequence")

    @property
    def uncalled(self) -> int:
        """Number of races not yet called, irrespective of who is leading."""

        return self.seats - self.democratic_won - self.republican_won

    @property
    def republican_trend(self) -> int:
        """Republican called wins plus current leads, not certified seats."""

        return self.republican_won + self.republican_leading


def load_inputs(snapshot_path: Path = SNAPSHOT, lock_path: Path = LOCK) -> tuple[HouseSnapshot, dict[str, Any]]:
    """Read a fixed AP snapshot and the untouched, pre-election forecast lock."""

    snapshot_data: dict[str, Any] = json.loads(snapshot_path.read_text(encoding="utf-8"))
    manifest: dict[str, Any] = json.loads(lock_path.read_text(encoding="utf-8"))
    snapshot = HouseSnapshot.from_dict(snapshot_data)
    if manifest.get("registration_date") != "2026-11-01":
        raise ValueError("Unexpected pre-election registration date")
    if manifest.get("forecast_information_as_of") != "2026-10-31":
        raise ValueError("Forecast information cut-off has changed")
    if manifest.get("scenario", {}).get("president_party_seats_defended") != 220:
        raise ValueError("Pre-election defended-seat assumption has changed")
    if manifest.get("target", {}).get("observed_2026_outcome") is not None:
        raise ValueError("The frozen 2026 outcome field must remain null")
    if manifest.get("scoring_protocol", {}).get("do_not_refit_on_2026_outcome") is not True:
        raise ValueError("Refitting guard missing from the pre-election manifest")
    return snapshot, manifest


def diagnostics(snapshot: HouseSnapshot, manifest: dict[str, Any]) -> dict[str, Any]:
    """Compute *conditional illustrations* without creating an observed score.

    The lock's official signed-error convention is ``prediction - observed``.
    Here we show the same arithmetic only against an explicitly hypothetical
    result, namely **if every AP leading candidate ultimately wins**.
    """

    defended: int = int(manifest["scenario"]["president_party_seats_defended"])
    bounds: list[float] = manifest["primary_forecast"]["model_based_95_prediction_interval"]
    lower: float = float(bounds[0])
    upper: float = float(bounds[1])
    if not all(math.isfinite(v) for v in (lower, upper)) or lower >= upper:
        raise ValueError("Invalid locked prediction interval")
    trend: int = snapshot.republican_trend
    conditional_change: int = trend - defended
    candidates: dict[str, float] = {
        key: float(value)
        for key, value in manifest["alternative_conditional_point_forecasts"].items()
    }
    if set(candidates) != {"ols", "student_t", "ridge", "recency_wls"}:
        raise ValueError("The four locked estimators must be preserved")
    candidate_comparison: dict[str, dict[str, float]] = {
        key: {
            "registered_point": value,
            "if_leads_hold_predicted_minus_conditional_change": value - conditional_change,
            "if_leads_hold_absolute_difference": abs(value - conditional_change),
        }
        for key, value in candidates.items()
    }
    # For integer House changes, in-interval GOP totals are [ceil(220+L), floor(220+U)].
    covered_min: int = math.ceil(defended + lower)
    covered_max: int = math.floor(defended + upper)
    sensitivity: list[float] = [float(v) for v in manifest["bootstrap_95_sensitivity_interval"]]
    return {
        "report_type": "provisional_house_scenario_diagnostics_not_official_scoring",
        "final_outcome_recorded": False,
        "publisher": snapshot.source_name,
        "source_url": snapshot.source_url,
        "reported_at_utc": snapshot.reported_at_utc,
        "retrieved_at_utc": snapshot.retrieved_at_utc,
        "snapshot_status": "live_unofficial",
        "called_democratic": snapshot.democratic_won,
        "called_republican": snapshot.republican_won,
        "uncalled_races": snapshot.uncalled,
        "uncalled_democratic_leads": snapshot.democratic_leading,
        "uncalled_republican_leads": snapshot.republican_leading,
        "democrats_called_majority": snapshot.democratic_won >= snapshot.majority,
        "republican_seats_if_all_leads_hold": trend,
        "republican_total_lower_bound_from_current_calls": snapshot.republican_won,
        "republican_total_upper_bound_from_current_calls": snapshot.republican_won + snapshot.uncalled,
        "locked_republican_seats_defended": defended,
        "illustrative_change_if_all_leads_hold": conditional_change,
        "if_leads_hold_interval_inclusion": lower <= conditional_change <= upper,
        "locked_ols_95_interval": [lower, upper],
        "locked_bootstrap_sensitivity_interval_not_calibrated": sensitivity,
        "if_leads_hold_bootstrap_sensitivity_band_inclusion": sensitivity[0] <= conditional_change <= sensitivity[1],
        "integer_republican_seats_in_locked_ols_interval": [covered_min, covered_max],
        "candidate_scenario_differences": candidate_comparison,
        "not_inferences": [
            "No final observed 2026 House party division exists in this record.",
            "AP leads are not election results; the if-leads-hold scenario is not a probability forecast.",
            "One election cannot establish interval coverage or statistical calibration.",
            "Do not use this file in place of the locked official scorer.",
        ],
    }


def write_json(report: dict[str, Any], destination: Path) -> None:
    """Serialize stable, human-auditable JSON for diffing in continuous integration."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def make_figures(snapshot: HouseSnapshot, manifest: dict[str, Any], report: dict[str, Any], directory: Path) -> None:
    """Render eight evidence-labeled figures from archived inputs only."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "figure.dpi": 120, "savefig.dpi": 180, "font.size": 11,
        "axes.labelcolor": PALETTE["charcoal"], "text.color": PALETTE["charcoal"],
        "axes.edgecolor": "#A7B0BA", "axes.spines.top": False,
        "axes.spines.right": False,
    })
    forecasts: dict[str, float] = manifest["alternative_conditional_point_forecasts"]
    main: float = float(forecasts["ols"])
    interval: list[float] = report["locked_ols_95_interval"]
    trend: int = report["illustrative_change_if_all_leads_hold"]

    def save(fig: Any, filename: str) -> None:
        fig.savefig(directory / filename, bbox_inches="tight", facecolor="white", metadata={"Creator": "Medium-Blog post-election audit"})
        plt.close(fig)

    # 1. Locked interval and provisional leading scenario are different claims.
    fig, ax = plt.subplots(figsize=(10.5, 3.4))
    ax.hlines(0, interval[0], interval[1], color=PALETTE["teal"], linewidth=11, alpha=.3)
    ax.plot(main, 0, marker="o", color=PALETTE["teal"], markersize=12, label="31 Oct OLS point")
    ax.plot(trend, 0, marker="D", color=PALETTE["gop"], markersize=10, label="4 Nov AP if all leads hold")
    ax.axvline(interval[0], color=PALETTE["muted"], linestyle="--", linewidth=1)
    ax.set_xlim(-57, -8)
    ax.set_ylim(-.9, .9)
    ax.set_yticks([])
    ax.set_xlabel("Republican House seat change (negative = losses)")
    ax.set_title("Frozen forecast, live scenario: one point close to the interval edge")
    ax.legend(loc="lower center", ncol=2, frameon=False)
    save(fig, "fig1_registered_interval_and_live_scenario.png")

    # 2. Category semantics: calls versus leads, never a final tally.
    fig, ax = plt.subplots(figsize=(11, 3.6))
    groups = [
        ("Democratic called", snapshot.democratic_won, PALETTE["dem"], 1),
        ("Republican called", snapshot.republican_won, PALETTE["gop"], 1),
        ("Democratic lead", snapshot.democratic_leading, "#90B2D8", 0),
        ("Republican lead", snapshot.republican_leading, "#DEAB9F", 0),
    ]
    start = 0
    for label, n, color, solid in groups:
        ax.barh(0, n, left=start, height=.42, color=color, edgecolor="white", label=f"{label}: {n}", hatch=None if solid else "///")
        start += n
    ax.set_xlim(0, 435)
    ax.set_yticks([])
    ax.set_xlabel("All 435 House districts: wins called and still-uncalled leads")
    ax.set_title("AP live categories cannot be collapsed into certified results")
    ax.legend(loc="lower center", bbox_to_anchor=(.5, -.65), ncol=2, frameon=False)
    save(fig, "fig2_called_seats_vs_uncalled_leads.png")

    # 3. Locked four-estimator comparison, with no post-hoc winner selection.
    fig, ax = plt.subplots(figsize=(10.5, 4.4))
    names = ["OLS", "Student-t", "Ridge", "Recency WLS"]
    keys = ["ols", "student_t", "ridge", "recency_wls"]
    for i, key in enumerate(keys):
        value = float(forecasts[key])
        ax.plot([value, trend], [i, i], color="#B0BCC7", linewidth=3)
        ax.plot(value, i, "o", color=PALETTE["teal"], markersize=9)
        ax.plot(trend, i, "D", color=PALETTE["gop"], markersize=7)
        ax.text(value+1, i+.17, f"{value:.2f}", fontsize=9)
    ax.set_yticks(range(4), names)
    ax.invert_yaxis()
    ax.set_xlim(-54, -22)
    ax.set_xlabel("Republican House seat change; red diamond = AP if leads hold")
    ax.set_title("All four forecasts were registered before the election")
    save(fig, "fig3_frozen_estimators_vs_live_scenario.png")

    # 4. Integer discontinuity in the interval-hit classification.
    fig, ax = plt.subplots(figsize=(10.5, 4.1))
    totals = list(range(158, 212))
    covered_min, covered_max = report["integer_republican_seats_in_locked_ols_interval"]
    heights = [1 if covered_min <= n <= covered_max else 0 for n in totals]
    colors = [PALETTE["teal"] if hit else PALETTE["gop"] for hit in heights]
    ax.bar(totals, heights, color=colors, width=.9)
    ax.axvline(snapshot.republican_trend, linestyle="--", color=PALETTE["charcoal"], label="AP if leads hold: 172")
    ax.set_ylim(-.05, 1.22)
    ax.set_yticks([0, 1], ["Outside", "Inside"])
    ax.set_xlabel("Republican House seats in a hypothetical final outcome (220 baseline)")
    ax.set_title("One seat can change interval inclusion, not the underlying uncertainty")
    ax.legend(frameon=False)
    save(fig, "fig4_integer_interval_hit_boundary.png")

    # 5. Signed error convention of the archived pre-election scorer.
    fig, ax = plt.subplots(figsize=(10.5, 4.1))
    totals = list(range(165, 192))
    for key, label, color in (
        ("ols", "OLS", PALETTE["teal"]),
        ("student_t", "Student-t", PALETTE["dem"]),
        ("ridge", "Ridge", PALETTE["yellow"]),
        ("recency_wls", "Recency WLS", PALETTE["gop"]),
    ):
        ax.plot(totals, [float(forecasts[key]) - (n - 220) for n in totals], label=label, color=color)
    ax.axhline(0, color=PALETTE["muted"], linewidth=1)
    ax.axvline(snapshot.republican_trend, linestyle="--", color=PALETTE["charcoal"], linewidth=1)
    ax.set_xlabel("Hypothetical final Republican seats (not yet observed)")
    ax.set_ylabel("Prediction minus hypothetical seat change")
    ax.set_title("Registered error sign: a positive value means losses underestimated")
    ax.legend(frameon=False, ncol=2)
    save(fig, "fig5_error_sensitivity_to_final_totals.png")

    # 6. Alternative bootstrap band is sensitivity, not a coverage guarantee.
    fig, ax = plt.subplots(figsize=(10.5, 3.6))
    bands = [
        ("OLS model-based 95%", report["locked_ols_95_interval"], PALETTE["teal"]),
        ("Pairs/residual sensitivity (not calibrated)", report["locked_bootstrap_sensitivity_interval_not_calibrated"], PALETTE["yellow"]),
    ]
    for i, (label, (lo, hi), color) in enumerate(bands):
        ax.plot([lo, hi], [i, i], color=color, linewidth=10, solid_capstyle="round")
        ax.text(hi+1, i, f"[{lo:.1f}, {hi:.1f}]", va="center", fontsize=9)
    ax.axvline(trend, color=PALETTE["gop"], linestyle="--", linewidth=2, label="AP if all leads hold")
    ax.set_xlim(-56, -5)
    ax.set_yticks(range(2), [b[0] for b in bands])
    ax.invert_yaxis()
    ax.set_xlabel("Republican House seat change")
    ax.set_title("Not every uncertainty band has the same statistical meaning")
    ax.legend(frameon=False, loc="lower right")
    save(fig, "fig6_model_interval_vs_bootstrap_sensitivity.png")

    # 7. Clear chronology: published inputs precede the election.
    fig, ax = plt.subplots(figsize=(11, 3.5))
    dates = ["31 Oct", "1 Nov", "3 Nov", "4 Nov", "Later"]
    labels = ["Forecast inputs frozen", "Registration committed", "Election", "AP live snapshot", "Official initial division"]
    xs = list(range(5))
    ax.hlines(0, 0, 4, color=PALETTE["pale"], linewidth=6)
    for i, (x, label) in enumerate(zip(xs, labels, strict=True)):
        color = PALETTE["muted"] if i == 4 else PALETTE["teal"]
        ax.plot(x, 0, "o", markersize=10, color=color)
        ax.text(x, .16 if i % 2 == 0 else -.17, label, ha="center", va="bottom" if i % 2 == 0 else "top", fontsize=9)
    ax.set_xticks(xs, dates)
    ax.set_xlim(-.35, 4.35)
    ax.set_ylim(-.62, .65)
    ax.set_yticks([])
    ax.set_title("The pre/post boundary is a research design, not a caption")
    for spine in ("left", "bottom"):
        ax.spines[spine].set_visible(False)
    save(fig, "fig7_ex_ante_to_unverified_and_final_timeline.png")

    # 8. Historical RMSE and one scenario discrepancy share units but differ in estimand.
    fig, ax = plt.subplots(figsize=(10.5, 3.9))
    historical_rmse: float = float(manifest["primary_forecast"]["forward_rmse_house_seats"])
    provisional_discrepancy: float = abs(main - trend)
    vals = [historical_rmse, provisional_discrepancy]
    bars = ax.barh([0, 1], vals, color=[PALETTE["dem"], PALETTE["gop"]], height=.48)
    ax.bar_label(bars, fmt="%.2f seats", padding=4)
    ax.set_yticks([0, 1], ["Historical 10-election forward RMSE", "One AP if-leads-hold discrepancy"])
    ax.invert_yaxis()
    ax.set_xlim(0, 22)
    ax.set_xlabel("House seats; descriptive scales, not a statistical test")
    ax.set_title("A single new error is not an RMSE estimate")
    save(fig, "fig8_history_vs_one_provisional_discrepancy.png")


def main() -> None:
    """Provide a deterministic audit command with optional output locations."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=SNAPSHOT)
    parser.add_argument("--lock", type=Path, default=LOCK)
    parser.add_argument("--output-dir", type=Path, default=ROOT)
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()
    snapshot, manifest = load_inputs(args.snapshot, args.lock)
    report = diagnostics(snapshot, manifest)
    output_dir: Path = args.output_dir
    write_json(report, output_dir / "results" / "provisional_diagnostics.json")
    if not args.no_figures:
        make_figures(snapshot, manifest, report, output_dir / "figures")
    print(f"AP live: {snapshot.republican_won} GOP called + "
          f"{snapshot.republican_leading} GOP leading = "
          f"{snapshot.republican_trend} if all leads hold; "
          "no final outcome has been scored.")


if __name__ == "__main__":
    main()
