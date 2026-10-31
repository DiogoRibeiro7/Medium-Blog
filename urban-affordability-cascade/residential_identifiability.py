"""Sharp non-identification bounds for Porto-to-AMP residential exchange.

The observed inputs are Census 2021 *destination* margins for residents who
lived in a different Portuguese municipality on 31 December 2019. No source
here identifies their prior municipality. This module proves how many
opposite directional stories satisfy the same published counts; its examples
are feasible allocations, NOT estimates of residential origin–destination
flows or of housing-price-induced displacement.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from extract_residential_mobility import DEFAULT_PROVENANCE, ROOT
from residential_mobility import load_residential


FIGURE = "fig13_residential_od_identification_set.png"
RESULT_JSON = "residential_od_identification_bounds.json"
RESULT_CSV = "residential_od_feasible_worlds.csv"


@dataclass(frozen=True, slots=True)
class DestinationMargins:
    """2021 residents previously in another Portuguese municipality.

    The AMP count contains Porto. `rest_amp` is the difference between two
    nested *destination* counts, not a newly observed migration flow.
    """

    porto: int
    amp: int

    def __post_init__(self) -> None:
        if any(isinstance(x, bool) or not isinstance(x, int) for x in (self.porto, self.amp)):
            raise TypeError("Census destination margins must be integer counts")
        if self.porto < 0 or self.amp < self.porto:
            raise ValueError("Nested Census destination margins are inconsistent")

    @property
    def rest_amp(self) -> int:
        """Intermunicipal in-movers now resident in the other 16 AMP municipalities."""
        return self.amp - self.porto

    @property
    def net_outward_bounds(self) -> tuple[int, int]:
        """Sharp bounds on Porto→rest minus rest→Porto among 2021 residents."""
        return -self.porto, self.rest_amp


@dataclass(frozen=True, slots=True)
class FeasibleAllocation:
    """One hypothetical residential OD allocation respecting observed margins."""

    scenario: str
    porto_to_rest_amp: int
    rest_amp_to_porto: int
    porto_inbound_from_elsewhere: int
    rest_amp_inbound_from_non_porto: int
    net_porto_to_rest_amp: int
    evidence_class: str = "hypothetical_feasible_allocation_not_observed"


def load_margins() -> DestinationMargins:
    """Derive nested margins from the source-hashed Census extract."""
    records = {record.geography_code: record for record in load_residential()}
    porto, amp = records["1312"], records["11A"]
    if amp.resident_total_2021 < porto.resident_total_2021:
        raise ValueError("AMP population must include Porto population")
    return DestinationMargins(
        porto=porto.moved_from_other_municipality,
        amp=amp.moved_from_other_municipality,
    )


def allocate(
    margins: DestinationMargins,
    *,
    scenario: str,
    porto_to_rest_amp: int,
    rest_amp_to_porto: int,
) -> FeasibleAllocation:
    """Construct a valid two-direction allocation with explicit residual origins.

    No conservation equation equates Porto outflows with Porto in-movers:
    the Census counts are classified by the *current* place of residence.
    The residuals can originate in other municipalities, including outside
    AMP. We do not assert the existence of any particular residual pairing.
    """
    if not isinstance(scenario, str) or not scenario.strip():
        raise ValueError("A nonempty scenario label is required")
    values = (porto_to_rest_amp, rest_amp_to_porto)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise TypeError("Hypothetical directional flows must be integer counts")
    if not (0 <= porto_to_rest_amp <= margins.rest_amp):
        raise ValueError("Porto-to-rest allocation exceeds other-AMP destination margin")
    if not (0 <= rest_amp_to_porto <= margins.porto):
        raise ValueError("Rest-to-Porto allocation exceeds Porto destination margin")
    return FeasibleAllocation(
        scenario=scenario,
        porto_to_rest_amp=porto_to_rest_amp,
        rest_amp_to_porto=rest_amp_to_porto,
        porto_inbound_from_elsewhere=margins.porto - rest_amp_to_porto,
        rest_amp_inbound_from_non_porto=margins.rest_amp - porto_to_rest_amp,
        net_porto_to_rest_amp=porto_to_rest_amp - rest_amp_to_porto,
    )


def contrasting_worlds(margins: DestinationMargins) -> list[FeasibleAllocation]:
    """Fixed examples that are observationally indistinguishable in this table."""
    proposals = (
        ("Zero within-AMP exchange", 0, 0),
        ("Hypothetical net outward", 25_000, 5_000),
        ("Hypothetical net inward", 5_000, 12_000),
        ("Sharp outward endpoint", margins.rest_amp, 0),
        ("Sharp inward endpoint", 0, margins.porto),
    )
    return [
        allocate(margins, scenario=name, porto_to_rest_amp=x, rest_amp_to_porto=y)
        for name, x, y in proposals
    ]


def plot_identification_set(margins: DestinationMargins, output: Path) -> None:
    """Draw the fully feasible OD rectangle, NOT a probability density."""
    width, height = margins.rest_amp, margins.porto
    if width == 0 or height == 0:
        raise ValueError("Nonempty observed margins required for the illustrative plot")

    fig, (ax, bound) = plt.subplots(
        1, 2, figsize=(13.1, 6.2), gridspec_kw={"width_ratios": [1.7, 1.0]}
    )
    ax.fill_between([0, width], 0, height, color="#d8ede9", alpha=0.85)
    ax.plot([0, height], [0, height], color="#a85842", lw=2, label="Zero net exchange")
    # The rectangle encodes all possible pair allocations; its area has no
    # likelihood interpretation and must never be used as a prior over flows.
    ax.plot([0, width, width, 0, 0], [0, 0, height, height, 0], color="#467f7b", lw=1.5)
    markers = (
        (25_000, 5_000, "A: +20,000", "#2a617b", (8, -21)),
        (5_000, 12_000, "B: −7,000", "#aa5c42", (8, -22)),
    )
    for x, y, label, color, offset in markers:
        if x <= width and y <= height:
            ax.scatter([x], [y], s=64, color=color, zorder=5)
            ax.annotate(label, (x, y), xytext=offset, textcoords="offset points",
                        fontsize=9, color=color)
    ax.set_xlim(-1_500, width * 1.04)
    ax.set_ylim(-400, height * 1.12)
    ax.set_xlabel("Hypothetical Porto → other AMP moves (people)")
    ax.set_ylabel("Hypothetical other AMP → Porto moves (people)")
    ax.set_title("Every point fits the same Census margins", fontsize=11.5)
    ax.grid(alpha=0.15)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", frameon=False, fontsize=9)

    lower, upper = margins.net_outward_bounds
    bound.hlines(0, lower, upper, color="#467f7b", lw=7, alpha=0.75)
    bound.scatter([lower, 0, upper], [0, 0, 0], s=[90, 50, 90],
                  c=["#a85842", "#5b6974", "#2d7670"], zorder=5)
    bound.annotate(f"{lower:,}", (lower, 0), xytext=(0, -34),
                   textcoords="offset points", ha="center", fontsize=10)
    bound.annotate(f"+{upper:,}", (upper, 0), xytext=(0, -34),
                   textcoords="offset points", ha="center", fontsize=10)
    bound.set_ylim(-1, 1)
    bound.set_xlim(lower - 9_000, upper + 9_000)
    bound.set_yticks([])
    bound.set_xlabel("Net exchange, Porto → other AMP (people)")
    bound.set_title("Sharp bounds from margins alone", fontsize=11.5)
    for spine in ("top", "right", "left"):
        bound.spines[spine].set_visible(False)
    bound.spines["bottom"].set_visible(False)
    bound.tick_params(axis="x", bottom=False, labelbottom=False)
    bound.text(0.5, 0.68, "Sign not identified", transform=bound.transAxes,
               ha="center", fontsize=11, color="#4e6375")

    fig.suptitle("Residential flows cannot be reconstructed from destination margins",
                 fontsize=14, y=0.98)
    fig.text(0.025, 0.025,
             "INE Censos 2021 via Município do Porto. Rectangle and markers are FEASIBLE, NOT OBSERVED flows. "
             "No probability or price causality implied.", fontsize=8.5, color="#526377")
    fig.subplots_adjust(left=0.075, right=0.975, bottom=0.17, top=0.82, wspace=0.24)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def run(output: Path = ROOT) -> dict[str, object]:
    """Generate the auditable sharp bounds, five example worlds, and Figure 13."""
    margins = load_margins()
    worlds = contrasting_worlds(margins)
    figure_dir, result_dir = output / "figures", output / "results"
    figure_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    plot_identification_set(margins, figure_dir / FIGURE)

    with (result_dir / RESULT_CSV).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(worlds[0])), lineterminator="\n")
        writer.writeheader()
        writer.writerows(asdict(world) for world in worlds)

    source = json.loads(DEFAULT_PROVENANCE.read_text(encoding="utf-8"))
    lower, upper = margins.net_outward_bounds
    data: dict[str, object] = {
        "evidence_class": "sharp_partial_identification_from_observed_destination_margins",
        "source_extract_sha256": source["derived_csv_sha256"],
        "source_archive_sha256": source["source_archive_sha256"],
        "reference_start": source["reference_start"],
        "census_reference": source["census_reference"],
        "observed_other_municipality_destination_margins": {
            "porto": margins.porto, "amp_including_porto": margins.amp,
            "other_16_amp": margins.rest_amp,
        },
        "sharp_bounds_given_these_margins_only": {
            "porto_to_other_amp": [0, margins.rest_amp],
            "other_amp_to_porto": [0, margins.porto],
            "net_outward_porto_to_other_amp": [lower, upper],
        },
        "worlds": [asdict(world) for world in worlds],
        "interpretation": (
            "Sharp logical bounds for cross-municipality exchange between Porto and the other "
            "16 AMP municipalities among 2021 residents. They are NOT statistical confidence "
            "intervals, estimated flows, historical net migration, or housing displacement."
        ),
        "why_unidentified": (
            "The published other-municipality category does not disclose the prior municipality "
            "and may include prior residence outside AMP. Either sign of exchange is feasible."
        ),
        "requirements_to_identify": [
            "municipality of residence on 31 Dec 2019 by municipality on 19 Apr 2021",
            "privacy-safe cross-tabulation with official geography codes and suppressed-cell rules",
            "tenure, household, housing cost and move-reason data for any price-displacement claim",
            "multi-period variation and an identification strategy for causal housing spillovers",
        ],
    }
    (result_dir / RESULT_JSON).write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return data


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    result = run(args.output)
    bounds = result["sharp_bounds_given_these_margins_only"]
    print(f"Reproduced non-identification bounds: {bounds}")


if __name__ == "__main__":
    main()
