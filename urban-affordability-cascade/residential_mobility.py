"""Descriptive Porto Census 2021 residential-origin contrasts by parish.

These are observed 2019-to-2021 retrospective *inbound* origin categories for
residents enumerated in April 2021. They cannot identify movers' origin
municipality, Porto's outward flows, move motives, or affordability effects.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from extract_residential_mobility import (
    DEFAULT_CSV,
    DEFAULT_PROVENANCE,
    FIELDS,
    FIELD_MAP,
    REQUIRED_GEOGRAPHIES,
    ROOT,
    ResidentialOrigin,
    verify_porto_reconciliation,
)


FIGURES = ("fig11_parish_intermunicipal_in_mobility.png", "fig12_parish_residential_mover_mix.png")
RESULTS = ("residential_mobility_parish_summary.csv", "residential_mobility_reconciliation.json")
SHORT_NAMES = {
    "131216": "Aldoar–Foz do Douro–Nevogilde",
    "131217": "Cedofeita–Sé–Miragaia union",
    "131218": "Lordelo do Ouro–Massarelos",
}
COLOURS = ("#34877f", "#e6a15a", "#b95b47", "#8471b2")


def load_residential(path: Path = DEFAULT_CSV) -> list[ResidentialOrigin]:
    """Verify source-pinned CSV, exact 11 identities and category partitions."""
    if path.resolve() == DEFAULT_CSV.resolve():
        metadata = json.loads(DEFAULT_PROVENANCE.read_text(encoding="utf-8"))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != metadata["derived_csv_sha256"]:
            raise ValueError("Observed residential-mobility extract changed without provenance")
    records: list[ResidentialOrigin] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != list(FIELDS):
            raise ValueError("Unexpected residential mobility schema")
        for row in reader:
            values: dict[str, int] = {}
            for field in FIELD_MAP:
                raw = row[field]
                try:
                    count = int(raw)
                except (ValueError, TypeError) as exc:
                    raise ValueError(f"Invalid count for {field}") from exc
                if str(count) != raw:
                    raise ValueError(f"Non-canonical integer for {field}")
                values[field] = count
            record = ResidentialOrigin(
                row["geography_code"], row["geography"], row["level"], **values
            )
            record.verify()
            reference = record.as_row()
            for field in FIELDS[-4:]:
                if field == "eligible_2019_population":
                    if str(reference[field]) != row[field]:
                        raise ValueError("Wrong 2019-eligible denominator")
                elif not math.isclose(float(reference[field]), float(row[field]), abs_tol=1e-6):
                    raise ValueError(f"Incorrect derived migration rate: {field}")
            records.append(record)
    verify_porto_reconciliation(records)
    return records


def parish_summary(records: Sequence[ResidentialOrigin]) -> list[dict[str, str | int]]:
    """Parish metrics and descriptive city-level contrasts (no significance)."""
    lookup = {item.geography_code: item for item in records}
    porto = lookup["1312"]
    parishes = [item for item in records if item.level == "parish"]
    rows: list[dict[str, str | int]] = []
    for item in sorted(parishes, key=lambda rec: rec.moved_from_other_municipality / rec.eligible_2019_population):
        pct = 100 * item.moved_from_other_municipality / item.eligible_2019_population
        city_pct = 100 * porto.moved_from_other_municipality / porto.eligible_2019_population
        rows.append({
            "geography_code": item.geography_code,
            "geography": item.geography,
            "eligible_2019_population": item.eligible_2019_population,
            "2019_other_municipality_residents": item.moved_from_other_municipality,
            "2019_other_municipality_pct_eligible": f"{pct:.6f}",
            "difference_from_porto_pct_points": f"{pct - city_pct:.6f}",
            "all_residence_movers": item.all_movers,
            "all_residence_movers_pct_eligible": f"{100 * item.all_movers / item.eligible_2019_population:.6f}",
        })
    return rows


def plot_intermunicipal(records: Sequence[ResidentialOrigin], out: Path) -> None:
    """Observed shares of current residents previously in another municipality."""
    lookup = {item.geography_code: item for item in records}
    rows = parish_summary(records)
    labels = [SHORT_NAMES.get(r["geography_code"], str(r["geography"])) for r in rows]
    shares = [float(r["2019_other_municipality_pct_eligible"]) for r in rows]
    fig, ax = plt.subplots(figsize=(11.7, 6.7))
    bars = ax.barh(labels, shares, color="#34877f", height=0.69)
    porto = lookup["1312"]
    amp = lookup["11A"]
    porto_share = 100 * porto.moved_from_other_municipality / porto.eligible_2019_population
    amp_share = 100 * amp.moved_from_other_municipality / amp.eligible_2019_population
    ax.axvline(porto_share, color="#a34e2d", lw=2, label=f"Porto: {porto_share:.2f}%")
    ax.axvline(amp_share, color="#193e6a", lw=1.7, ls="--", label=f"AMP: {amp_share:.2f}%")
    for bar, row in zip(bars, rows):
        ax.text(bar.get_width() + 0.12, bar.get_y() + bar.get_height()/2,
                f"{int(row['2019_other_municipality_residents']):,}",
                va="center", fontsize=8.7, color="#33475a")
    ax.set_xlim(0, max(shares) + 1.5)
    ax.set_xlabel("Share of 2021 residents who lived in another municipality on 31 Dec 2019 (%)")
    ax.set_title("Residential in-mobility varies across Porto's seven parishes", fontsize=13, pad=14)
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    ax.grid(axis="x", alpha=0.16)
    ax.set_axisbelow(True)
    fig.text(0.02, 0.025,
             "Source: INE Censos 2021 via Município do Porto (official archive). Not outward moves or an annual rate.",
             fontsize=8.5, color="#536579")
    fig.subplots_adjust(left=0.29, right=0.96, bottom=0.15, top=0.90)
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_mover_mix(records: Sequence[ResidentialOrigin], out: Path) -> None:
    """Partition *movers* by 2019 origin category; avoid total-population mixups."""
    parishes = sorted(
        (r for r in records if r.level == "parish"),
        key=lambda r: r.all_movers / r.eligible_2019_population,
    )
    labels = [SHORT_NAMES.get(r.geography_code, r.geography) for r in parishes]
    categories = (
        ("Moved within parish", "moved_within_same_parish"),
        ("Other Porto parish", "moved_from_other_parish_same_municipality"),
        ("Other municipality", "moved_from_other_municipality"),
        ("Abroad", "moved_from_abroad"),
    )
    fig, ax = plt.subplots(figsize=(12, 6.9))
    bottom = np.zeros(len(parishes))
    for (label, field), colour in zip(categories, COLOURS):
        values = np.array([100 * getattr(r, field) / r.all_movers for r in parishes])
        ax.barh(labels, values, left=bottom, color=colour, height=0.67, label=label)
        bottom += values
    if not np.allclose(bottom, 100):
        raise ValueError("Mover-origin categories are not an exhaustive partition")
    for i, item in enumerate(parishes):
        share = 100 * item.all_movers / item.eligible_2019_population
        ax.text(100.5, i, f"{share:.1f}% moved", va="center", fontsize=9, color="#3c4e5c")
    ax.set_xlim(0, 118)
    ax.set_xlabel("Composition of residents who moved since 31 Dec 2019 (%)")
    ax.set_title("Where Porto's 2021 residents lived before their move", fontsize=13, pad=14)
    ax.legend(ncol=2, bbox_to_anchor=(0.5, -0.10), loc="upper center", frameon=False, fontsize=9)
    fig.text(0.02, 0.022,
             "Denominator: movers only (not all residents). Origin municipality within Portugal is not identified.",
             fontsize=8.5, color="#536579")
    fig.subplots_adjust(left=0.29, right=0.96, bottom=0.29, top=0.91)
    fig.savefig(out, dpi=160)
    plt.close(fig)


def run(output: Path = ROOT) -> dict[str, object]:
    """Reproduce two purely descriptive figures and auditable results."""
    records = load_residential()
    lookup = {item.geography_code: item for item in records}
    figure_dir, result_dir = output / "figures", output / "results"
    figure_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    plot_intermunicipal(records, figure_dir / FIGURES[0])
    plot_mover_mix(records, figure_dir / FIGURES[1])

    rows = parish_summary(records)
    with (result_dir / RESULTS[0]).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    porto, amp = lookup["1312"], lookup["11A"]
    source = json.loads(DEFAULT_PROVENANCE.read_text(encoding="utf-8"))
    result: dict[str, object] = {
        "source": source,
        "geographies": len(records),
        "portugal_residents_2021": lookup["PT"].resident_total_2021,
        "porto_residents_2021": porto.resident_total_2021,
        "porto_parish_reconciliation": "exact for all eight source count fields",
        "porto_other_municipality_in_movers": porto.moved_from_other_municipality,
        "porto_other_municipality_share_eligible_pct": round(100 * porto.moved_from_other_municipality / porto.eligible_2019_population, 6),
        "amp_other_municipality_share_eligible_pct": round(100 * amp.moved_from_other_municipality / amp.eligible_2019_population, 6),
        "min_parish_other_municipality_pct": min(float(r["2019_other_municipality_pct_eligible"]) for r in rows),
        "max_parish_other_municipality_pct": max(float(r["2019_other_municipality_pct_eligible"]) for r in rows),
        "limits": [
            "not a directed municipality origin-destination matrix",
            "Porto inward residence changes only; cannot measure Porto-to-suburb exits",
            "2019-12-31 to 2021-04-19 is not an annual rate",
            "seven parishes are cross sections and do not identify affordability effects",
            "resident individuals, not household relocation counts",
            "AMP and Porto comparisons overlap geographically",
        ],
    }
    (result_dir / RESULTS[1]).write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    result = run(args.output)
    print(json.dumps({
        "porto_other_municipality_in_movers": result["porto_other_municipality_in_movers"],
        "porto_other_municipality_share_eligible_pct": result["porto_other_municipality_share_eligible_pct"],
    }, indent=2))


if __name__ == "__main__":
    main()
