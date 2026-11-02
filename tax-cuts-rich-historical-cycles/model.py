"""Illustrative tax feedback and public/private capital sensitivity model.

All inputs are hypothetical and dimensionless. No result is a historical estimate.
The model uses the standard library only and writes deterministic SVG/CSV outputs.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Scenario:
    """Parameters defining a hypothetical one-period policy response."""

    rate_before: float = 0.35
    rate_after: float = 0.28
    taxable_base: float = 100.0
    base_elasticity: float = 0.3
    capital_elasticity: float = 0.4
    public_productivity: float = 0.1
    private_capital_share: float = 0.3
    public_spending_share: float = 0.4

    def __post_init__(self) -> None:
        """Reject economically invalid or nonfinite parameters."""
        values = vars(self)
        if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in values.values()):
            raise TypeError("All parameters must be finite numeric values")
        if not 0 < self.rate_after < self.rate_before < 1:
            raise ValueError("Tax rates must satisfy 0 < after < before < 1")
        if self.taxable_base <= 0:
            raise ValueError("Tax base must be positive")
        if self.base_elasticity < 0 or self.capital_elasticity < 0 or self.public_productivity < 0:
            raise ValueError("Elasticities and productivity must be nonnegative")
        if not 0 <= self.private_capital_share <= 1:
            raise ValueError("Private capital share must lie within [0, 1]")
        if not 0 <= self.public_spending_share <= 1:
            raise ValueError("Public spending share must lie within [0, 1]")


def evaluate(scenario: Scenario) -> dict[str, float]:
    """Compute behavioral revenue feedback and an explicit production scenario.

    The tax base responds to the net-of-tax share with constant elasticity.
    Private K responds to the same ratio; public K changes only under a
    spending-offset rule. This is comparative statics, not an equilibrium model.
    """
    s = scenario
    net_ratio = (1 - s.rate_after) / (1 - s.rate_before)
    base_after = s.taxable_base * net_ratio**s.base_elasticity
    revenue_before = s.rate_before * s.taxable_base
    revenue_after = s.rate_after * base_after
    static_loss = (s.rate_before - s.rate_after) * s.taxable_base
    actual_loss = revenue_before - revenue_after
    feedback_fraction = (static_loss - actual_loss) / static_loss

    # A spending offset reduces public capital proportionally to lost revenue.
    private_capital_ratio = net_ratio**s.capital_elasticity
    public_capital_ratio = 1 - s.public_spending_share * actual_loss / revenue_before
    if public_capital_ratio <= 0:
        raise ValueError("The spending scenario exhausts public capital")
    output_ratio = (
        private_capital_ratio**s.private_capital_share
        * public_capital_ratio**s.public_productivity
    )
    return {
        "base_after": base_after,
        "revenue_before": revenue_before,
        "revenue_after": revenue_after,
        "static_loss": static_loss,
        "actual_loss": actual_loss,
        "feedback_fraction": feedback_fraction,
        "private_capital_ratio": private_capital_ratio,
        "public_capital_ratio": public_capital_ratio,
        "output_ratio": output_ratio,
    }


def elasticity_grid() -> list[dict[str, float]]:
    """Return a stable 3 x 3 sensitivity grid for the article."""
    output: list[dict[str, float]] = []
    for base_elasticity in (0.0, 0.5, 1.0):
        for public_productivity in (0.0, 0.3, 0.6):
            scenario = Scenario(base_elasticity=base_elasticity, public_productivity=public_productivity)
            result = evaluate(scenario)
            output.append({
                "base_elasticity": base_elasticity,
                "public_productivity": public_productivity,
                "revenue_feedback_pct": 100 * result["feedback_fraction"],
                "output_change_pct": 100 * (result["output_ratio"] - 1),
            })
    return output


def write_outputs(directory: Path) -> None:
    """Produce canonical deterministic CSV and self-contained SVG figure."""
    if not isinstance(directory, Path):
        raise TypeError("directory must be pathlib.Path")
    directory.mkdir(parents=True, exist_ok=True)
    rows = elasticity_grid()
    columns = ("base_elasticity", "public_productivity", "revenue_feedback_pct", "output_change_pct")
    with (directory / "sensitivity.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    # Three aligned panels are not needed for nine values: use a heatmap with
    # explicitly printed percent values and a fixed diverging sign convention.
    cells: list[str] = []
    for index, row in enumerate(rows):
        col, line = index // 3, index % 3
        value = row["output_change_pct"]
        colour = "#d9ead3" if value >= 0 else "#f4cccc"
        x, y = 210 + 180 * col, 115 + 100 * line
        cells.append(f'<rect x="{x}" y="{y}" width="172" height="92" fill="{colour}" stroke="#666"/>')
        cells.append(f'<text x="{x + 86}" y="{y + 52}" text-anchor="middle" font-size="23">{value:+.2f}%</text>')
    labels = [
        '<text x="300" y="88" text-anchor="middle">e = 0.0</text>',
        '<text x="480" y="88" text-anchor="middle">e = 0.5</text>',
        '<text x="660" y="88" text-anchor="middle">e = 1.0</text>',
    ]
    for i, productivity in enumerate((0.0, 0.3, 0.6)):
        labels.append(f'<text x="195" y="{168 + 100 * i}" text-anchor="end">g = {productivity:.1f}</text>')
    figure = '\n'.join([
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 790 480">',
        '<rect width="790" height="480" fill="white"/>',
        '<text x="395" y="32" text-anchor="middle" font-size="22" font-weight="bold">Hypothetical output response to a tax cut</text>',
        '<text x="395" y="58" text-anchor="middle" font-size="14">Rows: public capital productivity (g); columns: taxable-base elasticity (e)</text>',
        *labels, *cells,
        '<text x="395" y="446" text-anchor="middle" font-size="14">Invented parameters; colours show sign, not statistical significance</text>',
        '</svg>',
    ])
    (directory / "sensitivity.svg").write_text(figure + "\n", encoding="utf-8")


def main() -> None:
    """Command line entry point for deterministic output generation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "results")
    args = parser.parse_args()
    write_outputs(args.out)


if __name__ == "__main__":
    main()
