"""Observed commuting-connectivity diagnostics for the 17 AMP municipalities.

This exploratory analysis uses symmetric 2021 work/study interactions, not
residential relocations. Moran's I measures *association* across the fixed
network and provides no estimate of housing-price transmission or causation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analysis import MUNICIPALITIES, ROOT, load_panel, observed_summary
from extract_commuting import FIELDS, OUTPUT as PAIRS_CSV, PROVENANCE, municipality_codes_from_panel


RANDOM_SEED = 20261031
DEFAULT_PERMUTATIONS = 9999
FIGURE_NAMES = ("fig9_2021_commuting_interactions.png", "fig10_commuting_moran_null.png")
RESULT_NAMES = ("commuting_spatial_diagnostics.csv", "commuting_municipal_exposure.csv", "commuting_diagnostics_provenance.json")


def load_weights(path: Path = PAIRS_CSV) -> tuple[list[str], np.ndarray]:
    """Reconstruct one undirected 17x17 matrix from 136 unique pair rows.

    A validation failure is an error, not an invitation to fill missing links
    or combine the symmetric source appearances as two observed flows.
    """
    names = sorted(MUNICIPALITIES)
    idx = {name: i for i, name in enumerate(names)}
    code_to_name = municipality_codes_from_panel()
    matrix = np.zeros((len(names), len(names)), dtype=float)
    seen: set[tuple[str, str]] = set()

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != list(FIELDS):
            raise ValueError("Unexpected commuting-pair schema")
        for row in reader:
            a, b = row["municipality_a"], row["municipality_b"]
            ac, bc = row["municipality_a_code"], row["municipality_b_code"]
            if a not in idx or b not in idx or ac >= bc:
                raise ValueError("Invalid ordered municipality pair")
            if code_to_name.get(ac) != a or code_to_name.get(bc) != b:
                raise ValueError("Municipality code and name disagree")
            if (ac, bc) in seen:
                raise ValueError("Duplicate undirected pair")
            seen.add((ac, bc))
            try:
                count = int(row["commuting_interactions_2021"])
            except (ValueError, TypeError) as exc:
                raise ValueError("Non-integer interaction count") from exc
            if count < 0 or str(count) != row["commuting_interactions_2021"]:
                raise ValueError("Invalid nonnegative whole-number count")
            matrix[idx[a], idx[b]] = count
            matrix[idx[b], idx[a]] = count
    if len(seen) != 17 * 16 // 2:
        raise ValueError(f"Expected 136 distinct municipality pairs, got {len(seen)}")
    if not np.allclose(matrix, matrix.T) or np.any(np.diag(matrix)):
        raise ValueError("Expected zero-diagonal symmetric commuting matrix")
    if not np.all(matrix.sum(axis=1) > 0):
        raise ValueError("At least one municipality has no observed links")
    if path.resolve() == PAIRS_CSV.resolve():
        source_record = json.loads(PROVENANCE.read_text(encoding="utf-8"))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != source_record["derived_csv_sha256"]:
            raise ValueError("Pinned commuting pair CSV SHA-256 mismatch")
    return names, matrix


def weighted_neighbor_growth(growth: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Neighbor outcomes weighted by symmetric interactions, excluding self."""
    validate_weights(weights, n=len(growth))
    if not np.all(np.isfinite(growth)):
        raise ValueError("Missing or non-finite growth outcome")
    return (weights @ growth) / weights.sum(axis=1)


def validate_weights(weights: np.ndarray, *, n: int) -> None:
    """Reject malformed, signed, disconnected and diagonal-weight matrices."""
    if weights.shape != (n, n) or n < 3:
        raise ValueError("Expected an n-by-n matrix with at least three places")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError("Weights must be finite and nonnegative")
    if not np.allclose(weights, weights.T) or not np.allclose(np.diag(weights), 0):
        raise ValueError("Weights must be symmetric with zero diagonal")
    if np.any(weights.sum(axis=1) <= 0):
        raise ValueError("Every location must have a positive weighted degree")


def morans_i(growth: np.ndarray, weights: np.ndarray) -> float:
    """Global Moran's I = (n/S0)(z' W z)/(z' z), using raw symmetric W.

    For a random-label null on a fixed n-node graph, E[I] = -1/(n-1).
    """
    growth = np.asarray(growth, dtype=float)
    validate_weights(weights, n=len(growth))
    if not np.all(np.isfinite(growth)):
        raise ValueError("Non-finite outcome")
    centered = growth - growth.mean()
    ss = centered @ centered
    if ss <= 0:
        raise ValueError("Constant outcome has undefined Moran's I")
    return float((len(growth) / weights.sum()) * (centered @ weights @ centered) / ss)


def permutation_test(
    growth: np.ndarray, weights: np.ndarray, *, seed: int, permutations: int
) -> tuple[float, float, np.ndarray]:
    """Two-sided random-label test against the fixed graph, with plus-one p.

    Two-sided means at least as far from the theoretical random-label mean
    (-1/(n-1)) as the observed I. The seed and test direction are prespecified.
    """
    if permutations < 99:
        raise ValueError("At least 99 permutations are required")
    observed = morans_i(growth, weights)
    rng = np.random.default_rng(seed)
    centered = np.asarray(growth, dtype=float) - np.mean(growth)
    scale = len(centered) / (weights.sum() * (centered @ centered))
    # Fixed W and centered-outcome norm are permutation invariant, so calculate
    # them only once; merely permute municipality labels under the null.
    null = np.array([
        scale * (z @ weights @ z)
        for z in (rng.permutation(centered) for _ in range(permutations))
    ])
    expectation = -1.0 / (len(growth) - 1)
    extreme = np.count_nonzero(np.abs(null - expectation) >= abs(observed - expectation) - 1e-12)
    return observed, (1 + int(extreme)) / (permutations + 1), null


def weights_for_mode(matrix: np.ndarray, mode: str) -> np.ndarray:
    """Exploratory weighting alternatives; avoid a degenerate complete binary W."""
    if mode == "raw":
        return matrix.copy()
    if mode == "sqrt":
        return np.sqrt(matrix)
    if mode == "top3_union":
        # Keep the strongest three links named by either endpoint. The union
        # is required to preserve the undirected meaning of the source data.
        mask = np.zeros(matrix.shape, dtype=bool)
        for i in range(len(matrix)):
            ranked = sorted(
                (j for j in range(len(matrix)) if j != i and matrix[i, j] > 0),
                key=lambda j: (-matrix[i, j], j),
            )
            mask[i, ranked[:3]] = True
        return np.where(mask | mask.T, matrix, 0)
    raise ValueError(f"Unknown weighting mode: {mode}")


def growth_vectors(names: list[str]) -> dict[str, np.ndarray]:
    """Use original published endpoint medians; no invented intermediate data."""
    summary = {str(r["municipality"]): r for r in observed_summary(load_panel())}
    return {
        "sale_price_growth_2019_2025_pct": np.array(
            [float(summary[m]["sale_price_growth_2019_2025_pct"]) for m in names]
        ),
        "new_lease_rent_growth_2020_2025_pct": np.array(
            [float(summary[m]["new_lease_rent_growth_2020_2025_pct"]) for m in names]
        ),
    }


def _plot_heatmap(names: list[str], weights: np.ndarray, path: Path) -> None:
    """Observed symmetric interactions, on a log colour scale for legibility."""
    fig, ax = plt.subplots(figsize=(10.6, 9.4))
    im = ax.imshow(np.log10(1.0 + weights), cmap="YlGnBu", vmin=0)
    ax.set_xticks(range(len(names)), labels=names, rotation=90, fontsize=8.2)
    ax.set_yticks(range(len(names)), labels=names, fontsize=8.2)
    ax.set_title("2021 work/study interactions between AMP municipalities", pad=14, fontsize=13)
    ax.set_xlabel("Municipality (unordered pair; symmetric matrix)")
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    cb.set_label("log10(1 + reported pair interactions)")
    ax.text(0.0, -0.29, "Source: Census-derived workbook attributed to INE 0012340; not household moves.",
            transform=ax.transAxes, fontsize=8, color="#4a5865")
    fig.subplots_adjust(bottom=0.30, left=0.25, right=0.92, top=0.92)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_permutations(
    results: dict[str, tuple[float, float, np.ndarray]], path: Path
) -> None:
    """Show the prespecified random-label null and observed weighted Moran I."""
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.8), sharey=True)
    labels = ["Sale-price growth, 2019–2025", "New-contract rent growth, 2020–2025"]
    for ax, (name, (observed, p, null)), title in zip(axes, results.items(), labels):
        ax.hist(null, bins=43, color="#a9c9dc", edgecolor="white", linewidth=0.35)
        ax.axvline(observed, color="#aa4b2d", lw=2.4, label=f"Observed I = {observed:.3f}")
        ax.axvline(-1 / 16, color="#233f5c", lw=1.2, ls="--", label="Null mean = −0.0625")
        ax.set_title(title, fontsize=10.5)
        ax.set_xlabel("Moran's I (2021 interaction weights)")
        ax.text(0.97, 0.95, f"Two-sided p = {p:.3f}", ha="right", va="top",
                transform=ax.transAxes, fontsize=9.2)
        ax.legend(loc="upper left", fontsize=8, frameon=False)
    axes[0].set_ylabel("Random-label permutations")
    fig.suptitle("Exploratory spatial association, not a test of residential displacement", fontsize=12.5)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def run(output: Path = ROOT, *, permutations: int = DEFAULT_PERMUTATIONS) -> dict[str, object]:
    """Reproduce the observed weighted network and all fixed-seed diagnostics."""
    names, matrix = load_weights()
    growth = growth_vectors(names)
    figure_dir, result_dir = output / "figures", output / "results"
    figure_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    stats: list[dict[str, object]] = []
    primary: dict[str, tuple[float, float, np.ndarray]] = {}
    for mode in ("raw", "sqrt", "top3_union"):
        weights = weights_for_mode(matrix, mode)
        for j, (outcome, y) in enumerate(growth.items()):
            observed, p, null = permutation_test(
                y, weights, seed=RANDOM_SEED + 11 * j, permutations=permutations
            )
            stats.append({"outcome": outcome, "network": mode, "sample": "all_17",
                          "n_municipalities": 17, "observed_moran_i": f"{observed:.6f}",
                          "random_label_expected_i": f"{-1/16:.6f}",
                          "two_sided_p": f"{p:.6f}", "permutations": permutations})
            if mode == "raw":
                primary[outcome] = (observed, p, null)

    # Porto is a high-strength node; recompute the same prespecified raw test
    # without it, rather than pretending the result is independent of Porto.
    keep = [i for i, name in enumerate(names) if name != "Porto"]
    restricted = matrix[np.ix_(keep, keep)]
    for j, (outcome, y) in enumerate(growth.items()):
        observed, p, _ = permutation_test(
            y[keep], restricted, seed=RANDOM_SEED + 100 + 11 * j,
            permutations=permutations,
        )
        stats.append({"outcome": outcome, "network": "raw", "sample": "exclude_porto",
                      "n_municipalities": 16, "observed_moran_i": f"{observed:.6f}",
                      "random_label_expected_i": f"{-1/15:.6f}",
                      "two_sided_p": f"{p:.6f}", "permutations": permutations})

    prices = growth["sale_price_growth_2019_2025_pct"]
    rents = growth["new_lease_rent_growth_2020_2025_pct"]
    weighted_prices = weighted_neighbor_growth(prices, matrix)
    weighted_rents = weighted_neighbor_growth(rents, matrix)
    degrees = matrix.sum(axis=1)
    exposure = [{
        "municipality": name,
        "incident_pair_interactions_2021": int(degrees[i]),
        "share_of_all_amp_pair_incidents_pct": f"{100 * degrees[i] / degrees.sum():.6f}",
        "sale_growth_2019_2025_pct": f"{prices[i]:.6f}",
        "weighted_neighbor_sale_growth_pct": f"{weighted_prices[i]:.6f}",
        "new_rent_growth_2020_2025_pct": f"{rents[i]:.6f}",
        "weighted_neighbor_new_rent_growth_pct": f"{weighted_rents[i]:.6f}",
    } for i, name in enumerate(names)]

    _plot_heatmap(names, matrix, figure_dir / FIGURE_NAMES[0])
    _plot_permutations(primary, figure_dir / FIGURE_NAMES[1])
    _write_csv(result_dir / RESULT_NAMES[0], stats, list(stats[0]))
    _write_csv(result_dir / RESULT_NAMES[1], exposure, list(exposure[0]))

    source_record = json.loads(PROVENANCE.read_text(encoding="utf-8"))
    metadata: dict[str, object] = {
        "source": source_record,
        "interpretation": "undirected work/study interactions; not residential moves or causal price spillover",
        "pair_count": int(np.count_nonzero(np.triu(matrix, k=1))),
        "sum_unique_pair_interactions": int(np.triu(matrix, k=1).sum()),
        "doubling_warning": "Summing both halves of the symmetric matrix doubles each pair count",
        "network": "fixed symmetric 2021 weights, within the legally defined 17 AMP municipalities",
        "moran": "raw symmetric W; two-sided random-label p around E[I] = -1/(n-1)",
        "null_seed": RANDOM_SEED,
        "permutations": permutations,
        "primary": {key: {"moran_i": float(val[0]), "two_sided_p": float(val[1])}
                    for key, val in primary.items()},
        "limits": ["one cross-sectional network", "2021 connectivity overlaps 2019–2025 price windows",
                   "17 observations, multiple exploratory comparisons", "complete binary graph (all 136 pairs positive)",
                   "no direct official INE re-download verification yet", "no observed household relocations"],
    }
    (result_dir / RESULT_NAMES[2]).write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return metadata


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT)
    parser.add_argument("--permutations", type=int, default=DEFAULT_PERMUTATIONS)
    args = parser.parse_args(argv)
    outcome = run(args.output, permutations=args.permutations)
    print(json.dumps({"primary": outcome["primary"], "pairs": outcome["pair_count"]}, indent=2))


if __name__ == "__main__":
    main()
