"""Build a deterministic, fresh-kernel executable companion notebook."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
FILENAME = "feature_importance_is_not_causality.ipynb"


def markdown(source: str, identifier: str) -> dict[str, Any]:
    """Return a reproducibly identified Jupyter markdown cell."""

    return {"cell_type": "markdown", "id": identifier, "metadata": {},
            "source": source.splitlines(keepends=True)}


def code(source: str, identifier: str) -> dict[str, Any]:
    """Return an unexecuted Jupyter code cell."""

    return {"cell_type": "code", "id": identifier, "metadata": {},
            "source": source.splitlines(keepends=True),
            "execution_count": None, "outputs": []}


def notebook() -> dict[str, Any]:
    """Describe assumptions, all nine studies, and output verification."""

    cells = [
        markdown(
            "# A Feature Importance Score Is Not a Causal Effect\n\n"
            "Article #29. Nine controlled synthetic experiments, nine original "
            "figures, and eighteen machine-readable evidence tables. This notebook "
            "uses fast independent smoke replications and isolates all outputs in "
            "a fresh temporary directory.\n",
            "introduction",
        ),
        markdown(
            "## Four different targets\n\n"
            "1. **Predictive relevance:** does a feature improve future error "
            "under the specified evaluation design?\n"
            "2. **Fixed-model reliance:** does changing the input change a "
            "fixed predictor's risk?\n"
            "3. **Attribution:** which reference distribution and coalition "
            "game allocates a prediction among coordinates?\n"
            "4. **Causal effect:** what happens to an outcome under a "
            "well-defined intervention in an identified causal system?\n\n"
            "These answers do not become equal merely because all are called "
            "feature importance. The article derives the corresponding formulas.\n",
            "estimands",
        ),
        code(
            "from pathlib import Path\n"
            "import tempfile\n"
            "import numpy as np\n"
            "from IPython.display import Image, display\n"
            "from analysis import Design, EXPERIMENTS, FIGURES, SEED, run\n"
            "print('Seed:', SEED, '| studies:', ', '.join(EXPERIMENTS))\n",
            "imports",
        ),
        markdown(
            "## Exact correlation and attribution identities\n\n"
            "For an oracle f(x)=βx₁ and standardized bivariate normal features "
            "with correlation ρ, marginal permutation increases squared error "
            "in expectation by 2β²; conditional resampling of X₁ given X₂ "
            "increases it by 2β²(1−ρ²).\n\n"
            "For f(x₁,x₂)=x₁ and centered Gaussian features, an "
            "interventional/marginal Shapley game yields (x₁,0), while a "
            "conditional/observational game yields "
            "(x₁−ρx₂/2, ρx₂/2). Both satisfy the coalition axioms for "
            "their *different* value functions; neither identifies a "
            "structural causal effect without more assumptions.\n",
            "identities",
        ),
        code(
            "from analysis import exact_two_feature_shapley\n"
            "rho, x1, x2 = 0.8, 1.3, -0.4\n"
            "for game in ('interventional', 'observational'):\n"
            "    values = exact_two_feature_shapley(x1, x2, rho, game)\n"
            "    assert abs(sum(values)-x1) < 1e-12\n"
            "    print(game, values)\n",
            "checks",
        ),
        markdown(
            "## Run all nine short, independent replicates\n\n"
            "The canonical paper uses 18–24 independently seeded outer "
            "replications, depending on the experiment. This notebook uses "
            "three per study for speed. The generator, estimator and figure "
            "code are the *same functions*. The temporary destination "
            "ensures notebook executions do not overwrite evidence files.\n",
            "method",
        ),
        code(
            "with tempfile.TemporaryDirectory() as directory:\n"
            "    path = Path(directory)\n"
            "    summaries = run(Design(root=path, quick=True))\n"
            "    tables = sorted((path/'results').glob('*_quick.csv'))\n"
            "    figures = sorted((path/'figures').glob('*_quick.png'))\n"
            "    print('Evidence tables:', len(tables), '| original figures:', len(figures))\n"
            "    assert len(tables) == 18 and len(figures) == 9\n"
            "    for label, name in zip(EXPERIMENTS, FIGURES):\n"
            "        print('\\nStudy:', label, '| summary rows:', len(summaries[label]))\n"
            "        display(Image(filename=str(path/'figures'/f'{name}_quick.png')))\n",
            "execution",
        ),
        markdown(
            "## Reading the evidence responsibly\n\n"
            "- All observations are synthetic, not field measurements.\n"
            "- Error bars are 1.96 times the Monte Carlo standard error "
            "across independent outer draws; paired scenarios should "
            "not be analyzed as independent groups.\n"
            "- Random-forest impurity credit and holdout permutation "
            "have different units and different selection mechanisms.\n"
            "- Correlated predictors, interactions, post-outcome leakage, "
            "population shift, subgroup averaging, and winner selection "
            "can each change an importance story for a different reason.\n"
            "- None of these simulations licenses a claim about the causal "
            "effect of an observed predictor. See the full manuscript for "
            "identification conditions and reporting guidance.\n",
            "limitations",
        ),
    ]
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    """Write the exact unexecuted notebook source beside the builder."""

    target = ROOT / FILENAME
    target.write_text(json.dumps(notebook(), indent=2, ensure_ascii=False) + "\n",
                      encoding="utf-8")
    print(target.name)


if __name__ == "__main__":
    main()
