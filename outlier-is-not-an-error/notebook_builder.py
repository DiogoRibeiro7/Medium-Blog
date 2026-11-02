"""Build an executable Jupyter companion for article #31 without extra tools."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
FILENAME = "outlier_is_not_an_error.ipynb"


def markdown(source: str, identifier: str) -> dict[str, Any]:
    """Create one cell with a stable notebook v4 ID."""

    return {"cell_type": "markdown", "id": identifier, "metadata": {},
            "source": source.splitlines(keepends=True)}


def code(source: str, identifier: str) -> dict[str, Any]:
    """Create an unexecuted code cell; checked fresh in CI."""

    return {"cell_type": "code", "id": identifier, "metadata": {},
            "source": source.splitlines(keepends=True),
            "execution_count": None, "outputs": []}


def notebook() -> dict[str, Any]:
    """Generate nine original figures from the exact article study engine."""

    cells = [
        markdown(
            "# An Outlier Is Not an Error\n\n"
            "Article #31: nine seeded experiments, nine original figures, "
            "eighteen evidence CSV files. Every population, label, and shift "
            "is synthetic. A score is not a verified mistake.\n",
            "introduction",
        ),
        markdown(
            "## Define the target before considering robust estimators\n\n"
            "- For mixture (1−ε)N(0,1)+εN(K,1), the **population mean** "
            "equals εK. Suppressing its shift is useful only if the intended "
            "target is the dominant component's location.\n"
            "- Huber location here solves Σ clip((x−θ)/1, ±1.345)=0 with "
            "**known unit scale**. It is not an adaptive joint location-scale fit.\n"
            "- A Cauchy distribution has a center and median but no "
            "defined mathematical mean.\n"
            "- Student-t intervals target the *mixture mean*. Sign-test "
            "intervals target the *mixture median*. Their widths are not "
            "interchangeable.\n",
            "estimands",
        ),
        code(
            "from pathlib import Path\n"
            "from IPython.display import Image, display\n"
            "import numpy as np\n"
            "from analysis import (Design, EXPERIMENTS, FIGURES, SEED, "
            "mixture_median, expected_shortfall_lognormal, "
            "sign_median_interval, run)\n"
            "print('Seed:', SEED, '| experiments:', len(EXPERIMENTS))\n",
            "imports",
        ),
        code(
            "assert abs(mixture_median(0., 20.)) < 1e-9\n"
            "q, es = expected_shortfall_lognormal(.99, 1.1)\n"
            "assert es > q > 0\n"
            "low, high = sign_median_interval(np.linspace(-2, 2, 100))\n"
            "assert low < 0 < high\n"
            "print('Lognormal q99, ES99:', round(q, 3), round(es, 3))\n",
            "mathematical_checks",
        ),
        markdown(
            "## Reproduce the study code in an isolated temporary directory\n\n"
            "Quick mode runs three outer repetitions for most experiments "
            "(sixteen for CI coverage), with independent stable SeedSequence "
            "streams. It writes `_quick` files only and leaves canonical "
            "full-mode results untouched. Plots show ±1.96 times the "
            "outer-replication standard error, not sampling confidence "
            "intervals for a real population.\n",
            "execute_description",
        ),
        code(
            "import tempfile\n"
            "with tempfile.TemporaryDirectory() as directory:\n"
            "    root = Path(directory)\n"
            "    summaries = run(Design(root=root, quick=True))\n"
            "    tables = sorted((root/'results').glob('*_quick.csv'))\n"
            "    graphics = sorted((root/'figures').glob('*_quick.png'))\n"
            "    assert len(tables) == 18 and len(graphics) == 9\n"
            "    print('Evidence tables:', len(tables), '| figures:', len(graphics))\n"
            "    for study, figure in zip(EXPERIMENTS, FIGURES):\n"
            "        print('\\n', study, '| summary cells:', len(summaries[study]))\n"
            "        display(Image(filename=str(root/'figures'/f'{figure}_quick.png')))\n",
            "run_all_nine",
        ),
        markdown(
            "## Interpretation boundaries\n\n"
            "1. The contaminating distribution can be the *true* population, "
            "not an error mechanism. The robust estimate may target a "
            "different population functional than the mean.\n"
            "2. Cook's D is computed from an OLS model with an intercept; "
            "high leverage alone neither establishes a typo nor a large "
            "coefficient change.\n"
            "3. Robust covariance and Isolation Forest scores depend on "
            "training distribution, metric, and threshold. Neither "
            "establishes that an observation was recorded incorrectly.\n"
            "4. The legitimate mixture minority in study 7 is labeled "
            "valid by construction. Its anomaly score carries no moral "
            "or validity judgment.\n"
            "5. Study 8 deliberately changes a tail-risk estimand by "
            "deleting or capping valid outcomes. Study 9 recalibrates "
            "a 1% *review budget*, not a verified 1% error rate.\n\n"
            "For sources, alternative estimands, numerical values, reporting "
            "protocols, and operational safeguards see `article.md`.\n",
            "boundaries",
        ),
    ]
    return {"cells": cells,
            "metadata": {"kernelspec": {"display_name": "Python 3",
                                       "language": "python", "name": "python3"},
                         "language_info": {"name": "python", "version": "3.12"}},
            "nbformat": 4, "nbformat_minor": 5}


def main() -> None:
    """Write the canonical, intentionally unexecuted notebook with clean cells."""

    destination = ROOT/FILENAME
    destination.write_text(json.dumps(notebook(), indent=2, ensure_ascii=False)+"\n",
                           encoding="utf-8")
    print(destination.name)


if __name__ == "__main__":
    main()
