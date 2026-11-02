"""Generate article #33's deterministic, unexecuted Jupyter notebook."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
FILENAME = "the_test_set_is_not_an_oracle.ipynb"


def markdown(source: str, identifier: str) -> dict[str, Any]:
    """Create a stable-identity Markdown cell."""

    return {"cell_type": "markdown", "id": identifier, "metadata": {},
            "source": source.splitlines(keepends=True)}


def code(source: str, identifier: str) -> dict[str, Any]:
    """Create a clean code cell, with no cached output or execution count."""

    return {"cell_type": "code", "id": identifier, "metadata": {},
            "source": source.splitlines(keepends=True),
            "execution_count": None, "outputs": []}


def notebook() -> dict[str, Any]:
    """Show mathematical checks and run all nine lightweight study designs."""

    cells = [
        markdown(
            "# The Test Set Is Not an Oracle\n\n"
            "Article #33 examines what predictive evaluation can and cannot "
            "establish. Nine independent synthetic experiments cover iid CV, "
            "model search, nested CV, feature-selection leakage, temporal "
            "drift, look-ahead features, hierarchical groups, fold uncertainty, "
            "and adaptive test-set reuse.\n",
            "intro",
        ),
        markdown(
            "## What is being estimated?\n\n"
            "For a fixed learner A, cross-validation estimates a risk for a "
            "training procedure with approximately (K-1)/K of the available "
            "observations. Reusing those folds to select an algorithm makes "
            "their minimum an optimistic estimate of the selection process. "
            "Outer CV re-runs selection inside each training fold, but cannot "
            "make distribution shift or future-feature leakage disappear.\n\n"
            "All observations here are synthetic; Brier loss is lower-is-better "
            "and every probability is evaluated against a simulated outcome.\n",
            "estimands",
        ),
        code(
            "from pathlib import Path\n"
            "import os\n"
            "import sys\n"
            "import tempfile\n"
            "import numpy as np\n"
            "from IPython.display import Image, display\n"
            "# Import from the article directory, repository root, or an explicit hint.\n"
            "slug = 'the-test-set-is-not-an-oracle'\n"
            "hint = os.environ.get('VALIDATION_ARTICLE_DIR')\n"
            "location = Path.cwd().resolve()\n"
            "candidates = ([Path(hint).expanduser().resolve()] if hint else\n"
            "              [candidate for parent in (location, *location.parents)\n"
            "               for candidate in (parent, parent/slug)])\n"
            "study_dir = next((candidate for candidate in candidates\n"
            "                  if (candidate/'analysis.py').is_file() and\n"
            "                  (candidate/'the_test_set_is_not_an_oracle.ipynb').is_file()), None)\n"
            "if study_dir is None:\n"
            "    raise FileNotFoundError('Cannot find study; set VALIDATION_ARTICLE_DIR')\n"
            "sys.path.insert(0, str(study_dir))\n"
            "from analysis import Design, EXPERIMENTS, FIGURES, SEED, brier, run, stream\n"
            "print('Seed:', SEED, '| experiments:', len(EXPERIMENTS))\n",
            "imports",
        ),
        code(
            "assert len(EXPERIMENTS) == len(FIGURES) == 9\n"
            "labels = np.array([0, 1, 0, 1], dtype=np.int64)\n"
            "assert abs(brier(labels, np.full(4, .5))-.25) < 1e-12\n"
            "assert np.array_equal(stream(99, 'iid', 0).normal(size=6),\n"
            "                      stream(99, 'iid', 0).normal(size=6))\n"
            "print('Brier and random-stream identities passed')\n",
            "identities",
        ),
        markdown(
            "## Execute all nine small simulations\n\n"
            "This notebook uses three independent outer repetitions per "
            "experiment, compared with 20 to 70 for the archived full run. "
            "It writes a complete set of quick CSVs and figures in a temporary "
            "directory. Neither canonical evidence nor the random seed in "
            "the manuscript is altered. Error bars summarize Monte Carlo "
            "precision over repetitions, not the uncertainty of a real study.\n",
            "reproduce",
        ),
        code(
            "with tempfile.TemporaryDirectory() as directory:\n"
            "    destination = Path(directory)\n"
            "    reports = run(Design(root=destination, quick=True))\n"
            "    csv_files = sorted((destination/'results').glob('*_quick.csv'))\n"
            "    png_files = sorted((destination/'figures').glob('*_quick.png'))\n"
            "    assert len(csv_files) == 18 and len(png_files) == 9\n"
            "    print('Tables:', len(csv_files), 'Figures:', len(png_files))\n"
            "    for name, figure in zip(EXPERIMENTS, FIGURES):\n"
            "        print('\\n', name, '| summary cells:', len(reports[name]))\n"
            "        display(Image(filename=str(destination/'figures'/f'{figure}_quick.png')))\n",
            "execute_all_nine",
        ),
        markdown(
            "## Boundaries of the examples\n\n"
            "- A good iid CV estimate is conditional on a suitable data-generating "
            "and sampling process; it is not a transportability guarantee.\n"
            "- Nested CV evaluates selection with smaller outer training sets "
            "and does not certify every model's eventual deployment risk.\n"
            "- Feature selection on all labels and access to future features "
            "both leak information, through different mechanisms.\n"
            "- A random fold of history is not necessarily an honest sample "
            "of the next period or a new site.\n"
            "- Five folds are correlated through overlapping training sets; "
            "fold standard deviations are not independent replication errors.\n"
            "- Exact public score queries are a hypothetical, intentionally "
            "vulnerable oracle, not a procedure to apply to real benchmarks.\n\n"
            "When the kernel starts in an unrelated folder, define "
            "VALIDATION_ARTICLE_DIR as the absolute article path before "
            "starting the notebook. See the manuscript and study README "
            "for equations, sources, archival procedures, and limitations.\n",
            "limits",
        ),
    ]
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3",
                           "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    """Save an unexecuted notebook with stable cell IDs beside this script."""

    destination = ROOT/FILENAME
    destination.write_text(
        json.dumps(notebook(), indent=2, ensure_ascii=False)+"\n",
        encoding="utf-8",
    )
    print(destination.name)


if __name__ == "__main__":
    main()
