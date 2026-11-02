"""Build the article #32 Jupyter companion deterministically from source."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
FILENAME = "a_label_is_not_ground_truth.ipynb"


def markdown(source: str, identifier: str) -> dict[str, Any]:
    """One stable-ID notebook Markdown cell."""

    return {"cell_type": "markdown", "id": identifier, "metadata": {},
            "source": source.splitlines(keepends=True)}


def code(source: str, identifier: str) -> dict[str, Any]:
    """One notebook code cell without cached output or execution count."""

    return {"cell_type": "code", "id": identifier, "metadata": {},
            "source": source.splitlines(keepends=True),
            "execution_count": None, "outputs": []}


def notebook() -> dict[str, Any]:
    """Nine reproductions with explicit observed/latent label distinctions."""

    cells = [
        markdown(
            "# A Label Is Not Ground Truth\n\n"
            "Article #32: nine synthetic experiments, nine original figures, "
            "eighteen canonical evidence CSVs. A latent outcome, an annotator "
            "vote, a policy-observed label, and a model prediction are *not* "
            "interchangeable variables.\n",
            "intro",
        ),
        markdown(
            "## The label is a measurement channel\n\n"
            "For class-conditional flip rates eta0=P(tildeY=1|Y=0) and "
            "eta1=P(tildeY=0|Y=1), the recorded positive fraction q obeys "
            "q=eta0+(1-eta0-eta1)*pi. An inversion requires **known or "
            "identified** rates and eta0+eta1<1. The synthetic generator "
            "supplies these rates; real annotation records usually do not.\n\n"
            "Dawid–Skene EM assumes worker votes are independent *given* "
            "the latent class. Correlated workers can agree confidently "
            "on a shared mistake.\n",
            "theory",
        ),
        code(
            "from pathlib import Path\n"
            "import tempfile\n"
            "import numpy as np\n"
            "from IPython.display import Image, display\n"
            "from analysis import (Design, EXPERIMENTS, FIGURES, SEED, "
            "invert_noisy_prevalence, dawid_skene_binary, run)\n"
            "print('Seed', SEED, '| studies', len(EXPERIMENTS))\n",
            "imports",
        ),
        code(
            "assert abs(invert_noisy_prevalence(.2125, .05, .30)-.25) < 1e-12\n"
            "votes = np.tile([1,1,1,0,0], (40,1)).astype(int)\n"
            "posterior = dawid_skene_binary(votes)\n"
            "assert posterior.shape == (40,) and np.isfinite(posterior).all()\n"
            "print('Class-conditional inversion and EM numerical checks passed')\n",
            "identities",
        ),
        markdown(
            "## Execute independent quick studies without overwriting evidence\n\n"
            "This notebook uses three independent outer repetitions per "
            "study. Canonical full mode uses 20–36. Quick outputs are "
            "created in a temporary directory with `_quick` suffixes; "
            "the committed full-mode CSV and PNG evidence is untouched. "
            "Plot bars represent Monte Carlo precision over independent "
            "outer repetitions, not observational confidence intervals.\n",
            "execution",
        ),
        code(
            "with tempfile.TemporaryDirectory() as directory:\n"
            "    destination = Path(directory)\n"
            "    summaries = run(Design(root=destination, quick=True))\n"
            "    tables = sorted((destination/'results').glob('*_quick.csv'))\n"
            "    figures = sorted((destination/'figures').glob('*_quick.png'))\n"
            "    assert len(tables) == 18 and len(figures) == 9\n"
            "    print('Tables:', len(tables), 'Figures:', len(figures))\n"
            "    for study, figure in zip(EXPERIMENTS, FIGURES):\n"
            "        print('\\n', study, '| summary cells:', len(summaries[study]))\n"
            "        display(Image(filename=str(destination/'figures'/f'{figure}_quick.png')))\n",
            "execute_all_nine",
        ),
        markdown(
            "## Interpret carefully\n\n"
            "- Synthetic latent truth is available **only** because the "
            "generator created it. Label correction rates are not known "
            "just because a classifier assigns confident scores.\n"
            "- The class-conditional correction is oracle, not a free "
            "identification method.\n"
            "- Dawid–Skene here has complete votes, binary classes, "
            "pseudo-count regularization and an orientation anchored by "
            "initial majority vote. Shared worker errors break its model.\n"
            "- Randomized audit sampling protects against selective labels; "
            "an audit restricted to flagged cases cannot identify global "
            "prevalence.\n"
            "- A test label constructed in favor of model B is deliberate "
            "evaluation bias, not evidence that B generalized better.\n"
            "- Row-level splits can reuse the same entity in train and test; "
            "the appropriate unit of independence depends on deployment.\n\n"
            "Read `article.md` for derivations, sources, limitations, and "
            "a practical labeling-governance protocol.\n",
            "boundaries",
        ),
    ]
    return {"cells": cells,
            "metadata": {"kernelspec": {"display_name": "Python 3",
                                       "language": "python", "name": "python3"},
                         "language_info": {"name": "python", "version": "3.12"}},
            "nbformat": 4, "nbformat_minor": 5}


def main() -> None:
    """Write a clean nbformat v4 notebook beside the builder."""

    destination = ROOT/FILENAME
    destination.write_text(json.dumps(notebook(), indent=2, ensure_ascii=False)+"\n",
                           encoding="utf-8")
    print(destination.name)


if __name__ == "__main__":
    main()
