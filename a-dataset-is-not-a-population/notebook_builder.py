"""Build the clean, executable Article #34 notebook from versioned source.

The notebook performs isolated quick replications; it cannot overwrite the
committed 300-repetition tables and figures.
"""

from __future__ import annotations

from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parent
NOTEBOOK = ROOT / "a_dataset_is_not_a_population.ipynb"


def build() -> nbformat.NotebookNode:
    """Create a deterministic source notebook without cached cell outputs."""
    cells = [
        nbformat.v4.new_markdown_cell(
            "# A Dataset Is Not a Population — Reproducible Experiments\n\n"
            "Article #34 companion. All nine studies are **synthetic**. This "
            "notebook executes three independent replications per study in a "
            "temporary directory. It never changes canonical publication CSVs "
            "or PNGs. The complete 300-repetition study uses `analysis.py`."
        ),
        nbformat.v4.new_code_cell(
            "from pathlib import Path\n"
            "import os\nimport sys\nimport tempfile\n\n"
            "candidates = []\n"
            "if os.environ.get('POPULATION_ARTICLE_DIR'):\n"
            "    candidates.append(Path(os.environ['POPULATION_ARTICLE_DIR']))\n"
            "candidates.extend([Path.cwd(), Path.cwd() / 'a-dataset-is-not-a-population'])\n"
            "article_dir = next((p.resolve() for p in candidates if (p / 'analysis.py').is_file() "
            "and (p / 'figures.py').is_file()), None)\n"
            "if article_dir is None:\n"
            "    raise FileNotFoundError('Set POPULATION_ARTICLE_DIR to the companion directory')\n"
            "sys.path.insert(0, str(article_dir))\n"
            "from analysis import SEED, STUDIES, METHODS, simulate, write_evidence\n"
            "from figures import render_all\n"
            "print(f'Synthetic studies: {len(STUDIES)}, canonical seed: {SEED}')"
        ),
        nbformat.v4.new_markdown_cell(
            "## Reproduce the study contracts\n\n"
            "The temporary run checks method counts, replication identities, "
            "CSV evidence and plot generation. It is a **smoke test**, not a "
            "substitute for the full archived Monte Carlo estimates."
        ),
        nbformat.v4.new_code_cell(
            "with tempfile.TemporaryDirectory(prefix='population-notebook-') as directory:\n"
            "    output = Path(directory)\n"
            "    for study in STUDIES:\n"
            "        rows = simulate(study, repetitions=3, seed=SEED)\n"
            "        assert len(rows) == 3 * len(METHODS[study])\n"
            "        write_evidence(rows, output / 'data', '_quick')\n"
            "    figures = render_all(output / 'data', output / 'figures', '_quick')\n"
            "    assert len(figures) == 9 and all(p.stat().st_size > 5000 for p in figures)\n"
            "    assert len(list((output / 'data').glob('*_quick.csv'))) == 18\n"
            "    print('9 studies, 18 isolated CSVs, 9 isolated PNGs: PASS')"
        ),
        nbformat.v4.new_markdown_cell(
            "## Interpretation and limits\n\n"
            "- The mean errors are conditional on specified synthetic mechanisms.\n"
            "- Inverse response probabilities using latent outcomes are an "
            "oracle comparison, not an actionable estimator from respondents alone.\n"
            "- Marginal calibration does not identify missing joint distributions.\n"
            "- Covariate weighting requires support and conditional invariance.\n"
            "- Composition updates cannot correct changing conditional outcomes.\n"
            "- Monte Carlo confidence intervals around simulated bias are not "
            "population confidence intervals."
        ),
    ]
    for index, cell in enumerate(cells):
        # nbformat generates random cell IDs by default. Stable IDs make
        # source-to-notebook comparisons reproducible across separate runs.
        cell.id = f"population-study-{index:02d}"
    notebook = nbformat.v4.new_notebook(cells=cells)
    notebook.metadata.kernelspec = {
        "display_name": "Python 3", "language": "python", "name": "python3"
    }
    notebook.metadata.language_info = {"name": "python"}
    return notebook


def main() -> None:
    """Write stable notebook JSON with no executed outputs."""
    nbformat.write(build(), NOTEBOOK)
    print(NOTEBOOK)


if __name__ == "__main__":
    main()
