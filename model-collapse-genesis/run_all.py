"""Reproduce every number and figure in the article.

    python run_all.py            # downloads the corpus into ./data on first run

Takes a few minutes on a laptop. Outputs go to results/ and figures/.
"""

from __future__ import annotations

import contextlib
import io
from pathlib import Path

import experiment
import figures
import samples
import summarize
import theory

DATA, RESULTS, FIGS = Path("data"), Path("results"), Path("figures")

if __name__ == "__main__":
    theory.main(DATA, RESULTS)
    experiment.main(DATA, RESULTS, T=50, seeds=10)
    figures.main(RESULTS, FIGS)
    samples.main(DATA, RESULTS / "samples.txt")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        summarize.main(DATA, RESULTS)
    (RESULTS / "summary.txt").write_text(buf.getvalue())
    print(buf.getvalue())
