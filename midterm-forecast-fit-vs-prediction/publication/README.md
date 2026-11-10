# Scientific publication assets

The full manuscript for *The Forecast Is Not the Fit: Rebuilding an 89% Midterm
Election Model* and its Medium-ready edition are maintained in the private
[research-articles-medium repository](https://github.com/DiogoRibeiro7/research-articles-medium/tree/main/series/midterm-2026-forecast).
The editorial copies were verified before public [PR #563](https://github.com/DiogoRibeiro7/Medium-Blog/pull/563)
replaced them with short archive notices.

This public companion retains:

- The thirteen original [research figures](../figures/), four table-image figures,
  and fifteen [equation PNGs](equations/).
- [Historical sources and provenance](../data/PROVENANCE.md), reproducible
  [analysis code](../analysis.py), and the frozen
  [pre-election registration](../PRE_ELECTION_REGISTRATION.md).
- [Numerical evidence](../results/summary.json) and regression tests that
  validate the frozen model independently of private editorial text.

## Public checks

From the `midterm-forecast-fit-vs-prediction/` directory:

```bash
python prepare_medium_publication.py --check-scientific
python -m unittest discover -s tests -v
```

The first command checks the signatures and minimum sizes of all seventeen
figures and fifteen equation images, without needing a private checkout,
network credentials, or LaTeX. CI also independently regenerates the model,
source-sensitivity tables, figures, and registered forecast lock.

## Private editorial rendering

The historical renderer remains available for an authorized private source.
Explicitly provide **both** paths, outside the public research checkout:

```bash
python prepare_medium_publication.py --check \
  --source /path/to/private/midterm-forecast-fit-vs-prediction/article.md \
  --output /path/to/private/midterm-forecast-fit-vs-prediction/publication/medium_ready.md
```

For an intentional editorial rebuild, use `--build` with the same arguments.
Building equation PNGs requires `pdflatex` (standalone, amsmath, amssymb) and
`pdftoppm`. The builder can regenerate the public equation illustrations,
but only writes the editorial Markdown to the explicit private output path;
it refuses to overwrite any public compatibility notice.

These are *pre-election* artefacts with an information cutoff of 31 October
2026 and registration on 1 November 2026. Subsequent election scoring must
remain a separate retrospective and never rewrite the forecast lock.
