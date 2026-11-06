# The Iberian electricity price as a linear programme

Is the Iberian day-ahead electricity price the dual value of a linear programme built from
OMIE's public order books? How far do the market's non-convex orders move it, and how
much would 1,000 MW more supply at zero price lower it? A registered study of one year of
15-minute periods.

**Status: plan registered ([METHODS.md](METHODS.md)); data built
([docs/DATA.md](docs/DATA.md)); programmes solved and estimates reported against the plan
([docs/RESULTS.md](docs/RESULTS.md)); reproducible figures and values generated
([publication/values.json](publication/values.json)); the editorial manuscript archived
privately ([public notice](publication/article.md)); a critical review
([docs/REVIEW.md](docs/REVIEW.md)), and the corrections it led to, logged in the plan's
change log.**

## Running it

```bash
pip install -e ".[dev]"
python -m pytest -q          # offline, from the committed tables and results
iberian-lp programmes        # about 17 minutes; writes results/programmes.csv
iberian-lp estimates         # writes results/estimates.json and docs/RESULTS.md
iberian-lp publication       # regenerates publication/values.json and figures/
```

The complete Medium manuscript and its parameterized authoring template are maintained in
the private [research-articles-medium](https://github.com/DiogoRibeiro7/research-articles-medium)
repository. The [public notice](publication/article.md) preserves historical links.
The public publication command regenerates only numerical values and research figures; it
never rewrites the private editorial text. Tests still check the underlying numerical claims,
source provenance, generated values and figure reproducibility.

Each programme's dual interval is computed exactly from the integer steps and checked
against HiGHS. The tests re-solve every programme exactly from the committed steps, and
check one day against HiGHS.

Rebuilding the tables downloads 144 files (about 950 MB) from omie.es into `data/raw/`,
which is not committed:

```bash
iberian-lp download          # the 48 drawn days, one request a second
iberian-lp data              # about 4 minutes; writes data/derived/ and docs/DATA.md
```

## Data and source

Data: OMIE (OMI-Polo Español, S.A.), [omie.es](https://www.omie.es). OMIE's public
information may be used freely, keeping its original content intact and citing the
source. The raw files are not committed; their names, sizes and SHA-256 hashes are in
`data/derived/manifest.json`. The committed tables in `data/derived/` are derived from
them: the offered steps summed by price, and one row per zone-period.
