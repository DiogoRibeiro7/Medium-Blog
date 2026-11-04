# Public source capture and coverage audit

## Why this exists

The article's original 64-country scatterplot and the earlier reconstruction
are **not** interchangeable. Before publishing any regression coefficients,
we must record the exact development and poverty files used to derive them.

This PR introduces an auditable first stage. It deliberately **does not**
claim that the full original dataset or Forbes residence counts are recovered.

## Reproduce the public-source snapshot

On GitHub, open **Actions → Capture public billionaire-poverty source
snapshots → Run workflow** on the branch containing this pipeline.

The manual workflow:

1. Installs the isolated research dependencies and runs offline tests.
2. Downloads an archived Our World in Data / World Bank PIP $30-per-day
   poverty extract, plus World Bank GDP per capita PPP and population.
3. Computes SHA-256 checksums and byte sizes for all three exact sources.
4. Rejects missing, unexpectedly short, mutated or inconsistent sources.
5. Records coverage, unmatched country codes, and *observation-year* spans.
6. Uploads the input snapshots, source manifest and coverage report as a
   GitHub Actions artifact with 30-day retention.

Download and preserve that artifact separately for longer-term reproducibility.
**Do not** assume that a 30-day Actions artifact is a permanent scientific
archive, or that retrieving a World Bank API document freezes future revisions.

From a compatible local environment, the same commands are:

    cd billionaires-poverty-audit
    python -m pip install -r requirements.txt
    python -m unittest discover -s tests -p 'test_*.py' -v
    python source_pipeline.py fetch --output-dir data/source_snapshots
    python source_preflight.py \
      --snapshot-dir data/source_snapshots \
      --output results/public_source_coverage.json

## Semantics and safeguards

The source coverage report describes only the **intersection of available
OWID/WDI country codes in the requested 2020–2025 window**. No observations
are constructed, averaged, interpolated or converted to zero.

The OWID *Year* field is not guaranteed to be the interview date of the
underlying income or consumption survey. A separate PIP survey-vintage
extract and survey-type classification are required before the research
can assert that all observed inputs represent comparable periods.

The report does not contain Forbes counts. No residence/citizenship attribution
is guessed, and missing Forbes countries are not labelled as countries with
zero billionaires.

## Release checklist

- [ ] Manually execute the capture workflow and verify its success.
- [ ] Download and inspect the source artifact, country coverage and SHA-256
      manifest; attach permanent archive metadata if publishing.
- [ ] Verify and archive Forbes 2026 aggregate counts by *residence* and
      *citizenship* separately, without redistributing person records.
- [ ] Recover PIP survey interview years, welfare aggregates and PPP vintage.
- [ ] Construct both comparable country samples; reconcile the original
      plotted 64-country sample and its country-selection rules.
- [ ] Run the regression scripts on the pinned joined observations and
      compare every result against the provisional text.
- [ ] Produce the eight publication figures and the Medium figure ledger.
- [ ] Update the article, claim audit and quantitative conclusions only
      after the empirical figures have been independently reproduced.

The Python test fixtures are synthetic and test the **software contracts**,
not the truth of any country-level statistic.

## Download-timeout recovery (4 November 2026)

The second live-source workflow attempt passed all 37 offline tests but
failed with `TimeoutError: The read operation timed out` in the public
download step. This is **not** evidence that the source content or regression
was invalid. The downloader now records the exact source URL and filename
before each request, allows a 120-second socket timeout, and makes at most
two attempts with a bounded delay for timeouts, transient HTTP 5xx/429/408,
and incomplete HTTP reads. It never retries a hard HTTP 404.

Every accepted download must contain at least 100 bytes, must not be an HTML
error page, and must match any `Content-Length` response header. The SHA-256
manifest still uses the exact returned file bytes. A later revision of a live
OWID/WDI response will therefore yield a different checksum.

**These safeguards improve diagnostics and failure handling; they do not
guarantee that public data endpoints are available.** If the live capture
fails again, inspect the newly identified filename and URL in the CI log,
rather than changing the article's empirical figures or using synthetic
country observations.

