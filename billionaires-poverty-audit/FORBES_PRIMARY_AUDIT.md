# Forbes 2026 primary-country audit

The 54-country exploratory regressions are based on an incomplete secondary
transcription of billionaire counts attributed by **citizenship**. Before using
them for an external reply, we must inspect the source list and its country
definitions.

## What is verified

The earlier official-data capture and the 54-country citizenship regression
have passed GitHub Actions. The **Forbes source list itself has not yet been
independently reconciled** with the secondary transcription.

The March 2026 Forbes annual release reported **3,428 billionaires**. Its
person-list endpoint can change over time, and its schema is not a substitute
for a versioned statistical release.

## Primary Forbes checks

The parser in [forbes_primary.py](forbes_primary.py):

- Rejects invalid JSON, unsupported structures, duplicate person URIs and
  counts different from 3,428.
- Separates the explicitly named `countryOfCitizenship` field from the
  `country` field, whose residence semantics are **unverified**.
- Audits how many records are missing each geography field.
- Aggregates counts by exact country labels, with no invented mapping.
- Compares exact country names with the incomplete secondary table and
  emits unresolved names and differences.
- Retains the source URL, UTC retrieval time, exact raw-response byte count
  and SHA-256. **It does not retain identifiable person-level records.**

These are software validations, not a proof that the API has preserved
the original March 2026 list. A successful list-size check is necessary
but insufficient for a historically pinned Forbes edition.

## Reproduction

Run the deterministic offline checks:

    cd billionaires-poverty-audit
    python -m pip install -r requirements.txt
    python -m unittest discover -s tests -p 'test_forbes_primary.py' -v

To attempt the live audit, use the GitHub Actions manual workflow
**Forbes 2026 primary source audit**, or execute:

    python forbes_primary.py \
      --out-dir results/forbes_primary_2026 \
      --secondary data/forbes_2026_citizenship_secondary.csv

A remote access failure, source-schema mismatch, country-field missingness
or disagreement in the 3,428 figure **must be reported**. Do not suppress
the error, change the expected count to force success or treat countries
missing from one source as having zero billionaires.

## Publication boundary

The original chart uses an undisclosed 64-country selection. Even if the
country counts from this Forbes endpoint are recovered, this work does not
establish the chart's exact sample, the meaning of the API's `country` field,
wealth concentration, or any causal effect on poverty.

The next evidence tasks are a manually reviewed ISO3 mapping, comparison
against the original chart's country list if obtained, and updated sensitivity
models against **matched country-year welfare measures**.
