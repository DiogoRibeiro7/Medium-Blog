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

## Upstream 503 and publication status (5 November 2026)

GitHub Actions run 42053766035 failed in the live source job after two requests
to the Forbes person-list API. The upstream response was HTTP 503,
`first byte timeout`. The parser's offline tests and repository hygiene
passed. No primary Forbes country figures were recovered by that run.

The parser now supports a bounded **page-by-page fallback** if the full feed
temporarily fails. Every page must have its requested number of records and
the assembled annual list must contain the expected number of unique
identifiers. Original response-page SHA-256 digests, offsets and request URLs
are recorded. A combined in-memory payload is explicitly labelled **derived**,
not represented as untouched primary-source bytes.

The live Forbes job is intentionally **manual** (workflow_dispatch). It cannot
be a required CI condition on every PR while Forbes responds unreliably.
Offline contract tests remain required on PRs. A failed manual audit is a
real unresolved source-access blocker, not a reason to replace official
counts with inferred or fabricated data.

A successfully retrieved API list still needs historical-vintage reconciliation
against Forbes' published March 2026 annual release. The `country` field is
not equated with residence without an explicit source definition. No
publication figures or conclusions are approved by this change.

## Country-code reconciliation gate

After a **successful live Forbes retrieval**, the manual workflow now runs
`forbes_reconciliation.py` against the aggregate `counts_citizenship.csv` and
the incomplete secondary table. The Forbes auditor records SHA-256 hashes for
all its generated CSVs, and reconciliation refuses a country-count file that
differs from the one covered by `audit.json`. The reconciliation also refuses
test fixtures, an incorrect annual-list count, inconsistent citizenship
coverage, duplicate labels and attempts to substitute the ambiguous Forbes
API `country` field for `countryOfCitizenship`.

The output is a **review queue**, not a validated Forbes-to-ISO3 crosswalk:

| Output | Meaning |
| --- | --- |
| `reconciliation/exact_label_candidates.csv` | Exact label matches to the secondary transcription, with provisional ISO3 codes and count differences |
| `reconciliation/unmapped_primary_labels.csv` | Primary citizenship labels without an exact secondary match, for manual review |
| `reconciliation/secondary_without_exact_primary_label.csv` | Secondary labels absent as exact primary labels; **not** verified zero counts |
| `reconciliation/summary.json` | Source and aggregate digests, coverage, mapping-review counts and explicit publication blockers |

This step performs **no fuzzy country-name matching**. Exact spelling is not
independent confirmation of the country code, territorial definition or
citizenship assignment. Before accepting a crosswalk, inspect each source
label, territory and contested geography manually; retain the raw label,
selected ISO3 and documented evidence for each decision. Missing primary
labels must remain missing, never be imputed as zero.

For offline verification, run:

```bash
python -m unittest discover -s tests -p 'test_forbes*.py' -v
```

The manual source workflow remains necessary because the previously observed
HTTP 503 has not been resolved. Even a completed ISO3 review will **not**
establish a historical March 2026 source vintage, residence rather than
citizenship, the original chart's 64-country selection or a causal effect.

## Follow-up: live response schema (5 November 2026)

[Workflow run 42097705351](https://github.com/DiogoRibeiro7/Medium-Blog/actions/runs/42097705351)
passed its offline tests but failed during primary-source retrieval. Its
first request received HTTP 503; the second downloaded **6,502,586 bytes**
of JSON successfully. The parser then rejected the response because it could
not find the expected `personList.persons` array. This was a **schema** error
after network access succeeded, not evidence that Forbes returned an
incomplete person count. No country-level evidence was archived by that run.

A [public Forbes API client](https://github.com/TheAlgorithms/Python/blob/master/web_programming/get_top_billionaires.py)
uses the plural
`personList.personsLists`. The auditor and pagination fallback now recognize
that explicit path alongside `personList.persons` and top-level `persons`.
Ambiguous or unknown paths still fail with **key names only**, without
logging person-level data. The required **3,428** person count, URI
uniqueness, field typing and at least 95% citizenship-field coverage have
not been loosened.

The exact live response shape has **not yet been independently captured** in
the audit's aggregate-only results. Re-run the manual job after merging this
schema correction; if it still fails, inspect the safe key-name diagnostic
before considering another supported layout. No Forbes counts or new
regression conclusions are confirmed by this parser change alone.
