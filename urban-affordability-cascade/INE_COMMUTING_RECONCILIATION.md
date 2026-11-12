# INE 2021 commuting-source reconciliation: operator-controlled audit

**Status (10 November 2026): official indicator identified, but an independently
retrieved, validated official response has not been obtained.** No official
commuting matrix is added in this change; the 136 existing undirected pairs
remain labelled a third-party extraction attributed to INE.

## What is being tested

The existing 2021 municipal work/study network was reconstructed from a pinned,
checksummed third-party workbook. Its provenance links to
[INE indicator 0012340](https://www.ine.pt/xurl/indx/0012340/PT), described
as a Census 2021 municipal work/study interaction indicator. This is **not**
Census residential relocation or a directed origin–destination migration
matrix. The aim is to compare exactly the same 17 AMP municipalities and the
same symmetric counts with an *independently downloaded* official response.

A successful match is evidence about the **count reproduction**, not about
causal housing effects, or a certification of the source's classification.

## Obtain and inspect an official export

1. Open INE's [indicator page](https://www.ine.pt/xurl/indx/0012340/PT).
   Inspect metadata, territorial code vintage, all dimension names,
   measurement unit, suppression flags, and the 2021 reference period.
2. Confirm the API endpoint and parameters against the official metadata.
   The **candidate** URL below is not yet verified to return the assumed
   response schema. Do not assert a source reconciliation until it is checked.

   ```text
   https://www.ine.pt/ine/json_indicador/pindica.jsp?op=2&varcd=0012340&Dim1=S7A2021&lang=PT
   ```

3. Download the response yourself through an authorized workflow. Preserve
   **exact downloaded bytes** for SHA-256. This tool does not fetch from the
   internet. Do not put an unlicensed or personal-level export into the repo.
4. Make a CSV mapping of the official geographic dimension codes to the
   **fixed DICO** identities of all 17 AMP municipalities. It must have exactly
   the columns `source_code,dico`, no repeated source codes or DICO entries.
   These source codes are **not assumed identical** to DICO without checking.
5. Prepare a JSON receipt containing the following declarations and the SHA
   of the exact JSON bytes. These declarations are operator-supplied, not
   independent source authentication:

   ```json
   {
     "indicator_id": "0012340",
     "reference_period": "2021",
     "geography": "2021 Census, NUTS 2024 municipal geography",
     "primary_dimension_field": "geocod",
     "secondary_dimension_field": "dim_3",
     "count_field": "valor",
     "source_url": "https://www.ine.pt/ine/json_indicador/pindica.jsp?op=2&varcd=0012340&Dim1=S7A2021&lang=PT",
     "response_sha256": "<lowercase SHA-256 hex digest of exact response>",
     "downloaded_at_utc": "<ISO-8601 UTC timestamp>"
   }
   ```

   **The three named JSON data fields are an intake-schema hypothesis**, not
   a claim about the current INE API payload. If metadata disproves that
   schema, adapt the parser in a separately reviewed PR with fixtures from
   the documented official response. Do not fabricate a successful match.

6. Run from `urban-affordability-cascade/`:

   ```bash
   python ine_commuting_reconciliation.py \
     --source-json /secure/ine-0012340-2021.json \
     --receipt-json /secure/ine-0012340-receipt.json \
     --code-map-csv /secure/ine-geography-to-dico.csv \
     --output-json /tmp/porto-ine-commuting-reconciliation.json
   ```

## What must pass

The parser requires a single 2021 period, explicit declared dimensions and
values, a complete **17×17** local matrix (272 off-diagonal directed
*views* and 17 zero diagonals), integer/nonnegative nonsuppressed counts,
exact pair symmetry, one-to-one geography mapping, matching SHA-256
and complete 136 undirected frozen comparison pairs. Missing, suppressed,
duplicated, asymmetric or mismatched counts are rejected or reported explicitly.

For count discrepancies, the audit JSON records each mismatched unordered
municipal pair and the signed difference. The CLI returns a nonzero exit code
when at least one discrepancy is found. It never overwrites the frozen
analysis matrix, its provenance, or the housing article.

This workflow's automated tests use **explicitly synthetic INE-shaped JSON**
constructed from the *third-party* frozen matrix to exercise validation.
They do not verify the independent INE data. There is deliberately no
automated live data acquisition in GitHub Actions.

The core result remains **unverified against a fresh INE API extract**
until the full official response, territorial metadata, population universe,
measure definition, response checksum and pair-level comparison have all
been independently reviewed.

## Scientific boundary

A Census commuting network measures work/study relationships, not previous
residence, reasons for moving or direction of a residential displacement.
Even a perfect 136-pair match cannot validate the latter causal mechanism.
For directed relocation, see [OD_DATA_ACCESS.md](OD_DATA_ACCESS.md).

## Manually capture the official INE JSON before reconciliation

The [manual INE capture workflow](https://github.com/DiogoRibeiro7/Medium-Blog/blob/main/.github/workflows/ine-0012340-source-capture.yml)
requests indicator `0012340` metadata and a **candidate** 2021 data slice.
The candidate period selector `S7A2021` and payload schema are **unverified**
until the downloaded metadata and records are inspected.

Only `workflow_dispatch` can run the capture; no pull-request or push job
contacts INE. In GitHub, select **Actions → Capture INE Census 2021 commuting
source (manual) → Run workflow**. The capture runner calls
`ine_commuting_capture.py`, saving exact valid JSON response bytes, each
SHA-256 checksum, effective response URL, timestamp, schema hints and any
failure reason in `capture_manifest.json`. An unsuccessful source request
causes a failed job, while any other successful capture and the manifest are
still uploaded as a 30-day workflow artifact.

The script uses HTTPS GET to fixed `www.ine.pt` endpoints, rejects changed
redirects, enforces bounded response sizes and refuses malformed JSON. The
operator must inspect the actual period, dimensions, municipality codes,
measure, privacy flags and meaning. A captured JSON object does **not**
mean an independently verified municipal matrix. Every generated manifest
explicitly marks `semantics_verified=false` and
`frozen_pairs_reconciled=false`. If official dimensions differ from the
previously proposed intake format, update the parser only with reviewed
official examples, then use `ine_commuting_reconciliation.py` to compare all
136 undirected AMP pairs.

The new unit tests are offline and use only synthetic HTTP responses; they
cannot authenticate the source. The capture workflow does **not** overwrite
the pinned commuting dataset or its provenance. It is a first evidence-gathering
step toward that separate reconciliation.


## Offline integrity review of a captured workflow artifact

After running the manual capture workflow, download its
`ine-0012340-2021-unreconciled-source-capture` ZIP from GitHub Actions
and extract it to a local directory. Review the captured evidence without
reaching the network:

```bash
python ine_commuting_capture_review.py \
  --capture-dir /path/to/extracted/ine-0012340 \
  --output-json /tmp/ine-0012340-integrity-review.json
```

The reviewer checks that the declared response URLs still match the fixed
INE endpoints, the raw JSON bytes match their original SHA-256 checksums,
source sizes and structural descriptions are consistent, UTC capture
metadata is present, the expected two entries are accounted for, and
failed downloads are never mistaken for successful source files. It
reports missing or tampered inputs rather than repairing them.

The manual GitHub workflow also runs this review immediately after each
capture attempt and includes `integrity_review.json` in its artifact.
An incomplete capture retains explicit missing-source status. Automated
unit tests use synthetic fixtures only; they never contact INE.

**Integrity is not authenticity or statistical validity.** The report
sets `source_semantics_verified=false`,
`INE_publisher_authenticated_independently=false` and
`frozen_pairs_reconciled=false`. Matching the frozen 136-pair commuting
matrix remains a later step requiring official indicator metadata and
a reviewed geographic-code mapping. The current pinned matrix must not
be upgraded merely because this integrity review succeeds.
