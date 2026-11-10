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
