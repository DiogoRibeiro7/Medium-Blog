# INE 2021 commuting-source reconciliation: operator-controlled audit

**Status (13 November 2026): independently captured official Arouca, Espinho
and Santa Maria da Feira slices have been preserved and checked against
45 of the 136 frozen pairs.** All 48 observed off-diagonal source views
match the pinned third-party counts; the three reciprocal links among these
municipalities agree in both directions. The remaining **91 pairs** and
full-matrix symmetry are **not yet verified** against
official data. No official full commuting matrix is added; the 136 existing
undirected pairs remain labelled a third-party extraction attributed to INE.

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

The original **full-matrix** parser tests use explicitly synthetic
INE-shaped JSON constructed from the third-party frozen matrix; they do
not authenticate independent INE data. A separate small PR-triggered
source probe now exists, with an archived first real response and its
own offline regression tests described below.

The **complete 136-pair result** remains unverified against the official API
until the remaining municipality slices, reverse views, territorial metadata,
population universe and source definitions have all been independently reviewed.

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

Only `workflow_dispatch` can run **this manual capture workflow**; a separate
bounded PR-triggered probe is documented below. In GitHub, select
**Actions → Capture INE Census 2021 commuting
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


## Capturing the 2021 indicator in small municipality slices

The one-shot PR #625 capture on **12 November 2026** obtained valid metadata
from INE indicator `0012340` but the candidate all-municipality 2021
request timed out. The archived official metadata identifies **three**
dimensions: period (2021, `S7A2021`), geographic dimension 2 (residence
or destination), and geographic dimension 3 (destination or residence).
All **17 AMP DICO municipality codes** are present in both dimensions
and agree with the fixed municipal panel.

The [manual partition workflow](../../.github/workflows/ine-0012340-partitioned-capture.yml)
attempts a smaller candidate API request with `Dim2=<municipality DICO>`
or `Dim3=<municipality DICO>`. A real Arouca response now supports the
effectiveness of **`Dim2=0104` for that one slice**; other values and the
`Dim3` direction remain hypotheses until separately checked. Select
**Actions → INE 0012340 municipality-slice capture (manual) → Run workflow**.
Specify `start_index` (zero-based among sorted DICO codes), `count` (1–3),
and `dimension` (`Dim2` or `Dim3`). Start with 0 and count 2.
For each requested code, the runner saves raw JSON bytes if returned,
SHA-256, effective URL, observed period-key hints and errors in
`partition_manifest.json`; the artifact is retained for 30 days.

The official metadata is fetched and checked against the 17 expected
municipality names and codes. Its exact SHA-256 is recorded and compared
with the archived November 12 capture, **but a mismatch alone is not a
failure**, because INE includes a varying extraction timestamp. A changed
indicator, period, measurement unit, dimension or municipality identity
does fail validation. The candidate geographic-slice responses are
never silently combined, assigned an official origin/destination
direction, or used to replace the existing commuting matrix.

Only after one or more real data slices have been inspected and their
filter behavior established should the study proceed to the full
136-pair reconciliation. A successful transfer remains
**unverified source-shape evidence, not residential mobility or
causal displacement evidence**.

## Archived official Arouca slice and reproducible partial reconciliation

On **13 November 2026**, the bounded PR-triggered source probe in
[PR #655](https://github.com/DiogoRibeiro7/Medium-Blog/pull/655)
successfully retrieved an official `Dim2=0104` JSON response for Arouca;
the next selected municipality, Espinho (`0107`), timed out. The workflow
run produced a 30-day artifact, but the exact successful bytes and original
provenance are now retained in Git:

- [`data/ine_0012340_2021_dim2_0104_raw.json`](data/ine_0012340_2021_dim2_0104_raw.json):
  57,793 bytes, SHA-256
  `abbe42d2e98dd1522609da8d058663430f9b42fee99591634371379446c22635`.
- [`data/ine_0012340_2021_dim2_first_two_manifest.json`](data/ine_0012340_2021_dim2_first_two_manifest.json):
  the original manifest, including the failed Espinho request, source URL,
  timestamp, raw response digest and separately checked official metadata.
- [`data/ine_0012340_2021_dim2_0104_audit.json`](data/ine_0012340_2021_dim2_0104_audit.json):
  the deterministic, reviewed one-row numerical comparison; regenerate it
  with the command below.

The Arouca response contains **347 distinct `dim_3` codes**, including
geographies outside the AMP municipality set and higher-level aggregates.
It reports `geocod=0104` on **every** record, a single `2021` period and
an explicit successful API status. The audit uses **only the 17 exact AMP
DICO municipality counterparts**. It requires the Arouca diagonal to be
zero and matches each of the other **16 unique** interaction counts to the
checksummed frozen third-party CSV: **16 matches, zero differences**.
It never sums the 347 records as if all were municipalities.

To regenerate the report offline, from `urban-affordability-cascade/`:

```bash
python ine_commuting_partition_reconciliation.py \
  --source-json data/ine_0012340_2021_dim2_0104_raw.json \
  --manifest-json data/ine_0012340_2021_dim2_first_two_manifest.json \
  --municipality-code 0104 \
  --output-json /tmp/ine-0012340-arouca-partial-audit.json
```

The script rejects source checksum/size mismatches, changed API URLs,
incorrect indicator or period, failed filter behavior, duplicate or missing
AMP codes, invalid integer counts, inconsistent municipality names and a
nonzero diagonal. It reports any numerical disagreement without changing
research inputs. Offline tests pin the captured raw hash and independently
check all 16 pair values; mutation tests demonstrate the rejection paths.

The PR source probe also distinguishes **external INE availability** from
validation failures. A timeout or transient HTTP 429/5xx response is
recorded in the uploaded manifest, the job summary and a workflow warning;
it does not invalidate otherwise passing code checks. A missing/corrupt
manifest, changed municipality scope or non-network capture failure still
fails the job. A warning does **not** mean that a missing slice was obtained.

**Scope at this first capture:** it established one-row agreement, contingent
on the recorded GitHub Actions capture and official source metadata. The
original manifest's `captured_unverified` status remains historically accurate;
the separate audit is what establishes numerical reconciliation.

### Second successful probe: Arouca and Espinho

The subsequent GitHub Actions run
[#43404869490](https://github.com/DiogoRibeiro7/Medium-Blog/actions/runs/43404869490)
returned two successful 2021 `Dim2` municipality slices. Unlike the prior
attempt, neither request timed out. Its **complete artifact** (both exact
raw source files and its original manifest) is preserved under
[`data/ine_0012340_pr_run_43404869490/`](data/ine_0012340_pr_run_43404869490/):

| Raw response | Exact SHA-256 | AMP comparisons |
| --- | --- | --- |
| `ine_0012340_dim2_0104_candidate.json` (Arouca) | `4c209febabc788d01a64121121dea2eff27b792ba4c1fce527d01a471de46de7` | 16/16 match |
| `ine_0012340_dim2_0107_candidate.json` (Espinho) | `69d06467a55030f1ccd681a055c6ef2aaa68c7f82c3bca32201b63879244e37b` | 16/16 match |

Each response includes 347 unique `dim_3` categories and all 17 AMP
municipality identities. Both report only their requested `geocod` as the
first geography, a single 2021 period, zero municipality diagonals and
successful API status. The same offline parser verified each source against
the capture manifest and third-party CSV; its two machine-readable reports
are `audit_0104.json` and `audit_0107.json` in that directory. For example:

```bash
python ine_commuting_partition_reconciliation.py \
  --source-json data/ine_0012340_pr_run_43404869490/ine_0012340_dim2_0107_candidate.json \
  --manifest-json data/ine_0012340_pr_run_43404869490/partition_manifest.json \
  --municipality-code 0107 \
  --output-json /tmp/ine-0012340-espinho-partial-audit.json
```

The two lists of 16 undirected pairs overlap only at Arouca–Espinho:
both source directions report **34** interactions. Therefore the combined
source coverage is **31 distinct pairs out of 136**, not 32. The unobserved
105 pairs must not be inferred from the existing third-party workbook, and
symmetry outside the shared Arouca–Espinho link is untested. Pinned offline
regression tests verify both SHA-256 digests, all 32 source-to-workbook
comparisons, both reports and the single cross-source symmetry check.

Neither raw-byte integrity nor agreement with the third-party workbook
independently authenticates the publisher. The full 136-pair source matrix
is still unverified, and the observations are **2021 work/study interactions,
not migration or causal housing spillovers**.

### Third successful probe: Santa Maria da Feira

The next bounded PR run
[#43412012492](https://github.com/DiogoRibeiro7/Medium-Blog/actions/runs/43412012492)
selected exactly `start_index=2`, `count=1`, `dimension=Dim2`, corresponding
to DICO **0109 (Santa Maria da Feira)**. The official 2021 response succeeded.
Its exact 62,767-byte JSON, original manifest and deterministic offline audit
are archived together in
[`data/ine_0012340_pr_run_43412012492/`](data/ine_0012340_pr_run_43412012492/).
The source SHA-256 is
`f2062a66f2a3bb9d3f5217c13ecde623c1a76f506374c2516a8b5034a4fd95df`.

The response has 347 distinct `dim_3` codes, a single `2021` period,
`geocod=0109` throughout, official API success, and all 17 AMP counterparts
including a zero diagonal. The separate offline numerical audit matches
**all 16** nonself AMP interactions against the checksummed frozen CSV, with
no differences. The two previously archived reciprocal slices agree:

| Pair | Arouca / Espinho source | Santa Maria da Feira source |
| --- | ---: | ---: |
| Arouca–Santa Maria da Feira | 826 | 826 |
| Espinho–Santa Maria da Feira | 3,910 | 3,910 |

Together with the previously matched Arouca–Espinho count of 34, the three
reciprocal links are symmetric. Three 16-pair slices yield
`3 × 16 − 3 = 45` distinct unordered AMP pairs; **91 of 136 remain**.
This does not establish symmetry of any unobserved link.

Reproduce the third comparison offline from `urban-affordability-cascade/`:

```bash
python ine_commuting_partition_reconciliation.py \
  --source-json data/ine_0012340_pr_run_43412012492/ine_0012340_dim2_0109_candidate.json \
  --manifest-json data/ine_0012340_pr_run_43412012492/partition_manifest.json \
  --municipality-code 0109 \
  --output-json /tmp/ine-0012340-feira-partial-audit.json
```

The committed `audit_0109.json` is checked for exact deterministic equality
in offline tests, alongside the official response hash, all 16 counts,
the two reciprocal links and total unique-pair coverage. As before, this
is **source agreement for 2021 work/study interactions**, not residential
mobility, commuting direction or a causal housing-spillover estimate.
