# Publication snapshot metadata

`Blood_Pressure_Missingness` is a maintained case study inside the multi-project `Medium-Blog` repository. It therefore does **not** have an independent GitHub release lifecycle.

The project instead maintains a reproducible publication snapshot: a coherent set of privacy-safe public artifacts, citation metadata, an executable notebook, and a frozen archival environment that can be checked against the current source tree.

## Current snapshot version

The package metadata in `pyproject.toml` currently declares version `0.1.0`. The scoped `CITATION.cff` uses the same internal case-study version.

That version identifies the state of this maintained case study. It does not imply a repository-wide release, Git tag, GitHub Release, journal publication, or DOI.

## Integrity model

`PUBLICATION_MANIFEST.json` records SHA-256 hashes and byte sizes for the privacy-safe publication artifacts. Regenerate it with:

```bash
python -m blood_pressure_missingness.publication_manifest \
  --root . \
  --output PUBLICATION_MANIFEST.json
```

Validate the complete snapshot with:

```bash
python -m blood_pressure_missingness.publication_integrity --root .
```

The integrity gate checks:

- package and citation version consistency;
- scoped citation URL and Apache-2.0 licence metadata;
- exact publication-manifest agreement with current public artifacts;
- strict pinning of the archival publication environment.

## Snapshot maintenance

When the privacy-safe dataset or derived publication artifacts materially change:

1. regenerate the public outputs;
2. regenerate `PUBLICATION_MANIFEST.json` in the same refresh transaction;
3. run the publication-integrity gate;
4. run the ordinary compatibility CI job;
5. run the frozen `publication-lock` CI job;
6. review snapshot-sensitive narrative and methods text.

The repository commit remains the authoritative source identity. No subdirectory-specific GitHub release or tag is required.

## Archival metadata

Do not invent a DOI, release date, journal record, or standalone repository identity for this subproject. If the analysis is later published through an external venue, record that real publication metadata when it exists and keep it clearly distinct from the repository's internal project version.

## Privacy boundary

The publication snapshot contains only privacy-safe aggregate data and derived artifacts. Raw health rows, Google Sheet identifiers, credentials, calendar-dated private joins, and other secret-backed source material remain outside the public snapshot.
