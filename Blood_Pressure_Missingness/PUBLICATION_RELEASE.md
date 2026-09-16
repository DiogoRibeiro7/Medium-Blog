# Publication and release metadata

This directory is one case study inside the multi-project `Medium-Blog` repository. Publication metadata is therefore scoped to `Blood_Pressure_Missingness/` rather than placed at repository root.

## Current version

The package metadata in `pyproject.toml` currently declares version `0.1.0`.

The scoped `CITATION.cff` uses the same version. Until an actual archival release is created, it intentionally does **not** contain a DOI or release date.

## Release tag convention

Use a namespaced tag so releases from this case study cannot be confused with unrelated projects in the same repository:

```text
blood-pressure-missingness-v0.1.0
```

Future versions should follow the same pattern:

```text
blood-pressure-missingness-vMAJOR.MINOR.PATCH
```

## Executable release-readiness gate

Before considering a commit for tagging, run from this directory:

```bash
python -m blood_pressure_missingness.release_readiness --root .
```

The gate checks the tree itself. It requires:

- `pyproject.toml`, `CITATION.cff`, and the namespaced release tag to agree on the version;
- `CITATION.cff` to remain scoped to this case study and declare `Apache-2.0`;
- `RELEASE_MANIFEST.json` to match the current integrity-tracked public artifacts exactly;
- every archival dependency in `requirements-publication-lock.txt` to be pinned with `==`;
- no DOI or `date-released` field before those values exist in reality.

After an actual archival release exists, the DOI/date guard can be intentionally relaxed with:

```bash
python -m blood_pressure_missingness.release_readiness \
  --root . \
  --allow-archival-metadata
```

That flag does not create or validate a DOI. It only permits real archival metadata to be present after it has been added deliberately.

## Before creating a release

1. Run the executable release-readiness gate above.
2. Confirm the ordinary compatibility CI job is green.
3. Confirm the frozen `publication-lock` CI job is green.
4. Confirm the public notebook executes from the project directory.
5. Confirm public JSON artifacts reproduce from the committed privacy-safe snapshot.
6. Confirm `current_narrative` passes against the current aggregate snapshot.
7. Confirm no raw workbook, Google Sheet identifier, credentials, calendar-dated joins, or row-level health data are present.
8. Review the README and methods notes for snapshot-sensitive numbers.
9. Create the namespaced Git tag and GitHub release from the reviewed `main` commit.
10. Only after a real release exists, add the actual release date to `CITATION.cff` if desired.
11. Only after an external archival service has minted a DOI, add that real DOI to `CITATION.cff`.

## What not to invent

Do not pre-populate:

- a DOI that has not been minted;
- a release date before the release exists;
- a journal or article publication record that does not exist;
- a repository-wide version that implies unrelated projects share this case study's lifecycle.

The Git commit remains the authoritative identity before a tagged release exists.

## Licence

The repository is licensed under Apache License 2.0. The scoped citation metadata records `Apache-2.0` consistently with the root `LICENSE` file.
