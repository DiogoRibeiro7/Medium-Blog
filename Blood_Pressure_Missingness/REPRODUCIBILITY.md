# Reproducibility environments

The project uses two dependency surfaces for different purposes.

## Supported development environment

Normal development and routine CI use `requirements.txt` through the package's `pyproject.toml`.

Those dependencies are expressed as supported version ranges. This is intentional: the maintained codebase should continue to work across compatible releases rather than depend on one historical package snapshot.

Use:

```bash
python -m pip install -e .
python -m unittest discover -s tests -p 'test_*.py' -v
```

from the `Blood_Pressure_Missingness` directory.

## Frozen publication environment

`requirements-publication-lock.txt` records the exact third-party package versions from a known-green publication reference run:

- GitHub Actions run: `#155`
- PR exact head: `d97efb455e24a7aa2701ab5d732a35844279f1c9`
- operating system: Ubuntu 24.04.5 LTS
- Python: 3.13.15
- full regression/notebook suite: 122 tests passed

This lock exists because numerical software can change at very small floating-point scales between compatible releases. The global time-form work exposed an HC3 interval difference of roughly `1.6e-9` across allowed `statsmodels` versions: scientifically irrelevant, but enough to demonstrate why an archival environment is useful.

To recreate the frozen third-party environment:

```bash
python -m pip install -r requirements-publication-lock.txt
python -m pip install -e . --no-deps
python -m unittest discover -s tests -p 'test_*.py' -v
```

The `--no-deps` step is important. It installs the local package without allowing its broader dependency ranges to replace the frozen versions.

## What the lock does not mean

The lock is not the supported dependency policy for future development. It is an archival reproduction target for the current publication snapshot.

A future refreshed dataset or materially revised article may justify a new publication lock. In that case, the old lock should remain recoverable from the corresponding Git tag or release rather than being treated as a timeless environment.

The raw private Google Sheet, credentials, row-level health measurements, and calendar-dated joins remain outside both environments. Reproducibility of the public statistical outputs uses the committed privacy-safe aggregate snapshot wherever possible.