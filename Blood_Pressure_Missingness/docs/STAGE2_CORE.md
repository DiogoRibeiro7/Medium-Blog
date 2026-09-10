# Stage 2: core implementation ownership

The canonical module `blood_pressure_missingness.public_analysis` now owns the primary public-analysis implementation. The historical `analysis.py` module is retained only as a compatibility shim during the staged migration.

This slice is intentionally mechanical: the implementation blob is unchanged, so estimators, outputs, privacy guarantees, and numerical results remain identical.

Subsequent slices will migrate dependent analysis modules to package-relative imports before removing legacy compatibility paths.
