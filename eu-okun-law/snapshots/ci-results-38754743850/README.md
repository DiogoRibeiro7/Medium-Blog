# Archived CI outputs, workflow run 38754743850

Outputs of the *EU Okun empirical estimates* workflow, run
[38754743850](https://github.com/DiogoRibeiro7/Medium-Blog/actions/runs/38754743850)
(commit `e73ee68`), from its artifact `eu-okun-empirical-baselines`.

**Why this is frozen here.** GitHub keeps workflow artifacts for 30 days. The classical
results review ([`docs/CLASSICAL_RESULTS_REVIEW.md`](../../docs/CLASSICAL_RESULTS_REVIEW.md),
section 3) compares these files with outputs regenerated from
[`snapshots/eurostat-2026-10-15/`](../eurostat-2026-10-15/README.md). The raw files of that
snapshot are byte-identical to the artifact's `data/` directory. The files were copied from
the artifact on 2026-10-19, before it expired, and are not modified.

Reproduce the review, including the comparison:

```bash
poetry run eu-okun review --data-dir snapshots/eurostat-2026-10-15 --output-dir docs/review --report docs/CLASSICAL_RESULTS_REVIEW.md --published-dir snapshots/ci-results-38754743850 --provenance "CI artifact eu-okun-empirical-baselines, workflow run 38754743850 (commit e73ee68)"
```

| File | SHA-256 |
| --- | --- |
| `ols_hac.json` | `84d94cc59ae412f3181ab16f4c08f916aff92c3cd593b4088860d236e6432ce2` |
| `comparison/comparison.json` | `927d477c60a49df248174acbef6b8c2641003fe52fe216f0c7ca4818236da731` |
| `regimes/regimes.json` | `b74d88d1f810f84670bc6c34604679f3271c0860e52db026b52deb74da98190b` |
| `inference/stability.json` | `8278fa4d4b0ee170148a727352910d50cfe0f71a6e0981f98f36c217c2b6c6c1` |
| `inference/country_pairs.json` | `90dea6190cbb5f75a763e168aeda567787526a28fd39a6b87dbfed75f74b76d3` |
| `forecast/scores.json` | `43663feb55d614f51f1a795598fe67831745004eebd483bca39fb912abc01025` |
