# Linux replication, GitHub Actions run 38983863147

Written by the *EU Okun hierarchical Bayes* workflow (run 38983863147, ubuntu-latest) on
branch `study/eu-okun-bayes-regimes` at commit 4dc5d2a, with
`specs=pandemic_excluded,regime_shift,regime_shift_final_gdp,pandemic_excluded_dynamic,regime_shift_dynamic,drifting_association`.
It uses the same snapshot, lock file, seeds, sampler settings and single retry as the local
run.

- `regimes.json` and `recovery.json` are copied unchanged from the workflow artifact
  `eu-okun-bayes-reports`.
- `fits/<spec>/fit.json` (diagnostics, settings, environment) is copied unchanged from
  the artifact `eu-okun-hierarchical-bayes`.
- `first-attempt/<spec>/fit.json` holds the failed first attempts. `drifting_association`
  passed the retry. `regime_shift_final_gdp` failed again (R-hat 1.013 for `phi[9]`), as
  it did locally.

An earlier dispatch (run 38983097415, commit b26974d) fitted the same specifications but
stopped at the general report step, which requires a `primary` fit that was not
requested. That workflow condition was fixed in 4dc5d2a, and none of that run's results
are used.
