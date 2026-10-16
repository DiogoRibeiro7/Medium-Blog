# Linux replication, GitHub Actions run 38975791746

Written by the *EU Okun hierarchical Bayes* workflow (run 38975791746, ubuntu-latest) on
branch `study/eu-okun-bayes-dynamics` at commit 0f8a91d, with
`specs=primary,dynamic_factor,ardl,ardl_dynamic_factor`. It uses the same snapshot, lock file,
seeds, sampler settings and single retry as the local run.

- `dynamics.json` is copied unchanged from the workflow artifact `eu-okun-bayes-reports`
  (`bayes-dynamics/dynamics.json`).
- `fits/<spec>/fit.json` (diagnostics, settings, environment) is copied unchanged from
  the artifact `eu-okun-hierarchical-bayes`.
- `first-attempt/primary/fit.json` is the failed first attempt of `primary` on Linux
  (R-hat 1.012 for `phi[9]`, ESS 372 for `tau_alpha`). The single retry with 3000
  warm-up iterations passed.
