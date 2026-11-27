# Figure and source QA — 2026-11-27

Reviewed the successful [original PR artifact](https://github.com/DiogoRibeiro7/Medium-Blog/actions/runs/45929172619).

## Numerical checks against the downloaded CSVs

- Composition: A 35% in both groups; B 10% in both groups; pooled 320/1,000 = 32% versus 150/1,000 = 15%.
- Measurement: identical assumed 20% latent prevalence; reported 28% and 16%.
- Daily: seven days, 23 positives each day in both scenarios; 23 distinct positive individuals in persistent scenario and 100 in rotating scenario.

All inputs are synthetic. These are mathematical counterexamples, not Gallup microdata.

## Visual review

Original image dimensions: 1440 × 810 for composition and measurement; 1440 × 900 for daily. The two bar-chart footer captions overlapped x-axis labels. The revision places them within the managed Matplotlib layout as axis labels, adds value labels derived from the synthetic data, and makes the binary heatmap legend explicit. Re-run CI and visually inspect the replacement images before publication.

## Source recheck

- OWID's November 26, 2026 insight links the within-country claim to Gallup's July 9, 2024 report. Gallup describes the financial-status comparison as global and discusses its 2023 wave; the cited report does not establish within-country estimates.
- Confirmed Leblang and Wesselbaum (2026), DOI 10.1007/s10654-026-01424-z, including income heterogeneity by country income group.
- Confirmed Lykes and Kemmelmeier (2014), DOI 10.1177/0022022113509881; Heu et al. (2019), DOI 10.1177/0146167218796793; Luhmann et al., DOI 10.1038/s44159-022-00124-1; Paris et al. (2026), DOI 10.1525/collabra.161255.
- No survey microdata or empirical estimates were added.
