# Commercial data: selection, clustering, and limits on inference

Source: the checksum-verified [URBAN-WASTE dataset](https://zenodo.org/records/3355300), as parsed in [workflow 39582196872](https://github.com/DiogoRibeiro7/Medium-Blog/actions/runs/39582196872). The calculations below use the actual CSV outputs archived by that run.

- 3,321 total rows from 34 kitchen labels; 380 eligible whole-day rows (11.44%) from 25 kitchens.
- Eligible rows represent only 7.71% of the total observed source-weight sum.
- The most represented kitchen contributes 121/380 = 31.84% of eligible rows; the five most represented kitchens contribute 292/380 = 76.84%.
- The provisional pooled weight/whole-day-portions ratio is 0.24353 **unverified source weight units** per portion; median row ratio 0.16125.
- Dropping all row ratios above the empirical 95th percentile changes the provisional ratio of sums to 0.20644; this is an outlier-selection scenario, not a robust causal estimate.
- Leaving out one eligible kitchen at a time changes the provisional pooled ratio to approximately 0.20418–0.33135.
- The equal-kitchen mean of kitchen-specific ratios is approximately 1.20635, substantially different from the pooled ratio, and is sensitive to kitchens with very few usable records. This is not a recommended population estimate.

These ratios are *not* kilograms per meal. The source does not yet establish weight units, observation-window alignment, or whether all recorded waste originated from served portions. The data also lack menu price, food procurement cost, customer repeat visits, and a portion-size intervention.

The new R pipeline reproduces concentration, kitchen leave-one-out sensitivity and minimum-eligible-record thresholds directly from committed-code outputs. To run after the project real-data download and analysis:

```sh
Rscript scripts/analyse-commercial-selection.R
Rscript scripts/analyse-commercial-heterogeneity.R
```

No observed restaurant contribution-margin benefit is claimed. Future economic modelling must be an explicit set of *hypothetical* cost/demand scenarios, not an empirical interpretation of these unverified ratios.
