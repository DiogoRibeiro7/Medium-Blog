# URBAN-WASTE commercial source audit

The actual source schema was recovered from [GitHub Actions run
39464256676](https://github.com/DiogoRibeiro7/Medium-Blog/actions/runs/39464256676).
The publisher's CSV is `Food Tracking Device dataset.csv` in
[Zenodo 3355300](https://zenodo.org/records/3355300) (2019).

The prior successful run reported 1,586 rows and 13 fields from 29 distinct
kitchen labels. The fields are `City / kitchen`, `Timestamp`,
`Waste weight sum`, five `portions, ...` fields and five
`ordered portions, ...` fields (Whole day, Midday, Breakfast,
Lunch, Brunch). The weight field is text and its unit is not proven
from that schema. Several portion-count fields have extensive missingness.

This stage:
1. Parses numeric source weight values strictly without stripping units.
2. Rejects a changed source schema.
3. Counts available positive denominator observations for each meal period.
4. Counts records and measured weights by anonymised kitchen label.
5. Does not add potentially overlapping meal counts together.
6. Does not equate ordered with served portions.
7. Does not infer that waste weight represents plate waste, kitchen waste,
   kilograms, or monetary costs without an accompanying source codebook.

The resulting R outputs are `outputs/commercial_fields.csv`,
`outputs/commercial_kitchen_coverage.csv`, and
`outputs/commercial_denominator_coverage.csv`.

Run `Rscript scripts/analyse-commercial.R` after downloading the original
files. This is a descriptive source audit, not a business-profit study.
The next stage must verify source units and measurement definitions before
publishing waste-per-portion numbers.
