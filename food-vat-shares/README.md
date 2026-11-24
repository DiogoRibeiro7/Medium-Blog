# Who gains from lower VAT on food: euros and budget shares by income fifth, EU 2015–2025

Companion code for a Medium article on the claim that cutting VAT on food "benefits the rich more than the poor". The article measures both halves of that dispute, the euros each income fifth saves and the saving as a share of its budget, in every member state with a household budget survey by income fifth, with Portugal as one case.

## What it finds

All numbers are from `output.txt`. A fall of 1% in food prices, fully passed on, is the unit of the cross-country comparison; the shape of every result is the same for any size of cut.

- **Engel's law in the budget survey (wave 2015, 26 member states, imputed rents excluded).**
  - Food and non-alcoholic drink take a larger share of the poorest fifth's budget than of the richest fifth's in 23 of 26 countries; the share falls at every step in 17.
  - The ratio of the two shares has a median of 1.43 (Netherlands 0.91 to Cyprus 1.98). Portugal 1.53 (22.0% against 14.4%).
  - EU27: 18.8% against 13.8%.
- **Euros and shares at once.**
  - The richest fifth saves more euros per adult equivalent than the poorest in 26 of 26 countries (median ratio 1.52), and the poorest saves more as a share of its budget in 23 of 26.
  - Spending per adult equivalent differs more between the two fifths (median ratio 2.01) than the euros saved do.
  - EU27: the richest fifth receives 29.6% of the euros saved and makes 34.2% of all spending; the poorest 11.5% and 9.7%.
  - Kakwani index of the saving against spending: median −0.072, negative (progressive) in 23 of 26.
  - The ratio of richest to poorest real spending falls in exactly the 23 countries where the cut is progressive in shares; for a 6% to 0% cut on all food and drink the median fall is 0.44%.
- **Restaurants and cafés behave the other way.** The richest fifth spends a larger share on them in 25 of 26 countries (Poland is the exception); the richest fifth would receive a median 40.2% of a restaurant VAT saving and the poorest 7.2%; Kakwani median +0.088.
- **Other waves.** 2010: 22 of 25 countries progressive in shares. 2020 (CY, FR, MT left out because Eurostat repriced their 2015 surveys): 18 of 21; Belgium, Denmark and the Netherlands are not.
- **Against income instead of spending (Eurostat ICW experimental statistics).** The poorest fifth spends a larger share of its disposable income on food than the richest in every country, 26 of 26 in 2015 and 25 of 25 in 2020; median ratio 2.55, first against tenth decile 3.5.
- **Portugal (survey of 2022/23).**
  - Food and drink 17.1% of the budget in the poorest fifth, 10.0% in the richest; 19.3% and 6.8% of disposable income.
  - Taking VAT on all food and drink from 6% to 0% (prices 5.66% lower) would save the poorest fifth about 12 euros a household a month at 2025 food prices and the richest about 24; 0.97% and 0.57% of spending; 1.09% and 0.38% of income.
  - The richest fifth would receive 26.9% of the saving and makes 34.7% of spending; the poorest 13.6% and 10.3%.
  - Food at the standard 23% rate instead would cost 2.74% of the poorest fifth's spending and 1.60% of the richest's.
- **Who is in the richest fifth (EU-SILC 2025, incomes of 2024).**
  - In Portugal the richest fifth of people starts at 22,416 euros of equivalised disposable income a year: 1,868 euros a month in twelve payments, 1,601 in fourteen.
  - A single person taking home 1,500 euros a month sits at about the 67th percentile with twelve payments and the 76th with fourteen; a childless couple who each take home 1,500 euros fourteen times a year sits at about the 88th.
  - The same single person is in the richest fifth in 5 member states and below the median in 15.

## Data

The data are a canonical snapshot in the data registry, [DiogoRibeiro7/data](https://github.com/DiogoRibeiro7/data), `datasets/eurostat-food-vat-incidence`. It holds seven Eurostat JSON-stat responses retrieved on 24 November 2026: `hbs_str_t223`, `hbs_exp_t133`, `hbs_exp_t111`, `icw_aff_01`, `icw_sr_10`, `ilc_di01` and `prc_hicp_aind`. Eurostat's API serves only the current vintage, so the registry keeps the bytes.

The registry commit, URLs, sizes and SHA-256 are in [`../data_sources/eurostat-food-vat-incidence.json`](../data_sources/eurostat-food-vat-incidence.json). `fetch.py` downloads the files into `data/` and stops if any byte differs. No data are committed here: `data/` and `results/` are rebuilt on every run and ignored by git.

## Reproduce

```
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v    # the functions against answers known by construction
python fetch.py                            # the seven files, from the pinned registry commit
python vat_numbers.py > output.txt         # every number in the article; also writes results/
python figures.py                          # figures/
```

The workflow `.github/workflows/food-vat-shares.yml` runs all of this from a clean checkout and fails if `output.txt` changes.

## Method notes

- **The two measures.** For income fifth *q*, write *s* for the share of food in the budget and *c* for spending per adult equivalent. A fall *p* in food prices saves *p* · *s* · *c* euros and *p* · *s* of the budget. The first rises with income when food spending rises with income; the second falls with income when the food share does.
- **Price arithmetic.** A cut from rate *t*₀ to *t*₁, fully passed on, lowers the price by 1 − (1 + *t*₁)/(1 + *t*₀): 6% to 0% is 5.66%, not 6%.
- **Cash basis.** Imputed rents of owner-occupiers (CP042) are removed from the denominator, because nobody pays them and countries impute them differently. This lowers the measured progressivity in countries where richer households own more. For Portugal's 2020 wave Eurostat publishes divisions only, so the published total is used.
- **Kakwani index.** Concentration coefficient of the saving minus that of spending, with five equal groups ranked by income and the trapezoid rule. Negative means the saving is spread more equally than spending.
- **Inequality.** After a fall *p*, real spending of fifth *q* becomes *c* · (1 + *p* · *s*) to first order, so the ratio of richest to poorest falls exactly when the poorest fifth's food share is the larger.
- **Euros for Portugal.** PPS are converted with the euros-per-PPS factor implied by Eurostat's own mean spending per household (`hbs_exp_t111`, 0.894), then uprated to 2025 food prices with the HICP for food and non-alcoholic beverages (×1.32 from 2020). A 6% to 0% cut on all food and drink is an upper bound for any basket.
- **Income positions.** Linear interpolation between the decile cut-offs of `ilc_di01`, with zero income at the bottom; positions above the ninth decile are reported as 90 or more. Couples are equivalised with the modified OECD scale (1, 0.5, 0.3 for children under 14).

## Files

- **`fetch.py`**: download and verify the registry snapshot.
- **`jsonstat.py`**: read a JSON-stat 2.0 response into a table.
- **`vat_core.py`**: budget tables, profiles by fifth, concentration and Kakwani indices, the inequality identity, income positions.
- **`vat_numbers.py`**: all numbers, printed in order, and the tables in `results/`.
- **`figures.py`**: the five figures.
- **`tests/test_core.py`**: 8 checks.

## Licence

Code: Apache-2.0. Data: Eurostat, reused under the Commission's reuse policy (Decision 2011/833/EU), with acknowledgement of the source.
