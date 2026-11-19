# Frequent purchases, official inflation and perceived prices

Companion code for the Medium article on supermarket prices and felt inflation. The thesis is that the index misses what you buy often. Official inflation weights each item by the money spent on it, while people judge prices by what they pay for every week. Since 2002, frequent purchases have risen faster than the rest of the basket, and survey answers about prices follow them.

## What it finds

All numbers below are from `output.txt`.

**Two baskets in the euro area** (Eurostat's frequent out-of-pocket purchases, FROOPP, 42% of spending in 2026).
- **Since December 2001**, frequent purchases rose 92.4% to October 2026, all items 72.3%, everything else 58.0%.
- **Since December 2020**, the figures are 32.7%, 28.3% and 25.3%.
- **How often.** Frequent purchases rose faster than everything else in 243 of 298 months.

**Item by item, 2001 to 2025.** The median rise is 95% for ten things bought often, 81% for five bills and 23% for eight things bought rarely. Computers and TVs fell by about 80%.

**Every EU country.** FROOPP is published only for the euro area and the EU. Here it is rebuilt for each country from class indices and weights.
- **Validation.** For the euro area, the rebuilt index matches Eurostat's within 0.02 points a year until 2016.
- **Countries.** From December 2020 to December 2025, frequent purchases rose more than all items in 26 of 27 countries. The median gap is 4.2 points.

**What people say.** The survey asks how prices have changed over the last 12 months. The data cover 27 countries, 2002–2025, with 7,725 country-months.
- **Weight on frequent purchases.** With country fixed effects, the answers put a weight of 0.86 on frequent purchases (95% interval 0.71 to 1.04). Their share of spending is 0.50.
- **Where the weight goes.** Split six ways, the answers weigh food, drink and tobacco at 0.36 against a 0.24 share of spending. They weigh household energy and other bills (rent, water, phone, insurance) at about zero against 0.16.

**Memory.**
- **Longer horizons.** Adding the last 36 months of price change raises the within R² from 0.470 to 0.541. Almost all of that weight goes to frequent purchases.
- **Changeover.** The jump in answers at the 2002 euro cash changeover is not explained by prices of either basket.

**A felt index.** Weighted 0.86 on frequent purchases, euro-area prices rose 31.6% from December 2020 to October 2026. The published index rose 28.4%.

## Data

The analysis reads a snapshot in `data/`, taken from the Eurostat dissemination API on 19 November 2026:

- `prc_hicp_midx`, `prc_hicp_inw`: HICP monthly indices (2015 = 100) and item weights, ECOICOP ver.1, for the 27 EU countries and the euro area, to December 2025.
- `prc_hicp_minr`, `prc_hicp_iw`: HICP indices and weights under ECOICOP ver.2 (COICOP 2018), euro area, to October 2026.
- `ei_bsco_m`: European Commission consumer survey, "price trends over the last 12 months", balance, seasonally adjusted, to October 2026.

`python fetch_data.py` downloads the same series again and reports any difference from the snapshot. With `--write`, it replaces the snapshot. Seasonally adjusted survey balances are re-estimated every month, so later downloads differ slightly. Eurostat data are reused under the European Commission's reuse policy, with acknowledgement.

## Method notes

- **Rebuilding FROOPP.** The basket uses Eurostat's published list of ECOICOP classes. It is chained each December with the annual weights, as the HICP is. Pairs of classes that a country publishes only together (09.3.4/5, 09.5.3/4, 12.1.2/3) enter as the pair.
- **After 2017.** From 2017 Eurostat compiles special aggregates from five-digit subclasses. Its FROOPP weight is then 26 to 33 per mille lower than the class-level list, and the euro-area 12-month rates differ by 0.08 points on average.
- **Fixed effects.** Regressions use country fixed effects, and month fixed effects as a check. Intervals come from 999 resamples of whole countries.
- **Determinism.** Sums go through `math.fsum` or `np.bincount`, and small systems are solved in Python floats. The output does not depend on the processor's vector kernels.

## Reproduce

```
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v    # the methods on cases with a known answer
python felt_numbers.py > output.txt        # every number in the article, from the snapshot
python figures.py                          # figures/
python fetch_data.py                       # optional: compare the snapshot with Eurostat today
```

The workflow `.github/workflows/felt-inflation.yml` runs all of this. It fails if `output.txt` changes.

## Limits

- **Answers are a balance, not a rate.** The survey's balance counts how many people say prices rose a lot, moderately or slightly, minus those who say they stayed or fell. The weights above come from how the balance moves, not from anyone's estimate of inflation.
- **Baskets are averages.** FROOPP is defined from purchase frequencies in a few member states, and the weights are national averages. A household's own basket can differ a lot.
- **Correlation over time.** Food and energy shocks often arrive together. The split between groceries, bills and the rest rests on the times when they moved apart, and its intervals are wide for "other frequent purchases" and "everything else".

## Licence

Code: Apache-2.0.
