# How far ahead weather can be forecast: error growth in Lorenz's models and in the real atmosphere

Companion code for the Medium article on the two-week horizon of weather forecasts. The thesis is that the horizon is set by the atmosphere, through the growth of small errors, and not by the power of the computers that run the forecasts.

## What it finds

All numbers below are from `output.txt`.

**Small errors grow at a steady rate.**
- In Lorenz's 1963 model the largest Lyapunov exponent is 0.908, so an error doubles every 0.76 time units. Two runs that start 0.0002 apart, as in Lorenz's rounded printout, differ visibly after about 9 time units.
- In Lorenz's 1996 model (40 variables, one time unit taken as five days), small errors double every 2.05 days.

**With one scale of motion, precision always pays.** Every tenfold reduction of the starting error adds about 6.5 days to the horizon (ln 10 divided by the exponent: 6.8 days), at every level of precision.

**With two scales, it stops paying.**
- In the two-scale version (36 slow variables, each coupled to 10 fast ones), the fastest errors double every 8 hours.
- Cutting a large starting error tenfold still adds 8.5 days. Below a thousandth of the natural variability, each tenfold cut adds only about 1.3 days.
- A hundred thousand times more precision, from 1e-03 to 1e-08, buys 6.7 days, against 32 days in the one-scale model.

**Ensembles keep some skill longer.** In a perfect model with a 5% starting error, the anomaly correlation of a single forecast falls below 0.6 at day 10.0, and that of the 50-member ensemble mean at day 12.0. The ensemble's probability skill score is still 0.14 at day 15 and falls below 0.05 at day 20.5.

**The real record cannot show the mechanism.**
- In two years of NCEP/NCAR reanalysis, the closest same-season match of a day is typically 0.65 of the way to an unrelated day. Matches reach 95% of the unrelated level within 4 days.
- A linear stochastic copy, with the same memory and noise but no chaos, diverges the same way: the largest gap between the two curves is 0.05.

## Data

- **Reanalysis.** Near-surface air temperature from the NCEP/NCAR Reanalysis 1, 6-hourly, 2013 to 2014, North America at 2.5 degrees. This is the sample xarray distributes as tutorial data, repository [pydata/xarray-data](https://github.com/pydata/xarray-data), commit `a35297e`. `fetch_data.py` downloads it into `data/` and checks its SHA-256 digest. It is not committed.
- **Published doubling times.** `literature/doubling_times.csv` holds estimates from Charney et al. (1966) and Smagorinsky (1969), both as summarised by Bengtsson and Hodges (2006); from Lorenz (1982); and from Simmons and Hollingsworth (2002), ECMWF Technical Memorandum 342, Table 1.

## Reproduce

```
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v    # the methods on systems with a known answer
python fetch_data.py                       # the reanalysis sample, checked against its digest
python forecast_numbers.py > output.txt    # every number in the article, about four minutes
python figures.py                          # figures/
```

The workflow `.github/workflows/forecast-horizon.yml` runs all of this and fails if `output.txt` changes.

Chaotic runs amplify differences in the last bit of arithmetic. Everything that feeds back into a chaotic integration therefore uses elementwise operations, sums in a fixed order or `math.fsum`, and scales rounded to six decimals. The output is identical with numpy's AVX-512 kernels and with them disabled.

## Files

- **`chaos_core.py`**: the models (Lorenz 1963, Lorenz 1996 with one and two scales), a fourth-order Runge-Kutta step, Lyapunov exponents, error growth and horizons, perfect-model ensembles and their scores (anomaly correlation, fair CRPS), and the analogue test with its linear stochastic copy.
- **`forecast_numbers.py`**: all numbers, printed in order, and the tables in `results/`.
- **`figures.py`**: the six figures.
- **`tests/test_core.py`**:
  - the integrator against the exponential;
  - the two-scale model against a direct loop;
  - Lyapunov exponents of a linear system and of Lorenz 1963;
  - horizons under pure exponential growth (ln 10 divided by the rate per decade);
  - the fair CRPS against the closed form for a normal distribution;
  - a planted repeat found by the analogue search;
  - the memory kept by the linear copy.

## Limits

- **Toy models.** Lorenz's models are not the atmosphere. Days come from Lorenz's convention that one time unit is about five days, and the horizon is defined as the time to reach half the error of a guess from the climate.
- **Two scales are not many.** The two-scale model shows the flattening that Lorenz (1969) argued for, but with two scales the gain per tenfold never falls to zero.
- **Small reanalysis sample.** The sample covers two years of one field over one region.

## Licence

Code: Apache-2.0. The reanalysis sample is NOAA data, downloaded at run time and not redistributed here.
