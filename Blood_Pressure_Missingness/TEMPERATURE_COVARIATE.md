# Time-matched ambient temperature covariate

The blood-pressure analysis now includes ambient temperature as an exploratory
**covariate**. In statistical terminology, *covariate* is the appropriate term
here rather than *cofactor*.

## Exposure definition

For every private blood-pressure measurement, the refresh pipeline retrieves
hourly 2 m air temperature for:

- **Fiães, Santa Maria da Feira, Portugal**
- latitude `40.994459`
- longitude `-8.525370`
- timezone `Europe/Lisbon`

The source is the
[Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api),
using the `ecmwf_ifs` model and the `temperature_2m` variable in degrees Celsius.

Open-Meteo's historical product is a gridded model/reanalysis estimate. It must
not be described as an on-site thermometer reading.

## Time matching

The private Sheet already contains a local date and clock time for each
measurement. Those values are parsed in memory by `refresh_from_google_sheets.py`.

`temperature_covariate.py` converts Open-Meteo Unix timestamps to
`Europe/Lisbon` and matches each measurement to the nearest available hourly
temperature. A match more than 60 minutes away is rejected rather than silently
filled.

For an observed day \(d\), the temperature covariate is

\[
T_d
=
\frac{1}{n_d}
\sum_{i=1}^{n_d} T(t_{di}),
\]

where \(t_{di}\) is the actual clock time of blood-pressure reading \(i\) on
that day. Therefore \(T_d\) is **not** the meteorological daily mean. It is the
mean ambient temperature at the times when measurements were actually taken.

## Exploratory model

For each public daily metric \(Y_d\), the private refresh fits

\[
Y_d
=
\beta_0
+
\beta_1 d
+
\beta_2 \left(T_d-\bar T\right)
+
\varepsilon_d.
\]

The outcomes are:

- daily mean systolic pressure;
- daily mean diastolic pressure;
- daily mean pulse pressure;
- daily mean heart rate.

Each observed calendar day receives equal weight. Standard errors use the HC3
heteroskedasticity-robust covariance estimator.

The main quantities reported are the 30-day time slope and the temperature
coefficient per \(1\,^\circ\mathrm{C}\), both with 95% confidence intervals and
p-values.

This remains an exploratory association model. Temperature can be correlated
with season, time of day, activity, sleep, medication timing, and the
observation process. The coefficient must not be interpreted as a causal effect
of temperature on blood pressure.

## Privacy boundary

The temperature join happens only while the private Sheet is available in
memory. The repository does **not** persist:

- calendar dates from the health data;
- row-level measurement times;
- row-level matched temperatures;
- a day-indexed temperature series.

This matters because a detailed local weather sequence can itself act as a
quasi-identifier for calendar dates.

The committed output is only
`figures/temperature_covariate.json`, which contains source metadata,
non-temporal aggregate diagnostics, and fitted model coefficients.
