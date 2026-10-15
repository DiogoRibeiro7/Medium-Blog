# Out-of-sample validation protocol

The model target is the change in the quarterly unemployment rate at t,
expressed in percentage points. This is forecast at the end of t-1.

The information set includes unemployment and participation through t-1,
and real GDP growth at t-1 and t-2. This is an optimistic availability
assumption: official releases and historical revisions are **not**
reconstructed. The experiment is therefore pseudo-out-of-sample, not
real-time or vintage-correct forecasting.

The candidate methods are fixed in advance:

- zero unemployment change;
- historical expanding mean of the change;
- AR(1) of unemployment-rate changes;
- two lags of GDP growth;
- AR(1) plus two GDP growth lags;
- AR(1), two GDP lags, and lagged participation change.

The first scored quarter is 2018-Q1; each model is refitted using only
quarters strictly before the scored quarter and at least 32 observations.
All six methods use identical countries, origins and training rows. No
shuffled folds or future target observations are used.

MAE, RMSE, signed bias, skill relative to zero change, and skill
relative to unemployment-change AR(1) are calculated by country.
The output includes each origin, information cutoff and training count.
No causal impact follows from forecasting success, and observed GDP
for t-1 could have been unavailable when making a real-time forecast.

A future release can add synchronized time-block uncertainty for paired
forecast errors and evaluate genuine historical data vintages.
