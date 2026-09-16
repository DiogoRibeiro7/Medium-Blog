# Global systolic time-form sensitivity

## Question

The primary analysis summarises the global association between calendar time and observed daily mean systolic pressure with an equal-observed-day linear OLS model and HC3 covariance.

That slope is useful, but it is still one functional-form choice. The purpose of this sensitivity layer is therefore narrow:

> Does the sign and broad magnitude of the observed global association depend on representing the entire observed sample by one straight line?

This is a robustness analysis, not a search for a preferred trajectory.

## Common sample

All three summaries use the same **31 observed calendar days** from the privacy-safe public snapshot. Missing calendar days remain absent observations. No interpolation is performed across the 32-day internal gap.

The reporting scale is millimetres of mercury per 30 calendar days.

## 1. Equal-day linear OLS with HC3 covariance

The primary specification is

\[
y_i = \beta_0 + \beta_1 t_i + \varepsilon_i,
\]

where every observed calendar day receives equal weight. The reported quantity is

\[
30\widehat\beta_1.
\]

HC3 covariance is used for the interval. This is the same global linear estimand already reported by the primary analysis.

For the current snapshot,

\[
\boxed{-4.42\ \text{mmHg per 30 days}}
\]

with 95% HC3 interval

\[
[-6.38,-2.46].
\]

## 2. Quadratic endpoint-average change with HC3 covariance

The second specification allows modest curvature:

\[
y_i = \beta_0 + \beta_1 t_i + \beta_2 t_i^2 + \varepsilon_i.
\]

A quadratic does not have one constant slope, so reporting \(\widehat\beta_1\) would not be comparable to the linear 30-day summary. Instead, the analysis reports the fitted end-to-end average change across the observed calendar span:

\[
\Delta_Q
=
30\,
\frac{\widehat m(t_{\max})-\widehat m(t_{\min})}
{t_{\max}-t_{\min}},
\]

where

\[
\widehat m(t)=\widehat\beta_0+\widehat\beta_1t+\widehat\beta_2t^2.
\]

Its uncertainty is computed as an HC3 linear contrast of the fitted quadratic coefficients.

For the current snapshot,

\[
\boxed{-4.82\ \text{mmHg per 30 days}}
\]

with 95% HC3 interval

\[
[-6.68,-2.95].
\]

This quantity is an **endpoint-average change**, not the instantaneous derivative of the quadratic at any particular day.

## 3. Theil–Sen median pairwise slope

The third summary does not fit a polynomial. It takes the median of all pairwise slopes between observed days:

\[
\widehat\beta_{TS}
=
\operatorname{median}_{i<j}
\frac{y_j-y_i}{t_j-t_i}.
\]

The reported value is

\[
30\widehat\beta_{TS}.
\]

The interval is the nonparametric 95% Theil–Sen interval returned by the implementation.

For the current snapshot,

\[
\boxed{-5.10\ \text{mmHg per 30 days}}
\]

with 95% interval

\[
[-7.23,-2.78].
\]

## Current comparison

| Global summary | Estimate per 30 days | 95% interval |
|---|---:|---:|
| Equal-day linear OLS, HC3 | -4.42 | [-6.38, -2.46] |
| Quadratic endpoint-average, HC3 | -4.82 | [-6.68, -2.95] |
| Theil–Sen median pairwise slope | -5.10 | [-7.23, -2.78] |

All three current intervals are below zero.

The appropriate conclusion is therefore limited but useful:

\[
\boxed{
\text{the negative global association is not unique to one straight-line OLS summary}
}
\]

That does **not** imply that the underlying blood-pressure trajectory is globally linear, globally quadratic, smooth through the missing interval, or causally driven by time.

## Why the estimands must remain distinct

The three rows above answer related but different questions:

- linear OLS asks for the best equal-day constant linear slope under squared-error loss;
- the quadratic summary asks for the fitted average endpoint change after allowing one curvature term;
- Theil–Sen asks for the median pairwise calendar-time slope.

Agreement in sign is therefore a robustness result. Numerical closeness does not make the estimands interchangeable.

The analysis deliberately does not rank the three models, choose one by AIC/BIC, or extend the polynomial degree after seeing the data. Higher-degree polynomials and splines remain outside the current modelling programme.

## Relation to the long gap

This sensitivity analysis does not remove the central structural fact of the dataset: a 32-day interval contains no measurements.

The separate gap-aware decomposition shows that about four-fifths of the negative global time-pressure covariance is associated with separation between the two observed episodes around that gap. The global time-form sensitivity result should therefore be read alongside, not instead of, the gap-aware analysis.

In particular, none of these three global summaries identifies when or why the pre/post level difference arose inside the unobserved interval.

## Reproducibility

The implementation is in:

- `blood_pressure_missingness.analyses.global_time_form`

The committed privacy-safe result is:

- `figures/global_time_form_sensitivity.json`

It can be regenerated from the public snapshot alone:

```bash
python -m blood_pressure_missingness.analyses.global_time_form \
  --data data/analysis_snapshot.csv \
  --output figures/global_time_form_sensitivity.json
```

No private source rows, calendar dates, clinical thresholds, or causal interpretation are required.