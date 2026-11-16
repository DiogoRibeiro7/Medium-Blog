# HMC and the geometry of Bayesian inverse PDEs

This project is a reproducible computational companion to the Medium essay
*Hamiltonian Monte Carlo for Bayesian PINNs, Tested on a Posterior We Can Compute Exactly*.

## What it does—and what it does not

It uses a **known analytic forward solution**, not a trained PINN. This design
sets approximation error to zero, isolating posterior geometry and the behaviour
of HMC, Gaussian variational approximations and a MAP estimator.
The manuscript discusses how these phenomena carry over to B-PINNs and where
additional neural approximation and residual-likelihood assumptions enter.

The exact solution of the one-dimensional heat equation is

```text
u(x,t; A, κ) = A sin(πx) exp(-κπ²t),  x in [0,1], t ≥ 0.
```

It satisfies `u_t = κu_xx`, zero boundary values and initial condition
`u(x,0)=A sin(πx)` for all positive `A` and `κ`. Their log parameters have
independent Gaussian priors. Twenty-four synthetic noisy observations are
drawn at either narrowly or widely distributed measurement times; their spatial
locations and noise variance are held fixed.

## Reproduce

Tested with Python 3.13, NumPy 2.5, SciPy 1.18 and Matplotlib 3.11. From this
directory, use an isolated environment of your choice:

```bash
python -m pip install -e ".[test]"
python -m unittest discover -s tests -v
python experiment.py --output results --seed 20261116 --replications 100
```

The full run takes several minutes and writes
`results/metrics.json` and seven PNG figures under `results/figures/`.
`--quick` runs a reduced version (fewer replications, seeds and grid points) in
under a minute; CI uses it, and its numbers are not the committed ones. No data
are downloaded, and all random seeds are fixed. `render_equations.py` writes the
displayed equations used by the essay.

## What is compared

On the narrow-window data set:

1. A 480×480 deterministic integration of the two-dimensional posterior,
   treated as the numerical reference rather than an MCMC result (checked
   against a 960×960 grid).
2. MAP for the same posterior.
3. Reverse-KL Gaussian variational inference, both mean-field and
   full-covariance, optimised with 24×24 Gauss–Hermite quadrature (no
   stochastic VI noise).
4. Four-chain leapfrog HMC with three constant mass matrices: the identity, the
   Hessian of the negative log posterior at the MAP point, and the inverse of
   the exact posterior covariance (the dense metric that Stan-style warm-up
   adaptation tries to estimate). None of these is Riemannian HMC.

Follow-up studies (`studies.py`):

- **Identification.** Profile likelihood of κ with the amplitude maximised in
  closed form; exact-grid intervals as the prior on log κ is narrowed, widened
  or moved; local Hessian geometry at points along the ridge.
- **Samplers.** Eight seeds × four mean path lengths (4, 8, 15, 30 leapfrog
  steps) × three mass matrices. Rank-normalised split-Rhat, bulk and tail ESS
  and quantile MCSE (`diagnostics.py`, Vehtari et al. 2021, cross-checked
  against ArviZ in the tests), divergent transitions (energy error above
  1000), acceptance probability by position on the ridge, and the ratio of the
  tuned step size to the thinnest local width of the ridge in each metric's
  coordinates (leapfrog is unstable above 2).
- **Design** (`design.py`). Local Fisher information over windows of evenly
  spaced times; the Box–Lucas D-optimal two-time design; a two-time design
  targeted at κ alone, planned at the true κ, at the prior median and at two
  wrong guesses; each checked with exact-grid posteriors on 100 repeated data
  sets.
- **Calibration.** Coverage of the exact-grid 90% intervals over 100 repeated
  data sets at the original parameters, at seven fixed true values of κ from
  0.03 to 0.6, and with (A, κ) drawn from the prior (where nominal coverage
  holds by construction, so this checks the machinery).

## Scientific boundaries

- The analytic forward model is correctly specified by construction.
- The experiment is low-dimensional (two unknown log parameters), not a
  demonstration of scaling to thousands of neural weights.
- The residual term of a soft-constrained PINN is absent because the equation
  holds exactly. A full B-PINN would need an explicit residual-discrepancy
  model, prior over weights and inference over those weights.
- Credible-interval coverage at a fixed parameter value need not be nominal;
  the coverage study shows where it is not.
- The design results are local: they assume a value of κ, which is what the
  experiment is meant to find.
- The HMC sampler uses fixed path-length ranges rather than NUTS; the path-length
  sweep is there so that conclusions do not rest on one trajectory length.

## Outputs

- `posterior_geometry.png`: narrow design in physical and log coordinates, with
  mean-field and full-covariance VI contours.
- `predictive_uncertainty.png`: exact posterior and mean-field VI bands for
  u(0.5, t).
- `identification.png`: profile likelihood against posteriors under three prior
  widths, narrow and wide designs.
- `sampler_diagnostics.png`: traces, acceptance by position, stability margin
  and ESS per gradient across path lengths for the three mass matrices.
- `observation_geometry.png`: posterior geometry for the narrow window, the wide
  window and the κ-targeted two-time design.
- `design_tradeoff.png`: Laplace posterior SD of log κ over measurement windows,
  and the two-time trade-off between dispersion and decay.
- `coverage_by_truth.png`: interval coverage as the true κ moves away from the
  prior median.

The generated figures are original and can be reused with attribution to
Diogo Ribeiro and the companion project.
