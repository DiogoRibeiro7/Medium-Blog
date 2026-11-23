# Teaching Bayesian statistics problem first: a vaccine trial, the Challenger O-rings and a sampler check

Companion code for a Medium article arguing that Bayesian statistics is best taught from a problem: state what you believe, let software update it, watch the posterior move as evidence comes in, and bring in the algebra once it has something to explain. The course runs on two real problems and one check:

- **The BNT162b2 vaccine trial** (Polack et al. 2020). 8 of 170 cases in the vaccine arm, analysed with the protocol's Beta(0.700102, 1) prior on the share of cases in the vaccine arm.
- **The Space Shuttle O-ring flights before Challenger** (Dalal, Fowlkes and Hoadley 1989). A binomial logistic regression of thermally distressed O-rings on launch temperature, extrapolated to 31°F.
- **Simulation-based calibration** (Talts et al. 2018). A way to check a sampler without any algebra.

## What it finds

All numbers below are from `output.txt`.

**The vaccine prior, before any case.**
- The prior is worth 1.7 cases. Its mean corresponds to VE = 30%, but its median is 40.9%.
- It gives 38.4% to a harmful vaccine (VE below 0) and 14.5% to a vaccine that at least quadruples the risk.
- Its prior predictive for the vaccine arm's share of 170 cases runs from 2 to 158 (central 90%).

**One posterior, three routes.**
- A grid of 20,000 points, a Metropolis sampler (4 chains of 20,000) and the Beta-binomial formula agree to 0.02 percentage points: median VE 94.86%, 95% interval 90.35% to 97.63%.
- With the surveillance-time adjustment (2.214 against 2.222 thousand person-years) the interval is 90.3% to 97.6%, as published.

**The posterior as cases accumulate.**
- The protocol's success boundaries are reproduced: at most 6, 15, 25 and 35 vaccine-arm cases at 32, 62, 92 and 120 cases (P(VE > 30%) > 0.995), and 53 of 164 at the end (> 0.986). In VE terms: 76.9%, 68.1%, 62.7%, 58.8% and 52.3%.
- Every one of 5,000 random orders of the 170 observed cases passes every interim look.
- Updating one case at a time and updating all at once give the same posterior (largest relative gap 1.3e-14).
- Against a vaccine exactly 30% effective, the five-look rule calls success in 2.17% of 200,000 simulated trials. A final look alone gives 1.21%; checking after every case, 5.98%.

**Four priors.** Along one random order of arrival, the protocol, flat and enthusiast (VE = 90%, worth 20 cases) priors end at a median VE of 94.7% to 94.9%. A sceptic centred on VE = 0 and worth 100 cases ends at 72.8%, because it still carries 37% of the weight after 170 cases.

**The O-rings.**
- **Prior predictive.** A Normal(0, 10) "vague" prior puts 64.6% of its curves below 1% or above 99% at 70°F. The weakly informative prior used here puts 14.4% there.
- **Posterior.** With the weakly informative prior, the slope is negative with probability 98.8%. The chance that one ring shows distress is 4.9% at 70°F and 68.7% at 31°F (90% interval 14.2% to 96.0%). With the vague prior the 31°F median is 82.0%; maximum likelihood gives 81.8%.
- **Prior scale.** At 31°F the posterior median runs from 39.9% to 81.8% as the prior scale of the slope goes from 0.5 to 10. At 70°F it stays between 4.4% and 5.5%.
- **Flights with distress only.** Using only the 7 flights that had distress, the probability of a negative slope falls to 56.8%.

**Fake data and calibration.**
- **Fake data.** On 400 fake O-ring datasets, the 90% intervals cover 89.0% (intercept) and 88.8% (slope).
- **Calibration.** In 2,000 simulated trials, the correct sampler's 90% intervals cover 88.9% and its ranks are uniform (p = 0.57).
- **Broken samplers.** A sampler stopped too early covers 71.5%; one that counts the data twice covers 74.9%. Both pile the truth into the extreme ranks.

## Data

- **Vaccine trial.** Published counts and constants, entered in `bayes_core.py`:
  - cases and surveillance times from Polack et al. (2020), Table 3;
  - the prior, the looks and the thresholds from the trial protocol.
- **O-rings.** `vcd::SpaceShuttle` (Dalal, Fowlkes and Hoadley 1989), from Vincent Arel-Bundock's Rdatasets at commit `1dcc2bf`. `fetch_data.py` downloads it into `data/` (not committed) and checks it against `digests.txt`. Flight 4, with no O-ring count, is dropped, which leaves 23 flights.

## Reproduce

```
python -m pip install -r requirements.txt
python fetch_data.py                       # data/ at the pinned commit, checked against digests.txt
python -m unittest discover -s tests -v    # the methods on cases with a known answer
python bayes_numbers.py > output.txt       # every number in the article; about 45 seconds
python figures.py                          # figures/fig1 to fig6
```

The workflow `.github/workflows/bayes-problem-first.yml` runs all of this and fails if `output.txt` changes.

## Files

- **`bayes_core.py`**: the models and checks.
  - Vaccine trial: VE and the share of cases in the vaccine arm, the conjugate update, the grid, the random-walk Metropolis sampler, split R-hat, success boundaries and the decision-rule simulation.
  - O-rings: the logistic model, its priors, a 2-D grid posterior and a 2-D sampler.
  - Simulation-based calibration, with one correct and two broken samplers.
- **`bayes_numbers.py`**: all numbers, printed in order, and the arrays the figures use (in `results/`, not committed).
- **`figures.py`**: the six figures.
- **`tests/test_core.py`**: 15 tests. Among them: the published interval and boundaries, grid against formula, sequential against batch updating, recovery of a known logistic curve, and the calibration check catching double-counted data.

The sampler is written out in NumPy so every step can be read. In a course, PyMC, Stan or brms would play its role.
