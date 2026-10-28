# Testing MC dropout as a posterior

Reproduction code for the Medium article that tests whether the spread of Monte Carlo dropout behaves like a posterior: whether it shrinks as the training set grows, how much of the benchmark score that made it popular comes from the spread itself, whether it rises where the training data stop, and what it is still good for as a ranking score.

Everything is written in NumPy (networks, backpropagation, Adam, dropout, exact Gaussian processes), so the experiments run on a CPU without a deep-learning framework.

## Scripts

- `mcd.py` — shared code: the UCI benchmark loader (data, published splits and hyperparameter grids of the reference MC dropout experiments), batched fully connected networks with dropout before every dense layer trained side by side with Adam, optional validation checkpointing, MC prediction, an exact Gaussian-process posterior, and scoring helpers.
- `check_mcd.py` — checks before anything else runs: backpropagation against finite differences, fitting a known function, checkpointing, the exact mean and variance of inverted dropout on a linear model, the GP posterior against scikit-learn, the MC log-likelihood and the AUROC.
- `01_linear_floor.py` — dropout on the inputs of a linear regression, solved exactly: the spread of the regression function against the sample size and against the size of the effect, and the coverage of its 95% interval (figures 1 and 2).
- `02_concentration.py` — protein data, training subsets from 500 to 32,000 rows: the spread of MC dropout at fixed and tuned rates against a bootstrap of the same network, a deep ensemble and an exact GP posterior (figure 3, drawn by `figures.py`).
- `03_uci_benchmark.py` — the reference MC dropout UCI protocol (2018 grid-search version, 100x epochs) rerun on the published splits of yacht, concrete, energy and red wine (`--sets` adds the larger data sets; Boston housing is excluded), with the predictive variance split into the dropout spread and the tuned noise term, the score of Gaussians that drop the spread, and risk–coverage curves for four ranking scores (figures 4 and 6, drawn by `figures.py`).
- `04_shift.py` — power-plant data with a gap and a tail in ambient temperature removed from training: spread, coverage and detection of the shifted rows for MC dropout with one and two hidden layers, a deep ensemble and a GP (figure 5).
- `figures.py` — figures 3, 4 and 6 from the result tables.
- `check_reproduction.py` — compares a fresh `results/*.txt` with a committed copy.

Each script writes its numbers to `results/` and its figure to `figures/`. The article text is in `article.md`.

## Run

```bash
pip install -r requirements.txt
python check_mcd.py
python 01_linear_floor.py      # under a minute
python 02_concentration.py     # about 2 hours on one core (two replicates)
python 03_uci_benchmark.py     # about 1.5 hours on one core (yacht, concrete, energy, wine; 20 splits each)
python 04_shift.py             # about 40 minutes on one core
python figures.py
```

`MCD_QUICK=1` runs every network experiment with tiny settings; `MCD_RESULTS` and `MCD_FIGURES` redirect the output. A GitHub Actions workflow (`.github/workflows/mc-dropout-posterior.yml`) runs the checks, reruns the exact linear-model experiment and compares its numbers with `results/01_linear_floor.txt`, and smoke-runs the network experiments in quick mode. The network experiments are too long for CI; their committed results come from a full local run.

## Data

Downloaded on first run and cached in `data/` (not committed): the UCI regression data sets with the train/test splits and hyperparameter grids published with the MC dropout experiments, from a pinned commit of https://github.com/yaringal/DropoutUncertaintyExps (yacht hydrodynamics, concrete compressive strength, energy efficiency, red wine quality, kin8nm, combined cycle power plant, naval propulsion plant, protein tertiary structure). The original data are from the UCI Machine Learning Repository.

## Contact

Diogo Ribeiro — dfr@esmad.ipp.pt — ORCID 0009-0001-2022-7072
