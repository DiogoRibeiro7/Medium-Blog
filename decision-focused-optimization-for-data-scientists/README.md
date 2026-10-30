# The Lowest RMSE Is Not the Best Decision

Reproducible synthetic research companion to the long-form manuscript
[The Lowest RMSE Is Not the Best Decision: From Prediction to Stochastic Optimization](article.md).
This is article #7 in the [chat's chronological article index](../ARTICLES.md).

## Scientific questions

1. Can a conditional mean forecast have lower holdout RMSE but make a worse
   ordering decision than an estimated conditional quantile?
2. What is the cost-dependent *critical fractile* in an asymmetric
   single-period inventory decision?
3. How many demand scenarios does empirical sample-average approximation
   require before decision regret becomes stable?
4. What changes when two products share a capacity limit and the same
   nonanticipative allocation must apply before demand is known?
5. How do expected-cost and CVaR-weighted stochastic LP objectives change
   the allocation and the holdout distribution of losses?
6. What do the estimated value of the stochastic solution (VSS) and the
   expected value of perfect information (EVPI) measure?
7. How sensitive are frozen decisions to conditional volatility changes
   and a change in demand correlation that preserves marginal means?

The study uses **entirely synthetic data**. Its lognormal demand processes
are simple models chosen to separate statistical objects. The numbers are
not production KPIs, inventory recommendations, or estimates from a real
manufacturing or retail organization.

## Reproduce

From this directory:

```bash
python -m pip install -r requirements.txt
python analysis.py
python -m unittest discover -s tests -v
```

For a faster smoke test:

```bash
python analysis.py --quick
```

The fixed seed is 20261030. Independent named random streams separate
single-product model training, evaluation, oracle comparisons, SAA
replicates, multi-product LP scenarios and each stress-test population.
The full experiment uses 5,000 single-product training and 60,000 holdout
observations; 90 replicate scenario sets for SAA convergence; and 240
multi-product optimization scenarios, evaluated on independent sets of
70,000 demand vectors. The quick version is smaller, writes only
`_quick`-suffixed figures/tables, and leaves the full-study outputs
unchanged.

The [notebook](decision_focused_optimization.ipynb) covers each core
construction with a short run. The article-scoped GitHub Actions workflow
installs dependencies, runs mathematical tests including the fresh-kernel
notebook, and executes the quick reproduction.

## Experimental definitions

- Single-product demand is lognormal with a conditional log-mean linear in
  the selected basis (intercept, sine, feature); the generating log SD is
  0.45, estimated only on the training split.
- The asymmetric loss penalizes shortage at 8 units and excess at 2.
  The corresponding optimal critical fractile is 80%. The fitted mean is
  squared-error optimal, while the fitted quantile targets that loss.
- Two products have original marginal means 90 and 70 and Gaussian-copula
  correlation 0.65. The shared capacity is 190, with shortage penalties
  (8, 6) and excess penalties (2, 3).
- The deterministic policy orders the two marginal means. The risk-neutral
  SAA LP minimizes sample-average cost over 240 demand scenarios; additional
  policies optimize a mixture of expected cost and 90%-CVaR using the same
  scenarios, with risk weights 0.25, 0.5, 0.75, and 1.0.
- Nonanticipativity is enforced: one pair of order quantities is selected
  before the scenario demand outcomes. Only shortage/excess recourse slack
  varies across scenarios. The study does not include multi-period
  replenishment, holding costs, lead times, stock perishability, or
  supplier capacity uncertainty.
- Hindsight allocation uses the observed realized demand and is therefore
  **not feasible at decision time**. It is a theoretical information
  benchmark, never an input to the SAA policy.
- Under the conditional-volatility stress case, both marginal lognormal SDs
  rise by a factor 1.45 while marginal means stay fixed. Under the
  dependence stress case, Gaussian-copula correlation increases from
  0.65 to 0.92 while marginals stay fixed. All policy allocations are frozen.

## Original figures

| Figure | File | Interpretation |
| --- | --- | --- |
| 1 | [RMSE/cost reversal](figures/fig1_rmse_cost_reversal.png) | The lower-error point forecast produces a higher expected decision cost |
| 2 | [Critical fractile](figures/fig2_critical_fractile.png) | Shortage penalties change the optimal demand quantile |
| 3 | [Loss surface](figures/fig3_loss_surface.png) | Optimizing mean error, asymmetric cost, and tail risk are distinct |
| 4 | [SAA convergence](figures/fig4_saa_convergence.png) | Independent scenario samples generate variable decisions and regret |
| 5 | [Capacity frontier](figures/fig5_capacity_frontier.png) | Shared constraints couple two otherwise separable inventory problems |
| 6 | [Holdout regret](figures/fig6_out_of_sample_regret.png) | Compare realized mean and tail loss across frozen policies |
| 7 | [CVaR trade-off](figures/fig7_risk_cvar_tradeoff.png) | Risk weights change allocations and out-of-sample cost |
| 8 | [Stress and information](figures/fig8_shift_and_information_value.png) | Fixed policies under distribution shift; VSS and EVPI benchmarks |

All PNGs originate in [analysis.py](analysis.py), not adapted from
third-party charts.

## Machine-readable outputs

| CSV | Contents |
| --- | --- |
| `summary.csv` | Main parameters and figures' primary reported numbers |
| `single_product_predictions.csv` | First 500 holdout demand values, mean and quantile predictions and per-observation losses |
| `critical_fractile.csv` | Cost ratios, target quantiles, fitted/oracle quantities and realized costs |
| `single_product_loss_surface.csv` | Expected loss, 95th percentile and CVaR by candidate quantity |
| `saa_convergence.csv` | Scenario count, mean and SD of SAA quantity and estimated regret |
| `capacity_scenarios.csv` | Optimization-stage two-product demand scenarios |
| `allocation_policies.csv` | Decision vector, used capacity, mean cost and CVaR per policy |
| `capacity_frontier.csv` | Mean cost and CVaR on the binding-capacity boundary |
| `stress_tests.csv` | Fixed-policy cost metrics under original, volatility and dependence populations |
| `perfect_information.csv` | Mean-policy, SAA and hindsight costs, observed VSS and observed EVPI |

## Selected full-run results

| Quantity | Measured in synthetic holdout |
| --- | ---: |
| Mean forecast RMSE | 48.042 |
| 80% quantile forecast RMSE | 57.611 |
| Mean-based decision's expected asymmetric cost | 179.897 |
| Quantile-based decision's expected asymmetric cost | 148.383 |
| Deterministic mean allocation's expected cost | 231.272 |
| Risk-neutral scenario-LP allocation's expected cost | 204.000 |
| Risk-neutral allocation's 90%-CVaR | 633.779 |
| Pure-CVaR allocation's 90%-CVaR | 629.960 |
| Estimated VSS, deterministic mean minus stochastic policy | 27.272 |
| Estimated EVPI, stochastic minus clairvoyant policy | 131.100 |

These finite differences are from one seeded simulation, not population
identities. In particular, sample-averaged LP solutions are evaluated on
fresh holdout draws, so a risk weight can occasionally improve both
estimated mean loss and CVaR relative to another policy due to scenario
estimation error. The displayed trade-off is **not** claimed to be the true
population Pareto frontier.

## Research scope and limits

The paper derives the critical fractile, linear-programming slack
reformulation, CVaR epigraph, information-value definitions and the
relationship between copula dependence and separable expected cost.
The results depend on the generating laws, losses, feasibility constraint,
scenario count, chosen risk measure and distribution-shift mechanisms.
CVaR optimization in this paper is **risk-sensitive optimization**, not a
distributionally robust guarantee against all possible shifts.

We do not implement the learning algorithms of the SPO or differentiable
optimization papers cited in the manuscript. Those are motivation and
references for the broader field; this experiment instead compares
properly specified predictive summaries with downstream optimization.
