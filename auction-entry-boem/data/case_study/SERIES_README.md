# Game and Auction Theory for Data Scientists

Code for a six-part series of articles that introduces game theory and auction theory to readers who know basic probability and can read Python, and for a case study that applies the series to real auction data.

A predictive model treats its environment as fixed. A game is what you have when part of that environment is other people who are also optimising: competitors who reprice, bidders in the same auction, fraudsters who adapt to a detection rule. Parts 1 to 4 build the tools. Parts 5 and 6 apply them to auctions.

Each part ends with a worked data-science application on simulated data, and the case study uses 2,388 real bids from US offshore oil and gas lease sales. Every number quoted in an article is printed by a script in this repository.

## The series

| Part | Topic | Application | Folder | Article |
|---|---|---|---|---|
| 1 | Payoffs, best responses and Nash equilibrium | A pricing game built from an estimated logit demand model | [`part1/`](part1) | [Medium](https://medium.com/@diogo-ribeiro-1975/game-and-auction-theory-for-data-scientists-6ed2048c7167) |
| 2 | Mixed strategies and zero-sum games | Audit allocation by linear programming; backtest against worst case; cycling in adversarial training | [`part2/`](part2) | [Medium](https://medium.com/@diogo-ribeiro-1975/game-and-auction-theory-for-data-scientists-792a563995aa) |
| 3 | Cooperative games and the Shapley value | Feature attribution and the choice of value function; sampling error | [`part3/`](part3) | [Medium](https://medium.com/@diogo-ribeiro-1975/game-and-auction-theory-for-data-scientists-part-3-of-6-cooperative-games-and-the-shapley-value-3572fa60117b) |
| 4 | Games of incomplete information | Adverse selection in pricing; strategic classification; a first bidding game | [`part4/`](part4) | [Medium](https://medium.com/@diogo-ribeiro-1975/game-and-auction-theory-for-data-scientists-92803a10d8a6) |
| 5 | Auction formats, bidding and revenue equivalence | Monte Carlo check of revenue equivalence and where it fails; bid shading from logs; the winner's curse | [`part5/`](part5) | [Medium](https://medium.com/@diogo-ribeiro-1975/game-and-auction-theory-for-data-scientists-6fe7bc42b211) |
| 6 | Reserve prices, values recovered from bids and position auctions | The optimal reserve and how its payoff depends on competition; values recovered from first-price bids; a reserve chosen from data; rule changes in position auctions | [`part6/`](part6) | [Medium](https://medium.com/@diogo-ribeiro-1975/game-and-auction-theory-for-data-scientists-f4b30bfb2516) |

All six parts are published.

## Case study on real data

| Topic | Data | Folder | Article |
|---|---|---|---|
| Minimum bids, rejected bids and thin competition in US offshore oil and gas lease sales, 2017 to 2026 | BOEM bid files for 13 Gulf lease sales: 2,388 sealed first-price bids | [`case-study/`](case-study) | Forthcoming |

## Layout

Every part has the same structure:

```
partN/
├── games.py      computations; prints every number used in the article
├── figures.py    draws the figures from the same functions
├── output.txt    reference output of games.py
└── figures/      PNG files used in the article
```

The main script is called `games.py` in Part 1; later parts name it after their subject. The `case-study/` folder also has a `data/` folder, with BOEM's raw files and the bid table built from them, and a `build_dataset.py` script that builds and checks that table.

## Running the code

The scripts need Python 3.9 or later and three packages:

```bash
pip install numpy scipy matplotlib
```

They were last run with Python 3.13, NumPy 2.5, SciPy 1.18 and Matplotlib 3.11.

To reproduce Part 1:

```bash
cd part1
python games.py > output.txt   # about 10 seconds
python figures.py              # writes four PNG files to figures/
```

To reproduce Part 2:

```bash
cd part2
python zero_sum.py > output.txt   # about 5 seconds
python figures.py                 # writes five PNG files to figures/
```

To reproduce Part 3:

```bash
cd part3
python shapley.py > output.txt   # about 5 seconds
python figures.py                # writes four PNG files to figures/
```

To reproduce Part 4:

```bash
cd part4
python incomplete_information.py > output.txt   # about 5 seconds
python figures.py                               # writes four PNG files to figures/
```

To reproduce Part 5:

```bash
cd part5
python auctions.py > output.txt   # about 5 seconds
python figures.py                 # writes four PNG files to figures/
```

To reproduce Part 6:

```bash
cd part6
python auction_data.py > output.txt   # about 30 seconds
python figures.py                     # writes four PNG files to figures/
```

To reproduce the case study, which also needs `pandas` and `pdfplumber`:

```bash
pip install pandas pdfplumber
cd case-study
python build_dataset.py              # reads data/raw, writes data/bids.csv and data/build_check.txt
python offshore_bids.py > output.txt # a few seconds
python figures.py                    # writes four PNG files to figures/
```

The case study was run with pandas 3.0 and pdfplumber 0.11.

Random seeds are fixed, so `output.txt` should come out the same as the committed copy, up to floating-point differences in the last printed digit across library versions.

## Part 1 in brief

`part1/games.py` does five things:

1. Finds best responses, pure-strategy Nash equilibria and the result of iterated elimination of strictly dominated actions for any two-player game given as a pair of payoff matrices, and applies them to three small games.
2. Builds a duopoly pricing game from a logit demand model with an outside option, using the baseline parameters of Calvano, Calzolari, Denicolò and Pastorello (2020), on a 19 by 19 price grid and in continuous prices.
3. Simulates a randomised price experiment (400 store-days, 500 customers per day) and fits the demand model by maximum likelihood.
4. Bootstraps the experiment to put intervals on the equilibrium prices implied by the fitted model.
5. Compares a profit forecast that holds the rival's price fixed with realised profit once the rival replies.

The main results, per 1,000 potential customers:

| Quantity | Value |
|---|---|
| Nash equilibrium price (cost = 1) | 1.473 |
| Joint-profit-maximising price | 1.925 |
| Profit at equilibrium / at the joint maximum | 222.9 / 337.5 |
| Firm 1's forecast gain from re-optimising against a fixed rival | +8.5% |
| Firm 1's realised change after the rival's first reply | −25.5% |
| Firm 1's realised change at equilibrium | −29.7% |

All data in Part 1 are simulated. The pricing scenario is constructed to illustrate the method and is not an empirical claim about any market.

## Part 2 in brief

`part2/zero_sum.py` does five things:

1. Solves any two-player zero-sum game by linear programming, returning both players' optimal mixed strategies and the value, computed from two separate programmes.
2. Sets up an inspection game in which an auditor inspects one of six channels and a fraudster operates in one, and checks the linear-programming solution against a closed form.
3. Builds five inspection policies from a simulated log of fraud attempts and scores each on a backtest with bootstrap intervals and on its worst case against a fraudster who knows the policy.
4. Runs fictitious play and multiplicative weights on the same game for 100,000 rounds and tracks the exploitability of average play.
5. Compares simultaneous, alternating and extragradient steps on the bilinear game f(θ, φ) = θφ.

The main results, in expected loss per fraud attempt:

| Quantity | Value |
|---|---|
| Value of the six-channel game | 25.59 |
| Loss with no inspection | 40 |
| Backtest loss of the four policies confined to the attacked channels | 25.58 to 25.61 |
| Worst-case loss of those four policies | 25.59 to 36.00 |
| Exploitability of average play after 100,000 rounds, fictitious play / multiplicative weights | 0.11 / 0.06 |

All data in Part 2 are simulated. The stakes and detection rates are invented for the example.

## Part 3 in brief

`part3/shapley.py` does four things:

1. Computes exact Shapley values for any cooperative game by enumerating coalitions, estimates them by sampling orderings with standard errors, and checks whether a division is in the core.
2. Applies this to a three-channel marketing game, listing the marginal contributions under all six orderings.
3. Attributes one prediction of a linear model with correlated normal features under two value functions, conditional and interventional, both computed exactly, and repeats the comparison over a population and across correlations.
4. Measures the sampling error of permutation estimates on a twelve-feature game with interactions: interval coverage, ranking stability and error by number of orderings.

The main results:

| Quantity | Value |
|---|---|
| Shapley values of search, social and email (total 220) | 118.3, 73.3, 28.3 |
| Attribution to a feature with coefficient zero, interventional / conditional | 0.00 / 1.08 |
| Its share of total absolute attribution at correlation 0.9, conditional game | 28.7% |
| Interventional evaluation points outside the 99% region of the data at correlation 0.9 | 32.4% |
| Coverage of nominal 95% intervals with 100 orderings | 94.9% |
| Runs in which all twelve features are ranked correctly with 100 orderings | 12.9% |

All data in Part 3 are simulated, and the channel conversions are invented.

## Part 4 in brief

`part4/incomplete_information.py` does three things:

1. Simulates an insurer that cannot observe its customers' risk and reprices each year on the customers who bought the year before, computes the fixed point of that process exactly, and shows how break-even coverage depends on the share of risk variance a rating model explains.
2. Simulates applicants who can raise a reported score by a privately known amount, and compares a threshold learned on historical data with the threshold that minimises errors once applicants respond to it.
3. Checks the Bayesian Nash equilibrium of a first-price auction numerically, by computing for every type the gain from the best deviation.

The main results:

| Quantity | Value |
|---|---|
| Share of customers insured at launch / at the fixed point of repricing | 60.9% / 10.4% |
| Break-even premium at the fixed point, relative to the population's mean cost | 2.15 times |
| Break-even coverage with a rating model explaining 0%, 50%, 90% of the variance of log risk | 10.4%, 36.7%, 92.6% |
| Classification errors at the learned threshold before / after it is published | 0.0% / 17.4% |
| Errors at the best published threshold | 8.1% |
| Largest gain from deviating: truthful bidding / equilibrium formula (two bidders, uniform values) | 0.25 / 0.00 |

All data in Part 4 are simulated. The risk distribution, the willingness to pay and the manipulation costs are chosen for the example.

## Part 5 in brief

`part5/auctions.py` does four things:

1. Computes the expected payoff of every bid in a second-price auction against three distributions of rival bids, showing that bidding the value is best in each.
2. Simulates first-price and second-price auctions to compare the seller's revenue with symmetric risk-neutral bidders, with risk-averse bidders and with one strong and one weak bidder, checking each first-price equilibrium by the gain from the best deviation.
3. Chooses a buyer's bids from logs of past competing bids of different sizes and measures both the surplus they earn and how much the log overstates it.
4. Simulates the winner's curse in a common-value auction, computes the equilibrium shade, and reproduces the same selection effect when the best of several noisy estimates is picked.

The main results:

| Quantity | Value |
|---|---|
| Revenue, first price / second price, five symmetric bidders with uniform values | 0.6666 / 0.6665 |
| First-price revenue advantage with risk-averse bidders / with one strong and one weak bidder | +11.1% / +10.1% |
| First-price auctions won by the lower-value bidder, asymmetric case | 8.4% |
| Surplus earned by bids chosen from a log of 50 / 5,000 auctions, relative to the best possible | 97.4% / 99.9% |
| Amount by which the log overstates that surplus, 50 / 5,000 auctions | 9.4% / 0.4% |
| Naive winner's expected profit in a common-value auction with 2 / 5 / 20 bidders, in noise standard deviations | +0.57 / −0.50 / −1.41 |

All data in Part 5 are simulated.

## Part 6 in brief

`part6/auction_data.py` does three things:

1. Computes the seller's expected revenue with a reserve price for any number of bidders, finds the optimal reserve from the virtual value, and compares it with the revenue from one additional bidder and no reserve.
2. Simulates 500 first-price auctions with two bidders, recovers each bidder's value from the bids with the inversion of Guerre, Perrigne and Vuong (2000), chooses a reserve from the recovered values and from the raw bids, and scores both on the true revenue curve, with a bootstrap and 200 independent datasets.
3. Sets up a position auction under the generalised second-price rule, shows that truthful bidding is not an equilibrium, computes the lowest-revenue envy-free equilibrium and checks that its payments are those of the Vickrey-Clarke-Groves mechanism, inverts equilibrium bids into values, and compares a forecast of a rule change made by replaying logged bids with the equilibrium outcome.

The main results:

| Quantity | Value |
|---|---|
| Optimal reserve for values with a beta(2, 3) distribution, any number of bidders | 0.333 |
| Revenue gain from the optimal reserve with 2 / 3 / 5 bidders | +12.3% / +2.4% / +0.2% |
| Root mean squared error against true values: bids read as values / recovered values | 0.2017 / 0.0177 |
| Reserve chosen from raw bids / from recovered values | 0.192 / 0.343 |
| Share of the attainable revenue gain captured, raw bids / recovered values | 47.4% / 99.6% |
| The same share on average over 200 datasets | 50.6% / 97.1% |
| Simulated position auctions in which truthful bidding is not an equilibrium | 79.0% |
| Revenue change from a switch to VCG, forecast by replaying bids / in equilibrium | −7.5% / 0.0% |

All data in Part 6 are simulated. Recovered values are biased downward for bids within one kernel bandwidth of the highest bid (6.6% of bids in the dataset the article follows); the script prints the size of that bias.

## Case study in brief

`case-study/build_dataset.py` reads the files BOEM publishes for each lease sale (fixed-format text files for sales 249 to 261, bid recaps in PDF for BBG1 to BBG3), collapses joint bids into single bids, attaches the minimum bid and the outcome of BOEM's review of each high bid, and checks the result against the totals in BOEM's sale summaries: all 64 published totals match. `case-study/offshore_bids.py` then measures competition, the position of high bids relative to the minimum bid, the disagreement between bidders on contested blocks, what happened to blocks whose high bid BOEM rejected, and the break-even for a higher minimum bid. Sources and file descriptions are in `case-study/data/README.md`; BOEM states that its materials are generally in the public domain.

| Quantity | Value |
|---|---|
| Blocks receiving bids that drew a single bid | 90.1% |
| Median high bid relative to the minimum bid | 1.24 times |
| High bids within 10% of the minimum | 39.3% |
| Median high bid / second bid on contested blocks | 1.65 |
| High bids rejected by BOEM, sales with published reviews | 85 of 2,062 (4.1%) |
| Rejected blocks that later drew a higher bid / a higher bid that was accepted | 69.4% / 58.8% |
| Share of the bidders below $150 per acre who must stay for a $150 deep-water minimum to raise bonus revenue | 77% |

## Author

Diogo Ribeiro
ORCID: [0009-0001-2022-7072](https://orcid.org/0009-0001-2022-7072)
Contact: dfr@esmad.ipp.pt
