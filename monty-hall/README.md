# The Monty Hall problem, in four parts

Companion code for a four-part series of Medium articles on the Monty Hall problem. Each part takes one question the usual retelling compresses into a single number, and every number quoted in an article is printed by a script in this folder.

## The series

- **Part 1 — history and the calculations behind it** ([`part1/`](part1)). Bertrand's boxes (1889), Gardner's three prisoners (1959), Selvin's letters (1975), the Parade columns (1990–91), the 1991 dispute in *The American Statistician*. The standard game, the biased host and the ignorant host by exact enumeration; the n-door game in which the host opens k goat doors; how much evidence Monty Hall's 10 + 10 dining-room games and vos Savant's 200 + 200 classroom protocol actually carried; how long a sceptic's simulation has to run.
- **Part 2 — which rules of the game change the decision** ([`part2/`](part2)). The host's rules as four dials (tie-break, knowledge, placement, offer policy): exact posteriors for a general host, which dials can make staying better, the expected value of perfect information of each, the game in which the host may decline to offer (value max(1/3, m − 1/3) when he must offer in a share m of games), and a Bayesian contestant who learns the offer policy from past episodes.
- **Part 3 — the host is a missingness mechanism** ([`part3/`](part3)). The game as data missing not at random (ignorable versus full likelihood, coarsening at random, the identification region), clips selected on the outcome, and real data with the same structure: the Coat ratings of Schnabel et al. (2016), 24 self-selected and 16 randomly assigned per user. The selection on the hidden rating, how much of it observed covariates absorb, a 100-split offline evaluation of six rating models (naive, reweighted and true error), how large a randomised sample the correction needs, and the logged propensities of the Open Bandit Dataset sample.
- **Part 4 — why people do not switch.** The laboratory evidence and simulated learners. Forthcoming.

## Layout

```
monty-hall/
├── README.md
├── requirements.txt       environment used for the committed outputs
└── partN/
    ├── <subject>.py       computations; prints every number used in the article
    ├── figures.py         draws the figures from the same functions
    ├── output.txt         reference output of the main script
    ├── figures/           PNG files used in the article
    └── article.md         the article text
```

## Run

```bash
pip install -r requirements.txt
cd part1
python history_numbers.py > output.txt   # about 20 s
python figures.py                        # about 10 s
cd ../part2
python protocol_numbers.py > output.txt  # about 1 min
python figures.py                        # about 1 min
cd ../part3
python missing_data_numbers.py > output.txt  # about 5 min; downloads the Coat and Open Bandit samples
python figures.py                            # about 2.5 min
```

A GitHub Actions workflow (`.github/workflows/monty-hall.yml`) reruns each part on every change and checks that the output matches the committed `output.txt`.

## Data for Part 3

Downloaded on first run, checked against SHA-256 digests and cached in `part3/data/` (not committed):

- Coat (Schnabel, Swaminathan, Singh, Chandak and Joachims, ICML 2016), from a pinned copy in the AutoDebias repository (https://github.com/DongHande/AutoDebias); original source https://www.cs.cornell.edu/~schnabts/mnar/
- Open Bandit Dataset, small sample shipped with the Open Bandit Pipeline (Saito, Aihara, Matsutani and Narita, 2021; Apache-2.0), from a pinned commit of https://github.com/st-tech/zr-obp

## Sources for the experiments in Part 1

- Monty Hall's dining-room games (10 staying: 4 cars; 10 switching: 8 cars): J. Tierney, "Behind Monty Hall's Doors: Puzzle, Debate and Answer?", *The New York Times*, 21 July 1991, p. 1.
- The classroom protocol (200 games staying, then 200 switching, cups and a die): M. E. Brandt, "Don't Mess with Marilyn!", *The C Users Journal*, July 1992, describing the follow-up *Parade* column.

## Contact

Diogo Ribeiro — dfr@esmad.ipp.pt — ORCID 0009-0001-2022-7072
