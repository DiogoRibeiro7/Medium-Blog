# The Monty Hall problem, in four parts

Companion code for a four-part series of Medium articles on the Monty Hall problem. Each part takes one question the usual retelling compresses into a single number, and every number quoted in an article is printed by a script in this folder.

## The series

- **Part 1 — history and the calculations behind it** ([`part1/`](part1)). Bertrand's boxes (1889), Gardner's three prisoners (1959), Selvin's letters (1975), the Parade columns (1990–91), the 1991 dispute in *The American Statistician*. The standard game, the biased host and the ignorant host by exact enumeration; the n-door game in which the host opens k goat doors; how much evidence Monty Hall's 10 + 10 dining-room games and vos Savant's 200 + 200 classroom protocol actually carried; how long a sceptic's simulation has to run.
- **Part 2 — the answer is the protocol.** Biased, ignorant and adversarial hosts; the conditional question; Monty as a game; learning the host from data. Forthcoming.
- **Part 3 — the host is a selection mechanism.** Monty Hall as non-ignorable missingness, and the same structure in logged data. Forthcoming.
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
```

A GitHub Actions workflow (`.github/workflows/monty-hall.yml`) reruns each part on every change and checks that the output matches the committed `output.txt`.

## Sources for the experiments in Part 1

- Monty Hall's dining-room games (10 staying: 4 cars; 10 switching: 8 cars): J. Tierney, "Behind Monty Hall's Doors: Puzzle, Debate and Answer?", *The New York Times*, 21 July 1991, p. 1.
- The classroom protocol (200 games staying, then 200 switching, cups and a die): M. E. Brandt, "Don't Mess with Marilyn!", *The C Users Journal*, July 1992, describing the follow-up *Parade* column.

## Contact

Diogo Ribeiro — dfr@esmad.ipp.pt — ORCID 0009-0001-2022-7072
