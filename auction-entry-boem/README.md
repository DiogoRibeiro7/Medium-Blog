# One more bidder or a higher minimum bid: BOEM's Gulf lease sales, 2013–2026

Computational companion to a Medium article. Question: in BOEM's offshore oil and gas lease sales nine tracts in ten receive a single sealed bid. Bulow and Klemperer (1996) showed that, in the standard auction model, one more bidder is worth more to a seller than the best reserve price. How much is one more bidder worth in these data, and how does that compare with what a higher minimum bid could add?

The analysis separates two things the raw comparison mixes: the mechanical gain from one more sealed bid on the same tract, and the fact that the tracts that attract several bids are different tracts.

## Run

```bash
python -m pip install -r requirements.txt
python fetch.py --out data   # downloads (run by the data workflow; see below)
python build_dataset.py      # data/bids.csv and data/build_check.txt
python entry.py              # results/summary.txt and results/*.csv
python figures.py            # figures/fig1_*.png ... fig3_*.png
python -m pytest -q tests
```

`results/summary.txt` holds every number used in the article. The workflow `.github/workflows/auction-entry-boem.yml` rebuilds the bid table from BOEM's files, reruns the analysis and checks that `data/bids.csv`, `data/build_check.txt` and `results/summary.txt` are unchanged.

## Data (`fetch.py`, `build_dataset.py`, `data/`)

- **Sales 249 (August 2017) to BBG3 (August 2026):** the parsed table and BOEM files of the case study in the game- and auction-theory series, copied from [gitlab.com/DiogoRibeiro7/game-auction-theory-ds](https://gitlab.com/DiogoRibeiro7/game-auction-theory-ds) (folder `case-study`; the commit is in `data/case_study/SOURCE`). That table was checked there against BOEM's published totals for every sale.
- **Sales 231, 233, 235, 238, 241, 246, 247 and 248 (August 2013 to March 2017):** BOEM's own files for each sale (`https://www.boem.gov/Sale-<n>/`): the archive of fixed-format ASCII files (`PREBID`, `BID`, `MAPS`), the sale statistics and the final bid recap, in `data/boem/<n>/`. The ASCII files are read with the record layout of the case study's parser. Every sale's tracts, bids, sum of high bids and sum of all bids match BOEM's sale statistics exactly; Sale 233, which has no statistics file, matches BOEM's press release of 28 August 2013. Rejected high bids are read from the `R` marks of each final recap and checked against the recap's own count. Sale 247 has no final recap on its page, so its review outcome is unknown.

The workspace where the analysis was written cannot reach gitlab.com or boem.gov, so `.github/workflows/auction-entry-boem-data.yml` runs `fetch.py` on GitHub's runners and commits the files with `data/fetch_log.txt` (URL, status and SHA-256 of every response). BOEM states that material produced by federal agencies is generally in the public domain (https://www.boem.gov/copyright).

The minimum bid is $25 per acre, or fraction of an acre, in water shallower than 400 m and $100 per acre at 400 m or deeper in every sale here; BOEM raised the deep-water figure from $37.50 in Sale 218 (December 2011).

## Definitions

- **Tract and block.** A tract is a block offered in one sale. The same block appears in several sales when it was not leased, when its high bid was rejected, or when an earlier lease expired.
- **Ratio.** A bid divided by the tract's minimum bid. Comparisons are made within sale and depth band (cells), so that oil prices, sale terms and the $25/$100 minimum are held fixed.
- **Contested tract.** A tract with two or more bids. One tract whose high bid was below the minimum (BOEM's "invalid bid" in Sale 231) is left out.
- **Fixed-effect multipliers.** exp of the coefficient on a contested indicator in a regression of the log ratio with cell effects (and block effects, or company-by-sale effects), fixed effects absorbed by alternating projections, standard errors clustered by block.
- **A second bidder as a second draw.** (a) Each single-bid tract receives one more bid drawn from the ratios of the single-bid tracts in its own cell; the expected high bid is the minimum times E[max(own ratio, draw)]. (b) One more bid as far from the first as the two bids are on two-bid tracts: sum of high bids over sum of the means of the two bids. Intervals from 1,000 bootstrap draws of tracts.
- **A higher minimum bid.** For single-bid tracts bid below a new minimum: best case, every one of those bidders raises its bid to the new minimum; worst case, every one walks away. The break-even share is the share that must stay for the change to pay.

## Caveats

- Bids are sealed, and bidders do not know how many others bid on a tract; only 1.9% of single-bid tracts were won at exactly the minimum. The second-draw benchmarks therefore assume that a bidder's bid does not respond to the competition that actually turns up, only to the competition it expects. If one more bidder were also expected, bids would rise more than the benchmarks show.
- Neither benchmark identifies what attracts a second bidder. The fixed-effect comparisons hold fixed a block's permanent quality, not its changing prospects between sales a median of 5.7 years apart.
- The minimum-bid bounds are bounds: the data do not show how many bidders below a new minimum would pay it.

Contact: Diogo Ribeiro, dfr@esmad.ipp.pt

## License

Apache-2.0, as the rest of the repository.
