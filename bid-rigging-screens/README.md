# Screens for bid rigging on firms and countries a model has not seen

Computational companion to a Medium article. Question: statistical screens for bid rigging combined with machine learning classify cartel and competitive bidding with about 90% accuracy when they are evaluated on a random split of the same data. A competition agency uses them differently: on firms the model has not seen, in markets with a different auction format, and where cartels are a small share of what is screened. How much of the accuracy survives each of those changes, and how does the model compare with a single screen that needs no training?

## Run

```bash
python -m pip install -r requirements.txt
python fetch.py --out data   # Mendeley Data download (run by the data workflow; see below)
python screens.py            # results/summary.txt and results/*.csv (about 15 minutes on two cores)
python figures.py            # figures/fig1_*.png ... fig5_*.png
python -m pytest -q tests
```

`results/summary.txt` holds every number used in the article. The workflow `.github/workflows/bid-rigging-screens.yml` reruns the analysis and checks that it is unchanged.

## Data (`fetch.py`, `data/raw/`)

Wallimann, H. (2021). *Detecting bid-rigging coalitions in different countries and auction formats*, Version 1. Mendeley Data, [doi:10.17632/x8jwmbcpgn.1](https://doi.org/10.17632/x8jwmbcpgn.1), licence CC BY 4.0. The files were prepared for Imhof and Wallimann (2021), *International Review of Law and Economics* 68, 106016, which asks users of the data to cite that paper.

- `Swissdata.csv` and `Swissdata_addpred.csv`: three Swiss construction cartels (Ticino, See-Gaster, Graubünden), first-price sealed bids; 896 cartel and 646 competitive coalition-periods. The two files hold the same rows in a different order; the second adds percentiles of each screen and, for 745 rows from one market of 17 firms, the identifiers of the three firms. The analysis uses the second file and its 36 base screens.
- `Japanesedata.csv`: the Okinawa cartel, first-price sealed bids, 2003–2007; 207 cartel and 1,793 competitive coalition-periods, firm identifiers for all.
- `Italiandata.csv`: eight cartels in road works procured by the city of Turin in 2000–2003 by average-bid auctions, the validation data of Conley and Decarolis (2016); 1,474 cartel and 18,777 competitive coalition-periods (one empty row is dropped). The paper reports 21,340 competitive coalitions; the published file has 18,777.

A row is a coalition of three firms that bid together in at least three tenders, in one period (cartel or competitive). In each shared tender nine screens are computed from the three firms' bids, and each screen is summarised across tenders by its minimum, maximum, mean and median, giving 36 features. The screens are: coefficient of variation (`cvbid`), skewness (`skew1`), absolute difference between the two lowest bids (`D`, in the currency of the bids), relative distance (`RD1`), normalised distance (`RDnorm`), alternative distance (`RDalter`), percentage difference (`Diffperc`), spread (`Spread`) and the Kolmogorov-Smirnov statistic against a uniform distribution (`Kolmo_stat`). `number_project` is the number of tenders the coalition shares.

The workspace where the analysis was written cannot reach Mendeley Data, so `.github/workflows/bid-rigging-screens-data.yml` runs `fetch.py` on GitHub's runners and commits the files with `data/fetch_log.txt` (URL, status and SHA-256 of every response).

## Analyses (`screens.py`)

1. **Published protocol.** Balanced classes, random 75/25 split, 100 repetitions, random forest with 500 trees on the 36 screens; correct classification rates against those published in Imhof and Wallimann (2021, random forest with 1,000 trees).
2. **Single screens.** AUC of the median of each screen with no training, and the direction (higher or lower under collusion) in each data set.
3. **Unseen firms.** In Okinawa and the Swiss market with identifiers, 30% of firms are held out in each of 50 draws; the forest is trained on coalitions with none of them and tested on coalitions with one, two or three of them. Baselines: a random split with the same number of training rows, and the median coefficient of variation on the same test rows.
4. **Number of tenders.** How much the number of tenders shared by a coalition, which differs between the cartel and competitive periods, predicts the label by itself, and how the minimum and maximum of a screen depend on it.
5. **Another country.** Forests trained on one or two data sets and tested on the third; the same with screens converted to percentile ranks within each data set; logistic regressions on ranks; and the median coefficient of variation alone.
6. **Thresholds and investigation lists.** A threshold on the median coefficient of variation chosen in Switzerland and applied elsewhere; and the precision of the 5% and 10% of coalitions ranked most suspicious, at each data set's own share of cartels.

AUC is the probability that a randomly chosen cartel coalition scores as more suspicious than a randomly chosen competitive one.

## Caveats

- The label is the period. Cartel coalitions are observed while the cartel operated and competitive coalitions after it ended, so anything else that changed between the periods can help a classifier. The comparisons here hold the label fixed and change how the evaluation is done; they do not remove that limitation of the data.
- Firm identifiers exist for one Swiss market and for Okinawa only; the unseen-firm analysis is limited to them.
- The investigation lists use each data set's own share of cartels, which comes from how the data were assembled, not from the share of cartels in a real market.

Contact: Diogo Ribeiro, dfr@esmad.ipp.pt

## License

Apache-2.0, as the rest of the repository.
