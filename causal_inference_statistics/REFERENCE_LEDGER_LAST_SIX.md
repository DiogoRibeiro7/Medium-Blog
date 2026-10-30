# Reference ledger — historical and book source batch

Date: 2026-10-30.

## Verified metadata

Fifteen references were matched to original publisher pages, trusted library catalogues, journal issue records or institutional metadata. Each ledger row has a direct evidence URL and verification note. This certifies bibliographic metadata only, not manuscript claims or experiment reproducibility.

1. `5e5f86fabe10c8ac`: WorldCat first edition Fisher 1935 Oliver and Boyd. Source: https://search.worldcat.org/title/The-design-of-experiments/oclc/2417943
2. `be897d12ef69e453`: CiNii IMS Kang Schafer 2007 Statistical Science 22(4) DOI. Source: https://cir.nii.ac.jp/crid/1362544420582650880
3. `1b9471812874d5db`: WorldCat Kish 1965 Wiley book bibliographic record. Source: https://search.worldcat.org/title/Survey-sampling/oclc/569253346
4. `9ed2bb04445de7b7`: WorldCat VanderWeele 2015 Oxford book catalogue. Source: https://search.worldcat.org/title/Explanation-in-causal-inference-%3A-methods-for-mediation-and-interaction/oclc/913231727
5. `67c6902a0760ef6b`: JSTOR Journal of Business 1990 63(1) part 2 S125-S140. Source: https://www.jstor.org/stable/i340515
6. `7c78e50f3ec4de7b`: Statistica Sinica official 2001 volume 11 issue 4 table of contents pages 920-936. Source: https://www3.stat.sinica.edu.tw/statistica/j11n4/11-4.htm
7. `1ffc57dc35b3113f`: Econometrica Frisch Waugh 1933 original DOI. Source: https://doi.org/10.2307/1907330
8. `43ee0207e4652c95`: SIAM reference lists Neyman 1959 Cramer volume pages 213-234. Source: https://epubs.siam.org/doi/10.1137/1121003
9. `a658db6a23cce3c1`: Routledge Gustafson Bayesian partially identified models book metadata. Source: https://www.routledge.com/Bayesian-Inference-for-Partially-Identified-Models-Exploring-the-Limits-of-Limited-Data/Gustafson/p/book/9780367570538
10. `778944a50e4e4caa`: National Diet Library Robins 1986 Mathematical Modelling 7(9-12) 1393-1512. Source: https://ndlsearch.ndl.go.jp/books/R100000136-I1360011143883724544
11. `5d4d8bef9a29a6a7`: Springer van der Laan Robins 2003 book DOI. Source: https://link.springer.com/book/10.1007/978-0-387-21700-0
12. `b7ca8949a556e668`: SAGE Petersen et al 2012 positivity paper DOI. Source: https://journals.sagepub.com/doi/10.1177/0962280210386207
13. `286afdab45c24734`: Oxford JNCI Cornfield et al 1959 volume 22 issue 1 pages 173-203. Source: https://doi.org/10.1093/jnci/22.1.173
14. `8a0aeb1902ba40e9`: Routledge Schafer book copyright 1997 Chapman and Hall. Source: https://www.routledge.com/Analysis-of-Incomplete-Multivariate-Data/Schafer/p/book/9780367803025
15. `062025e4ebb9f9b4`: Springer Second Seattle Symposium Robins 2004 chapter pages 189-326. Source: https://link.springer.com/book/10.1007/978-1-4419-9076-1

## Counts

- Before: 189 verified, 21 pending, 210 distinct citation strings.
- After: **204 verified, 6 pending, 210 distinct citation strings**.

## Remaining and known defects

- `57e976391e403909` — Cochran, W. G. (1965). The Planning of Observational Studies of Human Populations. *Journal of the Royal Statistical Society: Series A*, 128(2), 234–266. https://doi.org/10.2307/2344179
- `975372e382093237` — Hajek, J. (1971). Comment on 'An Essay on the Logical Foundations of Survey Sampling'. In V. P. Godambe and D. A. Sprott (eds.), *Foundations of Statistical Inference*. Holt, Rinehart and Winston.
- `5bf3761043916309` — Spirtes, P., Glymour, C. and Scheines, R. (2000). *Causation, Prediction, and Search*, 2nd edition. MIT Press.
- `78055c6653d6daa2` — Rosenbaum, P. R. (1987). The Role of a Second Control Group in an Observational Study. *Statistical Science*, 2(3), 292–306. https://doi.org/10.1214/ss/1177013224
- `415d64b8a308f7f9` — Zivich, P. N., Cole, S. R. and Westreich, D. (2022). Transportability and Generalizability of Causal Effects: A Practical Guide. *American Journal of Epidemiology*, 191(4), 722–728.
- `3a50df32bb911eae` — Chernozhukov, V., Demirer, M., Duflo, E. and Fernández-Val, I. (2023). Generic Machine Learning Inference on Heterogeneous Treatment Effects in Randomized Experiments. *Journal of Political Economy*, 131(7), 1805–1851. https://doi.org/10.1086/722086

The remaining six are deliberately not promoted. Specific discrepancies to resolve:

1. **Cochran (1965)**: The DOI `10.2307/2344179` identifies the article at pages **234–255**; the ledger citation says 234–266, which includes a separate discussion (Oxford Academic issue listing). Correct the article bibliography in all occurrences and regenerate deterministic ledger IDs.
2. **Chernozhukov et al. (2023)**: DOI `10.1086/722086` actually identifies Salvanes, Willage and Willén's *The Effect of Labor Market Shocks across the Life Cycle* in *Journal of Labor Economics* (2024), not the cited causal-ML paper. The causal-ML study's final published version is in *Econometrica* **93(4), 1121–1164 (2025)**, DOI `10.3982/ECTA19303`, under the expanded title beginning *Fisher–Schultz Lecture: Generic Machine Learning Inference...*. Resolve whether the article intends the 2023 working-paper revision or the 2025 version of record; correct every affected bibliography entry accordingly.
3. **Rosenbaum (1987)**: The ledger DOI `10.1214/ss/1177013224` appears inconsistent with the author's publication list, which links the paper to `10.1214/ss/1177013232`. Confirm article versus discussion pagination and DOI before promotion.
4. **Spirtes et al. (2000)**: second-edition copyright 2000 is supported by library records, but MIT Press's current second-edition listing uses a 2001 publication date. State which edition/printing is intended.
5. **Hájek (1971)** and **Zivich et al. (2022)**: require direct source-to-citation reconciliation rather than guessing publication details.

No manuscript text, figures, simulations, or existing citation metadata was changed in this PR. The defects above are documented for a focused correction PR.
