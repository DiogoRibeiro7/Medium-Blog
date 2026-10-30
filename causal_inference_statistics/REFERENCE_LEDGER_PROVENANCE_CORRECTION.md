# Citation ledger provenance correction

Date: 2026-10-30.

## Finding

After PR #391, the ledger contained 173 records marked `verified`, but 25 had the literal placeholders `h` and `t` in the evidence URL and verification-note fields. These values do not identify a source or document a metadata check. A verified status without traceable evidence is not supportable.

## Correction

All 25 affected records have been returned to `pending`, with empty evidence fields. No reference strings, DOI fields, article content, figures, or simulations were changed. The corrected status is **148 verified / 62 pending / 210 distinct citation strings**.

This is a provenance correction, not a claim that the references are incorrect. Reverify each item against a publisher, DOI registration, recognized bibliographic index or original institutional record, checking the actual bibliographic fields, before promoting it again.

## Affected records

1. Freedman, D. A. (2008). On Regression Adjustments to Experimental Data. *Advances in Applied Mathematics*, 40(2), 180–193. https://doi.org/10.1016/j.aam.2006.12.003
2. Holland, P. W. (1986). Statistics and Causal Inference. *Journal of the American Statistical Association*, 81(396), 945–960. https://doi.org/10.1080/01621459.1986.10478354
3. Lin, W. (2013). Agnostic Notes on Regression Adjustments to Experimental Data: Reexamining Freedman's Critique. *The Annals of Applied Statistics*, 7(1), 295–318. https://doi.org/10.1214/12-AOAS583
4. Rosenbaum, P. R. and Rubin, D. B. (1983). The Central Role of the Propensity Score in Observational Studies for Causal Effects. *Biometrika*, 70(1), 41–55. https://doi.org/10.1093/biomet/70.1.41
5. Rubin, D. B. (1974). Estimating Causal Effects of Treatments in Randomized and Nonrandomized Studies. *Journal of Educational Psychology*, 66(5), 688–701. https://doi.org/10.1037/h0037350
6. White, H. (1980). A Heteroskedasticity-Consistent Covariance Matrix Estimator and a Direct Test for Heteroskedasticity. *Econometrica*, 48(4), 817–838. https://doi.org/10.2307/1912934
7. Abadie, A. and Imbens, G. W. (2006). Large Sample Properties of Matching Estimators for Average Treatment Effects. *Econometrica*, 74(1), 235–267. https://doi.org/10.1111/j.1468-0262.2006.00655.x
8. Abadie, A. and Imbens, G. W. (2008). On the Failure of the Bootstrap for Matching Estimators. *Econometrica*, 76(6), 1537–1557. https://doi.org/10.3982/ECTA6474
9. Austin, P. C. (2009). Balance Diagnostics for Comparing the Distribution of Baseline Covariates Between Treatment Groups in Propensity-Score Matched Samples. *Statistics in Medicine*, 28(25), 3083–3107. https://doi.org/10.1002/sim.3697
10. Austin, P. C. (2011). An Introduction to Propensity Score Methods for Reducing the Effects of Confounding in Observational Studies. *Multivariate Behavioral Research*, 46(3), 399–424. https://doi.org/10.1080/00273171.2011.568786
11. Caliendo, M. and Kopeinig, S. (2008). Some Practical Guidance for the Implementation of Propensity Score Matching. *Journal of Economic Surveys*, 22(1), 31–72. https://doi.org/10.1111/j.1467-6419.2007.00527.x
12. Ho, D. E., Imai, K., King, G. and Stuart, E. A. (2007). Matching as Nonparametric Preprocessing for Reducing Model Dependence in Parametric Causal Inference. *Political Analysis*, 15(3), 199–236. https://doi.org/10.1093/pan/mpl013
13. Imbens, G. W. (2004). Nonparametric Estimation of Average Treatment Effects Under Exogeneity: A Review. *Review of Economics and Statistics*, 86(1), 4–29. https://doi.org/10.1162/003465304323023651
14. King, G. and Nielsen, R. (2019). Why Propensity Scores Should Not Be Used for Matching. *Political Analysis*, 27(4), 435–454. https://doi.org/10.1017/pan.2019.11
15. Rosenbaum, P. R. and Rubin, D. B. (1985). Constructing a Control Group Using Multivariate Matched Sampling Methods That Incorporate the Propensity Score. *The American Statistician*, 39(1), 33–38. https://doi.org/10.1080/00031305.1985.10479383
16. Stuart, E. A. (2010). Matching Methods for Causal Inference: A Review and a Look Forward. *Statistical Science*, 25(1), 1–21. https://doi.org/10.1214/09-STS313
17. Cole, S. R. and Hernán, M. A. (2008). Constructing Inverse Probability Weights for Marginal Structural Models. *American Journal of Epidemiology*, 168(6), 656–664. https://doi.org/10.1093/aje/kwn164
18. Crump, R. K., Hotz, V. J., Imbens, G. W. and Mitnik, O. A. (2009). Dealing with Limited Overlap in Estimation of Average Treatment Effects. *Biometrika*, 96(1), 187–199. https://doi.org/10.1093/biomet/asn055
19. Horvitz, D. G. and Thompson, D. J. (1952). A Generalization of Sampling Without Replacement from a Finite Universe. *Journal of the American Statistical Association*, 47(260), 663–685. https://doi.org/10.1080/01621459.1952.10483446
20. Kang, J. D. Y. and Schafer, J. L. (2007). Demystifying Double Robustness: A Comparison of Alternative Strategies for Estimating a Population Mean from Incomplete Data. *Statistical Science*, 22(4), 523–539. https://doi.org/10.1214/07-STS227
21. Li, F., Morgan, K. L. and Zaslavsky, A. M. (2018). Balancing Covariates via Propensity Score Weighting. *Journal of the American Statistical Association*, 113(521), 390–400. https://doi.org/10.1080/01621459.2016.1260466
22. Lunceford, J. K. and Davidian, M. (2004). Stratification and Weighting via the Propensity Score in Estimation of Causal Treatment Effects: A Comparative Study. *Statistics in Medicine*, 23(19), 2937–2960. https://doi.org/10.1002/sim.1903
23. Westreich, D. and Cole, S. R. (2010). Invited Commentary: Positivity in Practice. *American Journal of Epidemiology*, 171(6), 674–677. https://doi.org/10.1093/aje/kwp436
24. Hernán, M. A., Hernández-Díaz, S. and Robins, J. M. (2004). A Structural Approach to Selection Bias. *Epidemiology*, 15(5), 615–625. https://doi.org/10.1097/01.ede.0000135174.63482.43
25. Pearl, J. (1995). Causal Diagrams for Empirical Research. *Biometrika*, 82(4), 669–688. https://doi.org/10.1093/biomet/82.4.669

## Further quality gate

A future ledger validation should reject any `verified` row whose source is missing, non-URL or placeholder, and should test bibliographic fields against the cited source. The audit count must not be presented as a publication-level certification: matching article claims to sources remains separate work.
