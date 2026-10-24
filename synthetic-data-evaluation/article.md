# Evaluating Synthetic Tabular Data: Fidelity, Utility, Privacy and Valid Inference

*Six generators fitted to the Adult census extract, scored against an independent real sample on three axes, and then asked a harder question: do confidence intervals computed from their output cover the truth? Mostly they do not, and the reason is arithmetic, not realism.*

In February 2024 I published [a survey of techniques for generating synthetic data](https://towardsdev.com/advanced-techniques-for-generating-synthetic-data-bf040ff8de66): duplication, resampling, the bootstrap, augmentation, SMOTE, simulation from distributions, ARIMA, differential privacy, GANs and VAEs. Every section ended at the same place, with a code snippet that produced an output. None of them asked the question that decides whether producing that output was worth doing: judged against what?

That question is the subject of this sequel. A synthetic table is an estimate of a joint distribution, released in the format of the data it was estimated from. Its rows look like records, so it invites being treated as records: summarised, modelled, shared, tested. Whether that treatment is legitimate depends on three things that a single "realism" score cannot capture, and on a fourth that is usually not checked at all. The three are fidelity (does the table reproduce the structure of the real data?), utility (does an analysis run on it give the answer the real data would give?) and privacy (does it reveal the people it was learned from?). The fourth is inferential validity: when an analyst computes a standard error from a synthetic table, does it mean what a standard error means?

The argument of the article can be stated in advance. Quality is not a property of a synthetic table. It is a property of the table relative to a question and a yardstick, and the three axes pull against each other, so a generator that is best on one is rarely best on the others. On the fourth axis the answer is sharper: an interval computed from synthetic data as if it were real is too narrow by a factor that depends on how the data were generated, not on how realistic they look, and making the synthetic table bigger makes the interval worse, not better. That factor can be derived in a few lines, it can be corrected for some generators, and for others nobody knows how to correct it.

## Three corrections to the 2024 code

Before building on the earlier article, three of its examples need correcting, because they share a flaw that turns out to be the theme of this one.

The ARIMA example fitted a model and called `forecast(steps=20)`, describing the result as synthetic data. A forecast is the conditional mean of the future path. It is a smooth curve with none of the noise the series actually has, so as "synthetic data" it has the right level and the wrong variance. To generate synthetic paths one simulates from the fitted model, drawing new innovations at each step (in statsmodels, the `simulate` method of the fitted results), and to respect the uncertainty in the fitted coefficients one would also redraw them. The snippet also plotted `len(foreach)`, which raises a `NameError`.

The bootstrap example drew a thousand resamples, averaged each, and printed the average of the averages as the "estimated mean". That number is the sample mean again, up to simulation noise. What the bootstrap buys is the spread of those thousand averages, a standard error or an interval. The example computed the one thing the bootstrap does not add and discarded the one thing it does.

The GAN example stacked dense layers without activation functions between them. A composition of affine maps is affine, so the generator was tanh(Az + b) whatever the number of layers drawn, and the discriminator was a logistic regression. Such a generator can only produce transformed Gaussian noise.

There were smaller problems too: `random.sample` over a whole list is a permutation, not new data; replacing categories at random breaks the relation between a record's features and its label; and the differential-privacy call should have been `diffprivlib.tools.histogram`, which returns the counts before the bin edges. A differentially private histogram is a private release, and it only becomes synthetic data once records are sampled from it.

The first two corrections share a pattern. In both, something that looked like data, a forecast path or a resampled mean, was reported without the variability it should carry. Part two of this article is the same mistake at full scale: a synthetic table looks like a sample, and an analyst who treats it as one inherits an uncertainty statement that is wrong.

## Two families of generator

It helps to separate generators by where their information comes from.

A mechanistic generator starts from equations that describe how the data come about, with noise and parameters added: an epidemic model, a queue, a stochastic differential equation for an interest rate, an agent-based market, a physics engine. The mechanism brings information from outside the data, namely theory, and that buys three things. The ground truth is known, every parameter and every causal link. The generator can produce regimes the real data never covered. And it can produce interventions: change a parameter, rerun, and the output is a counterfactual. The price is that its errors are systematic. More simulations shrink the Monte Carlo error and leave the model error untouched, which is the sim-to-real gap.

A data-driven generator estimates the distribution of the data at hand: a copula, a sequence of trees, an interpolation scheme, a neural network. It can only give back what is in the data, smoothed and with error. It knows nothing about interventions or unobserved regimes. Its errors are of two kinds, a bias from the family of distributions it can represent and a variance from having been estimated on a finite sample, and the second kind is the one that part two is about.

This article uses both. The Adult census extract is synthesised with data-driven generators, because that is the setting in which synthetic data are released. The coverage experiments need a parameter whose true value is known exactly, and the cleanest way to have one is a known mechanism, so the inference study starts there and returns to Adult afterwards with the whole file playing the population.

## The experiment

The data are the Adult extract of the 1994 US Census prepared by Barry Becker and Ronny Kohavi (Kohavi 1996), here in the copy distributed with the PMLB benchmark collection (Romano et al. 2022): 48,842 records and fifteen columns. Six are numeric (age, the census sampling weight `fnlwgt`, years of education, capital gain, capital loss and weekly hours) and nine are categorical (work class, education level, marital status, occupation, relationship in the household, race, sex, country of birth and whether income exceeds $50,000, which holds for 23.9% of records). Adult is one of the most common benchmarks in the synthetic tabular data literature, which is a reason to use it: the results here can be read against published ones.

The file is split at random into three disjoint thirds of about 16,280 records. The *training* third plays the confidential data; every generator sees only this. The *holdout* third is the test set for predictive utility and the set of non-members for the privacy tests. The *reference* third is the piece most evaluations leave out: an independent real sample of the same size, drawn from the same population. It is the score an ideal generator would get. Two real samples differ from each other, and without that baseline there is no way to tell whether a synthetic table that differs from the training data by some amount differs by more than sampling noise.

Six generators produce one synthetic table each, of the same size as the training third.

*Independent marginals* resamples every column on its own. It reproduces each column's distribution exactly and destroys every relationship between columns, which makes it a useful floor.

The *Gaussian copula* keeps each column's empirical distribution and joins them through a multivariate normal on transformed scores (Sklar 1959; the approach of the Synthetic Data Vault, Patki et al. 2016). Numeric columns enter through the randomised probability integral transform, so a point mass such as the 91.7% of zero capital gains is spread over its probability interval rather than piled at one score, and categorical columns are ordered by frequency and given an interval of [0, 1] in proportion to their share. A copula of this kind captures pairwise correlation on the normal-score scale and nothing else.

*Sequential CART* follows Reiter (2005) and the synthpop package (Nowok, Raab and Dibben 2016). The first column is drawn from its marginal; every later column is modelled by a classification or regression tree grown on the columns before it, with leaves of at least five records, and its synthetic value is copied from a real record drawn at random from the leaf the synthetic record falls into. Income comes last, so it is synthesised conditionally on everything else.

*SMOTE* is used as a whole-table generator, as in the benchmarks of Kotelnikov et al. (2023): within each income class, a synthetic record lies on the segment between a real record and one of its five nearest real neighbours, with numeric columns interpolated and each categorical column taken from one end or the other (Chawla et al. 2002).

*CTGAN* (Xu et al. 2019) and *TVAE*, from the same paper and the same `ctgan` package, are the deep generators: a conditional GAN with mode-specific normalisation of numeric columns, and a variational autoencoder. Each was trained for the default 300 epochs on a CPU.

Two more tables complete the comparison. The *training data itself*, released as if it were synthetic, is the degenerate generator with perfect fidelity and no privacy at all. The *independent real sample* is the reference third.

The results come in two parts. Part one scores these tables on fidelity, utility and privacy; part two turns to inference. All code, the synthetic tables of the two deep generators and every number quoted below are in the [companion folder of my Medium-Blog repository](https://github.com/DiogoRibeiro7/Medium-Blog/tree/main/synthetic-data-evaluation); a workflow reruns the whole analysis on every change and checks that the results are reproduced.

## Fidelity: what the table looks like

Fidelity is the axis that synthetic data products usually lead with, and the first lesson of the experiment is how little the most common fidelity measure says. Compare each column of a synthetic table with the same column of the training data, by the Kolmogorov–Smirnov statistic for numeric columns and the total variation distance for categorical ones, and average over the fifteen columns. The independent real sample scores 0.0085. Independent marginals score 0.0050, which is *better* than a second real sample, because resampling a column reproduces it up to resampling noise. The copula scores 0.0059 and CART 0.0051. Among the generators that are not neural networks, SMOTE is the worst at 0.0203, because interpolating between records invents values the data never had, most visibly in weekly hours, where the real distribution has a spike at 40 that interpolation smears. A generator that keeps no relation between any two columns wins the marginal comparison. Marginals are a necessary check and nearly worthless as evidence.

The two deep generators are much further away: 0.167 for CTGAN and 0.116 for TVAE. Nearly all of it comes from the capital columns. In the training data 91.7% of capital gains are exactly zero. In CTGAN's table 0.7% are, 7.7% are negative, and 55% of capital losses are negative; TVAE puts 13.7% of capital gains at zero and makes 32% of them negative. A network that models a numeric column as a mixture of continuous modes, as both do, spreads a point mass over a small interval around it, and the Kolmogorov–Smirnov statistic, which compares cumulative distributions, sees the whole point mass go missing. These are the models with the package defaults, trained for 300 epochs on a CPU. Tuning, longer training and the post-processing that libraries built on them add, rounding and clipping to the observed range, would remove some of these defects. The point of including them is not to rank generators but to see what each measure can and cannot detect.

The next level is the joint distribution of pairs. Cut each numeric column at the deciles of the training data, form all 105 two-way tables, and average the total variation distance between the synthetic and the training version of each. The independent real sample gives 0.021, the floor that sampling noise imposes. CART gives 0.015, below the floor, which is the first sign of a problem discussed under privacy: a table can only be closer to the training data than a fresh sample is if it is partly made of the training data. SMOTE gives 0.039, the copula 0.067 and independent marginals 0.077. TVAE gives 0.200 and CTGAN 0.211, the highest of all, largely because their capital columns are wrong in every pair they enter. On this measure, the copula preserves very little more than independence. Its correlation matrix on normal scores is a poor summary of how, for example, relationship, marital status and sex determine each other.

Those three columns make the point more sharply than any distance. In the census coding, a "husband" is male and a "wife" is female, and the education level and the years of education are two codings of one variable. The real data obey these rules almost exactly: no female husbands, 0.4% of wives coded male, and no record whose years of education disagree with its education level. The synthetic tables do not all obey them. Under independent marginals 34% of husbands are women and 82% of records have an education level that contradicts their years of education. The copula makes 19% of husbands women and 80% of education pairs inconsistent; it has no way to represent a deterministic relation between a categorical column and a numeric one. CART reproduces all three rules exactly, because each tree sees the earlier column and finds the deterministic split. SMOTE gets the sex rules right and 4.7% of education pairs wrong, the records interpolated between neighbours with different education. CTGAN makes 12% of husbands female, 16% of wives male and 24% of education pairs inconsistent; TVAE 1.8%, 26% and 15%. And 58% of CTGAN's rows and 32% of TVAE's contain a numeric value outside the range of the real data, mostly a negative capital gain or loss, which no record in the census reports. Logical constraints are cheap to check, they are known to anyone who knows the data, and they catch failures that no distribution distance at the level of pairs of columns can be expected to flag with the same clarity.

The most general fidelity test is a discriminator: train a classifier to tell real rows from synthetic ones and see how well it does. This is the propensity-score measure of Woo et al. (2009), the pMSE: the mean squared distance of the predicted probabilities from the share of synthetic rows. Snoke et al. (2018) derived its distribution under the hypothesis that the synthetic data come from the right model and turned it into a standardised utility measure. Here the classifier is gradient boosting, and the probabilities are cross-fitted over five folds so that no row is scored by a model that saw it. With equal numbers of real and synthetic rows an AUC of 0.5 means the classifier cannot tell them apart. The independent real sample scores 0.498, as it should. Independent marginals score 0.993 and the copula 0.987: a classifier separates their rows almost perfectly. SMOTE scores 0.750. CTGAN scores 0.999 and TVAE 0.991: their rows are even easier to tell apart from real ones than the copula's, and the capital columns alone would give them away.

Then something odd. CART scores 0.446, and the training data itself, released as if it were synthetic, scores 0.107. An AUC below one half is not better than indistinguishable; it means the classifier is systematically wrong, and the mechanism is copying. When a synthetic row is a copy or a near-copy of a real row, cross-validation puts one of the pair in the training folds with one label and the other in the test fold with the opposite label, and a flexible classifier that has memorised the first predicts the wrong label for the second. Pair-aware cross-fitting removes the effect: if each synthetic row is assigned to the same fold as its nearest real row, the training copy scores exactly 0.500 and CART 0.483. The pMSE is a worse alarm still, because squared deviations from one half do not care about direction: the training copy has a pMSE of 0.005, less than the independent real sample's 0.006. A discriminator AUC below 0.5, or a pMSE below the value for a real holdout sample, is a warning about privacy, not a compliment about fidelity.

## Utility: what an analysis gets out of it

Fidelity asks whether the table looks right; utility asks whether an analysis run on it gives the answer the real data would give. Which analysis matters, and the two most common choices answer different questions.

The machine-learning convention is to train a model on synthetic data and test it on real data, "train on synthetic, test on real" (Esteban, Hyland and Rätsch 2017). Here the model is gradient boosting for income above $50,000, with early stopping on a slice of the synthetic training table, and the test set is the real holdout third. A model trained on the real training third reaches an AUC of 0.923. Trained on the independent real sample, 0.924. On CART's table, 0.911; on SMOTE's, 0.900; on the copula's, 0.855. On TVAE's, 0.882; on CTGAN's, 0.874. Independent marginals give 0.418, no better than chance: with no relation between the features and income in the synthetic table, early stopping ends the training almost at once, and what is left is noise, which here happened to point slightly the wrong way.

By this measure the copula is a respectable generator: it loses 0.07 of AUC, and a model trained on its output still ranks people by income far better than chance. The second convention, comparing the coefficients of a statistical model, tells a different story. Figure 1 fits the same logistic regression of income on age, years of education, weekly hours, sex and being married to each table and shows how far each synthetic coefficient lies from the real one, in units of the real estimate's standard error. The grey band marks the spread expected between two independent real samples of the same size, ±1.96·√2 standard errors; the independent real sample's own coefficients lie inside it, between −2.0 and +1.4.

![Figure 1](https://raw.githubusercontent.com/DiogoRibeiro7/Medium-Blog/main/synthetic-data-evaluation/figures/fig1_coefficients.png)

*Figure 1. Coefficients of a logistic regression of income > $50K fitted to each table, minus the coefficient fitted to the real training data, in standard errors of the real estimate (symmetric log scale). The grey band is ±1.96·√2, the spread expected between two independent real samples of the same size.*

The copula's coefficients for education and marriage are 21 and 31 standard errors away from the real ones: the synthetic data say that a year of education multiplies the odds of a high income by 1.17 where the real data say 1.46, and that marriage multiplies them by 1.7 where the real data say 10.2. The same table that trains a decent classifier gives a qualitatively wrong account of the relation between income and its predictors, because a classifier can do well on ranking while being calibrated badly and getting individual relations wrong, and gradient boosting finds whatever signal the copula left. SMOTE's coefficients are between 1.1 and 7.9 standard errors off. CART's lie between −2.5 and +1.6, about as far from the training estimate as a second real sample. CTGAN's lie between −5.7 and +2.4 and TVAE's between −7.7 and +13.9; TVAE says that marriage multiplies the odds of a high income by 23. The interval-overlap measure of Karr et al. (2006), the average of the shares of the real and the synthetic 95% interval covered by their intersection, ranks the tables the same way: 0.74 on average over the five coefficients for the independent real sample, 0.58 for CART, 0.29 for SMOTE, 0.13 for both deep generators and 0.08 for the copula and for independent marginals.

The two conventions are what Snoke et al. (2018) call specific utility, the quality of one analysis, as opposed to general utility, a property of the whole distribution measured by the discriminator. Neither is a property of the table alone. A synthetic table good enough to develop and test a classifier on may be useless for estimating an effect, and the converse can happen too. Utility has to be stated together with the analysis it refers to.

## Privacy: what the table reveals

The privacy axis asks whether the synthetic table reveals the records it was learned from. A synthetic table contains no real record by construction only if the generator never copies, and several generators do.

The crudest check is exact copies: synthetic rows identical, in all fifteen columns, to some training row. The independent real sample has 0.06% of them, because a few people in Adult share every recorded attribute. CART has 5.8%. Every other generator has under 0.1%. CART's copies come from its construction: with leaves of at least five donors and fifteen columns synthesised in sequence, a synthetic record sometimes follows a single real record through every leaf.

A finer view is the distance from each synthetic record to the closest training record, the DCR. Distances here are computed on an embedding in which every numeric column is replaced by its rank in the training data and every categorical mismatch adds one unit, so a distance below 1 means a training record that agrees on every category and is close on the numeric columns. Figure 2 shows the distribution for each generator, against the same distance for the independent real sample, which is the natural benchmark: a real person who was not in the training data is at some typical distance from the closest person who was. SMOTE's records sit far closer to training records than real outsiders do (median DCR 0.063 against 0.339), because each is built on the segment between two of them. CART's are slightly closer (0.262). The copula's and the independent generator's are much further away (1.10 and 1.16), because their records are implausible combinations that no real person matches. CTGAN's records are as far from the training records as the copula's (median 1.12). TVAE's median is 0.51, and its curve has a sharp step near 0.46, which is what a negative capital gain produces: it ranks below every real value, at distance 0.46 from the rank of the zero that most real records carry.

![Figure 2](https://raw.githubusercontent.com/DiogoRibeiro7/Medium-Blog/main/synthetic-data-evaluation/figures/fig2_dcr.png)

*Figure 2. Cumulative distribution of the distance from each synthetic record to its closest record in the training data. The dashed line is the independent real sample. Numeric columns enter as ranks and each categorical mismatch adds one unit, which produces the steps at 1 and √2. The training data released as synthetic would sit at zero.*

Distance alone does not say whether closeness is a leak or a sign of good fidelity. Platzer and Reutterer (2021) proposed the comparison that does: for each synthetic record, is the closest real record in the training set or in an equally large holdout set the generator never saw? If the generator learned the distribution rather than the records, the answer is a coin toss. The share for the independent real sample is 0.501. For the copula and independent marginals it is 0.500. For CART it is 0.656, and for SMOTE 0.903. For CTGAN and TVAE it is 0.514 and 0.522, slightly above one half. Nine in ten SMOTE records are closer to a record the generator saw than to any record it did not see.

The direct test is a membership-inference attack (Shokri et al. 2017): given a person's record, can an attacker who holds the synthetic table tell whether that person was in the training data? The attack used here is the simplest one, scoring each candidate by the distance to the nearest synthetic record, with 5,000 training members and 5,000 holdout non-members. Carlini et al. (2022) argued that membership attacks should be reported by their true-positive rate at a low false-positive rate rather than by an average such as the AUC, because the harm is to the individuals who can be identified with confidence, not to the average record. Both are reported. The AUC is 0.498 for the independent real sample, 0.558 for CART and 0.714 for SMOTE. At a 1% false-positive rate, the attack identifies 0.9% of members from the independent real sample, which is chance, 7.0% from CART and 15.8% from SMOTE. CTGAN and TVAE give attack AUCs of 0.496 and 0.497 and true-positive rates of 0.8% and 1.1%, which is chance. For the training data itself both are 1. Stronger attacks exist, including ones designed for generative models (Hayes et al. 2019), so these rates are lower bounds on what a determined attacker could achieve.

Figure 3 puts the three axes on one page.

![Figure 3](https://raw.githubusercontent.com/DiogoRibeiro7/Medium-Blog/main/synthetic-data-evaluation/figures/fig3_three_axes.png)

*Figure 3. (a) Fidelity, the mean total variation distance between two-way tables of the synthetic and the training data (square-root scale), against predictive utility, the AUC on the real holdout of a model trained on the synthetic table. (b) Predictive utility against the share of training records identified by a distance-based membership attack at a 1% false-positive rate (log scale). Each point is one synthetic table of 16,281 rows; independent marginals, at an AUC of 0.42, are marked on the edge of both panels.*

The figure is the argument of part one in a picture. In panel (a), predictive utility broadly rises as the table gets closer to the training data, up to the training data itself, which is perfectly faithful and perfectly useful; the deep generators are the exception, far from the training data in two-way tables because of their capital columns and still better for prediction than the copula. In panel (b), the same ordering reappears as leakage. The generators that score best on fidelity and utility, CART and SMOTE, are the ones that reveal their training records most; the ones that leak nothing, the copula and independent marginals, are the ones whose output misstates the relations in the data. The two deep generators sit to the right of panel (a) and at chance in panel (b): their output is far enough from the training data in structure that nothing about individuals leaks through this attack, and still close enough for a classifier trained on it to reach an AUC of 0.87 to 0.88. No generator is best on all three axes at once, and the independent real sample, which is what a perfect generator would produce, sits where none of them does: as useful as the training data and no more revealing than chance.

Two caveats keep these privacy numbers in proportion. They are diagnostics, not guarantees: one attack failing says nothing about a cleverer one. Stadler, Oprisanu and Troncoso (2022) showed that synthetic data can look safe on similarity metrics of this kind while remaining vulnerable to attacks targeted at outliers, and that the protection they do provide is uneven across records. A formal guarantee requires a generator trained under differential privacy (Dwork et al. 2006), which bounds the influence of any single record and pays for it in fidelity and utility, so it is a point on the same frontier, chosen explicitly. And the second caveat runs the other way: a generator that leaks nothing because its output is wrong is not private in any useful sense, it is just wrong.

## Part two: the variance a synthetic table cannot show

Everything so far compared one synthetic table with one real table. An analyst who receives a synthetic table does something different: fits a model to it and reports estimates with standard errors. The standard errors come from the same formulas that would be used on real data, and those formulas know nothing about where the rows came from.

The arithmetic is short. Take one coefficient, and let U be the sampling variance of its estimate from the real data, which has n_obs rows. The generator is fitted to those rows, so its parameters, call them θ̂, carry estimation error of the same order. Now draw a synthetic table of n_syn rows from the fitted generator and write k = n_syn / n_obs. If the generator is correctly specified, the coefficient estimated on the synthetic table, q, varies around the value implied by θ̂ with variance U/k, the ordinary sampling variance of a table of n_syn rows. Its total variance around the truth has two parts:

```
Var(q) = Var(θ̂-implied value) + synthesis noise = U + U/k
```

The standard error the analyst computes from the synthetic table estimates only the second part, U/k. The first part, the error the generator inherited from the real sample, is invisible in the synthetic table, because every synthetic row was drawn from the same fitted θ̂ and the table has no way of showing how far θ̂ is from the truth. The analyst's variance is too small by the factor (U + U/k)/(U/k) = k + 1.

A nominal 95% interval whose variance is too small by a factor r covers with probability 2Φ(1.96/√r) − 1. With k = 1, a synthetic table the size of the real one, r = 2 and the interval covers 83.4% of the time. With k = 5 it covers 57.6%, and with k = 20 it covers 33.1%. Generating more synthetic rows shrinks the visible variance U/k towards zero while the invisible part U stays exactly where it was. The information in a synthetic table is bounded by the information in the real data it was learned from, and no number of draws from the fitted model adds any.

This is a known result, and the literature on statistical disclosure control worked out the remedies in the 1990s and 2000s (Drechsler 2011 gives a book-length treatment). Rubin (1993) proposed releasing fully synthetic data drawn from a posterior predictive distribution, Little (1993) proposed releasing partially synthetic data, and Raghunathan, Reiter and Rubin (2003) gave the combining rules for the fully synthetic case, by analogy with multiple imputation (Rubin 1987). The remedy has two ingredients. The first is to release m > 1 synthetic data sets, so that the analyst can average over them. The second is a rule for the variance of that average, and the right rule depends on how the data sets were made.

Two ways of making them matter here. In *simple synthesis* the generator is fitted once and all m data sets are drawn from it. In *proper synthesis* the parameters are redrawn before each data set, from their posterior distribution or, as here and in synthpop's `proper = TRUE` option, by refitting the generator on a bootstrap resample of the real data (the bootstrap standing in for a posterior draw, Rubin 1981). Proper synthesis deliberately adds a second dose of parameter uncertainty, so that the spread between the m data sets carries information about U.

Write q̄ for the average of the m estimates, ū for the average of their squared standard errors (each computed as if the data were real), and b for the variance between the m estimates. The same accounting gives the variance of q̄ in units of ū:

```
simple synthesis:  Var(q̄) = ū (k + 1/m)
proper synthesis:  Var(q̄) = ū (k + (1 + k)/m)
```

These are the two estimators that Raab, Nowok and Dibben (2016) give for completely synthesised data, without and with the posterior predictive draw. The Raghunathan–Reiter–Rubin rule for proper synthesis uses the between-set variance instead, (1 + 1/m) b − ū, which has the same expectation as the second formula but is estimated from m − 1 degrees of freedom, so it is noisy and can be negative; Reiter (2002) proposed replacing a negative value by k ū. Treating the m stacked data sets as one real data set corresponds to ū/m.

The formulas give exact coverage predictions for a correctly specified generator, and those predictions are worth stating before any simulation, because they show that the choice of rule is not a detail. With k = 1 and m = 5, the as-if-real rule covers 57.6% under simple synthesis and 54.1% under proper synthesis. The simple rule applied to proper synthesis covers 93.0%; the proper rule applied to simple synthesis covers 96.6%. Mismatching the rule to the synthesis is a mild error, while ignoring the synthesis altogether is a severe one.

## A known mechanism

The predictions above assume a correctly specified generator. To check them, and to see what happens to generators that are not correctly specified, the first experiment uses data whose generating process is known exactly:

```
x3 ~ Bernoulli(0.4)
(x1, x2) | x3 ~ Normal((0.5·x3, 0), correlation 0.5, unit variances)
y = 1 + 0.5·x1 − 0.3·x2 + 0.8·x3 + 0.4·x1·x3 + e,   e ~ Normal(0, 1)
```

The analyst regresses y on x1, x2, x3 and the interaction x1·x3 by least squares, so all four slopes have known true values. Each replication draws n = 1,000 records, builds synthetic data with five generators, under simple and under proper synthesis, and records whether each rule's 95% interval contains the true coefficient. The generators are the copula, CART, SMOTE and independent marginals from part one, plus a *parametric* generator that fits the true family (a Bernoulli, a bivariate normal within each level of x3, and the regression of y on x with normal errors) and samples from it. The parametric generator is the case the combining rules were derived for. There are 2,000 replications, so a coverage near 95% is estimated with a Monte Carlo standard error of about half a percentage point. The coverages quoted are averages over the four slopes, which behave alike for the parametric generator and differ for the others where noted.

Real data first: the ordinary interval computed from the 1,000 real records covers 94.9% of the time on average over the four slopes (between 94.4% and 95.4% for each), as it should.

Figure 4 shows the rest. Panel (a) is the size experiment: one synthetic data set with k times as many rows as the real one, analysed as if it were real. The simulated coverage of the parametric generator sits on the closed-form curve, 83.3%, 57.5% and 32.8% for k = 1, 5 and 20 under simple synthesis against predicted 83.4%, 57.6% and 33.1%. Proper synthesis, which adds its own parameter draw, is worse when analysed naively, 74.4%, 45.0% and 24.5% against predicted 74.2%, 44.5% and 24.0%. Sequential CART, which is not a parametric model of anything, lands on the same simple-synthesis curve: 82.2%, 55.6% and 31.8%. A tree-based generator with donors drawn from leaves behaves, for these coefficients, like a correctly specified model fitted once.

![Figure 4](https://raw.githubusercontent.com/DiogoRibeiro7/Medium-Blog/main/synthetic-data-evaluation/figures/fig4_coverage_mechanism.png)

*Figure 4. Coverage of nominal 95% intervals for the four slopes of the known mechanism (average over the four), 2,000 replications of n = 1,000. (a) One synthetic data set of k·n rows analysed as if it were real. (b, c) m synthetic data sets of n rows combined by four rules, under simple and proper synthesis. Lines are the closed forms of the previous section; points are simulations.*

Panels (b) and (c) combine m data sets of n rows. The rule matched to the synthesis is exact at every m: between 94.7% and 95.2% for the simple rule under simple synthesis, and between 94.7% and 95.0% for the proper rule under proper synthesis, at every m from 1 to 20. The as-if-real rule collapses as m grows, to 32.5% under simple synthesis and 31.7% under proper synthesis at m = 20, which is the size experiment again by another route. The two mismatched rules miss by the predicted small amounts: the proper rule on simple synthesis overcovers (98.3% and 96.7%, against predicted 98.4% and 96.6% at m = 1 and 5) and the simple rule on proper synthesis undercovers (88.8% and 93.2%, against predicted 89.0% and 93.0% at m = 1 and 5), and both converge to 95% as m grows, because the term that distinguishes them is divided by m.

The Raghunathan–Reiter–Rubin rule deserves its own sentence. Under proper synthesis, the setting it was derived for, it covers 97.8% and 99.0% at m = 5 and 20, above nominal, and it reaches that coverage with intervals about 3.4 times as long as the real-data interval at m = 5, where the matched proper rule needs about 1.18 times. Its variance estimate went negative in 21% of replications at m = 5 and fell back on Reiter's adjustment; and when it is positive, its estimated degrees of freedom, (m − 1)(1 − ū/((1 + 1/m) b))² (Reiter and Raghunathan 2007), are often small, which inflates the t quantile. Applied to simple synthesis, where the between-set spread contains no parameter uncertainty at all, it fell back in about half the replications and still reported coverage of 96.0% at m = 5, but only through intervals about 2.5 times the real length. A rule that is right on average and erratic in each case is a poor thing to hand to an analyst, which is the conclusion Raab, Nowok and Dibben (2016) reached from their own simulations.

So far the combining rules look like a solution. The generic generators show where it ends. The Gaussian copula cannot represent the interaction between x1 and x3, because a single correlation matrix has no way to make the slope of y on x1 depend on x3. Its estimates are biased, by +5.1, −1.1, −3.9 and −5.8 real-data standard deviations for x1, x2, x3 and the interaction, and no rule helps: under simple synthesis with the matched rule and m = 20, coverage of the interaction is 0.0% and of the x1 slope 0.1%. More synthetic data sets make the intervals around the wrong value narrower. SMOTE sits in between: interpolating between neighbours pulls records towards each other, which distorts the joint distribution moderately, with biases of +0.9, −1.0, −0.4 and +0.3 standard deviations, and coverage with the matched rule of 83.4% and 81.9% for the x1 and x2 slopes and about 92% for the other two at m = 20. Independent marginals, which destroy every relation, cover essentially never.

The lesson of the known mechanism has two halves. Variance is a solved problem for a generator whose structure is right: release several data sets, say how they were made, and use the matching rule. Bias is not a variance problem. A combining rule can only widen an interval around the estimate the generator produces, and if that estimate is wrong, the interval covers the wrong place, more confidently with every data set added.

## Adult as the population

A known mechanism is clean but kind: four variables, linear relations, normal errors. Real data are neither. The second experiment keeps the real data and borrows the device that survey statisticians use to study estimators: treat the whole file as the population. The 48,842 Adult records define the population, and the analyst's logistic regression of income above $50,000 on age, years of education, weekly hours, sex and marital status (married with a civilian spouse or not), fitted to all of them, defines the true coefficients. These are the population values of a working model, not causal effects, and the model is deliberately simple; the question is only whether intervals computed from synthetic data cover them. Each of 1,000 replications draws 2,000 records without replacement, builds five synthetic data sets of 2,000 rows with each generator under simple and proper synthesis, and applies every rule.

The real samples set the bar: their intervals cover between 93.9% and 95.3% of the time for the five coefficients. Sampling 2,000 out of 48,842 without replacement makes them very slightly conservative, which does not matter at this precision.

Figure 5 shows the synthetic results for each coefficient, in two panels: one synthetic data set analysed as if it were real, and five data sets combined with the rule that matches how they were made.

![Figure 5](https://raw.githubusercontent.com/DiogoRibeiro7/Medium-Blog/main/synthetic-data-evaluation/figures/fig5_coverage_adult.png)

*Figure 5. Coverage of nominal 95% intervals for the population coefficients of a logistic regression of income > $50K, with Adult (48,842 records) as the population and samples of 2,000. (a) One synthetic data set analysed as if it were real. (b) Five synthetic data sets combined with the rule that matches the synthesis (simple rule for simple synthesis, proper rule for proper synthesis). 1,000 replications.*

Sequential CART is again the generator that behaves best, and again the naive analysis fails it: one synthetic table analysed as real covers 75.9% of the time on average across the five coefficients. Five data sets with the matched rule bring that to 88.6% under simple synthesis and 91.5% under proper synthesis. That is a large improvement and still short of 95%, and the shortfall is concentrated in one coefficient: years of education, covered 77.7% and 78.5% of the time. The cause is bias: the average synthetic estimate of the education coefficient sits 1.1 to 1.3 real-data standard deviations below the truth. The most plausible mechanism, which I have not isolated, is that with 2,000 records and fifteen variables the trees cannot carve the income–education relation finely enough, and drawing donor values inside a leaf blurs what they do capture. The other four coefficients are biased by at most 0.6 standard deviations and covered between 89% and 97% of the time.

SMOTE moves estimates the other way: three coefficients are pushed up by about two standard deviations (+1.8, +2.1 and +2.0 for age, education and hours under simple synthesis). A likely explanation, again a conjecture rather than something measured here, is that interpolating between neighbours within each income class shrinks the spread inside each class and so sharpens the contrast between them. With the matched rule its coverage is 74.1% under simple synthesis and 81.9% under proper synthesis on average. The copula and the independent generator are not in the same conversation: their coefficients for education and marriage are biased by eight to thirteen standard deviations, and their coverage for those is zero under every rule.

One detail of Figure 5 is a trap worth naming. The independent generator, which destroys every relation in the data, covers the coefficient for sex in 99.6% of replications under the matched rule. That is not a success. Once marital status is in the model, the population coefficient for sex is small, 0.10 on the log-odds scale, and a generator that sets every association to zero produces estimates near zero, so its intervals contain a small true value by accident. Coverage of a near-zero parameter by a generator that has no relations in it is evidence about the parameter, not about the generator, which is one more reason to read coverage together with bias and never alone.

The Raghunathan–Reiter–Rubin rule behaves as it did under the known mechanism. Under proper synthesis it lifts CART to 97.5% and SMOTE to 95.1%, above the matched rules, but it does so with intervals 3.4 to 4.4 times as long as the real-data interval, against about 1.15 for the matched proper rule. Such a rule covers a biased estimate by making the interval wide enough to reach past the bias. That is honest about uncertainty in a way, but it is not the variance the rule claims to estimate, and an analyst who reports an interval three or four times longer than the real data would give has lost most of the information.

CTGAN and TVAE are absent from this experiment, for two reasons. The practical one is cost: fitting CTGAN once took about an hour on the two CPU cores used here, and a coverage study needs thousands of fits. The substantive one is that no variance rule exists for them. A neural generator is neither simple nor proper synthesis in the sense of the formulas above: its parameters are not a posterior draw, they are the end point of a stochastic optimisation whose variability across random seeds is not the sampling variability of θ̂. Decruyenaere et al. (2024a) studied the question for deep generators and found that analysing their output as if it were real inflates type I error rates well beyond the nominal level, even when the estimates themselves are unbiased, and that a previously proposed correction factor does not repair it; they trace this partly to estimators that converge more slowly than the usual rate, so the true standard error is larger than any formula built on that rate would say. Their follow-up proposes tailoring the generator to a planned analysis with tools from debiased and targeted machine learning, so that the estimator recovers a variance that can be approximated (Decruyenaere et al., 2024b). Without such a correction, the only defensible use of a deep generator's output for inference is to verify the result on the real data.

## What a synthetic data release should report

The experiments point to a short list of things that should accompany any synthetic table that will be used for more than a demonstration. None of them is new; most are in the statistical disclosure control literature of the last thirty years. What the experiments add is a measure of how much each one changes the conclusion.

**A yardstick.** Every metric needs a reference value, and the right one is an independent real sample of the same size, scored with the same code. Without it, a two-way table distance of 0.015 or a coefficient two standard errors from the training estimate is uninterpretable. Two real samples differ, and a generator should be judged by whether it differs from the training data by more than another real sample would. When no second sample exists, a random split of the real data serves.

**Fidelity beyond the margins.** Marginal distributions are close to worthless as evidence: independent marginals match them perfectly. Report two-way (and, where it matters, higher-order) table distances, a discriminator with folds that keep each synthetic record with its nearest real record, and the logical rules the real data obey. A discriminator AUC below one half is a warning, not a compliment.

**Utility for a stated question.** "Train on synthetic, test on real" answers a question about one predictive model and says nothing about coefficients, subgroups or causal contrasts; the copula's predictive score was respectable while its regression coefficients were dozens of standard errors away. If the table will be used for inference, compare the estimates of the models people will actually fit, against the reference sample's spread.

**Privacy measured against a holdout, with the attack's error rates.** Exact copies, distances to the closest training record next to the same distances for holdout records, and a membership-inference attack reported as a true-positive rate at a low false-positive rate, with the reminder that these diagnose one attack and guarantee nothing.

**The synthesis, documented as part of the data.** Whether the m data sets came from simple or proper synthesis, what m and k are, which generator and which variables were modelled in what order. These are not implementation details: they decide which variance formula is valid. An analyst who receives one synthetic table without this information cannot compute a correct standard error, and the default, treating it as real, gives intervals that cover about four times in five when the table is the same size as the real data and less when it is larger.

**One data set is not enough for inference.** Release several, or release one with an explicit statement that it is for exploration and code development and that inferences must be checked on the confidential data. The US Census Bureau's SIPP Synthetic Beta works this way: researchers develop and run their code on the synthetic files, Census staff rerun it on the confidential data, and the results are released after disclosure review (U.S. Census Bureau 2017).

## The 2024 list, read again

The techniques in the earlier article look different through the four axes, and it is worth going back through them once.

*Duplication* is the training copy taken one step further. It has perfect fidelity in the sense of part one and no privacy at all, and on the fourth axis it is the size experiment in its purest form: a table duplicated k times gives standard errors √k times too small, with no information added. Duplication has legitimate uses as a weighting device, but its output must never be analysed as if the copies were observations.

*Resampling and the bootstrap* are simple synthesis from the empirical distribution. A single bootstrap sample of the same size, analysed as if it were real, has twice the variance its standard errors report, the k = 1 case of the formula, and covers about 83% of the time. The bootstrap is a method for estimating sampling variability, and it does that well when one looks at the spread across many resamples. One resample, released as data, is a synthetic table like any other.

*SMOTE* was measured here: good predictive utility, coefficients biased by up to about two standard deviations of the real estimate, and the worst privacy of the generators that do not copy outright. Even in its original role, rebalancing classes for a classifier, it changes the base rate the model learns, which matters whenever the scores are read as probabilities.

*Simulation from a distribution* is the parametric generator of part two. When the distribution is right, it is the one case in which the variance of synthetic-data estimates is fully understood and the combining rules are exact. When the distribution is a mechanistic model, the same rules apply to the uncertainty in its calibrated parameters, and the question the coverage study cannot answer, whether the mechanism itself is right, is the sim-to-real gap.

*Time series* need the same care. A forecast is not a synthetic path, a simulated path from fitted parameters is simple synthesis, and a path from redrawn parameters is proper synthesis, with the same consequences for anything estimated from it.

*GANs and VAEs* are CTGAN and TVAE here: with their default settings, worse fidelity than the simplest tree-based method on this dataset, good privacy diagnostics partly because their output is far from any real record, and no variance theory that would let an analyst attach a valid standard error to an estimate computed from their output.

*Differential privacy* is the only item on the list that offers a guarantee, and it offers it on the privacy axis alone. A differentially private generator is a deliberate choice of a point on the trade-off, and its output still needs the inferential accounting of part two, with the added noise as one more term in the variance.

## Closing

The 2024 article treated synthetic data as a set of techniques for producing rows. The experiments here treat it as an estimator, which is what it is: a synthetic table is a function of the real data, released in the form of data, and like any estimator it has a bias, a variance and a target. Read that way, the four axes stop looking like a checklist and start looking like consequences of one fact.

Fidelity is the bias of the generator as an estimator of the joint distribution, measured in two-way tables, logical rules and against a real holdout rather than in the marginals. Utility is the same bias seen through one analysis, which is why it has to be stated together with that analysis. Privacy is the price of low bias achieved by staying close to the training records, paid by CART partially and by SMOTE heavily. Inferential validity is the variance, and on that axis the realism of the rows is irrelevant: a perfectly faithful generator, fitted once and sampled once, gives intervals that cover 83% of the time, and sampling it twenty times as much gives intervals that cover a third of the time.

Two results deserve to be remembered past the specific numbers. Making a synthetic table larger never adds information; it only makes the standard errors computed from it smaller, so that, analysed as real, a bigger synthetic table is a more confident wrong answer. And combining rules fix variance, not bias: for a generator whose structure is right, releasing several data sets with the matching rule restores nominal coverage exactly, and for a generator whose structure is wrong, more data sets narrow the interval around the wrong value.

For anyone who builds, buys or receives synthetic data, the practical consequence is a change of question. Not "is this data realistic?", which every generator here could be made to look like in some metric, but "realistic for what, compared with what, revealing what, and with which variance formula?" A synthetic table that comes with answers to those four questions can be used. One that comes without them is a demonstration.

---

**Data and code.** [Companion folder in Medium-Blog](https://github.com/DiogoRibeiro7/Medium-Blog/tree/main/synthetic-data-evaluation), Apache-2.0: the generators, the metrics, the two coverage studies, the CTGAN and TVAE tables, `results/summary.txt` with every number quoted here, and a workflow that reruns the analysis and checks the numbers. Data: the UCI Adult extract (Becker and Kohavi 1996) in the PMLB copy (Romano et al. 2022).

## References

Becker, B., & Kohavi, R. (1996). Adult [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5XW20

Carlini, N., Chien, S., Nasr, M., Song, S., Terzis, A., & Tramèr, F. (2022). Membership inference attacks from first principles. *2022 IEEE Symposium on Security and Privacy*, 1897–1914. https://doi.org/10.1109/SP46214.2022.9833649

Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic minority over-sampling technique. *Journal of Artificial Intelligence Research*, 16, 321–357. https://doi.org/10.1613/jair.953

Decruyenaere, A., Dehaene, H., Rabaey, P., Polet, C., Decruyenaere, J., Vansteelandt, S., & Demeester, T. (2024a). The real deal behind the artificial appeal: Inferential utility of tabular synthetic data. *Proceedings of the 40th Conference on Uncertainty in Artificial Intelligence*, PMLR 244, 966–996. https://proceedings.mlr.press/v244/decruyenaere24a.html

Decruyenaere, A., Dehaene, H., Rabaey, P., Polet, C., Decruyenaere, J., Demeester, T., & Vansteelandt, S. (2024b). Debiasing synthetic data generated by deep generative models. *Advances in Neural Information Processing Systems 37*. https://arxiv.org/abs/2411.04216

Drechsler, J. (2011). *Synthetic Datasets for Statistical Disclosure Control: Theory and Implementation*. Lecture Notes in Statistics 201. Springer. https://doi.org/10.1007/978-1-4614-0326-5

Dwork, C., McSherry, F., Nissim, K., & Smith, A. (2006). Calibrating noise to sensitivity in private data analysis. *Theory of Cryptography Conference (TCC 2006)*, LNCS 3876, 265–284. https://doi.org/10.1007/11681878_14

Esteban, C., Hyland, S. L., & Rätsch, G. (2017). Real-valued (medical) time series generation with recurrent conditional GANs. arXiv:1706.02633. https://arxiv.org/abs/1706.02633

Hayes, J., Melis, L., Danezis, G., & De Cristofaro, E. (2019). LOGAN: Membership inference attacks against generative models. *Proceedings on Privacy Enhancing Technologies*, 2019(1), 133–152. https://doi.org/10.2478/popets-2019-0008

Karr, A. F., Kohnen, C. N., Oganian, A., Reiter, J. P., & Sanil, A. P. (2006). A framework for evaluating the utility of data altered to protect confidentiality. *The American Statistician*, 60(3), 224–232. https://doi.org/10.1198/000313006X124640

Kohavi, R. (1996). Scaling up the accuracy of naive-Bayes classifiers: A decision-tree hybrid. *Proceedings of the Second International Conference on Knowledge Discovery and Data Mining (KDD-96)*, 202–207. AAAI Press.

Kotelnikov, A., Baranchuk, D., Rubachev, I., & Babenko, A. (2023). TabDDPM: Modelling tabular data with diffusion models. *Proceedings of the 40th International Conference on Machine Learning*, PMLR 202, 17564–17579. https://proceedings.mlr.press/v202/kotelnikov23a.html

Little, R. J. A. (1993). Statistical analysis of masked data. *Journal of Official Statistics*, 9(2), 407–426.

Nowok, B., Raab, G. M., & Dibben, C. (2016). synthpop: Bespoke creation of synthetic data in R. *Journal of Statistical Software*, 74(11), 1–26. https://doi.org/10.18637/jss.v074.i11

Patki, N., Wedge, R., & Veeramachaneni, K. (2016). The Synthetic Data Vault. *2016 IEEE International Conference on Data Science and Advanced Analytics (DSAA)*, 399–410. https://doi.org/10.1109/DSAA.2016.49

Platzer, M., & Reutterer, T. (2021). Holdout-based empirical assessment of mixed-type synthetic data. *Frontiers in Big Data*, 4, 679939. https://doi.org/10.3389/fdata.2021.679939

Raab, G. M., Nowok, B., & Dibben, C. (2016). Practical data synthesis for large samples. *Journal of Privacy and Confidentiality*, 7(3), 67–97. https://doi.org/10.29012/jpc.v7i3.407

Raghunathan, T. E., Reiter, J. P., & Rubin, D. B. (2003). Multiple imputation for statistical disclosure limitation. *Journal of Official Statistics*, 19(1), 1–16.

Reiter, J. P. (2002). Satisfying disclosure restrictions with synthetic data sets. *Journal of Official Statistics*, 18(4), 531–544.

Reiter, J. P. (2005). Using CART to generate partially synthetic public use microdata. *Journal of Official Statistics*, 21(3), 441–462.

Reiter, J. P., & Raghunathan, T. E. (2007). The multiple adaptations of multiple imputation. *Journal of the American Statistical Association*, 102(480), 1462–1471. https://doi.org/10.1198/016214507000000932

Romano, J. D., Le, T. T., La Cava, W., Gregg, J. T., Goldberg, D. J., Chakraborty, P., Ray, N. L., Himmelstein, D., Fu, W., & Moore, J. H. (2022). PMLB v1.0: An open-source dataset collection for benchmarking machine learning methods. *Bioinformatics*, 38(3), 878–880. https://doi.org/10.1093/bioinformatics/btab727

Rubin, D. B. (1981). The Bayesian bootstrap. *The Annals of Statistics*, 9(1), 130–134. https://doi.org/10.1214/aos/1176345338

Rubin, D. B. (1987). *Multiple Imputation for Nonresponse in Surveys*. Wiley.

Rubin, D. B. (1993). Discussion: Statistical disclosure limitation. *Journal of Official Statistics*, 9(2), 461–468.

Shokri, R., Stronati, M., Song, C., & Shmatikov, V. (2017). Membership inference attacks against machine learning models. *2017 IEEE Symposium on Security and Privacy*, 3–18. https://doi.org/10.1109/SP.2017.41

Sklar, A. (1959). Fonctions de répartition à n dimensions et leurs marges. *Publications de l'Institut de Statistique de l'Université de Paris*, 8, 229–231.

Snoke, J., Raab, G. M., Nowok, B., Dibben, C., & Slavković, A. (2018). General and specific utility measures for synthetic data. *Journal of the Royal Statistical Society: Series A*, 181(3), 663–688. https://doi.org/10.1111/rssa.12358

Stadler, T., Oprisanu, B., & Troncoso, C. (2022). Synthetic data – anonymisation groundhog day. *31st USENIX Security Symposium*, 1451–1468. https://www.usenix.org/conference/usenixsecurity22/presentation/stadler

U.S. Census Bureau (2017). *SIPP Synthetic Beta: Request guidelines*. https://www.census.gov/content/dam/Census/programs-surveys/sipp/methodology/SSB_Request_Guidelines_Dec2017.pdf

Woo, M.-J., Reiter, J. P., Oganian, A., & Karr, A. F. (2009). Global measures of data utility for microdata masked for disclosure limitation. *Journal of Privacy and Confidentiality*, 1(1), 111–124. https://doi.org/10.29012/jpc.v1i1.568

Xu, L., Skoularidou, M., Cuesta-Infante, A., & Veeramachaneni, K. (2019). Modeling tabular data using conditional GAN. *Advances in Neural Information Processing Systems 32*. https://papers.nips.cc/paper/8953-modeling-tabular-data-using-conditional-gan
