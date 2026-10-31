# The Gilded Age in Data, Power, and Political Economy

Reproducible companion for the Medium-ready historical essay
**[The Gold Was Real. So Was the Gilding: The Political Economy of America's Gilded Age](article.md)**.

## Scope and interpretation

The essay studies U.S. industrial transformation, monopoly power, workers,
wealth concentration, immigration, and exclusions from democratic rights from
roughly the 1870s through 1900. It deliberately distinguishes:

1. **Historical values transcribed from named publications** (railroad mileage,
   foreign-born population and percentage, and Massachusetts taxable wealth).
2. **Constructed endpoints** that compound published average annual growth
   rates from an economic-history study. These are not observed annual series.
3. **Hypothetical economic models** of networks, monopoly pricing, and the
   labor share of output. No historical parameters are calibrated or estimated.

There are no downloaded datasets, hidden dependencies, synthetic historical
observations, contemporary Gini values projected into the nineteenth century,
or unsupported causal claims.

## Reproduce

Python 3.10+:

    python -m pip install -r requirements.txt
    python analysis.py
    python -m unittest discover -s tests -v

From the repository root:

    cd gilded-age-political-economy
    python analysis.py --output /tmp/gilded-age-reproduction

Regeneration is deterministic. Eight PNGs are placed in the output folder's
figures subdirectory. Three CSV tables are placed in results. The notebook
[model_walkthrough.ipynb](model_walkthrough.ipynb) provides an independently
executed contract. A path-scoped GitHub Actions workflow runs the tests and
regeneration under Python 3.12.

## Historical data provenance

| Local CSV | Exact underlying source | Important scope |
| --- | --- | --- |
| [railroad_mileage.csv](data/railroad_mileage.csv) | U.S. Census Bureau, *Statistical Abstract of the United States* (1904 edition), Table 101, reprinting Poor's Railroad Manual; [official PDF](https://www2.census.gov/library/publications/1905/compendia/statab/27ed/1904-04.pdf) | Miles of U.S. railroad line in operation in 1870, 1880, 1890, 1900; different from total track miles or annual new construction |
| [foreign_born_population.csv](data/foreign_born_population.csv) | U.S. Census Bureau, *Ancestry and Foreign-Born Populations* (31 March 2022), long-term historical slide; [official PDF](https://www2.census.gov/about/training-workshops/2022/2022-03-31-ancestry-presentation.pdf) | Rounded decennial stocks in millions and percentages from 1850–1910; not annual migration flows |
| [massachusetts_taxable_wealth.csv](data/massachusetts_taxable_wealth.csv) | Hugh Rockoff (2008), *Great Fortunes of the Gilded Age*, NBER Working Paper 14555, Table 2, reporting research by Steckel and Moehling; [working paper PDF](https://www.nber.org/system/files/working_papers/w14555/w14555.pdf) | Percentage of **taxable wealth among male household heads in Massachusetts**; not national wealth distribution; no observation for 1890 |
| [rockoff_growth_rates.csv](data/rockoff_growth_rates.csv) | Hugh Rockoff (2008), same working paper, Table 1 (rows 3, 4, 7); primary estimates credited there to Johnston–Williamson, Friedman–Schwartz, and MeasuringWorth | Average reconstructed annual rates for 1870–1899; GDP per capita, national income per capita, and unskilled real wages have different definitions and measurement uncertainty |

The transcription is intentionally small enough for a reviewer to check each
cell against its source. The original documents contain additional context,
changes in definitions, and methodological issues. The manuscript discusses
why all comparisons are conditional on historical measurement choices.

## Figures and evidence class

| Figure | Interpretation | Status |
| --- | --- | --- |
| 1 | Railroad miles in operation at four decennial benchmarks | Historical transcription |
| 2 | Foreign-born stock versus its share of population | Historical transcription, rounded |
| 3 | Compounded endpoint indexes from Rockoff's published average growth rates | Transformed literature-derived estimates, not annual observations |
| 4 | Massachusetts taxable wealth shares for male household heads | Regional historical estimate, not a national Gini |
| 5 | Real wage index conditional on three arbitrary labor-share paths | Hypothetical accounting identity |
| 6 | Fixed costs, average costs and duplicated networks | Hypothetical economic mechanism |
| 7 | Linear demand, monopoly output and deadweight loss | Hypothetical economic mechanism |
| 8 | Selected events of labor politics, regulation, and political exclusion | Documented chronology, not quantitative inference |

## Mathematical checks

For the *illustrative* linear demand and cost system:

\[
P(Q)=a-bQ,\qquad MC=c,\qquad a>c\geq0,\quad b>0,
\]

the marginal-cost benchmark and unconstrained single-price monopoly satisfy

\[
Q_C=\frac{a-c}{b},\quad
Q_M=\frac{a-c}{2b},\quad
P_M=\frac{a+c}{2},\quad
DWL=\frac{(a-c)^2}{8b}.
\]

This static illustration **sets fixed cost to zero** so that an MC-pricing
benchmark does not require an external subsidy. A separate network-cost
exercise assumes \(C(Q)=F+cQ\), with average cost \(F/Q+c\).
It highlights that a single network may avoid duplicated fixed costs while
still requiring policy to constrain market power. Neither result estimates an
actual railroad's market structure or rates.

The *hypothetical* labor-share identity is

\[
\frac{w_t}{w_0}
=
\frac{A_t}{A_0}\frac{s_t}{s_0},
\]

where output per worker \(A_t\) grows by an assumed 2.5% annually and the
share \(s_t\) follows one of three arbitrarily selected paths. It assumes
compatible units and abstracts from hours, composition, and factor-price
measurement. This is not a reconstruction of nineteenth-century wages.

The compounding function applies a published mean annual rate \(g\) across
29 calendar-year intervals from 1870 to 1899:
\(I_{1899}=100(1+g)^{29}\). The intermediate points were **not observed**;
the chart displays only the constructed endpoint.

## Generated tables

- [source_based_summary.csv](results/source_based_summary.csv) – transformations
  of the transcribed historical series and literature estimates, with
  evidence classification and source field.
- [hypothetical_monopoly_sensitivity.csv](results/hypothetical_monopoly_sensitivity.csv)
  – alternative demand slopes under arbitrary model inputs.
- [hypothetical_labor_share_paths.csv](results/hypothetical_labor_share_paths.csv)
  – the three assumed labor-share trajectories and their implied wage indexes.

## Caveats for publication

- National economic growth does not measure the distribution of wealth or
  access to political rights.
- Data on Massachusetts male household heads cannot be casually described as
  the U.S. national top-one-percent wealth share.
- Compounded average growth rates do not recover annual wage trajectories.
- Pre-1913 U.S. income-tax based distributional data do not exist; later survey
  concepts are not directly comparable to late nineteenth-century tax records.
- Population data and census boundaries have historical exclusions and
  definitional changes, including the treatment of Indigenous peoples.
- Railroad mileage aggregates track expansion, not consumer welfare or a
  specific company's abuse of market power.
- Historical cases and regulation involve contested causation. Neither the
  diagrams nor this small dataset identifies their causal effects.
- The article is prepared for manual Medium publication; a GitHub PR does not
  publish or merge it.
