# A Law Is Not a Reform: Reproducible companion

Companion to **[A Law Is Not a Reform: What the Progressive Era Actually Changed](article.md)**, a Medium-ready historical essay on American political reform, economic power, administrative capacity, organized labor, suffrage, and exclusion between roughly 1890 and 1920.

## The evidence boundary

The article contains **eight original figures** and maintains a firm distinction between these two evidence classes:

1. **Figures 1–4 are source-backed**: three small data transcriptions from identified historical tables and a selectively curated, documented policy-event chronology.
2. **Figures 5–8 are hypothetical demonstrations**: invented parameters in simplified economic and political models. They are mathematical illustrations, **not historical measurements, intervention-effect estimates, recovered demographic data, or predictions**.

The article does not claim that an association between two historical time series estimates the causal effect of any single law. Dates identify legislative or judicial events, not precise treatment dates for populations.

## Run

Python 3.10 or newer:

    python -m pip install -r requirements.txt
    python analysis.py
    python -m unittest discover -s tests -v

From the repository root:

    cd progressive-era-reform-gap
    python analysis.py --output /tmp/progressive-reproduction

The first command generates eight PNGs in [figures/](figures/) and three results tables in [results/](results/). Reproduction is deterministic, uses no downloaded data, and makes no external network calls.

The [model walkthrough notebook](model_walkthrough.ipynb) imports exactly the same tested code. A path-scoped GitHub Actions workflow executes installation, figure generation, and tests under Python 3.12.

## Historical sources and exact numeric scope

| File | Source | Boundary |
| --- | --- | --- |
| [urban_population.csv](data/urban_population.csv) | U.S. Census Bureau, *Statistical Abstract of the United States: 1924*, Table 34, “Urban and Rural Population, by States, 1900, 1910, and 1920,” continental U.S. row. [Official scan](https://www2.census.gov/library/publications/1925/compendia/statab/47ed/1924-02.pdf). | Contemporary tabulation reports 40.0%, 45.8%, 51.4% urban (1900, 1910, 1920). Those are **not** the often-reproduced 39.6%, 45.6%, 51.2% from a differently revised historical series. Definitions and retrospective reclassification differ. Census source specifies 2,500-or-more urban places, with stated New England exceptions. |
| [union_membership.csv](data/union_membership.csv) | Leo Wolman, “Labor,” in *Recent Economic Changes in the United States*, 1929, Table 38, “Membership of American Trade Unions,” [NBER digitized chapter](https://www.nber.org/system/files/chapters/c4960/c4960.pdf). | Nine **selected** observations (1900–1920); membership reported by heterogeneous organizations, often including cross-border members, with no official census-like count. This is not union density or bargaining effectiveness. Do not join with modern BLS union-series data without conceptual harmonization. |
| [income_tax_rates.csv](data/income_tax_rates.csv) | National Taxpayer Advocate, “From Tax Collector to Fiscal Automaton,” *2011 Annual Report to Congress*, vol. 2, Table 2, [IRS-hosted report](https://www.taxpayeradvocate.irs.gov/wp-content/uploads/2020/09/IRS-TAS-ARC-2011_VOL-2.pdf). | Federal **statutory marginal** bracket rates by tax year, 1913–1920, including threshold for highest bracket. **Not average tax paid, tax receipts, effective burden, tax compliance or inequality.** Wartime changes drive much of the jump. |
| [reform_milestones.csv](data/reform_milestones.csv) | FDA, DOJ, Federal Reserve History, FTC, National Archives, Library of Congress, linked below. | Selected enacted laws, constitutional ratifications, a judicial outcome, and an antitrust judgment. Not a count of all reforms or a measure of policy effectiveness. |

## Event chronology references

- FDA, [1906 law and its enforcement](https://www.fda.gov/about-fda/changes-science-law-and-regulatory-authorities/part-i-1906-food-and-drugs-act-and-its-enforcement).
- DOJ, [Standard Oil judgment historical archive](https://www.justice.gov/atr/legacy-antitrust-judgment-standard-oil-company-of-new-jersey-et-al); the plotted 1911 point denotes the July judgment/decree rather than all litigation milestones.
- National Archives, [16th Amendment](https://www.archives.gov/milestone-documents/16th-amendment), [17th Amendment](https://www.archives.gov/milestone-documents/17th-amendment), [19th Amendment](https://www.archives.gov/milestone-documents/19th-amendment).
- Federal Reserve History, [Federal Reserve Act signed December 23, 1913](https://www.federalreservehistory.org/essays/federal-reserve-act-signed).
- FTC, [agency creation in September 1914](https://www.ftc.gov/about-ftc/history), [the Clayton and FTC Acts](https://www.ftc.gov/advice-guidance/competition-guidance/guide-antitrust-laws/antitrust-laws).
- Library of Congress, [Keating-Owen law of 1916 and its invalidation in 1918](https://guides.loc.gov/this-month-in-business-history/february/national-child-labor-committee-chartered).

## Figure audit

| Figure | Claim and exact interpretation | Evidence class |
| --- | --- | --- |
| 1 | Population classified urban 1900–1920, continental U.S. | Census transcription |
| 2 | Highest and lowest statutory marginal income tax rates, 1913–1920 | IRS table transcription |
| 3 | Selected reported trade-union membership counts | Published 1929 historical estimate |
| 4 | Ten selected policy and judicial milestones, 1906–1920 | Sourced chronology |
| 5 | Serial implementation bottleneck, .9 × .7 × .6 × .8 = .3024 | Hypothetical probabilities |
| 6 | Detection versus total jurisdictional reach under an arbitrary Poisson-monitoring law | Hypothetical model |
| 7 | Equal legal protection but unequal practical access in two invented groups | Hypothetical illustration |
| 8 | Different distribution of avoided harm in three invented groups | Hypothetical illustration |

## Mathematical model contracts

The implementation chain assumes sequential conditional probabilities:

\[
P(\mathrm{realized\ benefit})=p_L p_E p_C p_A.
\]

The expression is justified **only if** the four input numbers are interpreted as the appropriate conditional probabilities of passing successive gates. It is not a literal empirical identity for policy compliance, and independence between the gates is *not* assumed.

The monitoring illustration assumes independent Poisson arrival opportunities at each of \(cN\) legally covered targets with mean intensity \(B/(cN)\). Consequently:

\[
P(\mathrm{detected}\mid\mathrm{covered})=1-\exp(-B/(cN)),\qquad
R(c,B)=c\,[1-\exp(-B/(cN))].
\]

Here \(N=100\), \(B\in\{10,25,50\}\), and \(0<c\leq1\). Neither \(B\) nor \(N\) describes any Progressive Era agency.

For a wholly hypothetical group \(g\), avoided harm is:

\[
\Delta H_g=H_g a_g e \ell,
\]

where baseline harm \(H_g\), practical access \(a_g\), enforcement \(e\), and legal reach \(\ell\) are invented inputs. This product does not identify policy effects; no causal counterfactual is observed or fitted.

## Output tables

- [source_based_summary.csv](results/source_based_summary.csv): long-form subset of three verified transcriptions, with units and source identifiers.
- [hypothetical_capacity.csv](results/hypothetical_capacity.csv): 288 combinations of legal coverage and monitoring budgets; every row labelled hypothetical.
- [hypothetical_incidence.csv](results/hypothetical_incidence.csv): nine invented scenario/group results, similarly labelled.

## Publication checklist

- Keep figure titles, source notes, and the words **historical** / **hypothetical** when uploading images to Medium.
- Do not use statutory tax rates as measured effective household taxes.
- Do not interpret reported union membership as a causal effect of the Progressive Era, or as worker unionization rates.
- The 19th Amendment prohibited vote denial **on account of sex**, not other discriminatory restrictions or noncitizenship.
- Early twentieth-century “progressive” political programs were not unified or uniformly inclusive. State reforms, federal legislation, enforcement and judicial review must be distinguished.
- The end year 1920 is a narrative bracket, not the end date for all Progressive Era movements or their consequences.
