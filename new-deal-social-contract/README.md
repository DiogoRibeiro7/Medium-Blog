# A Safety Net Is Not a Social Contract

Reproducible companion for **[A Safety Net Is Not a Social Contract: What the New Deal Built—and Whom It Left Out](article.md)**, a Medium-ready essay linking the Gilded Age and Progressive Era histories to the U.S. Great Depression and the New Deal.

The research question is not simply whether recovery occurred. It is which parts of the New Deal were **emergency relief**, which built **durable institutions**, and how legal eligibility, measurement choices, regional discretion, and monetary policy shaped the interpretation.

## Reproduce

Python 3.10 or later:

    python -m pip install -r requirements.txt
    python analysis.py
    python -m unittest discover -s tests -v

To write artifacts to another directory:

    python analysis.py --output /tmp/new-deal-reproduction

The analysis regenerates eight original PNGs under [figures/](figures/) and three machine-readable CSVs under [results/](results/). The [Jupyter walkthrough](model_walkthrough.ipynb) imports the same code. A path-scoped GitHub Actions workflow verifies the documented contracts on pull requests. No network downloads, API keys, random states, or data interpolation are required.

## Evidence classes

Figures 1–5 are historical source transcriptions or a selective dated chronology:

| Figure | Provenance | Scope |
| --- | --- | --- |
| 1 | BLS annual retrospective unemployment rates | 1929–1939, 14+ civilian labor force, estimates constructed retrospectively; not monthly CPS observations |
| 2 | BEA NIPA Table 1.1.6 | Real annual GDP, chained 2017 dollars indexed to 1929 = 100; historical revisions depend on data vintage |
| 3 | Mathy (2026) in *BLS Monthly Labor Review*, chart 1, citing Darby (1975) | Two **published estimates** under different classifications of emergency relief workers; not two independent surveys or estimates of a program's causal effect |
| 4 | March 3, 1936 *Congressional Record* table on bank suspensions | Years 1929–1932, 1934–1935; **1933 is deliberately absent** because the original table describes that year's banking holiday and closures as noncomparable |
| 5 | FDIC, NARA, SEC, NLRB, SSA, DOL histories | Ten selected legal and administrative milestones; the calendar does not estimate welfare effects |

Figures 6–8 are **wholly hypothetical** mechanisms with invented parameters:

| Figure | Mechanism | Not an empirical claim |
| --- | --- | --- |
| 6 | Short-run withdrawals less cash on hand as insured-deposit fraction changes | Not a calibrated bank-run equilibrium or real FDIC effect |
| 7 | Wage replacement partly smooths a stipulated four-month unemployment episode | Not a reconstruction of 1930s benefit levels, prices, consumption or utility |
| 8 | Three invented groups gain different amounts when coverage widens | Not actual racial, gender, occupational or regional take-up rates |

The mathematical examples are not used to prove how much the New Deal changed any historical outcome.

## Historical data provenance

### BLS retrospective unemployment

[bls_unemployment.csv](data/bls_unemployment.csv) transcribes exactly the 1929–1939 annual rows of BLS series [LFU21000100 and LFU22000100](https://data.bls.gov/timeseries/LFU21000100&series_id=LFU22000100&from_year=1929&to_year=1939&periods_option=specific_periods&periods=Annual%2BData) for unemployment rates (%) and unemployed counts (thousands). The official BLS [historical methods explanation](https://www.bls.gov/cps/prev_yrs.htm) states that these observations were retrospectively constructed in the 1940s; the monthly Current Population Survey did not exist during the 1930s. The series is for a civilian labor force aged 14 and older, not a modern age-16-plus survey.

### BEA real GDP

[bea_real_gdp.csv](data/bea_real_gdp.csv) transcribes years 1929–1940 from U.S. BEA [NIPA Table 1.1.6, Real Gross Domestic Product, Chained Dollars](https://apps.bea.gov/iTable/?Series=A&categories=survey&isuri=1&nipa_table_list=6&reqid=19&select_all_years=1&step=3), gross domestic product row, billions of chained 2017 dollars, as surfaced in the October 2026 research session. It is **not** nominal GDP. The code divides all values by the 1929 observation and multiplies by 100. Historical national accounts and their base-year price system can be revised.

### Unemployment under competing relief-worker classifications

[relief_worker_classification.csv](data/relief_worker_classification.csv) reproduces the 1929–1939 entries from Gabriel Mathy's [2026 *Monthly Labor Review* study](https://www.bls.gov/opub/mlr/2026/article/private-sector-estimates-of-unemployment-in-the-1930s-and-the-development-of-official-labor-force-statistics.htm), chart 1. The chart attributes its two series to Michael Darby, [*Three-and-a-Half Million U.S. Employees Have Been Mislaid*](https://www.nber.org/papers/w0088) (NBER working paper, 1975). Its labels refer to whether workers on emergency relief projects are **counted as unemployed or as employed**. They should not be reinterpreted as treatment and control groups or used to infer New Deal effectiveness. The second series is not an alternative BLS household survey.

### Bank suspensions

[bank_suspensions.csv](data/bank_suspensions.csv) transcribes a contemporary [March 3, 1936 *Congressional Record* table](https://www.govinfo.gov/content/pkg/GPO-CRECB-1936-pt3-v80/pdf/GPO-CRECB-1936-pt3-v80-11-2.pdf), “Number and deposits of banks suspended, 1921–35,” subset 1929–1932 and 1934–1935. Values are 659, 1,352, 2,294, 1,456, 57, and 34.

The original table **does not provide a comparable 1933 annual figure**, given the extraordinary closures and subsequent licensing decisions during the bank holiday. The [FDIC history](https://www.fdic.gov/history/1930-1939) independently describes **about 4,000 suspensions in 1933**, but that approximate number is **not inserted into this CSV or connected as though it were a directly comparable observation**. Do not call a bank suspension a final liquidation or equate different historical failure counts.

### Dated events

[new_deal_events.csv](data/new_deal_events.csv) includes selectively chosen milestones from:

- [FDIC history](https://www.fdic.gov/history/1930-1939): Emergency Banking Act and the June 1933 Banking Act, including FDIC creation.
- [National Archives, CCC records](https://www.archives.gov/research/guide-fed-records/groups/035.html): emergency conservation-work administration established April 5, 1933.
- [National Archives, TVA Act](https://www.archives.gov/milestone-documents/tennessee-valley-authority-act): May 18, 1933.
- [SEC historical source](https://www.sec.gov/newsroom/speeches-statements/gensler-remarks-90thsec-060624): Securities Exchange Act signed June 6, 1934.
- [National Archives, WPA records](https://www.archives.gov/research/guide-fed-records/groups/069.html): May 6, 1935.
- [NLRB, passage of Wagner Act](https://www.nlrb.gov/about-nlrb/who-we-are/our-history/1935-passage-of-the-wagner-act): July 5, 1935; [Supreme Court validation](https://www.nlrb.gov/about-nlrb/who-we-are/our-history/1937-act-held-constitutional), April 1937.
- [SSA history](https://www.ssa.gov/policy/docs/ssb/v70n3/v70n3p1.html): Social Security Act, August 14, 1935.
- [Department of Labor](https://www.dol.gov/general/aboutdol/history/flsa1938): Fair Labor Standards Act signed June 25, 1938.

## Mathematical contracts

In the imaginary bank example, \(q\) is the insured share of deposits, \(p\) the proportion of uninsured deposits withdrawn, \(s=0.04\) the insured-depositor withdrawal fraction, and \(r=0.20\) the immediately liquid-asset share:

\[
G(q,p)=\max\{0,(1-q)p+qs-r\}.
\]

The expression is a **cash shortfall**, not bankruptcy probability. The illustration holds depositor behavior conditional on coverage fixed, ignores asset quality and emergency lending, and does not establish deposit insurance's historical causal effect.

For the artificial unemployment episode, \(e_t\in\{0,1\}\) is employment, \(w=100\) is monthly wage in arbitrary units, and \(b\in\{0,0.35,0.60\}\) is an invented unemployment replacement fraction:

\[
Y_t=w[e_t+(1-e_t)b].
\]

This is current cash income with no savings, debt, taxes, in-kind support, or price response; it is not a utility or welfare function.

For an invented group \(g\), the final figure shows:

\[
\Delta H_g=H_g\,c_g\,a_g\,e,
\]

with arbitrary baseline harms \(H_g\), legal coverage \(c_g\), practical access \(a_g\), and fixed assumed effectiveness \(e=0.6\). The figures do not estimate any actual group's exposure or New Deal benefit.

## Generated result tables

- [source_based_summary.csv](results/source_based_summary.csv): 29 source-backed observations with values, years, units and citations.
- [hypothetical_bank_stress.csv](results/hypothetical_bank_stress.csv): 441 invented insured-share/withdrawal combinations with explicit evidence class.
- [hypothetical_income_smoothing.csv](results/hypothetical_income_smoothing.csv): 36 invented monthly cash-flow outputs with explicit evidence class.

## Historical cautions

1. BLS and Mathy/Darby unemployment series are not all calculated using the same definition; relief employment classification changes the measured count.
2. Retrospective unemployment estimates do not establish what policymakers knew contemporaneously.
3. BEA chained-dollar figures should never be presented as nominal money earned in the 1930s.
4. The 1933 bank-holiday closures invalidate a simple, directly comparable annual suspension count; the publication intentionally leaves a gap.
5. The 1935 Social Security Act included several distinct programs. Retired-worker monthly benefits began in 1940 after the 1939 amendments, not immediately in 1935.
6. Agricultural and domestic workers were excluded from original contributory old-age coverage, with disproportionate racial consequences. The **causal motives** of legislators and administrators remain debated. Larry DeWitt's [SSA *Social Security Bulletin* study](https://www.ssa.gov/policy/docs/ssb/v70n4/v70n4p49.html) argues the documented direct mechanism was administrative feasibility; an unequal impact is not by itself proof of a particular policymaker's intent.
7. None of the charts identifies the effect of WPA, Social Security, FDIC, labor law, gold-policy changes, or wartime spending on unemployment or GDP.
8. A GitHub PR and merged manuscript do not constitute publication on Medium.
