# Federal receipts as percentage of GDP (1964–2018)

The file [federal_receipts_gdp_1964_2018.csv](federal_receipts_gdp_1964_2018.csv)
is a transcription of the published annual [FRED FYFRGDA188S table](https://fred.stlouisfed.org/data/FYFRGDA188S),
which identifies the U.S. Office of Management and Budget as the original
source and the Federal Reserve Bank of St. Louis as distributor/calculator.
It was accessed 2026-11-02; the FRED page identifies 2026-10-28 as its last
update. The frozen scope is exactly 1964 through 2018, matching the IRS
individual statutory top-rate companion. Five decimal places are retained.

FRED documents the derived statistic:

    FYFRGDA188S = 100 * ((FYFR / 1000) / GDPA)

FYFR is federal fiscal-year budget receipts, in millions of current dollars;
GDPA is annual gross domestic product, in billions of current dollars. This
is **total federal receipts**, including payroll taxes and sources beyond
individual income tax. It is not revenue solely from top-rate taxpayers, nor
corporate income-tax receipts.

In particular, the fiscal-year numerator and calendar-year GDP denominator
are not perfectly matched observation periods. Changes in nominal GDP,
business cycles, inflation, policy affecting other tax bases, payroll
contributions, capital-gains realizations, and administrative practices all
affect the series. Do not interpret it as an effective average tax rate.

Co-plotting receipts with the IRS top statutory rate is a **visual chronology**
only. It neither determines the revenue effect of a particular reform nor
identifies the location of a country on a Laffer curve. A complete empirical
analysis needs a macroeconomic and legislative counterfactual, tax-base
composition, incidence, and uncertainty.
