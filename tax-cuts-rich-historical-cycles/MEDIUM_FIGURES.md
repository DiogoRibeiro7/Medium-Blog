# Medium figure catalogue and upload notes

The SVG files in results/ are the versioned research originals. The accompanying GitHub Actions workflow rasterizes the 11 SVGs to 2,200-pixel-wide PNGs and uploads an artifact named **tax-cuts-rich-medium-png**. These PNGs are **not** committed. Download the zip from the latest successful **Historical tax-response model** workflow run for PR #451, extract it, and upload the individual PNGs to Medium. The artifact expires after 30 days. Local reproduction:

    python -m pip install "CairoSVG==2.7.1"
    python tax-cuts-rich-historical-cycles/export_medium_png.py --out tax-cuts-rich-historical-cycles/publication_png

Do not paste SVG links as publication figures. Medium captions below distinguish descriptions from causal inference, and the proposed alt text should be entered in Medium's image accessibility interface.

| PNG filename | Suggested caption | Suggested alt text |
| --- | --- | --- |
| sensitivity.png | Illustrative tax-response sensitivity across assumed taxable-base elasticity and public-capital productivity. Parameters are hypothetical, not historical estimates. | Nine-cell sensitivity figure showing different hypothetical output responses to a tax-rate reduction under changing assumptions. |
| historical_top_rate.png | Highest US regular-income statutory marginal tax rate, 1964–2018 (IRS/FRED). Not an effective tax rate. | Annual US highest statutory federal individual income-tax bracket rate from 1964 to 2018, declining across several major reforms. |
| tax_rate_and_receipts.png | Top statutory income-tax rate versus total federal receipts as a share of GDP, shown on separate scales; descriptive only (IRS and OMB/FRED). | Two historical panels compare top federal individual statutory rates and total federal revenue relative to GDP from 1964 to 2018. |
| growth_and_investment.png | Real US output growth and gross private investment/GDP, 1964–2018 (BEA/FRED). Includes housing and inventories. | Separate panels show annual real GDP growth and gross private domestic investment share of GDP over 55 years. |
| household_gini.png | US pretax household money-income inequality, 1967–2018 (Census/FRED). The chart marks the 1993 CPS redesign. | Household income Gini series divided at the 1993 survey-method break, with measured inequality generally higher in later decades. |
| cbo_2018_distribution.png | Mean household resources before versus after modeled federal taxes and means-tested transfers, selected 2018 quintiles (CBO). This is not the TCJA effect. | Paired household-income bars for selected low, middle and high quintiles, before and after taxes and transfers in 2018. |
| nonresidential_investment.png | Gross private investment and nonresidential fixed investment are different BEA aggregates. Neither reveals a causal effect of tax reforms. | Two time-series panels distinguish overall gross private domestic investment from private nonresidential fixed investment shares of GDP. |
| fiscal_balance_public_debt.png | Federal budget balance is a flow; publicly held debt is a stock, each shown relative to GDP (OMB/FRED). | Annual US federal budget surplus or deficit and public debt as separate percentage-of-GDP series. |
| revenue_outlays.png | Federal receipts and net outlays as shares of GDP, 1964–2018 (OMB/FRED); the difference is the fiscal balance, not an isolated tax-cut effect. | Historical two-panel plot of federal revenue and spending relative to GDP. |
| receipt_composition.png | US federal revenue by broad source, 1964–2018 (OMB FY2027 Table 2.3). Category amounts are descriptive, not top-income incidence estimates. | Historical US federal revenue composition by individual income taxes, corporate taxes and other broad sources. |
| outlay_composition.png | US federal spending by broad function, 1964–2018 (OMB FY2027 Table 3.1). Not attributable to individual tax acts. | Historical federal budget spending functions relative to GDP, including broad categories of federal outlays. |

## Publication quality checklist

- [ ] Download and inspect all 11 PNGs for legible text, tick labels, and uncut titles.
- [ ] Upload only figures actively referenced in the public narrative, in the corresponding sections.
- [ ] Attach each figure's caption and alt text; hyperlink the underlying source/provenance in the surrounding paragraph.
- [ ] Avoid converting the data into pasted Markdown tables; Medium may not render them correctly.
- [ ] Cross-check the final article's numbers against the frozen CSVs and the claim-audit ledger.
- [ ] Confirm the exact PR CI run succeeded before treating the artifact as ready.
