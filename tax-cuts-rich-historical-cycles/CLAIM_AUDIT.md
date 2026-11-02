# Claim and source audit — historical tax-cuts essay

Status: pre-publication review ledger (2026-11-02). This is a verification map, **not** a declaration that every historical claim has been independently audited. A source supports an observation or a model assumption only to the extent described here.

| Claim or exhibit | Primary evidence / repository path | Interpretation and remaining check |
| --- | --- | --- |
| Major federal tax-law enactments, 1964–2017 | Authenticated laws and congressional record; `data/legislative_chronology_sources.md` | Enactment dates, phase-ins and statute-level revenue claims checked; detailed 1986 phase-out/surcharge and pre-1913 legal history remain open. |\n| Highest U.S. individual statutory marginal rate, 1964–2018 | IRS Statistics of Income Table 23 via FRED IITTRHB; `data/top_marginal_rate_1964_2018.csv` | Descriptive top bracket, not average effective tax; cross-check 1970s earned-income cap and pre-1964 landmark claims against IRS schedules. |
| Federal receipts relative to GDP, 1964–2018 | OMB/FRED FYFRGDA188S; `data/federal_receipts_gdp_1964_2018.csv` | Total federal receipts, fiscal-year amount divided by annual GDP. Not top-earner revenue or a causal revenue estimate. |
| Federal receipt categories, 1964–2018 | OMB FY2027 Historical Table 2.3; `data/omb_receipts_by_source_1964_2018.csv` | Five categories, one decimal each; a different source vintage from FRED total receipts. Not a causal/incidence decomposition. |
| Broad federal spending categories, 1964–2018 | OMB FY2027 Historical Table 3.1; `data/omb_outlays_by_function_1964_2018.csv` | Six superfunctions, including negative offsetting receipts; function labels are not causal policy effects. |
| Federal balance and public debt, 1964–2018 | OMB/FRED FYFSGDA188S and FYPUGDA188S; `data/federal_balance_public_debt_1964_2018.csv` | Distinguish flow from stock; debt held by public differs from gross federal debt. |
| Net federal outlays versus receipts | OMB/FRED FYONGDA188S, FYFRGDA188S and FYFSGDA188S; `data/federal_net_outlays_gdp_1964_2018.csv` | Accounting identity reconciled within 0.00015 percentage points, not an attribution to tax statutes. |
| Real GDP growth and gross private domestic investment share | BEA/FRED; `data/real_growth_private_investment_1964_2018.csv` | GDP growth and investment/GDP have different denominators and units. Gross investment includes housing and inventories. |
| Private nonresidential fixed investment share | BEA/FRED A008RE1A156NBEA; `data/nonresidential_fixed_investment_share_1964_2018.csv` | Excludes housing and inventories; includes structures, equipment and IP. Need equipment/structures/IP component analysis before strong channel claims. |
| Census pretax household inequality, 1967–2018 | Census/FRED GINIALLRH; `data/household_gini_1967_2018.csv` | 1993 CPS redesign is a discontinuity; the series is not after-tax inequality or wealth. |
| CBO selected 2018 quintile resources | CBO, *Distribution of Household Income, 2018*; `data/cbo_2018_distribution.csv` | Before/after transfer-and-tax gap is **not** the effect of TCJA. Extend to full quintiles and historical years. |
| 2017 tax act estimated reductions in average federal rates | CBO, *Distribution of Household Income, 2018*, 2017 Tax Act section | Lowest two quintiles ≈1.3 percentage-point decrease; highest ≈2.7. Model-based prior-law counterfactual; independently verify incidence assumptions. |
| Major top-end tax cuts and average macroeconomic effects | Hope and Limberg (2022), DOI 10.1093/ser/mwab061 | Report study's scope, estimands, uncertainty and identifying assumptions; do not generalize to all instruments or episodes. |
| TCJA business-investment response | Chodorow-Reich, Zidar and Zwick (2024), JEP 38(3) | Firm-level identification and investment response must remain distinct from aggregate before/after plot. Audit working-paper versus published specifications. |
| Projected TCJA macro-fiscal effects | CBO (2018), *How the 2017 Tax Act Affects CBO's Projections* | Projection, not ex-post causal estimate. Cross-check original score, baseline, horizon and definition of feedback. |
| Postwar top-rate/growth descriptive association | CRS R42729 (2012) and contemporary critiques | Correlations do not identify the effect of top rates. Verify the critique is presented as a competing interpretation. |
| Illustrative tax-response sensitivity | `model.py`, `results/sensitivity.csv` | Invented parameters only; model results must never be described as measured historical effects. |

## Final review checks

- [ ] Finish pre-1913 constitutional chronology and detailed tax-base / 1986 surcharge audit. Major law numbers, enactment dates, phase-ins and 1964 Section 1 revenue-feedback claim are independently sourced in `data/legislative_chronology_sources.md`.
- [ ] Verify all paper titles, publication years, page numbers and links in the bibliography.
- [ ] Audit the quoted 2017-act incidence and investment-response estimates against primary texts, including horizons and uncertainty.
- [ ] Obtain or explicitly scope out historical CBO after-transfer/after-tax series and fixed-investment components.
- [x] Freeze OMB revenue-by-source categories with documented vintage differences.
- [x] Freeze OMB broad spending-by-function categories with a consistent FY2027 vintage.
- [ ] Audit finer spending categories where needed to evaluate public-investment tradeoffs.
- [ ] Regenerate charts, verify source provenance and byte-level deterministic artifacts in CI.
- [ ] Convert publication figures into Medium-compatible image files, add alt text and clear captions.
- [ ] Review the final manuscript and add it to the historical-series index only after verification.

The ledger is intentionally conservative: a descriptive time series may check a numerical statement while leaving the causal proposition untested.