# Sources and statistical definitions

Reviewed 6 November 2026. This is a provenance ledger, not a claim that each underlying publisher provides a causal estimate.

## Chart audit

1. **NSI Bulgaria**, *Statistical Yearbook 2025*, construction/housing-fund chapter, completed dwellings 2021–2024. https://www.nsi.bg/file/30721/StatBook2025_en.pdf
2. **NSI Bulgaria**, *Short-term statistics of residential buildings, annual 2025*, published 8 May 2026: 37,571 completed dwellings. https://www.nsi.bg/en/announcement/short-term-statistics-of-residential-buildings-9009
3. **NSI Bulgaria**, *Housing fund, 2025*: 4,360,805 dwellings at year-end; treated as a fixed denominator for Figure 2 only. https://www.nsi.bg/en/announcement/housing-fund-9085
4. **Eurostat**, *Quarterly House Price Index*, `prc_hpi_q`; Bulgaria second-quarter year-on-year growth in 2021–2025. https://ec.europa.eu/eurostat/databrowser/view/prc_hpi_q/default/table
5. **NSI Bulgaria**, *HPI methodology*; source change from sampled surveys to property-register administrative records in Q1 2022. https://www.nsi.bg/en/metadata/house-price-index-hpi-227

## Portugal

6. **INE via Pordata**, *Fogos concluídos em construções novas para habitação familiar*, total, 1995–2025, portal last updated 17 July 2026. For 2021–2025, values are provisional. This is new construction, not total completed rehabilitation or net stock increase. https://www.pordata.pt/portugal/fogos+concluidos+em+construcoes+novas+para+habitacao+familiar+total+e+por+tipologia+do+fogo-189
7. **INE/LNEC**, *O Parque Habitacional: Análise e Evolução, 2011–2021* (2024), Census occupancy categories 4,142,581 habitual, 1,104,881 secondary, 723,215 vacant; study also reports 136,800 quantitative housing-need cases and a constrained 154,075 immediately usable *market* vacancies after repair and normal-market-buffer filters. https://www.ine.pt/ine_novidades/semin/INEWS60/82-83/
8. **Banco de Portugal**, *Relatório de Estabilidade Financeira*, May 2025, housing buyer birthplace, tax residence and average transaction values, including the 29% foreign-born buyer share in 2024. https://www.bportugal.pt/sites/default/files/documents/2025-05/ref_5_2025_pt.pdf
9. **IMF**, *Portugal: 2026 Article IV Consultation*, Country Report No. 26/147, especially Annex V, demand, supply, stock utilisation, rental institutions and price/income developments. https://www.imf.org/-/media/files/publications/cr/2026/english/1prtea2026001.pdf
10. **IMF**, *Portugal Financial Sector Assessment Program: Technical Note on Systemic Risk Analysis*, Country Report No. 26/203: distinct measure of non-resident purchase share (reported average 6%), not numerically interchangeable with the Bank of Portugal's foreign-born 2024 transaction category. https://www.elibrary.imf.org/view/journals/002/2026/203/article-A001-en.xml

## Frozen European panel

All source transformations are inherited from the tested `european-housing-affordability` repository; the derived country-year CSVs are copied here as publication companion inputs. The frozen sources are:

11. **Eurostat**, `prc_hpi_q`: quarterly nominal House Price Index for all residential property purchases. https://ec.europa.eu/eurostat/databrowser/view/prc_hpi_q/default/table
12. **Eurostat**, `tipsho30`: real (deflated) HPI. https://ec.europa.eu/eurostat/databrowser/view/tipsho30/default/table
13. **Eurostat**, `tipsho60`: standardised house-price/income ratio. National ratio levels have different country-specific long-run reference means. https://ec.europa.eu/eurostat/databrowser/view/tipsho60/default/table
14. **Eurostat**, `sts_cobp_a`: annual dwelling **permits**, thousand units, residential buildings excluding community residences; not completions. https://ec.europa.eu/eurostat/databrowser/view/sts_cobp_a/default/table
15. **Eurostat**, `lfst_hhnhtych`: surveyed private-household counts, with Census break flags and non-comparable observed year-on-year jumps suppressed in the upstream project. https://ec.europa.eu/eurostat/databrowser/view/lfst_hhnhtych/default/table

The source manifest records the exact CSV-to-publisher mapping and limitations. This companion reproduces figures from archived **derived** CSVs; the source-level JSON-stat conversion and original archive hashes are documented in the originating research project and should be reproduced independently from official Eurostat endpoints for any updated dataset.

## Empirical literature: different estimands and settings

16. **Asquith, B. J., Mast, E., & Reed, D. (2023)**. *Local Effects of Large New Apartment Buildings in Low-Income Areas*. Review of Economics and Statistics 105(2), 359–375. Local US event-study evidence; not an estimate for Portugal. https://doi.org/10.1162/rest_a_01055
17. **Mast, E. (2023)**. *JUE Insight: The effect of new market-rate housing construction on the low-income housing market*. Journal of Urban Economics 133, 103383. US moving-chain evidence. https://doi.org/10.1016/j.jue.2021.103383
18. **Bratu, C., Harjunen, O., & Saarimaa, T. (2023)**. *JUE Insight: City-wide effects of new housing supply: Evidence from moving chains*. Journal of Urban Economics 133, 103528. Helsinki register evidence. https://doi.org/10.1016/j.jue.2022.103528
19. **Baum-Snow, N. (2023)**. *Constraints on City and Neighborhood Growth: The Central Role of Housing Supply*. Journal of Economic Perspectives 37(2), 53–74. Context on urban supply constraints. https://doi.org/10.1257/jep.37.2.53

These papers show that a stronger empirical design can answer a question a national annual two-series graphic cannot. Moving-chain evidence and local event studies do not alone establish the magnitude of aggregate European price effects, or the distributional impact of every policy intervention.
