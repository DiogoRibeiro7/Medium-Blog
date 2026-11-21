# Data sources

Every file in this folder is a hand transcription of published figures. Nothing is downloaded at run time. `vacancy_tax.py` checks each file against the values below and stops if a value differs.

## `census_2021_ihru_table11.csv`

IHRU/OHARU (September 2024), *Territórios com Falta ou Desadequação da Oferta Habitacional em Portugal*, Table 11 ("Alojamentos familiares clássicos vagos em edifícios por necessidades de reparação por tipo de vago, Portugal, Censos 2021"), printed page 32. [Report](https://www.portaldahabitacao.pt/documents/20126/58203/1_Estudo%20PNH-_VFinal_completo.pdf/49b45fff-df9f-0dce-75eb-bef8df6745e1). The report counts dwellings in buildings with no or only light repair needs as being in good condition (its footnote 13). Repair needs are recorded for the building, not for the dwelling.

The national totals in `vacancy_tax.py` (5,970,677 classic family dwellings; 4,142,581 habitual residences; 1,104,881 secondary or seasonal; 723,215 vacant) are from INE and LNEC, *O Parque Habitacional: Análise e Evolução 2011–2021* (February 2023), as reported by [barlavento](https://www.barlavento.pt/?p=347089).

## `municipal_identification.csv`

Vacant and ruined properties that municipalities reported to the tax authority for the aggravated IMI rate, tax year 2023 (first payment May 2024). Figures from the Ministry of Finance, reported by Lusa on 1 May 2024 ([barlavento](https://www.barlavento.pt/?p=346516); [idealista](https://www.idealista.pt/news/node/63805)). `vacant_general_rule` is Article 112(3) of the IMI Code (three times the rate); `vacant_pressure_zones` is Article 112-B (ten times in the first year). The unit is the property (*prédio* or *fração autónoma*), not the Census dwelling: a building that is not divided into autonomous units counts once however many dwellings it has. The OECD (2026) gives 12,803 vacant dwellings identified by municipalities in 2023, which agrees with the 12,776 here to within 0.2%.

## `vancouver_eht.csv`

City of Vancouver, Empty Homes Tax annual reports for the 2023 and 2024 vacancy reference years ([2024 report](https://vancouver.ca/files/cov/2024-empty-homes-tax-annual-report.pdf); [2025 report](https://vancouver.ca/files/cov/2025-empty-homes-tax-annual-report.pdf)). The declared vacancy rates for 2017–2020 are read from the reports' chart; they are the same in both reports. Counts for 2017–2020 are left empty because the reports restate them after compliance work and do not print the restated values. Vacant properties include those declared vacant, determined vacant after audit and deemed vacant for not declaring. These are monitoring data, not an estimate of the tax's effect.

## `rate_cut_utao.csv`

Parliamentary Budget Office (UTAO), assessment of the 2026 housing tax package, as reported by [idealista](https://www.idealista.pt/news/node/73966): the cut in the autonomous IRS rate on residential rents from 25% to 10% for rents up to €2,300 a month reduces that revenue by 54%, a direct cost of €279.1 million a year; excluding half of such rental income from IRC costs €29.8 million a year at 2026 prices.
