# Who builds Portugal's houses? The housebuilding sector measured layer by layer

Computational companion to a Medium article. The article states a conjecture and tests it against
public data: Portugal's housing shortage is not only a shortage of workers but the loss, in
2008–2013, of a whole building ecosystem with three layers (the people with the skills, the small
and medium firms that run building sites, and the small promoters who commission new dwellings),
and after the crisis the market rebuilt the cheapest layer to enter, the one that sells, faster
than the layers that build.

Each layer gives predictions that the data could have contradicted. Some of them it does.

## Run

```bash
python -m pip install -r requirements.txt
python fetch.py --out data    # optional: refresh the snapshot (Eurostat + INE, about 4 minutes)
python analysis.py            # results/summary.txt and the CSVs behind each figure
python figures.py             # figures/fig1_*.png ... fig6_*.png
python -m pytest -q tests
```

`results/summary.txt` holds every number quoted in the article, section by section. The tests check
that the INE subclasses add up to their division, that the two INE business-statistics indicators
agree where they overlap, that dwellings by promoter add up to the total, that the 2008 splice of the
Labour Force Survey is consistent, and that `analysis.py` reproduces the pinned summary exactly.

## What is measured

| Layer | Series | Source |
|---|---|---|
| Skills | employment in construction (1995–2025); age structure against all industries (1993–2025); occupations (ISCO-08); education of employees; graduates in architecture and construction; labour and materials cost of new housing; pay relative to the economy | Eurostat `nama_10_a64_e`, `nama_10_a64`, `lfsa_egana2d`, `lfsa_egana`, `lfsa_egan2`, `lfsa_eisn2`, `edat_lfs_9910`, `educ_uoe_grad02`, `lc_lci_r2_a`; INE 0011748 |
| Firms | enterprises, persons employed and production by CAE Rev. 3 subclass (2008–2024); enterprises by employment size class (2008–2024) | INE 0008466, 0008467, 0008475, 0013838, 0013839, 0013847; Eurostat `sbs_sc_con_r2`, `sbs_sc_ovw` |
| Promoters | developers (CAE 41.10); dwellings licensed by promoter (2002–2025) and completed by promoter (2011–2025); houses and apartment buildings licensed; completions (2004–2025) | INE 0008466-0013847 (41.10), 0013236, 0013249, 0013235, 0013244, 0013245, 0012778 |
| Sellers | agencies (68.31.1), finders (68.31.2), buying and selling own property (68.10), renting (68.20); housing transactions (2009–2025) | INE SCIE indicators above, 0012787 |

Section M of the summary compares three counts of the same workforce (national accounts, Labour
Force Survey, firms' accounts). They agree in 2008 and diverge by about 100,000 people by 2024,
which matters for anything said about who is building.

Section E puts every layer on one scale, the share of the 2008 (and, where available, 2002) level
that exists in the latest year, and the recovered share of the loss
`R = (last − trough) / (reference − trough)`.

## Caveats that change the reading

- Firm counts include sole proprietors registered for tax purposes, so counts of firms overstate
  capacity; persons employed and production are the better measures, and both are reported.
- The firm statistics start in 2008, after construction employment had already fallen 16% from its
  2002 peak; comparisons with 2008 understate the loss.
- In CAE Rev. 3 developers are in construction (41.10); from 2025, under CAE Rev. 4 / NACE Rev. 2.1,
  they move to real estate (68.12). Series built on these codes will break in 2025.
- The firm a developer, a trader or a contractor declares as its main activity is its own choice.
  Some building is done by firms classified as traders of own property (68.10) and vice versa.
- The Labour Force Survey has breaks in 2011 and 2021 and small samples for workers under 25
  after 2012. It counts households, and recent migrants are hard to reach in household surveys.
- Production is deflated with INE's construction cost index for new housing; nominal values are
  also reported.

## Data

`fetch.py` downloads every series. It is run by `.github/workflows/portugal-building-ecosystem-data.yml`,
which commits the snapshot and `data/fetch_log.txt` (URL, status, filters and SHA-256 of every
response). Raw responses are in `data/raw/`, unchanged.

Terms: INE data are reused under CC BY 4.0 ("Fonte: Instituto Nacional de Estatística, IP – Portugal");
Eurostat data under the Commission's reuse decision 2011/833/EU ("Source: Eurostat"). Neither is
relicensed by this repository's licence, which covers the code, figures and results.
