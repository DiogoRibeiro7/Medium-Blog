# Bibliography audit — 25-reference bulk verification

Date: 2026-10-29.

## Verification scope

Twenty-five previously pending journal citations were matched against publisher, journal, PubMed, or recognized institutional publication metadata. Each is marked **verified** in `reference_ledger.csv`, with an individual evidence URL and note covering its author/title, publication year, volume/issue, or page range.

The batch spans five groups:

- **Experimental design and causal foundations (7):** Freedman (2008), Holland (1986), Lin (2013), Rosenbaum and Rubin (1983), Rubin (1974), Horvitz and Thompson (1952), White (1980).
- **Matching, propensity scores and survey methods (11):** Austin (2009), Abadie and Imbens (2006, 2008), Caliendo and Kopeinig (2008), Ho et al. (2007), King and Nielsen (2019), Stuart (2010), Rosenbaum and Rubin (1985), Austin (2011), Kang and Schafer (2007), Imbens (2004).
- **Causal graphs and selection (2):** Pearl (1995); Hernán, Hernández-Díaz and Robins (2004).
- **Weighting and overlap (4):** Cole and Hernán (2008), Crump et al. (2009), Lunceford and Davidian (2004), Li, Morgan and Zaslavsky (2018).
- **Positivity (1):** Westreich and Cole (2010).

These groups total 25 distinct citations. The `verified_source` field contains a primary publisher/journal URL where available; the alternative sources are PubMed or institutional publication records. The `verification_note` field records what was matched. This PR does not alter their manuscript text because no definite metadata errors were found among these 25 checked citations.

## Deferred issue

Cochran (1965), *The Planning of Observational Studies of Human Populations*, is cited in the series with pages 234–266. Oxford Academic and Wiley's article record lists pages **234–255**, while some scholarly citations include the subsequent discussion through page 266. It remains pending until the citation convention is explicitly chosen; it was **not** counted among the 25 verified entries.

## Status

Ledger after this PR: **80 verified / 130 pending** distinct citation strings. Verifying one citation string does not automatically validate differently worded editions of the same work. The overall bibliography is not yet fully certified.

Remaining work should proceed in batches of 20–30 with similarly explicit source evidence, then audit mathematical notation and reproducible figures.
