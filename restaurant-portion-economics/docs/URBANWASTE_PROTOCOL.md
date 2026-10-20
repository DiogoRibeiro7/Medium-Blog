# URBAN-WASTE commercial dataset — coverage-first protocol

Source: [Eriksson and Lindgren (2019), Zenodo 3355300](https://zenodo.org/records/3355300);
publisher file `Food Tracking Device dataset.csv`, verified MD5
`a88a1d7b09abd17064087e62d1704797`.

The successful R workflow associated with PR #211 observed 1,586 records,
13 fields and 29 distinct kitchen/location labels. The source includes
`City / kitchen`, `Timestamp`, `Waste weight sum`, and portions
reported under several distinct meal-service headings.

The previous audit emitted two CSV decoding warnings; therefore those
figures alone were not sufficient to certify correct parsing of every
record. This pipeline first decodes the full raw file using UTF-8, then
Windows-1252 if needed, and rejects delimiters or schemas that cannot
be resolved uniquely. All fields remain source-character data.

The script reports:
- Number of records and distinct kitchen/location labels.
- Missing and distinct field counts.
- Lexical patterns of `Waste weight sum` (without assigning kg or grams).
- Availability, positivity and completeness of each portion-count field.
- Coverage by anonymised kitchen code, without claiming representative
  restaurants or pooling incompatible meal services.
- Repeated kitchen/timestamp combinations (flagged, not discarded).

No weight conversion is performed. Source `Waste weight sum` is
intentionally not presented as kg, grams or euros until a verified
codebook explains its values. Portion counts with different service
labels are not summed, or treated as interchangeable exposures.

Do not interpret the resulting aggregates as plate waste per meal,
food waste attributable to portion size, cost savings or a causal
effect. Full descriptive waste-per-portion calculations need audited
units, stage definitions and denominator comparability.

Reproduce in a network-enabled environment, from the project root:

```sh
Rscript -e 'install.packages("jsonlite")'
Rscript scripts/download-data.R
Rscript scripts/audit-urbanwaste.R
```

The script fails loudly if the source format is incompatible rather
than reporting partial or distorted records.
