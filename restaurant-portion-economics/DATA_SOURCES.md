# Public dataset register

Last source review: 2026-10-20. Source files are downloaded at execution time,
rather than copied into this repository. URLs and record IDs are preserved.

| Source | Record | Unit | Applicability |
|---|---|---|---|
| URBAN-WASTE: Food Tracking Device data | https://zenodo.org/records/3355300 | Recorded waste from hotels/restaurants; column schema must be audited | Primary commercial-sector descriptive source |
| Finnish school food-service waste (2024–25) | https://zenodo.org/records/20825233 | 155 days, plate, kitchen and diner measures reported in record | Secondary institutional comparison, **not restaurant sales** |
| Latvian school plate-waste experiment | https://zenodo.org/records/21458353 | Individual plate observations | Potential validation of behavioural mechanism, external setting |
| WASTEWISE food-waste database | https://zenodo.org/records/17067509 | Harmonised food supply-chain categories | Context only; check NC/SA restrictions |
| USDA FoodData Central | https://fdc.nal.usda.gov/download-datasets/ | Ingredient-specific nutrition/weights | Future menu-composition module |
| WRAP hospitality | https://www.wrap.ngo/taking-action/food-drink/sectors/hospitality-food-service | Sectoral waste benchmarks | Contextual cost benchmarking |

## Evidence hierarchy

The first two Zenodo records are initial ingestion targets. The data
retrieval script uses the Zenodo public record API to resolve files by their
publisher-listed filenames, checks publisher MD5 hashes and writes a source
manifest. No restaurant economic outcomes are inferred directly from
institutional catering waste. The availability of a record is not proof
of its suitability: inspect every codebook, measurement unit, sampling
frame, license and missing-value convention before quantification.

The project does not copy dataset files into Git. Citation and license
conditions remain attached to the upstream files. No inference of calories
from kilograms is justified without food composition and edible-food detail.

The public sources do not identify customer response to smaller portions at
unchanged prices. This parameter remains explicitly hypothetical.
