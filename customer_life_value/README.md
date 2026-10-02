# Customer lifetime value analysis

This historical notebook now consumes **Online Retail II** from the shared
`DiogoRibeiro7/data` registry instead of keeping a second 45 MB workbook in
this repository.

Pinned source identity:

- repository: `DiogoRibeiro7/data`
- commit: `e90eed21a474c0a54ab9bf658c01dbd778bf58f1`
- path: `datasets/online-retail-ii/raw/online_retail_II.xlsx`
- SHA-256: `bcbe73b35f5b7babf197fb0cb983a11f5d9ff929078d4aa53d171b1f2df2e980`
- sheet: `Year 2010-2011`

The notebook calls `../scripts/fetch_online_retail_ii.py`, which downloads the
exact pinned workbook only when needed and verifies SHA-256 before use. The
local cache lives under `.cache/` and is ignored by Git.

The former repository-local workbook and large CSV exports were removed after
all known consumers were migrated.
