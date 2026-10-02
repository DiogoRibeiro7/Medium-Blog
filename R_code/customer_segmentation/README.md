# Online Retail II consumer data

The customer-segmentation R analyses no longer keep a repository-local CSV copy.

They consume the canonical **Online Retail II** workbook from
`DiogoRibeiro7/data` using an immutable source identity:

- commit: `e90eed21a474c0a54ab9bf658c01dbd778bf58f1`
- path: `datasets/online-retail-ii/raw/online_retail_II.xlsx`
- SHA-256: `bcbe73b35f5b7babf197fb0cb983a11f5d9ff929078d4aa53d171b1f2df2e980`

Run any script in this directory from `R_code/customer_segmentation/`.
The helper `load_online_retail_ii.R`:

1. calls `../../scripts/fetch_online_retail_ii.py`;
2. downloads the exact pinned workbook only when it is not already cached;
3. verifies SHA-256 before use;
4. reads the `Year 2010-2011` sheet;
5. normalizes the historical field names for the R scripts:
   - `Invoice` → `InvoiceNo`
   - `Price` → `UnitPrice`
   - `Customer ID` → `CustomerID`

The cached workbook lives under `.cache/` and is not committed.

This migration replaces the old duplicated Git LFS CSV objects. The canonical
workbook remains the source of truth; no derived 45 MB CSV is stored in this
repository.
