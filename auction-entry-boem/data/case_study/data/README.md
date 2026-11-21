# Data

All files come from the Bureau of Ocean Energy Management (BOEM), Gulf of Mexico / Gulf of America OCS Region. BOEM states that "materials produced by federal agencies are in the public domain and may be reproduced without permission" (https://www.boem.gov/copyright). Please credit BOEM as the source.

## `raw/`

| File | Sale | Date | What it is |
|---|---|---|---|
| `sale-249.zip` | 249 | 16 August 2017 | Fixed-format ASCII files: PREBID (blocks receiving bids), BID (bids), MAPS and COMPANY (translation tables), FORMATS (record layouts) |
| `sale-250.zip` | 250 | 21 March 2018 | as above |
| `sale-251.zip` | 251 | 15 August 2018 | as above |
| `sale-252.zip` | 252 | 20 March 2019 | as above |
| `sale-253.zip` | 253 | 21 August 2019 | as above |
| `sale-254.zip` | 254 | 18 March 2020 | as above |
| `sale-256.zip` | 256 | 18 November 2020 | as above |
| `sale-257.zip` | 257 | 17 November 2021 | as above |
| `sale-259.zip` | 259 | 29 March 2023 | as above |
| `sale-261.zip` | 261 | 20 December 2023 | as above |
| `sale-bbg1-final-bid-recap.pdf` | BBG1 | 10 December 2025 | Final Bid Recap (all bids by area and block, with rejected tracts marked R) |
| `sale-bbg2-final-bid-recap.pdf` | BBG2 | 11 March 2026 | Final Bid Recap |
| `sale-bbg3-preliminary-bid-recap.pdf` | BBG3 | 12 August 2026 | Preliminary Bid Recap (BOEM's review not yet published when the data were assembled) |

Each sale's files are linked from its page under https://www.boem.gov/oil-gas-energy/leasing.

## `rejected_tracts.csv`

The high bids BOEM rejected after its bid adequacy review in sales 249 to 261, transcribed from the "Listing of rejected bids" in each sale's Final Bid Recap (the URL is in the `source` column). For BBG1 and BBG2 the rejections are read from the recaps directly.

`build_dataset.py` checks every entry against the bid data by lease number and block number, and checks the number of rejections per sale against BOEM's sale summary. One discrepancy was found and resolved: the Sale 256 Final Bid Recap lists lease G36949 as Viosca Knoll block 973 in its list of rejected bids, while its bid listing gives Viosca Knoll block 873 for the same lease, with a single bid of $1,085,748. The table uses the lease number and block 873, which agree with the sale's data files.

## `bids.csv`

Written by `build_dataset.py`. One row per bid; a joint bid by several companies is one row.

| Column | Meaning |
|---|---|
| `sale`, `sale_date` | sale number and date |
| `lease` | OCS lease number assigned to the block in that sale |
| `area`, `protraction`, `block` | area name, protraction code and block number |
| `acres` | block size |
| `bid_type`, `royalty`, `bid_code` | BOEM's bid system label, royalty rate and code |
| `depth_band`, `shallow` | water-depth band implied by the code; `shallow` is below 400 m |
| `term_years` | primary lease term (sales 249 to 261 only) |
| `n_bids`, `rank` | bids on the block in that sale; rank of this bid, 1 for the high bid |
| `amount`, `per_acre` | bonus bid in dollars, and per acre |
| `min_per_acre`, `minimum_bid` | $25 per acre below 400 m and $100 per acre at 400 m or more, applied to the acreage rounded up |
| `lead_company`, `companies`, `n_partners` | BOEM company numbers, first listed first |
| `tract_rejected` | BOEM rejected the high bid on this block |
| `review_known` | the outcome of BOEM's review is published (false for BBG3) |
| `source` | raw file the row comes from |

The script reproduces, for every sale, BOEM's published counts of tracts receiving bids, bids, rejected tracts, and the sums of all bids and of high bids.
