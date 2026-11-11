# Results

- [`publication/`](publication/): every number quoted in the revised manuscript, produced by [`../publication_analysis.py`](../publication_analysis.py) from the SHA-256-locked 54-country citizenship panel. `numbers.json` holds full precision; `summary.txt` is the readable version. Run `python publication_analysis.py --check` to verify them.

The earlier residence-based reconstruction (63 and 67 countries) has not been recovered, and no results from it are kept here. `analysis.py` still runs on an externally supplied `data/countries.csv` that follows [`DATA_CONTRACT.md`](../DATA_CONTRACT.md).
