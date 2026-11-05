# Frozen Forbes 2026 citizenship-country audit

This directory is the **unchanged eight-file, aggregate-only** output of
[GitHub Actions run 42099896845](https://github.com/DiogoRibeiro7/Medium-Blog/actions/runs/42099896845),
captured on 5 November 2026 at 20:05:46 UTC. It preserves the first successful
primary Forbes person-list geography extraction following PRs #514 and #515.

| Item | Evidence |
| --- | --- |
| Source endpoint | `https://www.forbes.com/forbesapi/person/billionaires/2026/position/true.json?limit=5000` |
| Source response bytes | `6,502,586` |
| Source response SHA-256 | `42fd8cec21e84360adb812fb206afdcb7814483ffd718b3087e8f73502c1d17c` |
| Workflow commit | `bfe1f9af0c909a38f7f2ff1151818895b116cf5d` |
| GitHub artifact ID | `13449541578` |
| Artifact ZIP SHA-256 | `cba0154f79cf6dfc616d42f71567bddde87da8c3184d682904b644a73052c0a7` |
| Artifact expiration | 5 December 2026 |

The ZIP was downloaded and verified against GitHub's published artifact digest;
each original member was then copied **byte-for-byte** into this directory.
The original eight member digests are pinned in `SHA256SUMS.txt`. The full
Forbes response was hashed but was **not retained**, so its raw bytes cannot
be independently rehashed from this repository.

`counts_citizenship.csv` contains 3,428 records aggregated into 80 exact Forbes
citizenship labels. `counts_api_country.csv` is retained to study field
coverage only; its 3,407 assigned records **must not** be called residence
counts. Reconciliation files compare primary labels with the **incomplete**
secondary transcription; exact spelling does not verify the ISO3 code.

The 76-row secondary table sums to 3,359. Of the 75 exact-name comparisons,
35 have nonzero differences; the net total gap is +69 primary entries. Five
unmatched primary names cover six people. These are genuine data conflicts
requiring source and attribution review, not automatically imputable zeros.

This is a **5 November live API snapshot**, not proven to be the immutable
March 2026 Forbes edition. The source's original vintage, ISO3 mapping,
the chart's original 64-country sample and residence attribution remain
unverified. The source crosswalk status remains explicitly **not approved
for regressions or publication**.

Run the reproducibility checks offline:

```bash
cd billionaires-poverty-audit
python -m unittest discover -s tests -p 'test_forbes_evidence.py' -v
```
