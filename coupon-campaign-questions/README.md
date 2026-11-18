# What a year of grocery coupon campaigns can and cannot say

Code for a Medium article by Diogo Ribeiro. The subject is a coupon programme, measured from its own receipts: 2,469 households, one year (2017), 27 direct-mail coupon campaigns and 2,102 coupon redemptions, from the 84.51° *Complete Journey* study. The article's point is about data science when producing an analysis is cheap: the number a dashboard or an automated pipeline returns depends on which question was asked, and the questions that change the answer (what would this method say if nothing had happened? whose margin? how many households would an experiment need?) are where the work is.

One script prints every number in the article; a second draws the five figures.

## Data

The R package [completejourney](https://github.com/bradleyboehmke/completejourney) (CC0) packages the 84.51° Complete Journey 2.0 files. `fetch_data.py` downloads the five tables used here from that repository pinned to commit `5b5d061`, checks each against a SHA-256 digest and converts it to Parquet in `.cache/` (ignored by git, about 27 MB):

- `transactions` — 1,469,307 receipt lines: household, basket, product, store, quantity, sales value, loyalty-card discount, manufacturer coupon discount, retailer coupon match, timestamp.
- `campaigns`, `campaign_descriptions` — who was sent each of the 27 campaigns, its type (A, B or C) and its dates.
- `coupons` — the products each coupon covers. Recipients of a Type B or C campaign got all of its coupons; recipients of a Type A campaign got 16 coupons chosen from a pool on the basis of their past purchases, and which 16 is not recorded.
- `coupon_redemptions` — household, coupon and day of each redemption.

## What the scripts do

`coupon_numbers.py`, section by section:

1. Describes the data and the 27 campaigns.
2. Computes the quick answers: redemptions, sales on couponed lines and in couponed baskets, and spend of redeemers against non-redeemers.
3. Shows who was sent the coupons: recipients spent two to two and a half times as much as other households before any campaign, and, depending on campaign type and look-back, 32% to 62% of redemptions were of products the household had bought in the 8 or 13 weeks before.
4. Runs the standard pipeline: match recipients to non-recipients on the eight weeks before each campaign, then difference in differences, pooled and per campaign.
5. Runs three designs (matched on the eight weeks before; all households, change from the eight weeks before; all households, each against its own linear trend over the 16 weeks before) and gives each a placebo: the same design applied eight weeks earlier, before that campaign's coupons existed. Also the event studies behind Figure 2 and the trend design's sensitivity to its fitting window.
6. Repeats the per-campaign question under the three designs and counts the "significant" campaigns.
7. Turns each design's estimate into the share of redemptions that were extra purchases, and computes the size of the holdout experiment that could answer that question, including a within-household design.
8. Computes the coupon depth from the receipts and the incremental share each margin needs for the coupons to pay for the manufacturer.
9. Sets the coupons against the loyalty-card discounts in the same receipts.

`coupon_core.py` holds the data loading, the campaign panels and the estimators. `figures.py` draws the figures from the tables in `results/`. `tests/` checks the estimators on synthetic panels with known answers, including a simulation in which matching on the baseline produces a "lift" with no treatment at all.

## Run

From this folder, with Python 3.13:

```bash
python -m pip install -r requirements.txt
python fetch_data.py
python -m unittest discover -s tests -v
python coupon_numbers.py > output.txt    # about half a minute
python figures.py
```

`output.txt` is the run the article quotes. All randomness comes from one seeded household bootstrap (2,000 draws) and one seeded split of coupons, so a rerun with the pinned versions reproduces it exactly. The workflow `.github/workflows/coupon-campaign-questions.yml` downloads the data, runs the tests, reruns the script, fails if the output differs from the committed `output.txt`, and checks that the five figures are drawn.

## Design choices and assumptions

A unit is a household and a campaign; a household enters a campaign's panel if it bought anything in the eight weeks before the start. Campaign weeks count from the first day; only weeks complete by 31 December 2017 are used; the post-period is the eight weeks after the calendar end. Outcomes are baskets containing a product the campaign's coupons cover (Types B and C) and total spend (Type A, whose per-household coupons are unknown). Matching is coarsened exact matching on cells of earlier behaviour; uncertainty is a nonparametric bootstrap over households, so a household in several campaigns is resampled once.

Each design rests on an assumption the data can only partly check. The placebo checks it in a window where the campaign's coupons did not exist; other campaigns ran in those windows too, so "no coupons" means none of this campaign's. The incremental share treats every extra covered basket as one that carried a redemption. The break-even calculation treats the manufacturer's contribution margin as a share of the shelf value of the couponed items, which the data do not contain; it is shown for a range of margins. The panel households are a sample of one retailer's loyalty-card customers in one year.

## Licence

Apache-2.0. Data: CC0 (completejourney package), originally from 84.51°. Contact: dfr@esmad.ipp.pt
