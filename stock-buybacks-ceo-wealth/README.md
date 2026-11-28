# Stock buybacks and CEO wealth: a timing check on aggregate data

Reproduction code for the article on whether stock buybacks move wealth to CEOs at the expense of other shareholders. It tests the theory with the aggregate series that are public, and states what those series cannot show.

## Run

```
pip install -r requirements.txt
./run_all.sh
```

The run takes a few seconds. There is no download step: the aggregate series are transcribed in `src/series.py` from public web pages, with the URL of every source listed in that file. No data files are stored in this repository.

## Scripts

- `series.py` holds the transcribed series: S&P 500 buybacks 2007-2024 (S&P Dow Jones Indices press releases), the EPI CEO-to-worker pay ratio 1965-2024 (realized and granted), and the annual S&P 500 price return 2005-2024.
- `timing_check.py` correlates buyback growth with pay-ratio growth, with and without the market return, runs a permutation test (20,000 shuffles, seed 7) and reports the long-run path of the pay ratio.
- `robustness.py` drops years, runs leave-one-out and one-year lead-lag checks, and splits the sample at 2015.
- `transfer_model.py` works through the arithmetic of a buyback: when it raises earnings per share, and who gains or loses when the buyback price differs from intrinsic value.
- `figures.py` draws the five figures in `figures/`.

## Notes

- The pay measures are the EPI averages for the 350 largest US firms by sales. 2024 is a projection by EPI from data available in August 2024.
- Buyback totals are the latest figure quoted in the S&P release for each year, in USD billion. S&P revises earlier years slightly (2011 is 405.1 in later releases and 409.0 in the first one; the later value is used).
- With 17 annual changes the correlations are descriptive. They cannot separate buybacks from the profits and valuations that drive both series.
- Contact: dfr@esmad.ipp.pt
