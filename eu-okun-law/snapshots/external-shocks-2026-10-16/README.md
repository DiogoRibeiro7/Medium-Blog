# External shock series, frozen 2026-10-16

Files downloaded unchanged from pinned commits of the authors' public repositories,
with SHA-256 hashes in `manifest.json`. `quarterly_shocks.csv` sums the three months of
each complete quarter and is re-derived from the raw files by the audit
(`eu_okun.external_shocks.audit_shock_snapshot`).

| File | Source (pinned commit) | Licence | Coverage (quarterly) |
| --- | --- | --- | --- |
| `raw/shocks_ecb_mpd_me_m.csv` | [marekjarocinski/jkshocks_update_ecb](https://github.com/marekjarocinski/jkshocks_update_ecb) @ `f7ffc82` | CC BY 4.0 | 1999-Q1 to 2025-Q3 |
| `raw/oilSupplyNewsShocks_2025M12.xlsx` | [dkaenzig/oilsupplynews](https://github.com/dkaenzig/oilsupplynews) @ `3496b6e` | CC BY 4.0 | 1975-Q1 to 2025-Q4 |

## Attribution

- ECB monetary policy and central bank information shocks: Jarociński, M. and Karadi, P.
  (2020). Deconstructing Monetary Policy Surprises - The Role of Information Shocks.
  *American Economic Journal: Macroeconomics* 12(2), doi:10.1257/mac.20180090. Built from
  the Euro Area Monetary Policy Event-Study Database of Altavilla, Brugnolini, Gürkaynak,
  Motto and Ragusa (2019), *Journal of Monetary Economics* 108.
- Oil supply news shocks: Känzig, D. R. (2021). The Macroeconomic Effects of Oil Supply
  News: Evidence from OPEC Announcements. *American Economic Review* 111(4),
  doi:10.1257/aer.20190964.

Both data sets are licensed under the
[Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/);
they are redistributed here unchanged, with this attribution. The quarterly aggregation
is a derived work by this project.

## Variables

- `MP_median`, `MP_pm`: monetary policy shock (median-rotation and "poor man's"
  sign-restriction decompositions), % p.a. on the 1-year OIS scale.
- `CBI_median`, `CBI_pm`: central bank information shock (same decompositions).
- `pc1_hf`: raw policy surprise factor (`MP + CBI`).
- `oil_surprise`: oil supply surprise, % change of Brent futures around OPEC announcements.
- `oil_news_shock`: oil supply news shock extracted from the VAR in Känzig (2021).
