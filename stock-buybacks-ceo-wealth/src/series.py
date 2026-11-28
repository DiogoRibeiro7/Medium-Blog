"""Published aggregate series used in the buybacks / CEO pay timing check.

Every value is transcribed from a public source (URL in SOURCES). Nothing here is
downloaded at run time, because the sources are web pages and press releases.
Buybacks: S&P Dow Jones Indices press releases, S&P 500 constituents, USD billions.
Pay ratios: Economic Policy Institute CEO pay data page (top 350 US firms by sales).
Price return: annual S&P 500 price return (excl. dividends), percent.
"""

BUYBACKS = {  # USD bn, calendar-year S&P 500 buybacks (latest figure quoted in the cited release)
    2007: 589.1, 2008: 339.6, 2009: 137.6, 2010: 298.8, 2011: 405.1, 2012: 398.9,
    2013: 475.6, 2014: 553.3, 2015: 572.2, 2016: 536.4, 2017: 519.4, 2018: 806.4,
    2019: 728.7, 2020: 519.7, 2021: 881.7, 2022: 922.7, 2023: 795.2, 2024: 942.5,
}

# EPI CEO-to-worker compensation ratio: (realized, granted)
PAY_RATIO = {
    1965: (20.62, 15.30), 1966: (21.85, 16.21), 1967: (23.08, 17.12), 1968: (24.31, 18.03),
    1969: (24.04, 17.83), 1970: (23.76, 17.63), 1971: (23.49, 17.43), 1972: (23.22, 17.23),
    1973: (22.95, 17.02), 1974: (24.51, 18.18), 1975: (26.07, 19.34), 1976: (27.63, 20.49),
    1977: (29.18, 21.65), 1978: (30.74, 22.80), 1979: (33.41, 24.78), 1980: (36.08, 26.76),
    1981: (38.75, 28.75), 1982: (41.42, 30.73), 1983: (44.09, 32.71), 1984: (46.76, 34.69),
    1985: (49.43, 36.67), 1986: (52.10, 38.65), 1987: (54.77, 40.63), 1988: (57.44, 42.61),
    1989: (60.11, 44.59), 1990: (75.64, 56.11), 1991: (91.16, 67.62), 1992: (106.69, 79.14),
    1993: (108.24, 99.20), 1994: (87.58, 117.85), 1995: (117.07, 130.09), 1996: (150.00, 176.33),
    1997: (223.10, 234.83), 1998: (304.24, 301.50), 1999: (275.26, 288.46), 2000: (379.63, 393.08),
    2001: (214.27, 325.71), 2002: (186.16, 234.92), 2003: (228.47, 226.40), 2004: (261.83, 231.92),
    2005: (319.52, 244.33), 2006: (322.41, 237.31), 2007: (328.49, 242.43), 2008: (200.03, 218.26),
    2009: (166.39, 181.24), 2010: (209.95, 204.10), 2011: (238.91, 212.40), 2012: (363.43, 205.70),
    2013: (318.54, 211.34), 2014: (320.26, 221.00), 2015: (320.30, 215.88), 2016: (269.28, 219.43),
    2017: (294.38, 233.51), 2018: (284.37, 228.79), 2019: (311.48, 229.36), 2020: (356.22, 212.73),
    2021: (408.47, 272.98), 2022: (361.29, 226.00), 2023: (276.47, 196.98), 2024: (280.72, 212.57),
}

PRICE_RETURN = {  # annual S&P 500 price return, percent
    2005: 3.00, 2006: 13.62, 2007: 3.53, 2008: -38.49, 2009: 23.45, 2010: 12.78, 2011: -0.00,
    2012: 13.41, 2013: 29.60, 2014: 11.39, 2015: -0.73, 2016: 9.54, 2017: 19.42, 2018: -6.24,
    2019: 28.88, 2020: 16.26, 2021: 26.89, 2022: -19.44, 2023: 24.23, 2024: 23.31,
}

# EPI average CEO compensation, thousands of 2024 dollars: (realized, granted); only these five years are given in dollars
REAL_PAY_2024USD_K = {1965: (1078, 820), 1978: (1925, 1466), 2000: (25392, 25388), 2023: (21707, 15554), 2024: (22983, 16967)}

SOURCES = {
    "buybacks": [
        "https://press.spglobal.com/2015-03-23-S-P-500-2014-Buybacks-Post-16-3-Increase",
        "https://press.spglobal.com/2017-03-22-S-P-500-Buybacks-Total-135-3-Billion-for-Q4-2016-Decline-for-Full-Year-2016",
        "https://press.spglobal.com/2018-03-21-S-P-500-Q4-2017-Buybacks-Rose-6-0-to-137-0-Billion-Full-Year-2017-Fell-3-2-to-519-4-Billion",
        "https://press.spglobal.com/2020-03-24-S-P-500-buybacks-up-3-2-in-Q4-2019-Full-Year-2019-down-9-6-from-record-2018-as-companies-brace-for-a-more-volatile-2020",
        "https://press.spglobal.com/2021-03-24-S-P-500-Buybacks-Increase-28-2-in-Q4-2020-from-Q3-2020-Full-Year-2020-down-28-7-from-2019",
        "https://press.spglobal.com/2023-03-21-S-P-500-Q4-2022-Buybacks-Tick-up,-As-2022-Sets-A-Record-Proforma-Buyback-Tax-Would-Have-Reduced-Operating-Earnings-by-0-51-for-2022",
        "https://press.spglobal.com/2025-03-19-S-P-500-Q4-2024-Buybacks-Increase-7-4-and-2024-Expenditure-Sets-New-Record-by-Increasing-18-5-Earnings-Per-Share-Increases-from-Buybacks-Decline-for-the-Quarter,-as-Q1-2025s-Impact-is-Expected-to-Increase",
        "https://press.spglobal.com/2013-03-27-S-P-500-Stock-Buybacks-Decrease-in-Fourth-Quarter-of-2012",
        "https://prnewswire.com/news-releases/sp-500-fourth-quarter-buybacks-rebound-37-post-595-decline-in-2009-89390177.html",
    ],
    "pay_ratio": ["https://www.epi.org/publication/ceo-pay/"],
    "price_return": ["https://en.wikipedia.org/wiki/S%26P_500"],
}
