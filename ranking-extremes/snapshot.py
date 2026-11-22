"""The country table of Euro Top Tech's "Happiness and taxes in Europe", as published on 22 November 2026.

Source: https://www.eurotoptech.com/research/happiness-taxes-europe ("Updated November 2026").
Read on 22 November 2026 between 18:30 and 18:45 UTC and transcribed by hand. The page recomputes from a
live database of crowdsourced reports, so later visits can show other numbers; this module is the
snapshot every result in the folder uses. The page offers no download, standard deviations or intervals.

Columns
    happiness  mean 0-10 rating of life satisfaction, over the country's rated reports
    services   mean 0-10 rating of satisfaction with what taxes pay for (infrastructure, healthcare,
               public services), over the same reports
    n          number of rated reports (both ratings present); the page labels 5-9 as "thin sample"
    tax        average effective tax rate, 1 - net/gross, in percent, over all approved reports
    savings    average savings rate, yearly savings / net pay, in percent, over all approved reports
    net_pay    average net pay, thousand euro, over all approved reports

Page totals: 1,673 approved reports, 782 with both ratings. Countries need at least 5 ratings to be
listed; the 29 listed countries hold 744 of the 782 rated reports.

The registry record for this source is external/eurotoptech-happiness-taxes-europe in
DiogoRibeiro7/data; it holds the digest returned by canonical_sha256().
"""

import hashlib

import pandas as pd

SOURCE_URL = "https://www.eurotoptech.com/research/happiness-taxes-europe"
READ_ON = "2026-11-22"
APPROVED_REPORTS = 1673
RATED_REPORTS = 782
THIN_BELOW = 10  # the page's "thin sample" label covers 5-9 rated reports

# country, ISO 3166-1 alpha-3, happiness, services, n, tax %, savings %, net pay (k EUR)
_ROWS = [
    ("Kosovo", "XKX", 8.80, 6.20, 5, 14, 45, 73),
    ("Lithuania", "LTU", 8.60, 5.60, 5, 33, 45, 75),
    ("Czechia", "CZE", 8.31, 8.08, 13, 22, 44, 73),
    ("Serbia", "SRB", 8.30, 5.95, 20, 24, 42, 62),
    ("Bulgaria", "BGR", 8.00, 6.00, 9, 17, 42, 57),
    ("Croatia", "HRV", 7.89, 6.22, 9, 31, 38, 61),
    ("Norway", "NOR", 7.83, 8.00, 6, 33, 32, 82),
    ("Poland", "POL", 7.83, 7.70, 67, 25, 47, 80),
    ("Belgium", "BEL", 7.82, 7.27, 11, 35, 43, 64),
    ("Estonia", "EST", 7.78, 7.89, 9, 23, 41, 56),
    ("Switzerland", "CHE", 7.73, 8.37, 46, 24, 39, 122),
    ("Greece", "GRC", 7.63, 4.63, 8, 29, 29, 35),
    ("Sweden", "SWE", 7.62, 7.62, 13, 30, 34, 52),
    ("Romania", "ROU", 7.52, 5.90, 31, 32, 47, 65),
    ("Spain", "ESP", 7.39, 6.71, 110, 31, 43, 55),
    ("Austria", "AUT", 7.29, 8.07, 14, 35, 41, 51),
    ("Ireland", "IRL", 7.26, 4.99, 17, 38, 35, 84),
    ("Italy", "ITA", 7.23, 6.13, 71, 30, 35, 41),
    ("Portugal", "PRT", 7.23, 5.74, 31, 28, 32, 43),
    ("Cyprus", "CYP", 7.17, 6.33, 6, 15, 51, 116),
    ("United Kingdom", "GBR", 7.16, 6.65, 31, 34, 35, 88),
    ("Netherlands", "NLD", 7.09, 7.57, 35, 33, 38, 83),
    ("Germany", "DEU", 7.01, 6.77, 112, 39, 34, 59),
    ("Denmark", "DNK", 7.00, 7.30, 10, 38, 37, 63),
    ("France", "FRA", 6.89, 6.89, 28, 32, 36, 62),
    ("Luxembourg", "LUX", 6.80, 8.40, 5, 30, 41, 79),
    ("Slovakia", "SVK", 6.71, 5.14, 7, 25, 39, 55),
    ("Finland", "FIN", 6.56, 7.89, 9, 36, 35, 81),
    ("Slovenia", "SVN", 6.50, 5.67, 6, 31, 43, 46),
]

COLUMNS = ["country", "iso3", "happiness", "services", "n", "tax", "savings", "net_pay"]

# Correlations quoted on the page on the same visit (Pearson; country level unless stated).
PAGE_CORRELATIONS = {
    "country tax rate vs country services rating": 0.06,
    "country tax burden vs country happiness": -0.46,
    "happiness vs services rating, individual level": 0.38,
}


def table():
    """The 29-row country table as a DataFrame, in the page's order (by happiness)."""
    d = pd.DataFrame(_ROWS, columns=COLUMNS)
    d["thin"] = d["n"] < THIN_BELOW
    return d


def canonical_text():
    """One line per country, fixed formatting: the bytes the registry digest is taken over."""
    lines = ["country,iso3,happiness,services,n,tax,savings,net_pay"]
    for r in _ROWS:
        lines.append(f"{r[0]},{r[1]},{r[2]:.2f},{r[3]:.2f},{r[4]},{r[5]},{r[6]},{r[7]}")
    return "\n".join(lines) + "\n"


def canonical_sha256():
    """SHA-256 of canonical_text(), recorded in the data registry."""
    return hashlib.sha256(canonical_text().encode("utf-8")).hexdigest()


if __name__ == "__main__":
    print(canonical_sha256())
