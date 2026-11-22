"""Year in which women first voted under state law, by state (48 contiguous states).

Source: Center for American Women and Politics (2014), "Women's Suffrage in the U.S. by State",
Rutgers University, which separates full, presidential and primary suffrage. Arkansas (1917)
and Texas (1918) gave women the vote in primaries, which in the one-party South decided the
election. Every other state is coded 1920, the year of the 19th Amendment.

The coding reproduces Online Appendix Table 2 of Miller (2008) exactly: every state in the
mortality sample sits in the same event-time column (see tests/test_dates.py).
"""

FULL = {"WY": 1890, "CO": 1893, "UT": 1896, "ID": 1896, "WA": 1910, "CA": 1911, "AZ": 1912,
        "KS": 1912, "OR": 1912, "MT": 1914, "NV": 1914, "NY": 1917, "MI": 1918, "OK": 1918,
        "SD": 1918}
PRESIDENTIAL = {"IL": 1913, "NE": 1917, "OH": 1917, "IN": 1917, "ND": 1917, "RI": 1917,
                "IA": 1919, "ME": 1919, "MN": 1919, "MO": 1919, "TN": 1919, "WI": 1919}
PRIMARY = {"AR": 1917, "TX": 1918}
STATES = ["AL", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "ID", "IL", "IN", "IA", "KS",
          "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
          "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT",
          "VT", "VA", "WA", "WV", "WI", "WY"]
AMENDMENT = 1920


def suffrage_year(state: str, coding: str = "miller") -> int:
    """First year of suffrage under the chosen coding.

    miller      full, presidential or primary suffrage, whichever came first (Miller's coding)
    full_only   only full suffrage counts; presidential and primary states get 1920
    lag         Miller's coding plus one year (a law passed late in a year binds the next one)
    """
    if coding == "full_only":
        return FULL.get(state, AMENDMENT)
    year = FULL.get(state) or PRESIDENTIAL.get(state) or PRIMARY.get(state) or AMENDMENT
    if coding == "lag":
        return min(year + 1, AMENDMENT) if year < AMENDMENT else AMENDMENT
    return year


def kind(state: str) -> str:
    if state in FULL:
        return "full"
    if state in PRESIDENTIAL:
        return "presidential"
    if state in PRIMARY:
        return "primary"
    return "amendment"
