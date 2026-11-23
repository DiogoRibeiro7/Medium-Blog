"""Core of rent-or-buy-us: readers for the five sources and the owner-versus-renter replay.

Everything is annual. A household buys (or rents) in the middle of year t and holds for H years:

- The owner pays a 20% down payment and 2% buying costs, takes a 30-year fixed mortgage at that
  year's average Freddie Mac rate, refinances whenever the yearly average rate is at least one point
  below the rate on the loan (1% of the balance, same payoff date), pays property tax, insurance
  (0.4% of value) and maintenance (1% of value), and sells after H years paying 6% of the price.
- The renter rents a comparable home and invests the down payment and buying costs instead.
- Each year, whoever has the lower housing cost can invest the difference. The share they invest is
  the saving share s: s = 1 is the disciplined saver, s = 0 spends the difference. The owner's
  principal repayments are saved whatever s is: that is the forced saving.

Wealth after H years is linear in s for both, so two replays (s = 0 and s = 1) give every s.
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

CA = frozenset({"Los Angeles", "San Francisco", "San Diego"})

# name: (Case-Shiller column, rent file, ACS GEO_ID code). Rent files are BLS CPI-U "rent of
# primary residence" from FRED (CUUS: semiannual or annual averages; CUUR: monthly) or DBnomics.
AREAS: dict[str, tuple[str, str, str]] = {
    "New York": ("NY-New York", "CUUSA101SEHA.csv", "35620"),
    "Boston": ("MA-Boston", "CUUSA103SEHA.csv", "14460"),
    "Chicago": ("IL-Chicago", "CUUSA207SEHA.csv", "16980"),
    "Detroit": ("MI-Detroit", "CUUSA208SEHA.csv", "19820"),
    "Minneapolis": ("MN-Minneapolis", "CUUSA211SEHA.csv", "33460"),
    "Washington": ("DC-Washington", "CUURS35ASEHA.dbnomics.json", "47900"),
    "Dallas": ("TX-Dallas", "CUUSA316SEHA.csv", "19100"),
    "Atlanta": ("GA-Atlanta", "CUUSA319SEHA.csv", "12060"),
    "Miami": ("FL-Miami", "CUUSA320SEHA.csv", "33100"),
    "Tampa": ("FL-Tampa", "CUUSA321SEHA.csv", "45300"),
    "Los Angeles": ("CA-Los Angeles", "CUUSS49ASEHA.dbnomics.json", "31080"),
    "San Francisco": ("CA-San Francisco", "CUUSA422SEHA.csv", "41860"),
    "Seattle": ("WA-Seattle", "CUUSA423SEHA.csv", "42660"),
    "San Diego": ("CA-San Diego", "CUUSA424SEHA.csv", "41740"),
    "Phoenix": ("AZ-Phoenix", "CUUSA429SEHA.csv", "38060"),
    "Denver": ("CO-Denver", "CUUSA433SEHA.csv", "19740"),
    "United States": ("National-US", "CUUR0000SEHA.csv", "US"),
}
METROS = [a for a in AREAS if a != "United States"]


# --------------------------------------------------------------------------- readers


def annual_fred(text: str) -> pd.Series:
    """Annual averages of a FRED graph CSV of a CPI series (monthly, semiannual or annual).

    A year is kept only when it is complete: both halves for a semiannual series (observations dated
    January and July), one value for an annual one, and at least 11 months for a monthly one (BLS did
    not publish October 2025, so 2025 has 11)."""
    d = pd.read_csv(io.StringIO(text))
    d.columns = ["date", "v"]
    d["v"] = pd.to_numeric(d.v, errors="coerce")
    d = d.dropna()
    d["date"] = pd.to_datetime(d.date)
    g = d.groupby(d.date.dt.year).v
    n = g.count()
    mode = int(n.mode().iloc[0])
    keep = n == mode if mode in (1, 2) else n >= 11
    return g.mean()[keep].rename_axis("year")


def annual_dbnomics(text: str) -> pd.Series:
    """Annual averages of a DBnomics BLS series: semiannual periods (YYYY-S1, YYYY-S2) need both
    halves; monthly or bimonthly periods (YYYY-MM) need at least six observations."""
    s = json.loads(text)["series"]["docs"][0]
    d = pd.DataFrame({"p": s["period"], "v": s["value"]})
    d = d[d.v != "NA"]
    d["v"] = d.v.astype(float)
    d["year"] = d.p.str[:4].astype(int)
    g = d.groupby("year").v
    need = 2 if d.p.str.contains("-S").any() else 6
    return g.mean()[g.count() >= need]


def annual_case_shiller(text: str, column: str) -> pd.Series:
    """Annual averages of a Case-Shiller NSA column; zeros (written before a series starts) are
    missing, and a year needs all 12 months."""
    d = pd.read_csv(io.StringIO(text), parse_dates=["Date"]).set_index("Date")
    s = d[column].replace(0, np.nan).dropna()
    g = s.groupby(s.index.year)
    return g.mean()[g.count() == 12].rename_axis("year")


def annual_pmms(text: str) -> pd.Series:
    """Yearly average of the weekly 30-year fixed rate, as a fraction."""
    d = pd.read_csv(io.StringIO(text), parse_dates=["observation_date"])
    d["MORTGAGE30US"] = pd.to_numeric(d.MORTGAGE30US, errors="coerce")
    return d.groupby(d.observation_date.dt.year).MORTGAGE30US.mean().rename_axis("year") / 100


def damodaran_returns(sheet: pd.DataFrame) -> pd.DataFrame:
    """Annual returns from the 'Returns by year' sheet read with header=None: the block that starts at
    the row whose first cell is 'Year'. Columns: stocks (S&P 500 with dividends), bills (3-month),
    bonds (10-year Treasury)."""
    start = sheet.index[sheet.iloc[:, 0].astype(str).str.strip() == "Year"][0]
    rows = []
    for _, r in sheet.loc[start + 1 :].iterrows():
        y = pd.to_numeric(r.iloc[0], errors="coerce")
        if pd.isna(y):
            break
        rows.append(
            {
                "year": int(y),
                "stocks": float(r.iloc[1]),
                "bills": float(r.iloc[3]),
                "bonds": float(r.iloc[4]),
            }
        )
    return pd.DataFrame(rows).set_index("year")


def acs_value(text: str, geo_id: str, column: str) -> float:
    d = pd.read_csv(io.StringIO(text), sep="|", dtype=str).set_index("GEO_ID")
    return float(d.loc[geo_id, column])


def geo_id(code: str, year: int) -> str:
    if code == "US":
        return "0100000US"
    return f"310M{7 if year >= 2023 else 6}00US{code}"


# --------------------------------------------------------------------------- inputs


@dataclass(frozen=True)
class Area:
    name: str
    price: pd.Series  # dollars, annual average, by year
    rent: pd.Series  # dollars per year for a home like the owner's, by year
    tax_rate: float  # property tax as a share of value (outside California)
    rooms_ratio: float  # median rooms, owner-occupied over renter-occupied


@dataclass(frozen=True)
class Market:
    rate: pd.Series  # 30-year fixed, fraction
    returns: pd.DataFrame  # stocks, bills, bonds
    cpi: pd.Series  # CPI-U all items, annual average


@dataclass(frozen=True)
class Assumptions:
    down: float = 0.20
    buy_cost: float = 0.02
    sell_cost: float = 0.06
    insurance: float = 0.004
    maintenance: float = 0.01
    refinance: bool = True
    refi_gap: float = 0.01
    refi_cost: float = 0.01
    prop13_rate: float = 0.012
    portfolio: str = "stocks"  # stocks, 6040 (60% stocks, 40% 10-year Treasuries) or bills
    tax_drag: float = 0.0  # yearly return lost to tax on the invested account
    rent_elasticity: float = 0.5  # comparable rent = median contract rent x rooms_ratio ** e
    anchor_year: int = 2023


def load_area(name: str, data: Path, a: Assumptions) -> Area:
    cs_col, rent_file, code = AREAS[name]
    cs_file = "national-month.csv" if name == "United States" else "cities-month-NSA.csv"
    price_idx = annual_case_shiller((data / "sp-corelogic-case-shiller-datahub/raw" / cs_file).read_text(), cs_col)
    rent_text = (data / "bls-cpi-rent-us-metros/raw" / rent_file).read_text()
    rent_idx = annual_dbnomics(rent_text) if rent_file.endswith(".json") else annual_fred(rent_text)
    y = a.anchor_year
    acs = data / "census-acs-housing-costs-msa/raw"
    g = geo_id(code, y)

    def get(table: str, col: str) -> float:
        return acs_value((acs / f"acsdt1y{y}-{table}.dat").read_text(), g, col)

    value = get("b25077", "B25077_E001")
    contract_rent = get("b25058", "B25058_E001")
    taxes = get("b25103", "B25103_E001")
    rooms_ratio = get("b25021", "B25021_E002") / get("b25021", "B25021_E003")
    price = price_idx / price_idx.loc[y] * value
    rent = rent_idx / rent_idx.loc[y] * contract_rent * 12 * rooms_ratio**a.rent_elasticity
    return Area(name, price, rent, taxes / value, rooms_ratio)


def load_market(data: Path) -> Market:
    rate = annual_pmms((data / "freddie-mac-pmms-fred/raw/MORTGAGE30US.csv").read_text())
    sheet = pd.read_excel(
        data / "damodaran-historical-returns/raw/histretSP.xls",
        sheet_name="Returns by year",
        header=None,
        engine="xlrd",
    )
    cpi = annual_fred((data / "bls-cpi-rent-us-metros/raw/CPIAUCNS.csv").read_text())
    return Market(rate, damodaran_returns(sheet), cpi)


# --------------------------------------------------------------------------- replay


def portfolio_returns(m: Market, a: Assumptions) -> pd.Series:
    r = m.returns
    x = {"stocks": r.stocks, "6040": 0.6 * r.stocks + 0.4 * r.bonds, "bills": r.bills}[a.portfolio]
    return x - a.tax_drag


def payment(balance: float, rate: float, months: int) -> float:
    r = rate / 12
    if months <= 0:
        return 0.0
    if r == 0:
        return balance / months
    return balance * r / (1 - (1 + r) ** -months)


@dataclass
class Outcome:
    area: str
    year: int
    horizon: int
    owner_equity: float  # home equity at sale, after selling costs
    owner_side: float  # owner's invested savings with s = 1 (years when owning was cheaper)
    renter_base: float  # down payment and buying costs, invested
    renter_side: float  # renter's invested savings with s = 1 (years when renting was cheaper)
    down_payment: float
    principal_paid: float
    price_gain: float
    selling_cost: float
    interest_paid: float
    price_to_rent: float
    refinances: int

    def owner(self, s: float) -> float:
        return self.owner_equity + s * self.owner_side

    def renter(self, s: float) -> float:
        return self.renter_base + s * self.renter_side

    def breakeven_s(self) -> float:
        """Saving share at which owner and renter end level (nan if they never do)."""
        slope = self.renter_side - self.owner_side
        if slope == 0:
            return float("nan")
        return (self.owner_equity - self.renter_base) / slope

    def category(self) -> str:
        lazy, disciplined = (
            self.owner(0) > self.renter(0),
            self.owner(1) > self.renter(1),
        )
        if lazy and disciplined:
            return "owning wins either way"
        if not lazy and not disciplined:
            return "renting wins either way"
        if lazy and not disciplined:
            return "saving decides"
        return "saving favours the owner"


def replay(area: Area, market: Market, start: int, horizon: int, a: Assumptions) -> Outcome:
    """One household that buys or rents in the middle of `start` and leaves after `horizon` years."""
    P, R, rate_of = area.price, area.rent, market.rate
    x = portfolio_returns(market, a)
    price0 = P[start]
    balance, rate, left = (1 - a.down) * price0, rate_of[start], 360
    renter_base = (a.down + a.buy_cost) * price0
    owner_side = renter_side = 0.0
    principal = interest = 0.0
    refis = 0
    for k in range(1, horizon + 1):
        year = start + k - 1
        extra = 0.0
        if a.refinance and k > 1 and left > 0 and rate_of[year] <= rate - a.refi_gap:
            extra += a.refi_cost * balance
            rate, refis = rate_of[year], refis + 1
        paid = 0.0
        if left > 0:
            pmt = payment(balance, rate, left)
            for _ in range(min(12, left)):
                i = balance * rate / 12
                balance -= pmt - i
                paid += pmt
                principal += pmt - i
                interest += i
                left -= 1
        if area.name in CA:
            tax = a.prop13_rate * price0 * 1.02 ** (k - 1)
        else:
            tax = area.tax_rate * P[year]
        owner_cost = paid + extra + tax + (a.insurance + a.maintenance) * P[year]
        diff = owner_cost - R[year]
        renter_side += max(diff, 0.0)
        owner_side += max(-diff, 0.0)
        growth = np.sqrt((1 + x[year]) * (1 + x[year + 1])) - 1  # mid-year to mid-year
        renter_base *= 1 + growth
        renter_side *= 1 + growth
        owner_side *= 1 + growth
    sale = P[start + horizon]
    equity = sale * (1 - a.sell_cost) - max(balance, 0.0)
    return Outcome(
        area.name,
        start,
        horizon,
        equity,
        owner_side,
        renter_base,
        renter_side,
        a.down * price0,
        principal,
        sale - price0,
        a.sell_cost * sale,
        interest,
        price0 / R[start],
        refis,
    )


def cohorts(
    area: Area,
    market: Market,
    horizon: int,
    a: Assumptions,
    first: int = 1987,
    last: int = 2025,
) -> list[Outcome]:
    years = set(area.price.index) & set(area.rent.index) & set(market.rate.index)
    out = []
    for t in sorted(years):
        end = t + horizon
        if t < first:
            continue
        if end > last or end not in area.price.index or any((t + k) not in area.rent.index for k in range(horizon)):
            continue
        if any(y not in market.returns.index for y in range(t, end + 1)):
            continue
        out.append(replay(area, market, t, horizon, a))
    return out


def table(outcomes: list[Outcome]) -> pd.DataFrame:
    rows = []
    for o in outcomes:
        rows.append(
            dict(
                area=o.area,
                year=o.year,
                horizon=o.horizon,
                owner_lazy=o.owner(0),
                renter_lazy=o.renter(0),
                owner_disc=o.owner(1),
                renter_disc=o.renter(1),
                category=o.category(),
                breakeven_s=o.breakeven_s(),
                down_payment=o.down_payment,
                principal_paid=o.principal_paid,
                price_gain=o.price_gain,
                selling_cost=o.selling_cost,
                interest_paid=o.interest_paid,
                price_to_rent=o.price_to_rent,
                refinances=o.refinances,
                renter_base=o.renter_base,
                renter_side=o.renter_side,
                owner_side=o.owner_side,
            )
        )
    return pd.DataFrame(rows)


def with_(a: Assumptions, **kw) -> Assumptions:
    return replace(a, **kw)
