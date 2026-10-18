"""Feasibility of a foreign-demand shift-share instrument (``SHIFT_SHARE_PROTOCOL.md``).

The instrument for country i's GDP growth is ``Z[i,t] = sum_p e[i,p] g[p,t]``:

- ``e[i,p]`` is the value added of i absorbed in partner p's final demand, as a share of
  i's total value added, averaged over 2005-2007 (OECD TiVA 2025, measure FDVA);
- ``g[p,t]`` is p's quarterly real GDP growth (OECD quarterly national accounts; Eurostat
  for the intra-EU comparison set).

Before any IV estimate, the protocol asks whether the instrument built on defensible
partners (distant, non-European demand) can be strong enough. This module downloads and
freezes the inputs, builds the instrument, and computes its identifying variation after
country effects, quarter effects and its own lags. It also computes the benchmark expected
first-stage t under a fixed calibration, Rotemberg-type partner and country weights, and
the balance of exposure on industry and export shares. No unemployment series is used.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
from scipy import stats

from eu_okun.contract import EU27, EurostatDataError, quarter_sequence

ISO3: dict[str, str] = {
    "AT": "AUT", "BE": "BEL", "BG": "BGR", "CY": "CYP", "CZ": "CZE", "DE": "DEU",
    "DK": "DNK", "EE": "EST", "EL": "GRC", "ES": "ESP", "FI": "FIN", "FR": "FRA",
    "HR": "HRV", "HU": "HUN", "IE": "IRL", "IT": "ITA", "LT": "LTU", "LU": "LUX",
    "LV": "LVA", "MT": "MLT", "NL": "NLD", "PL": "POL", "PT": "PRT", "RO": "ROU",
    "SE": "SWE", "SI": "SVN", "SK": "SVK",
}
P1: tuple[str, ...] = ("USA", "CAN", "MEX", "BRA", "ARG", "CHN", "JPN", "KOR", "IND", "IDN",
                       "AUS", "ZAF")
NEAR: tuple[str, ...] = ("GBR", "CHE", "NOR", "TUR")
PARTNER_SETS: dict[str, tuple[str, ...]] = {
    "P1": P1,
    "P2": P1 + NEAR,
    "P3": tuple(ISO3[g] for g in EU27),          # leave-one-out: own country dropped
}
SET_LABELS = {"P1": "P1, distant demand (primary)", "P2": "P2, P1 + near extra-EU",
              "P3": "P3, intra-EU leave-one-out"}
SHARE_YEARS = ("2005", "2006", "2007")
FIRST_ORIGIN, LAST_ORIGIN = "2012-Q1", "2026-Q2"
SHIFT_START = "2010-Q1"
LAGS = 4
PANDEMIC = ("2020-Q1", "2021-Q4")
WEAK_T = 3.16
AR_CRITICAL = float(stats.norm.ppf(0.95))     # 90% Anderson-Rubin (Fieller) set
BETAS = (0.5, 1.0, 2.0)

TIVA = "https://sdmx.oecd.org/sti-public/rest/data/OECD.STI.PIE"
QNA = "https://sdmx.oecd.org/public/rest/data/OECD.SDD.NAD,DSD_NAMAIN1@DF_QNA,1.1"
LICENCE = "CC BY 4.0 (OECD terms and conditions)"


def _sources() -> dict[str, dict[str, str]]:
    eu3 = "+".join(ISO3[g] for g in EU27)
    areas = "+".join((*P1, *NEAR, *(ISO3[g] for g in EU27), "W"))
    return {
        "tiva_fdva": {
            "file": "tiva_fdva.csv",
            "url": (f"{TIVA},DSD_TIVA_FDVA@DF_FDVA,1.1/FDVA.{eu3}._T.{areas}._T.USD.A"
                    f"?startPeriod={SHARE_YEARS[0]}&endPeriod={SHARE_YEARS[-1]}"
                    "&dimensionAtObservation=AllDimensions"),
            "citation": "OECD (2025), Trade in Value Added (TiVA) 2025 edition: Origin of "
                        "value added in final demand.",
        },
        "tiva_mainlv": {
            "file": "tiva_mainlv.csv",
            "url": (f"{TIVA},DSD_TIVA_MAINLV@DF_MAINLV,1.1/VALU+EXGR.{eu3}._T+BTE.W.USD.A"
                    f"?startPeriod={SHARE_YEARS[0]}&endPeriod={SHARE_YEARS[-1]}"
                    "&dimensionAtObservation=AllDimensions"),
            "citation": "OECD (2025), Trade in Value Added (TiVA) 2025 edition: Principal "
                        "indicators, levels.",
        },
        "oecd_qna": {
            "file": "oecd_qna.csv",
            "url": (f"{QNA}/Q.Y.{'+'.join((*P1, *NEAR))}.S1..B1GQ._Z._Z._Z.PC.L.G1.T0102"
                    f"?startPeriod={SHIFT_START}&dimensionAtObservation=AllDimensions"),
            "citation": "OECD (2026), Quarterly National Accounts: GDP, volume, growth on the "
                        "previous quarter, seasonally adjusted.",
        },
    }


# ----------------------------------------------------------------------------------
# Acquisition and parsing
# ----------------------------------------------------------------------------------


def _download(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": "Medium-Blog/eu-okun-law",
                                    "Accept": "application/vnd.sdmx.data+csv; charset=utf-8"})
    try:
        with urlopen(request, timeout=180) as response:
            return bytes(response.read())
    except (HTTPError, URLError, TimeoutError) as exc:
        raise RuntimeError(f"OECD download failed for {url}: {exc}") from exc


def _records(raw: bytes) -> list[dict[str, str]]:
    rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))))
    if not rows or "OBS_VALUE" not in rows[0] or "TIME_PERIOD" not in rows[0]:
        raise EurostatDataError("Unexpected SDMX-CSV response.")
    return rows


def _value(row: Mapping[str, str]) -> float:
    value = float(row["OBS_VALUE"])
    if not math.isfinite(value):
        raise EurostatDataError("Non-finite OECD observation.")
    return value


def parse_exposure(fdva: bytes) -> list[dict[str, str]]:
    """e[i,p] per year and averaged over 2005-2007: FDVA(i, p) / FDVA(i, world)."""
    values: dict[tuple[str, str, str], float] = {}
    for row in _records(fdva):
        if row["VALUE_ADDED_SOURCE_ACTIVITY"] != "_T" or row["FINAL_DEMAND_ACTIVITY"] != "_T":
            raise EurostatDataError("TiVA FDVA response is not for all activities.")
        values[(row["VALUE_ADDED_SOURCE_AREA"], row["FINAL_DEMAND_AREA"],
                row["TIME_PERIOD"])] = _value(row)
    partners = sorted({p for (_, p, _) in values} - {"W"})
    out = []
    for geo in EU27:
        source = ISO3[geo]
        for partner in partners:
            shares = []
            for year in SHARE_YEARS:
                total = values.get((source, "W", year))
                part = values.get((source, partner, year))
                if total is None or part is None or total <= 0:
                    raise EurostatDataError(f"TiVA FDVA lacks {source}->{partner} in {year}.")
                shares.append(part / total)
            out.append({"geo": geo, "partner": partner,
                        **{f"share_{y}": repr(s) for y, s in zip(SHARE_YEARS, shares,
                                                                 strict=True)},
                        "share": repr(float(np.mean(shares)))})
    return out


def parse_balance(mainlv: bytes) -> list[dict[str, str]]:
    """Industry (B-E) share of value added and gross exports / value added, 2005-2007."""
    values: dict[tuple[str, str, str, str], float] = {}
    for row in _records(mainlv):
        values[(row["MEASURE"], row["REF_AREA"], row["ACTIVITY"], row["TIME_PERIOD"])] = _value(
            row)
    out = []
    for geo in EU27:
        source = ISO3[geo]
        industry, exports = [], []
        for year in SHARE_YEARS:
            total = values[("VALU", source, "_T", year)]
            industry.append(values[("VALU", source, "BTE", year)] / total)
            exports.append(values[("EXGR", source, "_T", year)] / total)
        out.append({"geo": geo, "industry_share": repr(float(np.mean(industry))),
                    "export_share": repr(float(np.mean(exports)))})
    return out


def parse_shifts(qna: bytes) -> list[dict[str, str]]:
    """Partner GDP growth (% on the previous quarter), seasonally adjusted volumes."""
    out = []
    for row in _records(qna):
        if row["ADJUSTMENT"] != "Y" or row["TRANSFORMATION"] != "G1":
            raise EurostatDataError("QNA response is not seasonally adjusted q/q growth.")
        out.append({"partner": row["REF_AREA"], "quarter": row["TIME_PERIOD"],
                    "growth": repr(_value(row))})
    return sorted(out, key=lambda r: (r["partner"], r["quarter"]))


PROCESSED = {"exposure.csv": ("tiva_fdva", parse_exposure),
             "balance.csv": ("tiva_mainlv", parse_balance),
             "shifts.csv": ("oecd_qna", parse_shifts)}


def _write_rows(path: Path, rows: Sequence[Mapping[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch_shift_share(output_dir: Path) -> dict[str, Any]:
    """Download TiVA and QNA, freeze the raw responses, derive the processed CSVs."""
    raw_dir, processed = output_dir / "raw", output_dir / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed.mkdir(parents=True, exist_ok=True)
    files = []
    for key, source in _sources().items():
        target = raw_dir / source["file"]
        target.write_bytes(_download(source["url"]))
        files.append({"key": key, **source, "licence": LICENCE, "sha256": _sha256(target),
                      "bytes": target.stat().st_size})
    for name, (key, parser) in PROCESSED.items():
        _write_rows(processed / name, parser((raw_dir / _sources()[key]["file"]).read_bytes()))
    manifest = {
        "description": "Inputs of SHIFT_SHARE_PROTOCOL.md (no outcome data)",
        "downloaded_at_utc": datetime.now(UTC).isoformat(),
        "files": files,
        "processed": {name: _sha256(processed / name) for name in PROCESSED},
        "share_years": list(SHARE_YEARS),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n",
                                              encoding="utf-8", newline="\n")
    return manifest


def audit_shift_share(directory: Path) -> dict[str, Any]:
    """Check raw hashes and that every processed CSV re-derives from the raw responses."""
    manifest: dict[str, Any] = json.loads((directory / "manifest.json").read_text(
        encoding="utf-8"))
    by_key = {f["key"]: f for f in manifest["files"]}
    for entry in manifest["files"]:
        if _sha256(directory / "raw" / entry["file"]) != entry["sha256"]:
            raise EurostatDataError(f"{entry['file']}: SHA-256 mismatch.")
    for name, (key, parser) in PROCESSED.items():
        path = directory / "processed" / name
        if _sha256(path) != manifest["processed"][name]:
            raise EurostatDataError(f"{name} does not match its manifest hash.")
        derived = parser((directory / "raw" / by_key[key]["file"]).read_bytes())
        if _read_rows(path) != derived:
            raise EurostatDataError(f"{name} does not re-derive from the raw response.")
    return manifest


# ----------------------------------------------------------------------------------
# The instrument
# ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class Inputs:
    """Exposure shares, partner shifts and balance covariates (from a verified snapshot)."""

    exposure: dict[str, dict[str, float]]          # geo -> partner (ISO3) -> share
    shifts: dict[str, dict[str, float]]            # partner (ISO3) -> quarter -> growth
    balance: dict[str, dict[str, float]] = field(default_factory=dict)


def load_inputs(directory: Path, eu_growth: Mapping[str, Mapping[str, float]]) -> Inputs:
    """Read the snapshot; ``eu_growth`` (geo -> quarter -> GDP growth) supplies P3's shifts."""
    exposure: dict[str, dict[str, float]] = {}
    for row in _read_rows(directory / "processed" / "exposure.csv"):
        exposure.setdefault(row["geo"], {})[row["partner"]] = float(row["share"])
    shifts: dict[str, dict[str, float]] = {}
    for row in _read_rows(directory / "processed" / "shifts.csv"):
        shifts.setdefault(row["partner"], {})[row["quarter"]] = float(row["growth"])
    for geo, series in eu_growth.items():
        shifts[ISO3[geo]] = dict(series)
    balance = {row["geo"]: {"industry_share": float(row["industry_share"]),
                            "export_share": float(row["export_share"])}
               for row in _read_rows(directory / "processed" / "balance.csv")}
    return Inputs(exposure, shifts, balance)


def partners_for(set_key: str, geo: str) -> tuple[str, ...]:
    """The set's partners; the intra-EU set leaves the country itself out."""
    partners = PARTNER_SETS[set_key]
    return tuple(p for p in partners if p != ISO3.get(geo))


@dataclass(frozen=True)
class Panel:
    """Instrument components by partner: (countries, quarters, partners) array."""

    countries: tuple[str, ...]
    quarters: tuple[str, ...]
    partners: tuple[str, ...]
    components: np.ndarray       # e[i,p] g[p,t], NaN where the shift is missing

    @property
    def z(self) -> np.ndarray:
        return np.asarray(self.components.sum(axis=2))


def instrument(inputs: Inputs, set_key: str, first: str = SHIFT_START,
               last: str = LAST_ORIGIN, countries: Sequence[str] = EU27) -> Panel:
    """Z[i,t] split by partner. A missing shift makes the component (and Z) NaN."""
    quarters = quarter_sequence(first, last)
    partners = PARTNER_SETS[set_key]
    comp = np.zeros((len(countries), len(quarters), len(partners)))
    for i, geo in enumerate(countries):
        own = partners_for(set_key, geo)
        for k, p in enumerate(partners):
            if p not in own:
                continue                           # leave-one-out: contributes zero
            share = inputs.exposure[geo][p]
            series = inputs.shifts.get(p, {})
            for j, q in enumerate(quarters):
                value = series.get(q)
                comp[i, j, k] = share * value if value is not None else np.nan
    return Panel(tuple(countries), tuple(quarters), partners, comp)


# ----------------------------------------------------------------------------------
# Feasibility
# ----------------------------------------------------------------------------------


def origins(quarters: Sequence[str], first: str = FIRST_ORIGIN,
            last: str = LAST_ORIGIN) -> list[int]:
    """Origin indices: in [first, last], with [t-4, t] clear of 2020-Q1..2021-Q4."""
    out = []
    for t, q in enumerate(quarters):
        if q < first or q > last or t < LAGS:
            continue
        if any(PANDEMIC[0] <= w <= PANDEMIC[1] for w in quarters[t - LAGS: t + 1]):
            continue
        out.append(t)
    return out


def _dummies(labels: Sequence[Any]) -> np.ndarray:
    levels = sorted(set(labels))
    return np.asarray([[1.0 if lab == lev else 0.0 for lev in levels] for lab in labels])


def residual_maker(country: Sequence[int], time: Sequence[int],
                   covariates: np.ndarray | None) -> np.ndarray:
    """Annihilator M of country effects, quarter effects and covariates."""
    blocks = [_dummies(country), _dummies(time)[:, 1:]]
    if covariates is not None and covariates.size:
        blocks.append(covariates)
    x = np.column_stack(blocks)
    return np.asarray(np.eye(len(country)) - x @ np.linalg.pinv(x))


@dataclass(frozen=True)
class Calibration:
    beta: float
    sigma: float
    label: str
    expected_t: float
    p_bounded: float


@dataclass
class Feasibility:
    """Identifying variation and benchmark power of one partner set."""

    set_key: str
    n: int
    quarters: int
    s: float
    sigma_u: float
    calibrations: list[Calibration]
    partner_weights: dict[str, float]
    country_weights: dict[str, float]
    instrument_sd: float

    @property
    def central(self) -> Calibration:
        return next(c for c in self.calibrations if c.label == "central")

    @property
    def beta_needed(self) -> float:
        return WEAK_T * self.sigma_u / math.sqrt(self.s)

    @property
    def quarters_needed(self) -> float:
        return self.quarters * (WEAK_T / self.central.expected_t) ** 2


def p_bounded(t: float, critical: float = AR_CRITICAL) -> float:
    """P(|t + N(0,1)| > critical): the Fieller set is bounded iff |first-stage t| > c."""
    return float(stats.norm.sf(critical - t) + stats.norm.cdf(-critical - t))


def feasibility(panel: Panel, sigma_hat: float, set_key: str,
                first: str = FIRST_ORIGIN) -> Feasibility:
    """S = sum of Z~^2 after country and quarter effects and four lags of Z."""
    z = panel.z
    rows, country, time = [], [], []
    for t in origins(panel.quarters, first):
        for i in range(len(panel.countries)):
            window = z[i, t - LAGS: t + 1]
            if np.all(np.isfinite(window)) and np.all(np.isfinite(panel.components[i, t])):
                rows.append((i, t))
                country.append(i)
                time.append(t)
    if not rows:
        raise ValueError(f"No complete instrument observations for {set_key}.")
    lags = np.asarray([[z[i, t - lag] for lag in range(1, LAGS + 1)] for i, t in rows])
    m = residual_maker(country, time, lags)
    zt = m @ np.asarray([z[i, t] for i, t in rows])
    s = float(zt @ zt)
    comps = np.asarray([panel.components[i, t] for i, t in rows])      # (n, partners)
    partner_weights = {p: float((m @ comps[:, k]) @ zt / s)
                       for k, p in enumerate(panel.partners)}
    country_weights = {panel.countries[i]: float(sum(zt[k] ** 2 for k, (ci, _) in
                                                     enumerate(rows) if ci == i) / s)
                       for i in sorted(set(country))}
    calibrations = []
    for beta in BETAS:
        for sigma, label in ((sigma_hat, "estimated"), (sigma_hat / 2, "halved")):
            expected = beta * math.sqrt(s) / sigma
            name = "central" if beta == 1.0 and label == "estimated" else (
                f"beta {beta:g}, sigma_u {label}")
            calibrations.append(Calibration(beta, sigma, name, expected, p_bounded(expected)))
    return Feasibility(set_key, len(rows), len(set(time)), s, sigma_hat, calibrations,
                       partner_weights, country_weights,
                       float(np.std(zt, ddof=1)))


def decision(results: Mapping[str, Feasibility]) -> str:
    """Protocol, section 6: go only if P1's central expected |t| reaches 3.16."""
    return "go" if results["P1"].central.expected_t >= WEAK_T else "no go"


# ----------------------------------------------------------------------------------
# GDP residual scale and balance
# ----------------------------------------------------------------------------------


def sigma_u(panel: Mapping[tuple[str, str], Mapping[str, float | None]],
            quarters: Sequence[str], first: str = FIRST_ORIGIN,
            countries: Sequence[str] = EU27) -> tuple[float, int]:
    """Residual sd of GDP growth after country and quarter effects and four lags each of
    GDP growth, the unemployment change and the participation change (protocol, 5)."""
    def get(geo: str, q: str, name: str) -> float:
        value = panel.get((geo, q), {}).get(name)
        return math.nan if value is None else float(value)

    rows, country, time, ys = [], [], [], []
    for t in origins(quarters, first):
        for i, geo in enumerate(countries):
            if t < LAGS + 1:
                continue
            controls = []
            for lag in range(1, LAGS + 1):
                q, prev = quarters[t - lag], quarters[t - lag - 1]
                controls += [get(geo, q, "gdp"),
                             get(geo, q, "unemployment") - get(geo, prev, "unemployment"),
                             get(geo, q, "participation") - get(geo, prev, "participation")]
            y = get(geo, quarters[t], "gdp")
            if np.all(np.isfinite(controls)) and math.isfinite(y):
                rows.append(controls)
                country.append(i)
                time.append(t)
                ys.append(y)
    m = residual_maker(country, time, np.asarray(rows))
    resid = m @ np.asarray(ys)
    k = len(set(country)) + len(set(time)) - 1 + len(rows[0])
    return math.sqrt(float(resid @ resid) / (len(ys) - k)), len(ys)


def balance(inputs: Inputs, set_key: str = "P1") -> dict[str, float]:
    """Cross-country correlation of total exposure with industry and export shares."""
    exposure = [sum(inputs.exposure[g][p] for p in partners_for(set_key, g)) for g in EU27]
    out = {}
    for name in ("industry_share", "export_share"):
        values = [inputs.balance[g][name] for g in EU27]
        out[name] = float(np.corrcoef(exposure, values)[0, 1])
    return out
