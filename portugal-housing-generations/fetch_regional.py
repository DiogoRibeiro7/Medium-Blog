"""Download and pin the INE series used by the regional and affordability panels.

For each indicator in ``INDICATORS`` (or ``AFFORDABILITY`` with ``--set
affordability``) the script requests INE's metadata and then the data, as many
periods per request as INE's 40,000-value limit allows, and writes every response
body unchanged into a gzipped JSON-lines container
``data/ine_<code>_<YYYY-MM-DD>.jsonl.gz``. Each line records the URL, retrieval
time, HTTP status, text encoding, byte count and SHA-256 of the raw body next
to the body itself, so ``regional.py`` can check every response it reads. The
first line is a header with the number of responses that follow. With ``--set
rental`` it also downloads the Eurostat datasets in ``EUROSTAT_RENTAL``, writing
each raw body with a ``.meta.json`` retrieval record (URL, time, size, SHA-256).

Nothing is parsed or transformed here. Run on a GitHub runner (the workflow job
``fetch-regional``); standard library only.

Run:  python fetch_regional.py --out data [--set regional|affordability]
"""

from __future__ import annotations

import argparse
import datetime as dt
import gzip
import hashlib
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

META_URL = "https://www.ine.pt/ine/json_indicador/pindicaMeta.jsp"
DATA_URL = "https://www.ine.pt/ine/json_indicador/pindica.jsp"

# varcd -> (short name, years, fixed filters on non-time, non-geographic dimensions)
INDICATORS: dict[str, tuple[str, list[int], dict[str, str]]] = {
    # Resident population, annual estimates (INE), totals for both sexes and all ages.
    "0012917": ("population_nuts2024", list(range(2021, 2026)), {"Dim3": "T", "Dim4": "T"}),
    "0008273": ("population_nuts2013", list(range(2011, 2024)), {"Dim3": "T", "Dim4": "T"}),
    # Dwellings in new construction for family housing, all typologies.
    "0013248": ("completions_nuts2024", list(range(2011, 2026)), {"Dim3": "T"}),
    "0013235": ("permits_nuts2024", list(range(2011, 2026)), {"Dim3": "T"}),
    # Census private households.
    "0014406": ("households_census_series", [2011, 2021], {}),
    "0014627": ("households_census2021_nuts2024", [2021], {"Dim3": "T", "Dim4": "T", "Dim5": "T"}),
}
# Price and income series for the affordability panel. Periods: "q4" = the fourth-quarter
# codes of a quarterly indicator (12-month windows ending in December), "all" = every period.
AFFORDABILITY: dict[str, tuple[str, str, dict[str, str]]] = {
    # Median value per m2 of family dwellings sold, last 12 months, method 2022, NUTS 2024; all categories.
    "0012234": ("sales_price_m2_nuts2024", "q4", {}),
    # The same statistic under the 2018 method, NUTS 2013 (to 2021 Q3).
    "0009490": ("sales_price_m2_method2018_nuts2013", "q4", {}),
    # Median bank appraisal value per m2, annual, municipalities 2024.
    "0012256": ("appraisal_m2_nuts2024", "all", {}),
    # Median declared gross income per tax household / net of IRS / per taxpayer (IRS-based, NUTS 2024).
    "0012712": ("income_gross_household_nuts2024", "all", {}),
    "0012741": ("income_net_household_nuts2024", "all", {}),
    "0012749": ("income_gross_taxpayer_nuts2024", "all", {}),
    # Median declared gross income per tax household, NUTS 2013 (earlier years).
    "0009763": ("income_gross_household_nuts2013", "all", {}),
}
# Rent, tenure and housing-cost series for the rental-market analysis (see rental.py).
RENTAL: dict[str, tuple[str, str, dict[str, str]]] = {
    # New lease contracts (INE local rent statistics, tax-authority records): median rent per m2,
    # number of contracts and quartiles; method 2026, NUTS 2024, calendar years 2020-2025.
    "0014698": ("new_lease_median_rent_m2_method2026", "all", {}),
    "0014697": ("new_lease_contracts_method2026", "all", {}),
    "0014711": ("new_lease_rent_m2_quartiles_method2026", "all", {}),
    # The same statistics under the 2021 method (NUTS 2024, 2020-2024; NUTS 2013, 2017-2023).
    "0012600": ("new_lease_median_rent_m2_method2021_nuts2024", "all", {}),
    "0012601": ("new_lease_contracts_method2021_nuts2024", "all", {}),
    "0009631": ("new_lease_median_rent_m2_method2021_nuts2013", "all", {}),
    "0009632": ("new_lease_contracts_method2021_nuts2013", "all", {}),
    # New leases whose tenant is resident in Portugal: median rent per m2 and contracts, 12 months to Q4.
    "0014700": ("new_lease_median_rent_m2_resident_tenants", "q4", {}),
    "0014699": ("new_lease_contracts_resident_tenants", "q4", {}),
    # Censuses: rented share and mean monthly rent of rented dwellings (the whole stock of tenancies).
    "0012521": ("census2021_rented_share_nuts2024", "all", {}),
    "0012522": ("census2021_mean_rent_nuts2024", "all", {}),
    "0006993": ("census2011_rented_share", "all", {}),
    "0006995": ("census2011_mean_rent", "all", {}),
    "0000977": ("census1991_2001_rented_share", "all", {}),
    "0000777": ("census2001_mean_rent", "all", {}),
    # Census 2021 tenancies: contract period, rental support, years in the dwelling, rent band by contract period.
    "0012508": ("census2021_tenancies_by_contract_period", "all", {"Dim3": "T"}),
    "0012529": ("census2021_tenancies_by_rental_support", "all", {"Dim4": "T"}),
    "0011522": ("census2021_unsupported_tenancies_by_contract_period_nuts2013", "all", {"Dim3": "2"}),
    "0012528": ("census2021_tenancies_by_years_in_dwelling", "all", {"Dim3": "T"}),
    "0011501": ("census2021_tenancies_rent_band_by_contract_period_portugal", "all", {"Dim2": "PT"}),
    # Housing cost overburden rate (EU-SILC as published by INE), NUTS 2.
    "0006260": ("overburden_nuts2_a", "all", {}),
    "0014472": ("overburden_nuts2_b", "all", {}),
}
# Eurostat datasets for the rental-market analysis: file tag -> dataset and query (Portugal only).
EUROSTAT_URL = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/"
EUROSTAT_RENTAL: dict[str, str] = {
    "ilc_lvho07c_PT": "ilc_lvho07c?geo=PT&unit=PC",
    "ilc_lvho02_PT_by_poverty": "ilc_lvho02?geo=PT&hhcomp=TOTAL&unit=PC",
    "ilc_mded02_PT": "ilc_mded02?geo=PT&hhcomp=TOTAL&unit=PC",
    "ilc_di09_PT": "ilc_di09?geo=PT&statinfo=MED_EI&unit=EUR&sex=T&age=TOTAL",
    "ilc_lvho25_PT": "ilc_lvho25?geo=PT&sex=T&age=Y_GE18&unit=PC",
    "ilc_lvho07d_PT": "ilc_lvho07d?geo=PT&unit=PC",
    "prc_hicp_aind_PT_rents": "prc_hicp_aind?geo=PT&geo=EA20&coicop=CP00&coicop=CP041&unit=INX_A_AVG",
}
SETS = {"regional": "INDICATORS", "affordability": "AFFORDABILITY", "rental": "RENTAL"}
PAUSE_SECONDS = 10
MAX_CELLS = 39_000  # INE serves at most 40,000 values per request


LOG: list[Path] = []


def log(message: str) -> None:
    """Print a progress line and append it to the log file, if one is set."""
    line = f"{dt.datetime.now(dt.timezone.utc):%H:%M:%S} {message}"
    print(line, flush=True)
    for path in LOG:
        with path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")


def request(url: str, timeout: float = 300.0, attempts: int = 6) -> tuple[bytes, int, str]:
    """GET ``url``; on HTTP 429 or a network error wait (honouring Retry-After) and retry."""
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Medium-Blog reproducibility fetch"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (fixed https host)
                charset = resp.headers.get_content_charset() or "utf-8"
                return resp.read(), resp.status, charset
        except urllib.error.HTTPError as exc:
            if exc.code != 429:
                raise
            last_error = exc
            time.sleep(min(int(exc.headers.get("Retry-After") or 60 * (attempt + 1)), 300))
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            last_error = exc
            log(f"network error on attempt {attempt + 1}: {exc!r}")
            if attempt >= 2:
                break  # an unreachable host does not come back within minutes; fail and let the job be rerun
            time.sleep(30)
    assert last_error is not None
    raise last_error


def record(url: str, body: bytes, status: int, charset: str, kind: str) -> dict[str, object]:
    """One container line: provenance fields plus the body decoded losslessly."""
    text = body.decode(charset)
    if text.encode(charset) != body:
        raise ValueError(f"Body of {url} does not round-trip through {charset}")
    return {
        "kind": kind,
        "url": url,
        "retrieved_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "http_status": status,
        "encoding": charset,
        "bytes": len(body),
        "sha256": hashlib.sha256(body).hexdigest(),
        "body": text,
    }


def cells_per_period(metadata: bytes, charset: str, filters: dict[str, str]) -> int:
    """Values in one period: geographic categories times every other dimension left unfiltered."""
    item = json.loads(metadata.decode(charset))[0]
    categories = item["Dimensoes"]["Categoria_Dim"][0]
    total = 1
    for dim in item["Dimensoes"]["Descricao_Dim"]:
        n = dim["dim_num"]
        if n == "1" or f"Dim{n}" in filters:
            continue
        total *= sum(1 for key in categories if key.startswith(f"Dim_Num{n}_"))
    return total


def period_codes(metadata: bytes, charset: str, selection: str | list[int]) -> list[str]:
    """Period codes to request: explicit years (annual), every period, or fourth quarters."""
    if isinstance(selection, list):
        return [f"S7A{y}" for y in selection]
    item = json.loads(metadata.decode(charset))[0]
    categories = item["Dimensoes"]["Categoria_Dim"][0]
    codes = [v[0] for k, v in categories.items() if k.startswith("Dim_Num1_")]
    codes = [c["categ_cod"] for c in sorted(codes, key=lambda c: int(c["categ_ord"]))]
    if selection == "all":
        return codes
    if selection == "q4":
        return [c for c in codes if c.startswith("S5A") and c.endswith("4")]
    raise ValueError(f"Unknown period selection {selection!r}")


def returned_periods(body: bytes, charset: str) -> set[str]:
    """Period labels present in a data response (empty if INE reports a failure)."""
    item = json.loads(body.decode(charset))[0]
    if "Verdadeiro" not in item.get("Sucesso", {}):
        return set()
    return {period for period, rows in item.get("Dados", {}).items() if rows}


def fetch_indicator(code: str, selection: str | list[int], filters: dict[str, str]) -> tuple[list[dict[str, object]], str, list[str]]:
    """Metadata plus the data, as many periods per request as INE's cell limit allows."""
    lines = []
    meta_url = f"{META_URL}?{urllib.parse.urlencode({'varcd': code, 'lang': 'PT'})}"
    meta_body, meta_status, meta_charset = request(meta_url)
    lines.append(record(meta_url, meta_body, meta_status, meta_charset, kind="metadata"))
    periods = period_codes(meta_body, meta_charset, selection)
    per_request = max(1, MAX_CELLS // cells_per_period(meta_body, meta_charset, filters))
    for i in range(0, len(periods), per_request):
        batch = periods[i:i + per_request]
        query = {"op": "2", "varcd": code, "lang": "PT", "Dim1": ",".join(batch), **filters}
        url = f"{DATA_URL}?{urllib.parse.urlencode(query)}"
        log(f"{code}: requesting {len(batch)} periods")
        body, status, charset = request(url)
        log(f"{code}: {len(body)} bytes")
        if len(returned_periods(body, charset)) != len(batch):
            raise ValueError(f"{code}: response for {batch} has periods {sorted(returned_periods(body, charset))}")
        lines.append(record(url, body, status, charset, kind="data"))
        time.sleep(PAUSE_SECONDS)
    return lines, f"all geographic levels; up to {per_request} periods per request", periods


def fetch_eurostat(out: Path, today: str) -> int:
    """Download each Eurostat dataset in ``EUROSTAT_RENTAL``; write the raw body and a retrieval record."""
    failures = 0
    for tag, query in EUROSTAT_RENTAL.items():
        url = EUROSTAT_URL + query
        try:
            body, status, _charset = request(url)
            document = json.loads(body)
        except Exception as exc:  # noqa: BLE001 - report and continue with the others
            log(f"eurostat {tag}: FAILED {exc!r}")
            failures += 1
            continue
        path = out / f"eurostat_{tag}_{today}.json"
        path.write_bytes(body)
        record = {"url": url, "retrieved_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                  "http_status": status, "bytes": len(body), "sha256": hashlib.sha256(body).hexdigest(),
                  "dataset_label": document.get("label"), "dataset_updated": document.get("updated")}
        path.with_suffix(".meta.json").write_text(json.dumps(record, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        log(f"eurostat {tag}: {len(body)} bytes -> {path.name}")
        time.sleep(2)
    return failures


def main() -> int:
    """Fetch every indicator and write one container per indicator."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "data")
    parser.add_argument("--only", nargs="*", default=None, help="indicator codes to fetch (default: all in the set)")
    parser.add_argument("--set", choices=sorted(SETS), default="regional", help="which group of indicators to fetch")
    parser.add_argument("--log", type=Path, default=None, help="also append progress lines to this file")
    args = parser.parse_args()
    if args.log:
        LOG.append(args.log)
    try:  # one short request first: if INE is unreachable from this runner, stop now rather than after hours of retries
        request(f"{META_URL}?varcd=0008273&lang=PT", timeout=45, attempts=1)
        log("INE reachable")
    except Exception as exc:  # noqa: BLE001
        log(f"INE unreachable: {exc!r}")
        return 2
    args.out.mkdir(parents=True, exist_ok=True)
    today = dt.datetime.now(dt.timezone.utc).date().isoformat()
    failures = 0
    sets: dict[str, dict[str, tuple[str, str | list[int], dict[str, str]]]] = {
        "regional": dict(INDICATORS), "affordability": dict(AFFORDABILITY), "rental": dict(RENTAL)}
    chosen = sets[args.set]
    if args.set == "rental" and not args.only:
        failures += fetch_eurostat(args.out, today)
    for code, (name, selection, filters) in chosen.items():
        if args.only and code not in args.only:
            continue
        try:
            lines, mode, periods = fetch_indicator(code, selection, filters)
        except Exception as exc:  # noqa: BLE001 - report and continue with the others
            log(f"{code} {name}: FAILED {exc!r}")
            failures += 1
            continue
        header = {"kind": "header", "indicator": code, "name": name, "filters": filters, "geography": mode,
                  "periods": periods, "responses": len(lines)}
        if isinstance(selection, list):
            header["years"] = selection
        path = args.out / f"ine_{code}_{today}.jsonl.gz"
        with gzip.open(path, "wt", encoding="utf-8", compresslevel=9) as fh:
            for line in (header, *lines):
                fh.write(json.dumps(line, ensure_ascii=False) + "\n")
        log(f"{code} {name}: {len(lines)} responses, {mode} -> {path.name}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
