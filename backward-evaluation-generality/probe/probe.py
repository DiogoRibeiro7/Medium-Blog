"""Availability probe for the generality study (run before the pre-registration).

    python probe/probe.py --out probe

Records metadata only, never values:

* Open Power System Data, time series package: the version, and for every column named
  ``<CC>_load_actual_entsoe_transparency`` the first and last hour with data, the number of hours
  with data and the number of complete days (24 hours with data);
* Wikipedia: for 193 UN member states and 118 chemical elements, the canonical English Wikipedia
  title (redirects resolved) and, from the Wikimedia pageviews API (agent = user, daily,
  2015-07-01 to 2026-10-31), the first and last day returned and the number of days returned.

Outputs ``probe/opsd_columns.tsv``, ``probe/wikipedia_titles.tsv`` and ``probe/probe_log.txt``.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

UA = "backward-evaluation-generality/1.0 (https://github.com/DiogoRibeiro7/Medium-Blog; research code)"
OPSD_META = "https://data.open-power-system-data.org/time_series/latest/datapackage.json"
OPSD_CSV = "https://data.open-power-system-data.org/time_series/{version}/time_series_60min_singleindex.csv"
MW_API = "https://en.wikipedia.org/w/api.php"
PV_API = ("https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/user/"
          "{title}/daily/20150701/20261031")

COUNTRIES = """Afghanistan|Albania|Algeria|Andorra|Angola|Antigua and Barbuda|Argentina|Armenia|Australia|Austria|
Azerbaijan|The Bahamas|Bahrain|Bangladesh|Barbados|Belarus|Belgium|Belize|Benin|Bhutan|Bolivia|
Bosnia and Herzegovina|Botswana|Brazil|Brunei|Bulgaria|Burkina Faso|Burundi|Cambodia|Cameroon|Canada|Cape Verde|
Central African Republic|Chad|Chile|China|Colombia|Comoros|Republic of the Congo|Costa Rica|Ivory Coast|Croatia|Cuba|
Cyprus|Czech Republic|Democratic Republic of the Congo|Denmark|Djibouti|Dominica|Dominican Republic|East Timor|Ecuador|
Egypt|El Salvador|Equatorial Guinea|Eritrea|Estonia|Eswatini|Ethiopia|Fiji|Finland|France|Gabon|The Gambia|
Georgia (country)|Germany|Ghana|Greece|Grenada|Guatemala|Guinea|Guinea-Bissau|Guyana|Haiti|Honduras|Hungary|Iceland|
India|Indonesia|Iran|Iraq|Republic of Ireland|Israel|Italy|Jamaica|Japan|Jordan|Kazakhstan|Kenya|Kiribati|North Korea|
South Korea|Kuwait|Kyrgyzstan|Laos|Latvia|Lebanon|Lesotho|Liberia|Libya|Liechtenstein|Lithuania|Luxembourg|Madagascar|
Malawi|Malaysia|Maldives|Mali|Malta|Marshall Islands|Mauritania|Mauritius|Mexico|Federated States of Micronesia|
Moldova|Monaco|Mongolia|Montenegro|Morocco|Mozambique|Myanmar|Namibia|Nauru|Nepal|Netherlands|New Zealand|Nicaragua|
Niger|Nigeria|North Macedonia|Norway|Oman|Pakistan|Palau|Panama|Papua New Guinea|Paraguay|Peru|Philippines|Poland|
Portugal|Qatar|Romania|Russia|Rwanda|Saint Kitts and Nevis|Saint Lucia|Saint Vincent and the Grenadines|Samoa|
San Marino|São Tomé and Príncipe|Saudi Arabia|Senegal|Serbia|Seychelles|Sierra Leone|Singapore|Slovakia|Slovenia|
Solomon Islands|Somalia|South Africa|South Sudan|Spain|Sri Lanka|Sudan|Suriname|Sweden|Switzerland|Syria|Tajikistan|
Tanzania|Thailand|Togo|Tonga|Trinidad and Tobago|Tunisia|Turkey|Turkmenistan|Tuvalu|Uganda|Ukraine|
United Arab Emirates|United Kingdom|United States|Uruguay|Uzbekistan|Vanuatu|Venezuela|Vietnam|Yemen|Zambia|Zimbabwe"""
ELEMENTS = """Hydrogen|Helium|Lithium|Beryllium|Boron|Carbon|Nitrogen|Oxygen|Fluorine|Neon|Sodium|Magnesium|Aluminium|
Silicon|Phosphorus|Sulfur|Chlorine|Argon|Potassium|Calcium|Scandium|Titanium|Vanadium|Chromium|Manganese|Iron|Cobalt|
Nickel|Copper|Zinc|Gallium|Germanium|Arsenic|Selenium|Bromine|Krypton|Rubidium|Strontium|Yttrium|Zirconium|Niobium|
Molybdenum|Technetium|Ruthenium|Rhodium|Palladium|Silver|Cadmium|Indium|Tin|Antimony|Tellurium|Iodine|Xenon|Caesium|
Barium|Lanthanum|Cerium|Praseodymium|Neodymium|Promethium|Samarium|Europium|Gadolinium|Terbium|Dysprosium|Holmium|
Erbium|Thulium|Ytterbium|Lutetium|Hafnium|Tantalum|Tungsten|Rhenium|Osmium|Iridium|Platinum|Gold|Mercury (element)|
Thallium|Lead|Bismuth|Polonium|Astatine|Radon|Francium|Radium|Actinium|Thorium|Protactinium|Uranium|Neptunium|
Plutonium|Americium|Curium|Berkelium|Californium|Einsteinium|Fermium|Mendelevium|Nobelium|Lawrencium|Rutherfordium|
Dubnium|Seaborgium|Bohrium|Hassium|Meitnerium|Darmstadtium|Roentgenium|Copernicium|Nihonium|Flerovium|Moscovium|
Livermorium|Tennessine|Oganesson"""


def split(block: str) -> list[str]:
    return [t.strip() for t in block.replace("\n", "").split("|") if t.strip()]


def get(url: str, tries: int = 3, timeout: int = 120) -> bytes:
    last: Exception | None = None
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "*/*"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                raise
            last = e
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            last = e
        time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"failed after {tries} tries: {last}")


def log(out: Path, msg: str) -> None:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    with open(out / "probe_log.txt", "a", encoding="utf-8") as f:
        f.write(f"{stamp}\t{msg}\n")


def probe_opsd(out: Path) -> None:
    meta = json.loads(get(OPSD_META))
    version = meta.get("version") or meta.get("id", "unknown")
    log(out, f"OPSD latest version: {version}")
    url = OPSD_CSV.format(version=version)
    body = get(url, timeout=600)
    log(out, f"{url}\tsha256 {hashlib.sha256(body).hexdigest()}\t{len(body)} bytes")
    head = pd.read_csv(io.BytesIO(body), nrows=0).columns
    cols = [c for c in head if re.fullmatch(r"[A-Z]{2}_load_actual_entsoe_transparency", c)]
    df = pd.read_csv(io.BytesIO(body), usecols=["utc_timestamp", *cols], parse_dates=["utc_timestamp"])
    rows = []
    for c in cols:
        s = df.set_index("utc_timestamp")[c].dropna()
        if s.empty:
            rows.append([c, "", "", 0, 0])
            continue
        per_day = s.groupby(s.index.floor("D")).size()
        rows.append([c, f"{s.index.min():%Y-%m-%d %H:%M}", f"{s.index.max():%Y-%m-%d %H:%M}", len(s),
                     int((per_day == 24).sum())])
    pd.DataFrame(rows, columns=["column", "first_hour", "last_hour", "hours_with_data", "complete_days"]).to_csv(
        out / "opsd_columns.tsv", sep="\t", index=False)
    other = [c for c in head if "load_actual" in c and c not in cols]
    log(out, f"OPSD: {len(cols)} country-level load columns; {len(other)} other load_actual columns: {', '.join(other)}")


def resolve(titles: list[str]) -> dict[str, tuple[str, bool]]:
    """Original title -> (canonical title, missing?)."""
    res: dict[str, tuple[str, bool]] = {}
    for i in range(0, len(titles), 50):
        chunk = titles[i : i + 50]
        q = urllib.parse.urlencode({"action": "query", "titles": "|".join(chunk), "redirects": 1, "format": "json",
                                    "formatversion": 2})
        doc = json.loads(get(f"{MW_API}?{q}"))["query"]
        norm = {d["from"]: d["to"] for d in doc.get("normalized", [])}
        redir = {d["from"]: d["to"] for d in doc.get("redirects", [])}
        pages = {p["title"]: p for p in doc.get("pages", [])}
        for t in chunk:
            c = norm.get(t, t)
            c = redir.get(c, c)
            p = pages.get(c, {})
            res[t] = (c, bool(p.get("missing", False)) or not p)
        time.sleep(0.2)
    return res


def probe_wikipedia(out: Path) -> None:
    rows = []
    for group, titles in (("country", split(COUNTRIES)), ("element", split(ELEMENTS))):
        log(out, f"Wikipedia {group}: {len(titles)} candidate titles")
        canon = resolve(titles)
        for t in titles:
            c, missing = canon[t]
            first = last = ""
            n = 0
            status = "missing page" if missing else "ok"
            if not missing:
                url = PV_API.format(title=urllib.parse.quote(c.replace(" ", "_"), safe=""))
                try:
                    items = json.loads(get(url))["items"]
                    days = sorted(it["timestamp"][:8] for it in items)
                    if days:
                        first, last, n = days[0], days[-1], len(days)
                except Exception as e:  # noqa: BLE001 - recorded
                    status = f"error: {str(e)[:80]}"
                time.sleep(0.05)
            rows.append([group, t, c, status, first, last, n])
    pd.DataFrame(rows, columns=["group", "candidate", "canonical", "status", "first_day", "last_day", "days_returned"]
                 ).to_csv(out / "wikipedia_titles.tsv", sep="\t", index=False)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="probe")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, fn in (("OPSD", probe_opsd), ("Wikipedia", probe_wikipedia)):
        try:
            fn(out)
        except Exception as e:  # noqa: BLE001 - recorded, the other source still runs
            log(out, f"{name}: ERROR {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
