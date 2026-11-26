"""Profile the Companies House bulk files before building the founder-exits cohort.

Runs on a GitHub runner (Companies House is not reachable from every machine). It downloads the
latest Free Company Data Product and the PSC snapshot, streams through both, and writes
aggregate counts only (no names, no dates of birth, no company-level rows) to a JSON file:

    python ch_probe.py --output-dir out/

If COMPANIES_HOUSE_API_KEY is set, it also makes a handful of API calls to check the advanced
search and filing-history endpoints, and records status codes and counts.
"""

import argparse
import base64
import collections
import csv
import io
import json
import os
import re
import sys
import time
import urllib.request
import zipfile
from datetime import UTC, date, datetime
from pathlib import Path

DOWNLOAD = "https://download.companieshouse.gov.uk/"
API = "https://api.company-information.service.gov.uk"
UA = "Mozilla/5.0 (compatible; Medium-Blog uk-founder-exits probe)"

TECH_SIC = {
    "58210", "58290", "62011", "62012", "63110", "63120", "72110", "72190", "21100", "21200", "32500",
    "26110", "26120", "26200", "26301", "26309", "26400", "26511", "26512", "26513", "26514", "26520",
    "26600", "26701", "26702", "26800",
}  # fmt: skip
CONTRACTOR_SIC = {"62020", "62030", "62090"}
COHORT_FROM, COHORT_TO = date(2016, 7, 1), date(2020, 12, 31)
UK_PLACE = re.compile(
    r"england|wales|scotland|united kingdom|\buk\b|northern ireland|great britain|companies house|companies act"
)


def get(url, timeout=600, headers=None):
    req = urllib.request.Request(url, headers={"User-Agent": UA, **(headers or {})})
    return urllib.request.urlopen(req, timeout=timeout)


def latest_link(page, pattern):
    html = get(DOWNLOAD + page).read().decode("utf-8", "replace")
    links = sorted(set(re.findall(pattern, html)))
    if not links:
        raise RuntimeError(f"no file matching {pattern} on {page}")
    return links[-1]


def download(name, target):
    with get(DOWNLOAD + name, timeout=3600) as r, open(target, "wb") as f:
        while chunk := r.read(1 << 22):
            f.write(chunk)
    return target.stat().st_size


def parse_date(s):
    s = (s or "").strip()
    for fmt in ("%d/%m/%Y", "%Y-%m-%d"):
        try:
            return time.strptime(s, fmt)
        except ValueError:
            pass
    return None


def as_date(t):
    return date(t.tm_year, t.tm_mon, t.tm_mday) if t else None


def sic_codes(row):
    codes = []
    for k in ("SICCode.SicText_1", "SICCode.SicText_2", "SICCode.SicText_3", "SICCode.SicText_4"):
        v = (row.get(k) or "").strip()
        if v:
            codes.append(v.split(" ")[0])
    return codes


def profile_basic(zpath, profile):
    status, years, acct, tech_year, contractor_year = (collections.Counter() for _ in range(5))
    dissolution_nonempty = rows = 0
    cohort = {}
    with zipfile.ZipFile(zpath) as z:
        inner = z.namelist()[0]
        with z.open(inner) as raw:
            reader = csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8", errors="replace"))
            reader.fieldnames = [f.strip() for f in reader.fieldnames]
            profile["basic_columns"] = reader.fieldnames
            for row in reader:
                rows += 1
                status[row.get("CompanyStatus", "").strip()] += 1
                if (row.get("DissolutionDate") or "").strip():
                    dissolution_nonempty += 1
                inc = as_date(parse_date(row.get("IncorporationDate")))
                if inc:
                    years[inc.year] += 1
                codes = sic_codes(row)
                if any(c in TECH_SIC for c in codes) and inc:
                    tech_year[inc.year] += 1
                    if COHORT_FROM <= inc <= COHORT_TO:
                        num = row.get("CompanyNumber", "").strip()
                        cohort[num] = (inc.isoformat(), row.get("CompanyStatus", "").strip())
                        acct[row.get("Accounts.AccountCategory", "").strip()] += 1
                if any(c in CONTRACTOR_SIC for c in codes) and inc:
                    contractor_year[inc.year] += 1
    profile["basic"] = {
        "rows": rows,
        "status": dict(status.most_common()),
        "dissolution_date_nonempty": dissolution_nonempty,
        "incorporation_year": {str(k): v for k, v in sorted(years.items()) if k >= 2000},
        "tech_sic_by_incorporation_year": {str(k): v for k, v in sorted(tech_year.items()) if k >= 2000},
        "contractor_sic_by_incorporation_year": {str(k): v for k, v in sorted(contractor_year.items()) if k >= 2000},
        "tech_cohort_live_size": len(cohort),
        "tech_cohort_account_category": dict(acct.most_common()),
    }
    return cohort


def norm(s):
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def profile_psc(zpath, cohort, basic_numbers, profile):
    kinds, ceased, nat, corp_country, corp_keys, natures = (collections.Counter() for _ in range(6))
    lines = 0
    psc_companies = set()
    cohort_stats = collections.defaultdict(lambda: {"ind": 0, "corp": 0, "corp_foreign": 0, "first_notified": None})
    with zipfile.ZipFile(zpath) as z:
        inner = z.namelist()[0]
        with z.open(inner) as raw:
            for line in io.TextIOWrapper(raw, encoding="utf-8", errors="replace"):
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                num, d = rec.get("company_number"), rec.get("data") or {}
                if not num:
                    continue
                lines += 1
                psc_companies.add(num)
                kind = d.get("kind", "")
                kinds[kind] += 1
                if d.get("ceased_on"):
                    ceased[kind] += 1
                for n in d.get("natures_of_control") or []:
                    natures[n] += 1
                if "individual" in kind:
                    nat[norm(d.get("nationality"))] += 1
                if "corporate" in kind or "legal-person" in kind:
                    ident = d.get("identification") or {}
                    for k in ident:
                        corp_keys[k] += 1
                    corp_country[norm(ident.get("country_registered") or ident.get("place_registered"))] += 1
                if num in cohort:
                    c = cohort_stats[num]
                    if "individual" in kind:
                        c["ind"] += 1
                    elif "corporate" in kind or "legal-person" in kind:
                        c["corp"] += 1
                        ident = d.get("identification") or {}
                        place = norm(
                            " ".join(
                                str(ident.get(k, ""))
                                for k in ("country_registered", "place_registered", "legal_authority")
                            )
                        )
                        if place and not UK_PLACE.search(place):
                            c["corp_foreign"] += 1
                    nd = d.get("notified_on")
                    if nd and (c["first_notified"] is None or nd < c["first_notified"]):
                        c["first_notified"] = nd
    lag = collections.Counter()
    for num, c in cohort_stats.items():
        inc = date.fromisoformat(cohort[num][0])
        if c["first_notified"]:
            days = (date.fromisoformat(c["first_notified"][:10]) - inc).days
            lag["<=30" if days <= 30 else "31-365" if days <= 365 else ">365"] += 1
    companies_not_in_basic = len(psc_companies - basic_numbers)
    profile["psc"] = {
        "records": lines,
        "companies": len(psc_companies),
        "companies_not_in_basic_file": companies_not_in_basic,
        "kinds": dict(kinds.most_common()),
        "ceased_by_kind": dict(ceased.most_common()),
        "natures_of_control_top": dict(natures.most_common(25)),
        "individual_nationality_top": dict(nat.most_common(40)),
        "corporate_identification_keys": dict(corp_keys.most_common()),
        "corporate_country_top": dict(corp_country.most_common(40)),
        "tech_cohort_companies_with_psc": len(cohort_stats),
        "tech_cohort_with_corporate_psc": sum(1 for c in cohort_stats.values() if c["corp"]),
        "tech_cohort_with_foreign_corporate_psc": sum(1 for c in cohort_stats.values() if c["corp_foreign"]),
        "tech_cohort_first_notified_lag_days": dict(lag),
    }


def api_checks(profile):
    key = os.environ.get("COMPANIES_HOUSE_API_KEY", "").strip()
    if not key:
        profile["api"] = "COMPANIES_HOUSE_API_KEY not set; API checks skipped"
        return
    auth = "Basic " + base64.b64encode(f"{key}:".encode()).decode()
    out = {}

    def call(path):
        try:
            with get(API + path, timeout=60, headers={"Authorization": auth}) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, None

    code, body = call(
        "/advanced-search/companies?incorporated_from=2017-03-01&incorporated_to=2017-03-31&sic_codes=62012&size=5000"
    )
    out["advanced_search_status"] = code
    if body:
        items = body.get("items") or []
        out["advanced_search_hits"] = body.get("hits")
        out["advanced_search_items"] = len(items)
        out["advanced_search_status_counts"] = dict(collections.Counter(i.get("company_status") for i in items))
        out["advanced_search_item_keys"] = sorted({k for i in items[:50] for k in i})
        sample = [i["company_number"] for i in items[:20]]
        sh01 = 0
        for num in sample:
            c2, fh = call(f"/company/{num}/filing-history?category=capital&items_per_page=100")
            if fh:
                sh01 += any((it.get("type") == "SH01") for it in fh.get("items") or [])
            out.setdefault("filing_history_status", collections.Counter())[str(c2)] += 1
            time.sleep(0.6)
        out["filing_history_sample_with_sh01"] = f"{sh01} of {len(sample)}"
        num = sample[0] if sample else None
        if num:
            c3, _psc = call(f"/company/{num}/persons-with-significant-control")
            out["psc_endpoint_status"] = c3
            c4, off = call(f"/company/{num}/officers")
            out["officers_endpoint_status"] = c4
            if off:
                out["officer_item_keys"] = sorted({k for i in off.get("items") or [] for k in i})
    if "filing_history_status" in out:
        out["filing_history_status"] = dict(out["filing_history_status"])
    profile["api"] = out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--work-dir", type=Path, default=Path("/tmp/ch"))
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    profile = {"run_date": datetime.now(UTC).date().isoformat()}
    try:
        basic_name = latest_link("en_output.html", r"BasicCompanyDataAsOneFile-\d{4}-\d{2}-\d{2}\.zip")
        psc_name = latest_link("en_pscdata.html", r"persons-with-significant-control-snapshot-\d{4}-\d{2}-\d{2}\.zip")
        profile["files"] = {"basic": basic_name, "psc": psc_name}
        t0 = time.time()
        profile["files"]["basic_bytes"] = download(basic_name, args.work_dir / "basic.zip")
        profile["files"]["psc_bytes"] = download(psc_name, args.work_dir / "psc.zip")
        profile["download_seconds"] = round(time.time() - t0)
        cohort = profile_basic(args.work_dir / "basic.zip", profile)
        with zipfile.ZipFile(args.work_dir / "basic.zip") as z, z.open(z.namelist()[0]) as raw:
            reader = csv.reader(io.TextIOWrapper(raw, encoding="utf-8", errors="replace"))
            header = [h.strip() for h in next(reader)]
            idx = header.index("CompanyNumber")
            numbers = {r[idx].strip() for r in reader if len(r) > idx}
        profile_psc(args.work_dir / "psc.zip", cohort, numbers, profile)
    except Exception as e:  # keep whatever was measured
        profile["error"] = f"{type(e).__name__}: {e}"
    api_checks(profile)
    (args.output_dir / "ch_probe.json").write_text(json.dumps(profile, indent=2, default=str))
    print(json.dumps({k: v for k, v in profile.items() if k in ("files", "error", "api")}, indent=2, default=str))
    return 1 if "error" in profile else 0


if __name__ == "__main__":
    sys.exit(main())
