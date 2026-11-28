"""Download the OECD SDMX extracts used in the analysis (nothing is stored in the repository).

Usage: python3 fetch_data.py RAW_DIR

Writes to RAW_DIR: am.csv, DF_LE.csv, DF_HEALTH_LVNG_TC.csv, DF_HEALTH_LVNG_AC.csv,
gdp_all.csv, pophist.csv, excess.csv, sha_slim.csv
"""
import csv
import io
import sys
import time
import urllib.request

BASE = "https://sdmx.oecd.org/public/rest/data"
FLOWS = {
    "am.csv": "OECD.ELS.HD,DSD_HEALTH_STAT@DF_AM,/all?startPeriod=2000&format=csvfile",
    "DF_LE.csv": "OECD.ELS.HD,DSD_HEALTH_STAT@DF_LE,/all?startPeriod=2000&format=csvfile",
    "DF_HEALTH_LVNG_TC.csv": "OECD.ELS.HD,DSD_HEALTH_LVNG@DF_HEALTH_LVNG_TC,/all?startPeriod=2000&format=csvfile",
    "DF_HEALTH_LVNG_AC.csv": "OECD.ELS.HD,DSD_HEALTH_LVNG@DF_HEALTH_LVNG_AC,/all?startPeriod=2000&format=csvfile",
    "gdp_all.csv": "OECD.SDD.NAD,DSD_NAMAIN10@DF_TABLE1_EXPENDITURE_HVPVOB,/all?startPeriod=2000&format=csvfile",
    "pophist.csv": "OECD.ELS.SAE,DSD_POPULATION@DF_POP_HIST,/all?startPeriod=2000&format=csvfile",
    "excess.csv": "OECD.ELS.HD,DSD_HEALTH_MORTALITY@DF_MORTALITY_EXCESS,/all?startPeriod=2020&format=csvfile",
}

# Health expenditure (SHA): ask only for the series the analysis uses.
# Dimension order: REF_AREA.FREQ.MEASURE.UNIT_MEASURE.FINANCING_SCHEME.FINANCING_SCHEME_REV.FUNCTION.
#                  MODE_PROVISION.PROVIDER.FACTOR_PROVISION.ASSET_TYPE.PRICE_BASE
SHA_KEY = ".A.EXP_HEALTH.USD_PPP_PS+PT_B1GQ+PT_EXP_HLTH._T+HF1+HF3.._T+HC6._T._T..."
SHA_URL = f"OECD.ELS.HD,DSD_SHA@DF_SHA,/{SHA_KEY}?startPeriod=2000&format=csvfile"


def get(path, tries=3):
    url = f"{BASE}/{path}"
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "oecd-health-analysis/1.0"})
            with urllib.request.urlopen(req, timeout=600) as r:
                return r.read().decode("utf-8")
        except Exception as exc:  # network hiccups on large downloads
            print("retry", k + 1, exc, file=sys.stderr)
            time.sleep(5 * (k + 1))
    raise RuntimeError(f"could not download {url}")


def main(raw):
    for name, path in FLOWS.items():
        text = get(path)
        if text.strip() in ("", "NoRecordsFound"):
            raise RuntimeError(f"{name}: no records returned")
        with open(f"{raw}/{name}", "w", encoding="utf-8") as f:
            f.write(text)
        print(name, text.count("\n"), "lines")

    keep_units = {"USD_PPP_PS", "PT_B1GQ", "PT_EXP_HLTH"}
    rows = []
    for x in csv.DictReader(io.StringIO(get(SHA_URL))):
        if x["MODE_PROVISION"] != "_T" or x["PROVIDER"] != "_T" or x["UNIT_MEASURE"] not in keep_units:
            continue
        rows.append((x["REF_AREA"], x["UNIT_MEASURE"], x["FINANCING_SCHEME"], x["FUNCTION"],
                     x["PRICE_BASE"], x["TIME_PERIOD"], x["OBS_VALUE"]))
    with open(f"{raw}/sha_slim.csv", "w", newline="", encoding="utf-8") as g:
        w = csv.writer(g)
        w.writerow(["area", "unit", "fs", "fn", "price", "year", "value"])
        w.writerows(rows)
    print("sha_slim.csv", len(rows), "rows")


if __name__ == "__main__":
    main(sys.argv[1])
