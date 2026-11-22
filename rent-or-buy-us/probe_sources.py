"""One-off probe: which upstream files exist and what they hold, before the sources are registered.

    python probe_sources.py --out probe

Downloads candidate FRED series (BLS rent indices by metro, CPI-U, Freddie Mac mortgage rates,
Treasury rates), Census ACS medians by metropolitan area, and Damodaran's historical returns
(live and from the Wayback Machine). Writes each response to the output folder and one line per
request to probe/log.tsv (URL, final URL, HTTP status, bytes, SHA-256). Nothing is committed.
"""

from __future__ import annotations

import argparse
import hashlib
import time
import urllib.error
import urllib.request
from pathlib import Path

HEADERS = {"User-Agent": "rent-or-buy-us/0.1 (research code; dfr@esmad.ipp.pt)", "Accept": "*/*"}
AREAS_OLD = ["A101", "A103", "A207", "A208", "A210", "A211", "A311", "A316", "A319", "A320", "A321",
             "A421", "A422", "A423", "A424", "A425", "A429", "A433"]
AREAS_NEW = ["S11A", "S12A", "S23A", "S23B", "S24A", "S35A", "S35B", "S35C", "S35D", "S37A", "S48A",
             "S48B", "S49A", "S49B", "S49D", "S49E"]
FRED = (["CUUR0000SEHA", "CUUR0000SA0", "CPIAUCNS", "MORTGAGE30US", "MORTGAGE15US", "GS10", "TB3MS"]
        + [f"CUUR{a}SEHA" for a in AREAS_OLD + AREAS_NEW] + [f"CUUR{a}SA0" for a in AREAS_OLD])
ACS_YEARS = [2010, 2015, 2019, 2023]
ACS = ("https://api.census.gov/data/{y}/acs/acs1?get=NAME,B25064_001E,B25077_001E,B25103_001E,B25105_001E,"
       "B25003_001E,B25003_002E&for=metropolitan%20statistical%20area/micropolitan%20statistical%20area:*")
DAMODARAN = "https://pages.stern.nyu.edu/~adamodar/pc/datasets/histretSP.xls"
WAYBACK = ["https://web.archive.org/web/2025id_/" + DAMODARAN, "https://web.archive.org/web/2026id_/" + DAMODARAN,
           "https://archive.org/wayback/available?url=pages.stern.nyu.edu/~adamodar/pc/datasets/histretSP.xls"]


def fetch(url: str, path: Path, log) -> None:
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=60) as r:
                body = r.read()
                path.write_bytes(body)
                log.write(f"{url}\t{r.geturl()}\t{r.status}\t{len(body)}\t{hashlib.sha256(body).hexdigest()}\t{path.name}\n")
                return
        except urllib.error.HTTPError as e:
            log.write(f"{url}\t-\t{e.code}\t0\t-\t{path.name}\n")
            return
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            if attempt == 2:
                log.write(f"{url}\t-\terror {e}\t0\t-\t{path.name}\n")
            time.sleep(4 * (attempt + 1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="probe")
    out = Path(ap.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "log.tsv", "w") as log:
        log.write("url\tfinal_url\tstatus\tbytes\tsha256\tfile\n")
        for sid in FRED:
            fetch(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}", out / f"fred_{sid}.csv", log)
            time.sleep(0.5)
        for y in ACS_YEARS:
            fetch(ACS.format(y=y), out / f"acs1_{y}.json", log)
        fetch(DAMODARAN, out / "damodaran_live.xls", log)
        for i, u in enumerate(WAYBACK):
            fetch(u, out / f"damodaran_wayback_{i}.bin", log)


if __name__ == "__main__":
    main()
