"""Availability probe for the model-choice test set (run before the pre-registration).

    python probe/probe.py --out probe

For seven fixed lists of English Wikipedia articles that this project has never downloaded, records
metadata only, never values: the canonical title (normalisation and redirects resolved), whether the
page exists, and from the Wikimedia pageviews API (agent = user, daily, 2015-07-01 to 2026-10-31)
the first and last day returned and the number of days returned.

Output: ``probe/titles.tsv`` and ``probe/probe_log.txt``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "gen_probe", HERE.parent.parent / "backward-evaluation-generality" / "probe" / "probe.py")
gen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gen)

LISTS = {
    "us_state": """Alabama|Alaska|Arizona|Arkansas|California|Colorado|Connecticut|Delaware|Florida|Georgia (U.S. state)|
Hawaii|Idaho|Illinois|Indiana|Iowa|Kansas|Kentucky|Louisiana|Maine|Maryland|Massachusetts|Michigan|Minnesota|Mississippi|
Missouri|Montana|Nebraska|Nevada|New Hampshire|New Jersey|New Mexico|New York (state)|North Carolina|North Dakota|Ohio|
Oklahoma|Oregon|Pennsylvania|Rhode Island|South Carolina|South Dakota|Tennessee|Texas|Utah|Vermont|Virginia|
Washington (state)|West Virginia|Wisconsin|Wyoming""",
    "city": """Tokyo|Delhi|Shanghai|São Paulo|Mexico City|Cairo|Mumbai|Beijing|Dhaka|Osaka|New York City|Karachi|Buenos Aires|
Chongqing|Istanbul|Kolkata|Manila|Lagos|Rio de Janeiro|Tianjin|Kinshasa|Guangzhou|Los Angeles|Moscow|Shenzhen|Lahore|
Bangalore|Paris|Bogotá|Jakarta|Chennai|Lima|Bangkok|Seoul|Nagoya|Hyderabad|London|Tehran|Chicago|Chengdu""",
    "solar_system": """Sun|Mercury (planet)|Venus|Earth|Mars|Jupiter|Saturn|Uranus|Neptune|Pluto|Ceres (dwarf planet)|
Eris (dwarf planet)|Haumea|Makemake|Moon|Io (moon)|Europa (moon)|Ganymede (moon)|Callisto (moon)|Titan (moon)|Enceladus|
Triton (moon)|Phobos (moon)|Deimos (moon)|Charon (moon)""",
    "programming_language": """Python (programming language)|Java (programming language)|C (programming language)|C++|
C Sharp (programming language)|JavaScript|PHP|Ruby (programming language)|Go (programming language)|
Rust (programming language)|Swift (programming language)|Kotlin (programming language)|R (programming language)|MATLAB|
Perl|Haskell|Scala (programming language)|Lua (programming language)|Fortran|COBOL|Lisp (programming language)|
Julia (programming language)|TypeScript|SQL|Pascal (programming language)|Assembly language|Erlang (programming language)|
Prolog|Dart (programming language)|Elixir (programming language)""",
    "instrument": """Piano|Guitar|Violin|Cello|Double bass|Flute|Clarinet|Oboe|Bassoon|Saxophone|Trumpet|Trombone|French horn|
Tuba|Harp|Drum kit|Pipe organ|Accordion|Harmonica|Banjo|Mandolin|Ukulele|Xylophone|Bagpipes|Sitar|Synthesizer|
Electric guitar|Bass guitar|Harpsichord|Viola""",
    "sport": """Association football|Basketball|Cricket|Tennis|Baseball|American football|Rugby union|Ice hockey|Golf|
Volleyball|Badminton|Table tennis|Boxing|Swimming (sport)|Cycling|Track and field|Gymnastics|Wrestling|Fencing|
Rowing (sport)|Sailing|Skiing|Figure skating|Handball|Field hockey|Surfing|Skateboarding|Snowboarding|Judo|Taekwondo|
Karate|Archery|Triathlon|Formula One|Chess""",
    "mammal": """Dog|Cat|Horse|Cattle|Pig|Sheep|Goat|Lion|Tiger|Leopard|Cheetah|Elephant|Giraffe|Zebra|Hippopotamus|
Rhinoceros|Gorilla|Chimpanzee|Orangutan|Brown bear|Polar bear|Giant panda|Red fox|Wolf|Rabbit|House mouse|Brown rat|Bat|
Dolphin|Blue whale|Orca|Kangaroo|Koala|Platypus|Camel|Deer|Moose|Raccoon|Squirrel|Hedgehog""",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="probe")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for group, block in LISTS.items():
        titles = gen.split(block)
        canon = gen.resolve(titles)
        for t in titles:
            c, missing = canon[t]
            first = last = ""
            n = 0
            status = "missing page" if missing else "ok"
            if not missing:
                url = gen.PV_API.format(title=urllib.parse.quote(c.replace(" ", "_"), safe=""))
                try:
                    items = json.loads(gen.get(url))["items"]
                    days = sorted(it["timestamp"][:8] for it in items)
                    if days:
                        first, last, n = days[0], days[-1], len(days)
                except Exception as e:  # noqa: BLE001 - recorded
                    status = f"error: {str(e)[:80]}"
                time.sleep(0.05)
            rows.append([group, t, c, status, first, last, n])
    pd.DataFrame(rows, columns=["group", "candidate", "canonical", "status", "first_day", "last_day", "days_returned"]
                 ).to_csv(out / "titles.tsv", sep="\t", index=False)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    (out / "probe_log.txt").write_text(f"{stamp}\t{len(rows)} candidates probed (metadata only)\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
