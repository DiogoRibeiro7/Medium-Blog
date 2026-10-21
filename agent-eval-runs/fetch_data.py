"""Build data/runs.csv from the agent runs that Sierra Research released with tau-bench and tau2-bench.

Every source file is pinned to a commit, so the table cannot drift if the upstream repositories change.
The downloads total about 440 MB; the table they produce is about 530 KB and is committed, so the
analysis runs without this step. Both upstream repositories are MIT licensed.

    python fetch_data.py                 # download, then write data/runs.csv
    python fetch_data.py --check         # rebuild in memory and compare with the committed table
    python fetch_data.py --local DIR     # read from DIR/tau-bench and DIR/tau2-bench, clones at the pinned commits
"""
import csv
import io
import json
import sys
import urllib.request
from pathlib import Path

TAU1 = "https://raw.githubusercontent.com/sierra-research/tau-bench/59a200c6d575d595120f1cb70fea53cef0632f6b/historical_trajectories/"
TAU2 = "https://raw.githubusercontent.com/sierra-research/tau2-bench/57e54143025bec219b5726884651c98eede202e2/data/tau2/results/final/"

# (benchmark, domain, model label, url)
SOURCES = [("tau-bench", d, m, TAU1 + f"{f}-{d}.json")
           for m, f in [("gpt-4o", "gpt-4o"), ("claude-3.5-sonnet", "sonnet-35-new")]
           for d in ["airline", "retail"]]
SOURCES += [("tau2-bench", d, m, TAU2 + f"{f}_{d}_{cfg}_gpt-4.1-2025-04-14_4trials.json")
            for m, f, cfg in [("claude-3.7-sonnet", "claude-3-7-sonnet-20250219", "default"),
                              ("gpt-4.1", "gpt-4.1-2025-04-14", "default"),
                              ("o4-mini", "o4-mini-2025-04-16", "default"),
                              ("gpt-4.1-mini", "gpt-4.1-mini-2025-04-14", "base")]
            for d in ["airline", "retail", "telecom"]]

COLUMNS = ["benchmark", "domain", "model", "task_id", "trial", "reward", "agent_cost", "user_cost"]
OUT = Path(__file__).parent / "data" / "runs.csv"


def rows(benchmark, domain, model, payload):
    """One row per run: which task, which repetition, pass or fail, and what the run cost in dollars."""
    if benchmark == "tau-bench":                        # a list of runs, no costs recorded
        for r in payload:
            yield [benchmark, domain, model, str(r["task_id"]), int(r["trial"]), int(r["reward"]), "", ""]
    else:                                               # a dict whose "simulations" are the runs
        for s in payload["simulations"]:
            yield [benchmark, domain, model, str(s["task_id"]), int(s["trial"]), int(s["reward_info"]["reward"]),
                   f"{s['agent_cost'] or 0:.6f}", f"{s['user_cost'] or 0:.6f}"]


def load(url, local=None, attempts=4):
    """The JSON at url, read from a local clone when one is given, else downloaded with a few retries."""
    if local:                                           # --local DIR: DIR holds clones of both repositories at the pinned commits
        repo = "tau-bench" if "/tau-bench/" in url else "tau2-bench"
        return json.loads((Path(local) / repo / url.split("/", 6)[6]).read_text(encoding="utf-8"))
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(url, timeout=600) as r:
                return json.loads(r.read())
        except (OSError, ValueError) as e:              # a cut transfer surfaces as IncompleteRead or a JSON error
            if attempt == attempts:
                raise
            print(f"  retry {attempt}: {type(e).__name__}", file=sys.stderr)


def build(local=None):
    table = []
    for benchmark, domain, model, url in SOURCES:
        print(f"reading {url.rsplit('/', 1)[1]}", file=sys.stderr)
        table.extend(rows(benchmark, domain, model, load(url, local)))
    # airline and retail tasks have numeric ids, telecom tasks have descriptive ones
    table.sort(key=lambda x: (x[0], x[1], x[2], (0, int(x[3]), "") if x[3].isdigit() else (1, 0, x[3]), x[4]))
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(COLUMNS)
    w.writerows(table)
    return buf.getvalue()


if __name__ == "__main__":
    local = sys.argv[sys.argv.index("--local") + 1] if "--local" in sys.argv else None
    text = build(local)
    if "--check" in sys.argv:
        same = OUT.read_text() == text
        print("data/runs.csv matches the pinned sources" if same else "data/runs.csv differs from the pinned sources")
        sys.exit(0 if same else 1)
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(text)
    print(f"wrote {OUT} with {text.count(chr(10)) - 1} runs", file=sys.stderr)
