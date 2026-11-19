"""Measure three agent organisations across tasks in the traces released with MAST.

Cemri et al. (2025), "Why Do Multi-Agent LLM Systems Fail?", released execution traces in
github.com/multi-agent-systems-failure-taxonomy/MAST. This script reads three groups of them
at a pinned commit and writes one row per run, with counts, token totals, flags and
extracted answers only (no trace text is copied):

    data/chatdev_runs.csv   ChatDev on 30 MMLU questions (two runs each, the unpaired
                            mmlu_0.log set aside) and on the 30 ProgramDev software tasks
    data/metagpt_runs.csv   MetaGPT (coder, tester, reviewer) on the same 30 + 30 tasks
    data/ag2_runs.csv       AG2 on 200 GSM-Plus problems in three configurations: the
                            original two-agent chat, an improved prompt, and a new topology
                            with a third, verifying agent; scored against the GSM-Plus answer
                            key (github.com/qtli/GSM-Plus, pinned commit)

Usage:
    python fetch_traces.py            # fetch both pinned commits and write the tables
    python fetch_traces.py --check    # rebuild them and confirm they equal the committed files
    python fetch_traces.py --mast DIR --gsmplus FILE   # use existing checkouts
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

MAST = ("https://github.com/multi-agent-systems-failure-taxonomy/MAST",
        "a70542e541b2104ef8fcd785778179e173fb8d70")
GSMPLUS = ("https://github.com/qtli/GSM-Plus", "3474129ec12fcd3e8ac08cb037aca1928efca98c")
HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

STAGES = {
    "DemandAnalysis": "decide", "LanguageChoose": "decide",
    "Coding": "code", "CodeComplete": "code",
    "CodeReviewComment": "review", "CodeReviewModification": "review",
    "TestErrorSummary": "test", "TestModification": "test",
    "EnvironmentDoc": "docs", "Manual": "docs",
}
STAGE_NAMES = ("decide", "code", "review", "test", "docs", "other")
CLEAN = "The software run successfully without errors"

CHATDEV_FIELDS = [
    "benchmark", "task", "run", "platform", "total_tokens", "completion_tokens", "duration_s",
    "llm_calls", "utterances", "phases", "roles", "code_lines", "doc_files", "modality",
    "uses_gui_library", "review_rounds", "review_comments", "test_reports", "first_test_clean",
    "any_test_clean", "test_error_summaries", *[f"tokens_{s}" for s in STAGE_NAMES],
]
METAGPT_FIELDS = ["benchmark", "task", "messages", "sequence", "coder_messages", "execution_output"]
AG2_FIELDS = ["condition", "trace", "gsmplus_row", "perturbation", "answer", "terminated",
              "correct", "gold_in_transcript", "messages", "words"]


def sparse_checkout(target: Path, repo: str, commit: str, paths: list[str]) -> Path:
    run = lambda *a: subprocess.run(a, cwd=target, check=True, capture_output=True)
    target.mkdir(parents=True, exist_ok=True)
    run("git", "init", "-q")
    run("git", "remote", "add", "origin", repo)
    run("git", "sparse-checkout", "set", "--no-cone", *paths)
    run("git", "fetch", "-q", "--depth", "1", "--filter=blob:none", "origin", commit)
    run("git", "checkout", "-q", "FETCH_HEAD")
    return target


# ----------------------------------------------------------------------------- ChatDev
def placeholders(text: str, key: str) -> list[str]:
    raw = re.findall(r"'" + key + r"': '((?:[^'\\]|\\.)*)'", text)
    return [r.encode("latin-1", "backslashreplace").decode("unicode_escape", "replace") for r in raw]


def parse_chatdev(path: Path, benchmark: str, task: str, run: int) -> dict:
    t = path.read_text(encoding="utf-8", errors="replace")

    def post(key: str) -> str:
        m = re.findall(r"\*\*" + key + r"\*\*=\$?([-\d.]+)", t)
        return m[-1] if m else ""

    records = re.split(r"\n(?=\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} INFO\])", t)
    phase, conclusions, roles = None, [], set()
    for rec in records:
        m = re.search(r"\*\*(.+?)<->(.+?) on : ([A-Za-z]+), turn (\d+)\*\*", rec)
        if m:
            roles.update([m.group(1).strip(), m.group(2).strip()])
            phase = m.group(3)
        if "**[Seminar Conclusion]**" in rec:
            conclusions.append((phase, rec.split("**[Seminar Conclusion]**:", 1)[1].strip()))

    stage_tokens, current = Counter(), None
    for m in re.finditer(r"\*\*phase_name\*\* \| (\w+)|\*\*\[OpenAI_Usage_Info Receive\]\*\*\n"
                         r"prompt_tokens: (\d+)\ncompletion_tokens: (\d+)", t):
        if m.group(1):
            current = m.group(1)
        else:
            stage_tokens[STAGES.get(current, "other")] += int(m.group(2)) + int(m.group(3))

    reviews = [c for p, c in conclusions if p == "CodeReviewComment"]
    # each execution of the program by the test engineer is logged as a [Test Reports] record
    reports = [r.strip() for r in re.findall(r"\*\*\[Test Reports\]\*\*:\n(.*?)(?=\n\[\d{4}-\d{2}-\d{2} |\Z)", t, re.S)]
    codes = placeholders(t, "codes")
    modality = placeholders(t, "modality")
    gui = re.compile(r"\bimport (?:tkinter|pygame|turtle|PyQt\d|kivy)\b|\bfrom (?:tkinter|pygame|PyQt\d|kivy)\b")
    if "/Users/" in t or "IMKClient" in t:
        platform = "macos"
    elif "/home/" in t:
        platform = "linux"
    else:
        platform = ""
    return {
        "benchmark": benchmark, "task": task, "run": run, "platform": platform,
        "total_tokens": post("num_total_tokens"), "completion_tokens": post("num_completion_tokens"),
        "duration_s": post("duration"), "llm_calls": t.count("**[OpenAI_Usage_Info Receive]**"),
        "utterances": post("num_utterances"), "phases": len(conclusions), "roles": len(roles),
        "code_lines": post("code_lines"), "doc_files": post("num_doc_files"),
        "modality": modality[-1].strip().lower() if modality else "",
        "uses_gui_library": int(any(gui.search(c) for c in codes)),
        "review_rounds": len(reviews),
        "review_comments": sum(1 for c in reviews if not c.startswith("<INFO> Finished")),
        "test_reports": len(reports),
        "first_test_clean": int(bool(reports) and reports[0].startswith(CLEAN)),
        "any_test_clean": int(any(r.startswith(CLEAN) for r in reports)),
        "test_error_summaries": sum(1 for p, _ in conclusions if p == "TestErrorSummary"),
        **{f"tokens_{s}": stage_tokens[s] for s in STAGE_NAMES},
    }


def chatdev_rows(mast: Path) -> list[dict]:
    rows = []
    for p in sorted((mast / "mmlu" / "chatdev_mmlu").glob("mmlu_*_*.log")):
        q, r = re.match(r"mmlu_(\d+)_(\d+)\.log", p.name).groups()
        rows.append(parse_chatdev(p, "mmlu", f"q{int(q):02d}", int(r)))
    for d in sorted((mast / "programdev" / "chatdev").iterdir()):
        if d.is_dir():
            rows.append(parse_chatdev(next(d.glob("*.log")), "programdev", d.name, 0))
    return sorted(rows, key=lambda x: (x["benchmark"], x["task"], x["run"]))


# ----------------------------------------------------------------------------- MetaGPT
EXECUTION = re.compile(r"exitcode|Traceback \(most recent call last\)|=+ test session starts|"
                       r"\d+ passed in|\d+ failed in|collected \d+ items")


def metagpt_rows(mast: Path) -> list[dict]:
    rows = []
    for bench, folder, pat in (("mmlu", mast / "mmlu" / "metagpt_mmlu", r"mmlu_(\d+)\.txt"),
                               ("programdev", mast / "programdev" / "metagpt", r"programdev_(\d+)\.txt")):
        for p in sorted(folder.glob("*.txt")):
            t = p.read_text(encoding="utf-8", errors="replace")
            seq = re.findall(r"NEW MESSAGES:\n\n?(\w+):", t)
            rows.append({
                "benchmark": bench, "task": f"t{int(re.match(pat, p.name).group(1)):02d}",
                "messages": len(seq), "sequence": ">".join(s.replace("Simple", "") for s in seq),
                "coder_messages": seq.count("SimpleCoder"),
                "execution_output": int(bool(EXECUTION.search(t))),
            })
    return sorted(rows, key=lambda x: (x["benchmark"], x["task"]))


# ----------------------------------------------------------------------------- AG2
PROBLEM = {"org": r"Problem:\n(.*?)\nuser: ", "prompt": r"Problem Statement: \n(.*?)\nuser: ",
           "topology": r"^assistant: (.*?)\nuser: "}


def boxed_all(text: str) -> list[str]:
    """Contents of every \\boxed{...}, with nested braces; also matches a lost backslash."""
    out = []
    for m in re.finditer(r"(?:\\boxed|\x08oxed|(?<![A-Za-z])oxed)\{", text):
        i = j = m.end()
        depth = 1
        while j < len(text) and depth:
            depth += {"{": 1, "}": -1}.get(text[j], 0)
            j += 1
        out.append(text[i:j - 1])
    return out


def to_value(s: str):
    s = s.strip().strip("'\"").replace("\\$", "").replace("$", "").replace(",", "")
    s = s.replace("\\%", "").replace("%", "")
    s = re.sub(r"\\text\{[^}]*\}", "", s).replace("\\,", "").replace("\\!", "").strip()
    if re.fullmatch(r"(?i)'?none'?|no answer|n/a", s):
        return "None"
    m = re.fullmatch(r"\\[dt]?frac\{(-?[\d.]+)\}\{(-?[\d.]+)\}", s)
    if m:
        return float(m.group(1)) / float(m.group(2))
    m = re.fullmatch(r"(-?\d+)\s*/\s*(\d+)", s)
    if m:
        return int(m.group(1)) / int(m.group(2))
    m = re.search(r"-?\d*\.?\d+", s)
    return float(m.group(0)) if m else None


def same(pred, gold: str) -> bool:
    if gold == "None":
        return pred == "None"
    g = to_value(gold)
    if isinstance(pred, float) and isinstance(g, float):
        return abs(pred - g) <= 1e-4 * max(1.0, abs(g)) or round(pred, 2) == round(g, 2)
    return False


def fmt(v) -> str:
    if v is None:
        return ""
    if v == "None":
        return "None"
    return f"{v:.6g}"


def ag2_rows(mast: Path, gsmplus_file: Path) -> list[dict]:
    gp = [json.loads(line) for line in gsmplus_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    norm = lambda s: re.sub(r"\s+", " ", s).strip().lower()
    index = {norm(r["question"]): i for i, r in enumerate(gp)}
    rows = []
    for cond in ("org", "prompt", "topology"):
        for i in range(200):
            t = (mast / "math_interventions" / f"{cond}_traces" / f"trace_{i}.txt").read_text(
                encoding="utf-8", errors="replace")
            problem = re.search(PROBLEM[cond], t, re.S).group(1).strip()
            k = index[norm(problem)]
            body = t.split(problem, 1)[1]
            if cond == "topology":
                found = re.findall(r"SOLUTION_FOUND\s*[:\[]?\s*(.*)", body)
                if found:
                    b = boxed_all(found[-1])
                    final = b[-1] if b else found[-1].strip().rstrip("]").strip()
                else:
                    final = None
            else:
                b = boxed_all(body)
                final = b[-1] if b else None
            pred = to_value(final) if final is not None else None
            gold = str(gp[k]["answer"])
            numbers = {fmt(to_value(x)) for x in re.findall(r"-?\d[\d,]*\.?\d*", body)}
            rows.append({
                "condition": cond, "trace": i, "gsmplus_row": k, "perturbation": gp[k]["perturbation_type"],
                "answer": fmt(pred), "terminated": int(final is not None),
                "correct": int(pred is not None and same(pred, gold)),
                "gold_in_transcript": int(gold != "None" and fmt(to_value(gold)) in numbers),
                "messages": len(re.findall(r"^(?:assistant|user): ", t, re.M)),
                "words": len(body.split()),
            })
    return rows


# ----------------------------------------------------------------------------- output
def table(rows: list[dict], fields: list[str]) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=fields, lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue()


def build(mast: Path, gsmplus_file: Path) -> dict[str, str]:
    return {"chatdev_runs.csv": table(chatdev_rows(mast), CHATDEV_FIELDS),
            "metagpt_runs.csv": table(metagpt_rows(mast), METAGPT_FIELDS),
            "ag2_runs.csv": table(ag2_rows(mast, gsmplus_file), AG2_FIELDS)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="rebuild and compare with the committed tables")
    ap.add_argument("--mast", type=Path, help="existing MAST traces/ directory")
    ap.add_argument("--gsmplus", type=Path, help="existing gsmplus_test.jsonl")
    args = ap.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        mast = args.mast or sparse_checkout(Path(tmp) / "mast", *MAST, [
            "traces/mmlu/", "traces/programdev/", "traces/math_interventions/"]) / "traces"
        gsm = args.gsmplus or sparse_checkout(Path(tmp) / "gsmplus", *GSMPLUS,
                                              ["dataset/gsmplus_test.jsonl"]) / "dataset" / "gsmplus_test.jsonl"
        out = build(mast, gsm)
    DATA.mkdir(exist_ok=True)
    if args.check:
        bad = [n for n, text in out.items() if not (DATA / n).exists() or (DATA / n).read_text() != text]
        print("identical" if not bad else f"differs: {', '.join(bad)}")
        return 1 if bad else 0
    for name, text in out.items():
        (DATA / name).write_text(text)
        print(f"wrote data/{name}: {text.count(chr(10)) - 1} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
