"""Measure the ChatDev and MetaGPT MMLU traces released with MAST.

Cemri et al. (2025), "Why Do Multi-Agent LLM Systems Fail?", released the execution
traces of several multi-agent systems in github.com/multi-agent-systems-failure-taxonomy/MAST.
Two of them, ChatDev and MetaGPT, were run on the same 30 MMLU multiple-choice
questions (traces/mmlu). Both are organised as a software company: ChatDev has a CEO,
CPO, CTO, programmer, code reviewer and test engineer; MetaGPT's configuration here
has a coder, a tester and a reviewer.

This script reads the traces at a pinned commit and writes one row per run with counts
and extracted answers only (no trace text is copied):

    data/chatdev_runs.csv   60 runs, two per question (the unpaired file mmlu_0.log is set aside)
    data/metagpt_runs.csv   30 runs, one per question

Usage:
    python fetch_traces.py            # clone the pinned commit and write the two tables
    python fetch_traces.py --check    # rebuild them and confirm they equal the committed files
    python fetch_traces.py --traces DIR   # use an existing checkout of traces/mmlu
"""

from __future__ import annotations

import argparse
import csv
import io
import re
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

REPO = "https://github.com/multi-agent-systems-failure-taxonomy/MAST"
COMMIT = "a70542e541b2104ef8fcd785778179e173fb8d70"
HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

CHATDEV_FIELDS = [
    "question", "run", "subject", "cost_usd", "prompt_tokens", "completion_tokens",
    "total_tokens", "duration_s", "llm_calls", "utterances", "phases", "roles",
    "modality", "language", "uses_tkinter", "code_lines", "doc_files", "manual_lines",
    "review_comments", "review_generalise", "first_answer", "final_answer",
    "final_answer_kind", "answer_changed", "stated_answer", "gui_test_errors",
    "virtual_display_errors", "test_rounds", "tokens_decide", "tokens_code",
    "tokens_review", "tokens_test", "tokens_docs", "tokens_other", "agent_words",
]
STAGES = {
    "DemandAnalysis": "decide", "LanguageChoose": "decide",
    "Coding": "code", "CodeComplete": "code",
    "CodeReviewComment": "review", "CodeReviewModification": "review",
    "TestErrorSummary": "test", "TestModification": "test",
    "EnvironmentDoc": "docs", "Manual": "docs",
}
METAGPT_FIELDS = [
    "question", "subject", "messages", "coder_messages", "tester_messages",
    "reviewer_messages", "coder_answer", "coder_answer_source", "tester_expected",
    "tester_other_expected", "tester_restates_coder", "reviewer_disputes_answer",
    "reviewer_asks_scope", "messages_after_dispute_by_coder", "duration_s",
    "coder_words", "tester_words", "reviewer_words",
]

ANSWER_PATTERNS = [
    r"The answer is \((\d)\)",
    r"^\s*return\s+\(?(\d)\)?\s*(?:#.*)?$",
    r"\b(?:correct_answer|correct_option|correct_index|correct_answer_index|answer_index|answer)\s*=\s*(\d)\b",
    r"[\"'](?:correct_answer|correct_option|answer)[\"']\s*:\s*(\d)\b",
    r"\bcorrect_\w*\s*=\s*(\d)\b",
    r"\]\s*,\s*(\d)\s*\)",
    r"^\s*(\d)\s*,?\s*#.*[Cc]orrect",
]
GENERALISE = re.compile(
    r"hard-?cod|dynamic|any (?:question|scenario|input)|all questions|other questions|"
    r"different (?:question|scenario)|generali[sz]|user input|input (?:a|the|their) question",
    re.I,
)


def checkout(target: Path) -> Path:
    """Sparse, shallow checkout of traces/mmlu at the pinned commit."""
    run = lambda *a: subprocess.run(a, cwd=target, check=True, capture_output=True)
    run("git", "init", "-q")
    run("git", "remote", "add", "origin", REPO)
    run("git", "sparse-checkout", "set", "traces/mmlu")
    run("git", "fetch", "-q", "--depth", "1", "--filter=blob:none", "origin", COMMIT)
    run("git", "checkout", "-q", "FETCH_HEAD")
    return target / "traces" / "mmlu"


def answers_in(text: str) -> list[str]:
    found: list[str] = []
    for p in ANSWER_PATTERNS:
        found += re.findall(p, text, flags=re.M)
    return found


def classify(digits: list[str]) -> tuple[str, str]:
    distinct = sorted(set(digits))
    if len(distinct) == 1:
        return distinct[0], "single"
    return "", "branching" if distinct else "none"


def placeholders(text: str, key: str) -> list[str]:
    raw = re.findall(r"'" + key + r"': '((?:[^'\\]|\\.)*)'", text)
    return [r.encode("latin-1", "backslashreplace").decode("unicode_escape", "replace") for r in raw]


def subject_of(text: str) -> str:
    m = re.search(r"multiple choice questions \(with answers\) about (\S+?)\.", text)
    return m.group(1) if m else ""


def parse_chatdev(path: Path) -> dict:
    t = path.read_text(encoding="utf-8", errors="replace")
    q, r = re.match(r"mmlu_(\d+)_(\d+)\.log", path.name).groups()

    def post(key: str) -> str:
        m = re.findall(r"\*\*" + key + r"\*\*=\$?([-\d.]+)", t)
        return m[-1] if m else ""

    records = re.split(r"\n(?=\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} INFO\])", t)
    phase, conclusions, roles, agent_words = None, [], set(), 0
    for rec in records:
        m = re.search(r"\*\*(.+?)<->(.+?) on : ([A-Za-z]+), turn (\d+)\*\*", rec)
        if m:
            roles.update([m.group(1).strip(), m.group(2).strip()])
            phase = m.group(3)
            # message body = text after the header, minus the bracketed echo of the role prompt
            body = rec[m.end():].lstrip("\n")
            body = re.sub(r"^\[.*?\]\n\n", "", body, count=1, flags=re.S)
            agent_words += len(body.split())
        if "**[Seminar Conclusion]**" in rec:
            conclusions.append((phase, rec.split("**[Seminar Conclusion]**:", 1)[1].strip()))

    stage_tokens = Counter()
    current = None
    for m in re.finditer(r"\*\*phase_name\*\* \| (\w+)|\*\*\[OpenAI_Usage_Info Receive\]\*\*\nprompt_tokens: (\d+)\ncompletion_tokens: (\d+)", t):
        if m.group(1):
            current = m.group(1)
        else:
            stage_tokens[STAGES.get(current, "other")] += int(m.group(2)) + int(m.group(3))

    reviews = [c for p, c in conclusions if p == "CodeReviewComment"]
    substantive = [c for c in reviews if not c.startswith("<INFO> Finished")]
    codes = placeholders(t, "codes")
    first = classify(answers_in(codes[0]))[0] if codes else ""
    final, kind = classify(answers_in(codes[-1])) if codes else ("", "none")
    agent_text = "\n".join(c for _, c in conclusions)
    stated = Counter(re.findall(r"The answer is \((\d)\)", agent_text))
    reports = placeholders(t, "test_reports")
    test_phases = [p for p, _ in conclusions if p == "TestErrorSummary"]
    modality = placeholders(t, "modality")
    language = placeholders(t, "language")
    return {
        "question": int(q), "run": int(r), "subject": subject_of(t),
        "cost_usd": post("cost"), "prompt_tokens": post("num_prompt_tokens"),
        "completion_tokens": post("num_completion_tokens"), "total_tokens": post("num_total_tokens"),
        "duration_s": post("duration"), "llm_calls": t.count("**[OpenAI_Usage_Info Receive]**"),
        "utterances": post("num_utterances"), "phases": len(conclusions), "roles": len(roles),
        "modality": modality[-1].strip().lower() if modality else "",
        "language": re.sub(r"[^A-Za-z+#]", "", language[-1]) if language else "",
        "uses_tkinter": int(any("tkinter" in c for c in codes)),
        "code_lines": post("code_lines"), "doc_files": post("num_doc_files"),
        "manual_lines": post("manual_lines"),
        "review_comments": len(substantive),
        "review_generalise": int(any(GENERALISE.search(c) for c in substantive)),
        "first_answer": first, "final_answer": final, "final_answer_kind": kind,
        "answer_changed": int(bool(first) and first != final),
        "stated_answer": stated.most_common(1)[0][0] if stated else "",
        "gui_test_errors": int(any(re.search(r"TclError|no display name", x) for x in reports)),
        "virtual_display_errors": int(any(re.search(r"[Xx]vfb|pyvirtualdisplay", x) for x in reports)),
        "test_rounds": len(test_phases),
        **{f"tokens_{k}": stage_tokens[k] for k in ("decide", "code", "review", "test", "docs", "other")},
        "agent_words": agent_words,
    }


# MetaGPT coders sometimes compute the index instead of writing it. Index lookups
# (options.index("...")) are resolved below with ast.literal_eval, which reads literals
# and runs nothing. The one arithmetic case is resolved by hand:
#   question 7: 3 hours / (1/2 hour per subject) = 6 subjects; options [4, 2, 6, 5] -> index 2.
HAND_RESOLVED = {7: "2"}
DISPUTE = re.compile(r"(?:answer|premise)[^.]{0,60}\bis incorrect\b|built around an incorrect answer", re.I)
SCOPE = re.compile(r"edge case|broader range|other (?:inputs|questions|scenarios)|robust|variety of inputs|"
                   r"different (?:inputs|questions|scenarios)|generali[sz]", re.I)


def literal_index(code: str) -> str:
    """Resolve answer = options.index(<string>) when both are literals in the coder's code."""
    import ast
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return ""
    lists, strings = {}, {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                value = ast.literal_eval(node.value)
            except (ValueError, SyntaxError, TypeError):
                continue
            (lists if isinstance(value, list) else strings if isinstance(value, str) else {})[node.targets[0].id] = value
    m = re.search(r"(\w+)\.index\((\w+)\)", code)
    if m and m.group(1) in lists and m.group(2) in strings and strings[m.group(2)] in lists[m.group(1)]:
        return str(lists[m.group(1)].index(strings[m.group(2)]))
    return ""


def positive_asserts(tester_message: str) -> list[str]:
    """Digits in assert ... == "The answer is (d)" outside pytest.raises blocks, in order."""
    out = []
    for test in re.split(r"\n(?=def test_)", tester_message):
        if "pytest.raises" in test:
            continue
        out += re.findall(r"==\s*[fr]?[\"']The answer is \((\d)\)", test)
    return out


def parse_metagpt(path: Path) -> dict:
    t = path.read_text(encoding="utf-8", errors="replace")
    q = int(re.match(r"mmlu_(\d+)\.txt", path.name).group(1))
    blocks = re.split(r"\n\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\] NEW MESSAGES:\n", t)
    stamps = re.findall(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]", t)
    msgs = []
    for body in blocks[2::2]:
        body = body.split("\n" + "-" * 80, 1)[0].strip()
        m = re.match(r"(\w+):\s*(.*)", body, re.S)
        if m:
            msgs.append((m.group(1), m.group(2)))
    coder = [b for s, b in msgs if s == "SimpleCoder"]
    tester = [b for s, b in msgs if s == "SimpleTester"]
    reviewer = [b for s, b in msgs if s == "SimpleReviewer"]
    literal, kind = classify(answers_in(coder[0])) if coder else ("", "none")
    coder_answer = literal or (literal_index(coder[0]) if coder else "") or HAND_RESOLVED.get(q, "")
    source = "literal" if literal else "index lookup" if coder and literal_index(coder[0]) else "hand" if q in HAND_RESOLVED else ""
    first_tests = positive_asserts(tester[0]) if tester else []
    all_tests = [d for b in tester for d in positive_asserts(b)]
    from datetime import datetime
    ts = [datetime.strptime(s, "%Y-%m-%d %H:%M:%S") for s in stamps]
    words = lambda xs: sum(len(x.split()) for x in xs)
    return {
        "question": q, "subject": subject_of(t), "messages": len(msgs),
        "coder_messages": len(coder), "tester_messages": len(tester), "reviewer_messages": len(reviewer),
        "coder_answer": coder_answer, "coder_answer_source": source,
        "tester_expected": first_tests[0] if first_tests else "",
        "tester_other_expected": "".join(sorted(set(all_tests) - {coder_answer})),
        "tester_restates_coder": int(bool(first_tests) and first_tests[0] == coder_answer),
        "reviewer_disputes_answer": int(any(DISPUTE.search(b) for b in reviewer)),
        "reviewer_asks_scope": int(any(SCOPE.search(b) for b in reviewer)),
        "messages_after_dispute_by_coder": sum(1 for s, _ in msgs[next((i for i, (s, b) in enumerate(msgs) if s == "SimpleReviewer" and DISPUTE.search(b)), len(msgs)):] if s == "SimpleCoder"),
        "duration_s": int((max(ts) - min(ts)).total_seconds()) if ts else "",
        "coder_words": words(coder), "tester_words": words(tester), "reviewer_words": words(reviewer),
    }


def table(rows: list[dict], fields: list[str]) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=fields, lineterminator="\n")
    w.writeheader()
    for row in rows:
        w.writerow(row)
    return buf.getvalue()


def build(traces: Path) -> dict[str, str]:
    chatdev = sorted((parse_chatdev(p) for p in (traces / "chatdev_mmlu").glob("mmlu_*_*.log")),
                     key=lambda d: (d["question"], d["run"]))
    metagpt = sorted((parse_metagpt(p) for p in (traces / "metagpt_mmlu").glob("mmlu_*.txt")),
                     key=lambda d: d["question"])
    return {"chatdev_runs.csv": table(chatdev, CHATDEV_FIELDS),
            "metagpt_runs.csv": table(metagpt, METAGPT_FIELDS)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="rebuild and compare with the committed tables")
    ap.add_argument("--traces", type=Path, help="existing traces/mmlu directory")
    args = ap.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        traces = args.traces or checkout(Path(tmp))
        out = build(traces)
    DATA.mkdir(exist_ok=True)
    if args.check:
        bad = [name for name, text in out.items() if (DATA / name).read_text() != text]
        print("identical" if not bad else f"differs: {', '.join(bad)}")
        return 1 if bad else 0
    for name, text in out.items():
        (DATA / name).write_text(text)
        print(f"wrote data/{name}: {text.count(chr(10)) - 1} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
