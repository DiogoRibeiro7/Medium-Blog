"""Same organisation, different work: three agent systems across tasks in the MAST traces.

Companion code for the Medium follow-up to "Copying the Org Chart Into Multi-Agent LLM
Systems". Prints every number in the article and draws every figure. Calls no model.

Sections
  1  ChatDev, organised as a software company, on 30 MMLU questions (60 runs) and on the
     30 ProgramDev software tasks it was built for (30 runs): phases, budget, where the
     tokens went, review and testing.
  2  MetaGPT (coder, tester, reviewer) on the same two task sets.
  3  AG2 on 200 GSM-Plus problems in three configurations: the original two-agent chat, an
     improved prompt, and a new topology that adds a verifying agent; scored against the
     GSM-Plus answer key and compared problem by problem.

Run:  python org_task_fit.py > results/output.txt
"""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
DATA, FIG = HERE / "data", HERE / "figures"
FIG.mkdir(exist_ok=True)
RNG = np.random.default_rng(20261119)
N_BOOT = 10_000

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "font.size": 10.5, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True, "legend.frameon": False,
    "lines.linewidth": 2.0, "savefig.dpi": 200,
})


def header(title: str) -> None:
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


def read(name: str) -> list[dict]:
    with open(DATA / name, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def num(rows, key) -> np.ndarray:
    return np.array([float(r[key]) for r in rows])


def med_iqr(x, fmt="{:,.0f}") -> str:
    x = np.asarray(x, dtype=float)
    return (f"median {fmt.format(np.median(x))} (IQR {fmt.format(np.quantile(x, .25))}-"
            f"{fmt.format(np.quantile(x, .75))})")


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from the discordant counts b and c."""
    n = b + c
    return 1.0 if n == 0 else float(min(1.0, 2 * stats.binom.cdf(min(b, c), n, 0.5)))


# ---------------------------------------------------------------------------------------
# 1. ChatDev on two kinds of work
# ---------------------------------------------------------------------------------------
header("1. ChatDev on 30 MMLU questions and on the 30 ProgramDev software tasks (MAST, commit a70542e)")
cd = read("chatdev_runs.csv")
B = {"mmlu": [r for r in cd if r["benchmark"] == "mmlu"],
     "programdev": [r for r in cd if r["benchmark"] == "programdev"]}
LABEL = {"mmlu": "MMLU questions", "programdev": "ProgramDev software tasks"}
STAGES = ["decide", "code", "review", "test", "docs", "other"]
SNAME = {"decide": "Choose product format and language", "code": "Write the program", "review": "Code review",
         "test": "Run, test and fix", "docs": "Requirements file and user manual", "other": "Reflection and other"}

for b, rows in B.items():
    tasks = sorted(set(r["task"] for r in rows))
    print(f"{LABEL[b]}: {len(rows)} runs on {len(tasks)} tasks; platform recorded in the logs: "
          + ", ".join(f"{k} {v}" for k, v in Counter(r["platform"] for r in rows).items()))
for b, rows in B.items():
    print(f"\n{LABEL[b]}")
    print("  roles taking part: " + ", ".join(f"{k} roles in {v} runs" for k, v in sorted(Counter(r['roles'] for r in rows).items())))
    print("  phase executions: " + med_iqr(num(rows, "phases")) + "; code-review rounds: "
          + ", ".join(f"{k} in {v}" for k, v in Counter(r['review_rounds'] for r in rows).items()))
    print("  product format chosen: " + ", ".join(f"{k} {v}" for k, v in Counter(r['modality'] for r in rows).most_common()))
    for k, lab in [("total_tokens", "tokens"), ("completion_tokens", "tokens written by agents"),
                   ("llm_calls", "model calls"), ("duration_s", "seconds"), ("code_lines", "lines of code")]:
        print(f"  {lab}: " + med_iqr(num(rows, k)))
    print(f"  code imports a GUI library (tkinter, pygame, turtle, Qt, kivy): {sum(int(r['uses_gui_library']) for r in rows)} of {len(rows)}")

# Stage shares, pooled over runs, with a bootstrap over tasks (MMLU runs come in pairs per question).
def pooled_shares(rows):
    tok = np.array([[float(r[f"tokens_{s}"]) for s in STAGES] for r in rows])
    return tok.sum(axis=0) / tok.sum()


def boot_shares(rows):
    by_task = defaultdict(list)
    for r in rows:
        by_task[r["task"]].append(r)
    tasks = list(by_task)
    out = np.empty((N_BOOT, len(STAGES)))
    for i in range(N_BOOT):
        pick = RNG.integers(0, len(tasks), len(tasks))
        sample = [r for j in pick for r in by_task[tasks[j]]]
        out[i] = pooled_shares(sample)
    return out


print("\nShare of all tokens by stage, pooled over runs (95% bootstrap interval, resampling tasks):")
share, boot = {}, {}
for b, rows in B.items():
    share[b], boot[b] = pooled_shares(rows), boot_shares(rows)
for j, s in enumerate(STAGES):
    lo = {b: np.quantile(boot[b][:, j], [.025, .975]) for b in B}
    diff = boot["programdev"][:, j] - boot["mmlu"][:, j]
    print(f"  {SNAME[s]:<36s} MMLU {share['mmlu'][j]:6.1%} ({lo['mmlu'][0]:.1%}-{lo['mmlu'][1]:.1%})   "
          f"ProgramDev {share['programdev'][j]:6.1%} ({lo['programdev'][0]:.1%}-{lo['programdev'][1]:.1%})   "
          f"difference {share['programdev'][j] - share['mmlu'][j]:+6.1%} ({np.quantile(diff, .025):+.1%} to {np.quantile(diff, .975):+.1%})")
over = {b: (share[b][STAGES.index("decide")] + share[b][STAGES.index("docs")]) for b in B}
over_b = {b: boot[b][:, STAGES.index("decide")] + boot[b][:, STAGES.index("docs")] for b in B}
d = over_b["programdev"] - over_b["mmlu"]
print(f"  format/language meetings + documentation together: MMLU {over['mmlu']:.1%}, ProgramDev {over['programdev']:.1%}; "
      f"difference {over['programdev'] - over['mmlu']:+.1%} ({np.quantile(d, .025):+.1%} to {np.quantile(d, .975):+.1%})")
per_run = {b: np.array([(float(r['tokens_decide']) + float(r['tokens_docs'])) / float(r['total_tokens']) for r in rows])
           for b, rows in B.items()}
for b in B:
    print(f"  per run, meetings + documentation share: {LABEL[b]} " + med_iqr(per_run[b] * 100, "{:.1f}") + " %")
mw = stats.mannwhitneyu(per_run["mmlu"], per_run["programdev"])
print(f"  Mann-Whitney U on the per-run shares: U = {mw.statistic:.0f}, p = {mw.pvalue:.3f} "
      "(runs treated as independent; MMLU runs are two per question)")
ratio = {b: float(np.median(num(rows, 'tokens_decide') + num(rows, 'tokens_docs'))) for b, rows in B.items()}
print("  per run, meetings + documentation tokens: " + ", ".join(f"{LABEL[b]} median {v:,.0f}" for b, v in ratio.items()))

print("\nCode review")
for b, rows in B.items():
    rc = num(rows, "review_comments")
    silent = [r for r in rows if r["review_comments"] == "0"]
    sh = (sum(float(r["tokens_review"]) for r in silent) / sum(float(r["total_tokens"]) for r in silent)) if silent else float("nan")
    print(f"  {LABEL[b]}: reviewer commented in at least one round in {int(np.sum(rc > 0))} of {len(rows)} runs; "
          f"all three rounds ended 'Finished' in {len(silent)}; review share of tokens in those silent runs {sh:.1%}")

print("\nTesting: every execution of the program by the test engineer is a 'test report'")
for b, rows in B.items():
    print(f"  {LABEL[b]}: test reports per run " + med_iqr(num(rows, "test_reports")) + f"; first execution ran without error in "
          f"{sum(int(r['first_test_clean']) for r in rows)} of {len(rows)}; some execution ran without error in "
          f"{sum(int(r['any_test_clean']) for r in rows)}; runs with a bug-summary step {sum(int(r['test_error_summaries']) > 0 for r in rows)}")

# Figure 1: stage shares
fig, ax = plt.subplots(figsize=(8.4, 4.2))
order = sorted(range(len(STAGES)), key=lambda j: share["programdev"][j])
for y, j in enumerate(order):
    for b, col, dy in (("mmlu", ORANGE, 0.13), ("programdev", BLUE, -0.13)):
        lo, hi = np.quantile(boot[b][:, j], [.025, .975]) * 100
        ax.plot([lo, hi], [y + dy, y + dy], color=col, linewidth=2.2, solid_capstyle="round")
        ax.plot(share[b][j] * 100, y + dy, "o", color=col, markersize=8, markeredgecolor=SURFACE, markeredgewidth=1.5,
                label=LABEL[b] if y == 0 else None)
ax.set_yticks(range(len(order)))
ax.set_yticklabels([SNAME[STAGES[j]] for j in order])
ax.set_xlabel("Share of all tokens (%), with 95% bootstrap interval over tasks")
ax.set_xlim(0, 62)
ax.grid(axis="y", visible=False)
ax.legend(loc="lower right", fontsize=9)
fig.suptitle("ChatDev ran the same phases on both kinds of work; the tokens went to different ones", x=0.01,
             ha="left", fontweight="bold", fontsize=11.5)
fig.text(0.01, 0.01, "60 runs on 30 MMLU questions (Linux server) and 30 runs on 30 ProgramDev tasks (macOS). "
         "Data: MAST traces (Cemri et al. 2025).", fontsize=8, color=INK2)
fig.tight_layout(rect=(0, 0.04, 1, 0.94))
fig.savefig(FIG / "fig1_chatdev_two_tasks.png")
plt.close(fig)

# ---------------------------------------------------------------------------------------
# 2. MetaGPT on the same two task sets
# ---------------------------------------------------------------------------------------
header("2. MetaGPT (coder, tester, reviewer) on the same 30 MMLU questions and 30 ProgramDev tasks")
mg = read("metagpt_runs.csv")
for b in ("mmlu", "programdev"):
    rows = [r for r in mg if r["benchmark"] == b]
    seqs = Counter(r["sequence"] for r in rows)
    print(f"  {LABEL[b]}: {len(rows)} runs; message sequences: " + "; ".join(f"{k} in {v}" for k, v in seqs.items())
          + f"; coder messages per run: {sorted(set(int(r['coder_messages']) for r in rows))}; "
          f"runs with any execution output (exit code, traceback, test-runner summary): {sum(int(r['execution_output']) for r in rows)}")

# ---------------------------------------------------------------------------------------
# 3. AG2: adding a verifier, or changing the prompt
# ---------------------------------------------------------------------------------------
header("3. AG2 on 200 GSM-Plus problems: original chat, improved prompt, added verifier")
ag = read("ag2_runs.csv")
COND = {"org": "original two-agent chat", "prompt": "improved prompt", "topology": "added verifying agent"}
A = {c: {int(r["trace"]): r for r in ag if r["condition"] == c} for c in COND}
same_rows = all(A["org"][i]["gsmplus_row"] == A[c][i]["gsmplus_row"] for c in COND for i in range(200))
print(f"Each configuration's 200 traces are matched to the GSM-Plus test set by question text; "
      f"same problem at each index in all three: {same_rows}")
ptypes = Counter(A["org"][i]["perturbation"] for i in range(200))
print("Perturbation types among the 200: " + ", ".join(f"{k} {v}" for k, v in sorted(ptypes.items())))
CT = "critical thinking"
print(f"'{CT}' problems lack information needed for an answer; their key is 'None'.")

acc = {}
for c, lab in COND.items():
    k = sum(int(A[c][i]["correct"]) for i in range(200))
    lo, hi = wilson(k, 200)
    acc[c] = k / 200
    term = sum(int(A[c][i]["terminated"]) for i in range(200))
    print(f"  {lab:<26s} correct {k}/200 = {k / 200:.1%} (95% Wilson {lo:.1%}-{hi:.1%}); ended with a final answer {term}/200; "
          f"messages {med_iqr([int(A[c][i]['messages']) for i in range(200)])}; words {med_iqr([int(A[c][i]['words']) for i in range(200)])}")

print("\nBy problem type")
groups = {"missing information (critical thinking)": [i for i in range(200) if A["org"][i]["perturbation"] == CT],
          "all other types": [i for i in range(200) if A["org"][i]["perturbation"] != CT]}
sub = {}
for g, idx in groups.items():
    line = []
    for c in COND:
        k = sum(int(A[c][i]["correct"]) for i in idx)
        lo, hi = wilson(k, len(idx))
        sub[(g, c)] = (k, len(idx), lo, hi)
        line.append(f"{COND[c]} {k}/{len(idx)} = {k / len(idx):.1%} ({lo:.1%}-{hi:.1%})")
    print(f"  {g}: " + "; ".join(line))

print("\nPaired comparisons on the same problems (exact McNemar test on discordant pairs)")
pairs = {}
for a_, b_ in (("org", "prompt"), ("org", "topology"), ("prompt", "topology")):
    for g, idx in [("all 200", list(range(200)))] + list(groups.items()):
        only_a = sum(1 for i in idx if A[a_][i]["correct"] == "1" and A[b_][i]["correct"] == "0")
        only_b = sum(1 for i in idx if A[a_][i]["correct"] == "0" and A[b_][i]["correct"] == "1")
        both = sum(1 for i in idx if A[a_][i]["correct"] == "1" and A[b_][i]["correct"] == "1")
        pairs[(a_, b_, g)] = (only_a, only_b, both)
        print(f"  {COND[a_]} vs {COND[b_]}, {g}: right only with the first {only_a}, only with the second {only_b}, "
              f"both right {both}; net {only_b - only_a:+d}; p = {mcnemar_exact(only_a, only_b):.3f}")

print("\nThe verifying agent's runs that never declared a final answer")
unterm = [i for i in range(200) if A["topology"][i]["terminated"] == "0"]
print(f"  {len(unterm)} of 200 hit the conversation's message cap without a SOLUTION_FOUND line; messages in those runs: "
      f"{sorted(set(int(A['topology'][i]['messages']) for i in unterm))}")
print(f"  the correct number appears somewhere in the agents' messages in {sum(int(A['topology'][i]['gold_in_transcript']) for i in unterm)} of them; "
      f"the original chat got {sum(int(A['org'][i]['correct']) for i in unterm)} of the same {len(unterm)} problems right")
k_t = sum(int(A["topology"][i]["correct"]) for i in range(200) if A["topology"][i]["terminated"] == "1")
n_t = 200 - len(unterm)
print(f"  accuracy of the verifying configuration on the {n_t} runs that ended with an answer: {k_t}/{n_t} = {k_t / n_t:.1%}")
k_o = sum(int(A["org"][i]["correct"]) for i in range(200) if A["topology"][i]["terminated"] == "1")
print(f"  the original chat on the same {n_t} problems: {k_o}/{n_t} = {k_o / n_t:.1%}")
other_term = [i for i in groups["all other types"] if A["topology"][i]["terminated"] == "1"]
lost = sum(1 for i in other_term if A["org"][i]["correct"] == "1" and A["topology"][i]["correct"] == "0")
won = sum(1 for i in other_term if A["org"][i]["correct"] == "0" and A["topology"][i]["correct"] == "1")
print(f"  other problem types, runs that ended with an answer ({len(other_term)}): added verifier "
      f"{sum(int(A['topology'][i]['correct']) for i in other_term)}/{len(other_term)}, original chat "
      f"{sum(int(A['org'][i]['correct']) for i in other_term)}/{len(other_term)}; lost {lost}, gained {won}, "
      f"p = {mcnemar_exact(lost, won):.3f}")
print(f"  unterminated runs by type: " + ", ".join(f"{k} {v}" for k, v in Counter(A['topology'][i]['perturbation'] for i in unterm).items()))
wt = np.median([int(A['topology'][i]['words']) for i in range(200)]) / np.median([int(A['org'][i]['words']) for i in range(200)])
wp = np.median([int(A['prompt'][i]['words']) for i in range(200)]) / np.median([int(A['org'][i]['words']) for i in range(200)])
print(f"  median words per run relative to the original chat: improved prompt {wp:.1f}x, added verifier {wt:.1f}x")

# Figure 2: accuracy by problem type
fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.9), sharey=False)
cols = {"org": INK2, "prompt": BLUE, "topology": ORANGE}
for ax, (g, idx) in zip(axes, groups.items()):
    for x, c in enumerate(COND):
        k, n, lo, hi = sub[(g, c)]
        ax.plot([x, x], [lo * 100, hi * 100], color=cols[c], linewidth=2.2, solid_capstyle="round")
        ax.plot(x, k / n * 100, "o", color=cols[c], markersize=9, markeredgecolor=SURFACE, markeredgewidth=1.5)
        ax.text(x + 0.1, k / n * 100, f"{k}/{n}", va="center", fontsize=8.5, color=INK2)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["original\nchat", "improved\nprompt", "added\nverifier"])
    ax.set_xlim(-0.4, 2.6)
    ax.set_title(("Missing-information problems" if "missing" in g else "All other problems") + f" (n = {len(idx)})",
                 loc="left", fontsize=10)
    ax.grid(axis="x", visible=False)
axes[0].set_ylabel("Correct (%), 95% Wilson interval")
axes[0].set_ylim(-3, 100)
axes[1].set_ylim(60, 100)
fig.suptitle("AG2 on 200 GSM-Plus problems: the prompt's gain sits in one problem type; the verifier's does not hold",
             x=0.01, ha="left", fontweight="bold", fontsize=11.5)
fig.text(0.01, 0.01, "One released run per configuration on the same 200 problems, scored against the GSM-Plus key "
         "(github.com/qtli/GSM-Plus). Data: MAST traces.", fontsize=8, color=INK2)
fig.tight_layout(rect=(0, 0.04, 1, 0.94))
fig.savefig(FIG / "fig2_ag2_by_type.png")
plt.close(fig)

# Figure 3: paired changes against the original chat
fig, ax = plt.subplots(figsize=(8.6, 3.8))
rowsets = [("improved prompt,\nmissing-information problems", ("org", "prompt", "missing information (critical thinking)")),
           ("improved prompt,\nall other problems", ("org", "prompt", "all other types")),
           ("added verifier,\nmissing-information problems", ("org", "topology", "missing information (critical thinking)")),
           ("added verifier,\nall other problems", ("org", "topology", "all other types"))]
for y, (lab, key) in enumerate(reversed(rowsets)):
    lost, gained, _ = pairs[key]
    a_, b_, g = key
    idx = groups[g]
    never = sum(1 for i in idx if A[a_][i]["correct"] == "1" and A[b_][i]["correct"] == "0" and A[b_][i]["terminated"] == "0")
    wrong = lost - never
    ax.barh(y, gained, color=AQUA, height=0.55, label="gained" if y == 0 else None)
    ax.barh(y, -wrong, color=ORANGE, height=0.55, label="lost: wrong final answer" if y == 0 else None)
    ax.barh(y, -never, left=-wrong, color=SURFACE, edgecolor=ORANGE, hatch="////", linewidth=1.2, height=0.55,
            label="lost: no final answer (message cap)" if y == 0 else None)
    ax.text(gained + 0.5, y, f"+{gained}", va="center", fontsize=9, color=INK2)
    ax.text(-lost - 0.5, y, f"−{lost}" if lost else "0", va="center", ha="right", fontsize=9, color=INK2)
ax.axvline(0, color=INK2, linewidth=1)
ax.set_yticks(range(len(rowsets)))
ax.set_yticklabels([r[0] for r in reversed(rowsets)], fontsize=9.5)
ax.set_xlim(-30, 20)
ax.set_xticks([-25, -20, -15, -10, -5, 0, 5, 10, 15])
ax.set_xlabel("Problems lost (left) and gained (right) against the original chat, on the same problems")
ax.grid(axis="y", visible=False)
ax.legend(loc="upper left", fontsize=8.5)
fig.suptitle("Problem by problem: what each change to the AG2 configuration won and lost", x=0.01, ha="left",
             fontweight="bold", fontsize=11.5)
fig.text(0.01, 0.01, "Discordant pairs only; problems both configurations got right, or both got wrong, are not shown.",
         fontsize=8, color=INK2)
fig.tight_layout(rect=(0, 0.05, 1, 0.93))
fig.savefig(FIG / "fig3_ag2_paired.png")
plt.close(fig)

header("Figures written")
for f in sorted(FIG.glob("*.png")):
    print(f"  figures/{f.name}")
