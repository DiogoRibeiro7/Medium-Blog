"""Org charts for language-model agents: released traces and three small models.

Companion code for the Medium article on multi-agent LLM systems designed as copies of a
human organisation chart. One script prints every number in the article and draws every
figure. It calls no model.

Sections
  1  The released traces: ChatDev and MetaGPT, organised as software companies, answering
     the same 30 MMLU multiple-choice questions (data/*.csv, built by fetch_traces.py).
  2  Separation of duties with correlated and anchored reviewers (one-factor probit model,
     exact by Gauss-Hermite quadrature, checked by Monte Carlo).
  3  Deliberation with conformity, authority and evidence-driven correction (simulation).
  4  The last human: a fixed review budget against a growing team (deterministic model).

Run:  python org_chart_agents.py > results/output.txt
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize, stats

HERE = Path(__file__).resolve().parent
DATA, FIG = HERE / "data", HERE / "figures"
FIG.mkdir(exist_ok=True)
RNG = np.random.default_rng(20261118)

# Reference palette (light mode): surfaces, ink, first three categorical slots, blue ramp.
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
BLUE_RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "font.size": 10.5, "axes.titlesize": 11.5, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True, "legend.frameon": False,
    "lines.linewidth": 2.0, "savefig.dpi": 200,
})


def header(title: str) -> None:
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


def read(name: str) -> dict[str, np.ndarray]:
    """Columns as arrays: numeric where every value parses as a number, otherwise strings ('' = missing)."""
    import csv
    with open(DATA / name, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    out = {}
    for k in rows[0]:
        col = [r[k] for r in rows]
        try:
            out[k] = np.array([float(v) for v in col])
            if np.all(out[k] == np.round(out[k])) and "." not in "".join(col):
                out[k] = out[k].astype(int)
        except ValueError:
            out[k] = np.array(col, dtype=object)
    return out


def cat(name: str, col) -> np.ndarray:
    """A column that holds answer indices, read as strings ('' when missing)."""
    return np.array([str(v) if str(v) != "" else "" for v in col], dtype=object)


def q(x, p):
    return float(np.quantile(np.asarray(x, dtype=float), p))


# ---------------------------------------------------------------------------------------
# 1. The released traces
# ---------------------------------------------------------------------------------------
header("1. ChatDev and MetaGPT on 30 MMLU questions (MAST traces, commit a70542e)")
cd = read("chatdev_runs.csv")
mg = read("metagpt_runs.csv")
import csv as _csv
for table, name, cols in ((cd, "chatdev_runs.csv", ("first_answer", "final_answer", "stated_answer")),
                          (mg, "metagpt_runs.csv", ("coder_answer", "tester_expected", "tester_other_expected"))):
    with open(DATA / name, newline="", encoding="utf-8") as fh:
        rows = list(_csv.DictReader(fh))
    for c in cols:
        table[c] = np.array([r[c] for r in rows], dtype=object)
n_cd, n_mg = len(cd["question"]), len(mg["question"])
print(f"ChatDev runs: {n_cd} ({len(set(cd['question']))} questions x 2 runs); MetaGPT runs: {n_mg}")
print(f"Subjects: {len(set(mg['subject']))} distinct MMLU subjects; professional_law questions: "
      f"{int(np.sum(mg['subject'] == 'professional_law'))}")

print("\n1a. What the ChatDev organisation did with a multiple-choice question")
print(f"  roles taking part per run: {', '.join(str(int(x)) for x in sorted(set(cd['roles'])))}")
mods, counts = np.unique(cd["modality"], return_counts=True)
print("  product modality chosen in the opening 'demand analysis' meeting: "
      + ", ".join(f"{m} {c}" for m, c in zip(mods, counts)))
langs, lc = np.unique(cd["language"], return_counts=True)
print("  programming language chosen: " + ", ".join(f"{l} {c}" for l, c in zip(langs, lc)))
print(f"  runs whose code imports tkinter (a desktop GUI): {int(cd['uses_tkinter'].sum())} of {n_cd}")
for k, label in [("code_lines", "lines of code"), ("doc_files", "documentation files"),
                 ("manual_lines", "lines of user manual")]:
    print(f"  {label}: median {q(cd[k], .5):.0f} (range {cd[k].min()}-{cd[k].max()})")
print(f"  runs whose test phase failed on 'no display' (TclError): {int(cd['gui_test_errors'].sum())} of {n_cd}")
print(f"  runs that then tried a virtual display (Xvfb/pyvirtualdisplay) and failed on it: "
      f"{int(cd['virtual_display_errors'].sum())} of {n_cd}")
print(f"  test rounds per run: median {q(cd['test_rounds'], .5):.0f}, max {cd['test_rounds'].max()}")

print("\n1b. Resources per question (ChatDev)")
for k, label, fmt in [("total_tokens", "tokens", "{:,.0f}"), ("completion_tokens", "tokens written by agents", "{:,.0f}"),
                      ("agent_words", "words written by agents", "{:,.0f}"), ("cost_usd", "cost, USD", "{:.3f}"),
                      ("duration_s", "wall-clock seconds", "{:.0f}"), ("llm_calls", "model calls", "{:.0f}"),
                      ("utterances", "utterances", "{:.0f}"), ("phases", "phase executions", "{:.0f}")]:
    x = cd[k].astype(float)
    print(f"  {label}: median " + fmt.format(q(x, .5)) + ", IQR " + fmt.format(q(x, .25)) + "-"
          + fmt.format(q(x, .75)) + ", total " + fmt.format(x.sum()))
words_per_token = cd["agent_words"].sum() / cd["completion_tokens"].sum()
print(f"  words per completion token, pooled: {words_per_token:.2f}")
READING_WPM = 238  # Brysbaert (2019): average silent reading rate for English non-fiction
minutes = cd["agent_words"] / READING_WPM
print(f"  minutes to read what the agents wrote, at {READING_WPM} words/min: median {q(minutes, .5):.1f}, "
      f"IQR {q(minutes, .25):.1f}-{q(minutes, .75):.1f}")

stages = ["decide", "code", "review", "test", "docs", "other"]
stage_names = {"decide": "Choose product format and language", "code": "Write the program",
               "review": "Code review", "test": "Test and fix",
               "docs": "Requirements file and user manual", "other": "Reflection and other"}
tok = {s: cd[f"tokens_{s}"].astype(float) for s in stages}
total = sum(t.sum() for t in tok.values())
print("\n1c. Where the ChatDev tokens went (pooled over 60 runs)")
for s in stages:
    print(f"  {stage_names[s]:<38s} {tok[s].sum() / total:6.1%}   median per run {q(tok[s], .5):,.0f}")

print("\n1d. The code reviewer")
rc = cd["review_comments"]
print(f"  review rounds per run: 3 scheduled; runs in which the reviewer raised any comment: "
      f"{int(np.sum(rc > 0))} of {n_cd}; runs where every round ended '<INFO> Finished': {int(np.sum(rc == 0))}")
print(f"  share of tokens spent in review in runs with no review comment at all: "
      f"{tok['review'][rc == 0].sum() / sum(t[rc == 0].sum() for t in tok.values()):.1%}")
gen = cd["review_generalise"] == 1
print(f"  runs where a review comment asked to replace the hard-coded answer with logic for other questions: "
      f"{int(gen.sum())}")
kinds = cd["final_answer_kind"]
print(f"  final code commits to one answer: {int(np.sum(kinds == 'single'))}; branching logic with several "
      f"possible answers: {int(np.sum(kinds == 'branching'))}; no answer index in code: {int(np.sum(kinds == 'none'))}")
print(f"  of the {int(gen.sum())} runs with a generalisation request, final code commits to one answer in "
      f"{int(np.sum(gen & (kinds == 'single')))}; first version committed to one answer in "
      f"{int(np.sum(gen & (cd['first_answer'] != '')))}")
stated = cd["stated_answer"] != ""
print(f"  runs where an agent message states 'The answer is (X)' as the task asked: {int(stated.sum())} of {n_cd}")

print("\n1e. Agreement (no ground truth is used anywhere)")
single = {(int(a), int(b)): c for a, b, c, k in zip(cd["question"], cd["run"], cd["final_answer"], kinds) if k == "single"}
pairs = [(single[(i, 0)], single[(i, 1)]) for i in range(30) if (i, 0) in single and (i, 1) in single]
agree = sum(a == b for a, b in pairs)
print(f"  ChatDev, run 0 vs run 1, questions where both runs commit to one answer: {len(pairs)}; same answer: {agree}")
mga = {int(i): str(a) for i, a in zip(mg["question"], mg["coder_answer"])}
cross = [(single[(i, r)], mga[i]) for (i, r) in single]
print(f"  ChatDev run vs MetaGPT on the same question (runs committing to one answer): "
      f"{sum(a == b for a, b in cross)} of {len(cross)} agree")

print("\n1f. MetaGPT: coder, tester, reviewer")
fmt_set = lambda x: "/".join(str(int(v)) for v in sorted(set(x)))
print(f"  messages per run: {fmt_set(mg['messages'])}; coder messages per run: {fmt_set(mg['coder_messages'])}; "
      f"tester: {fmt_set(mg['tester_messages'])}; reviewer: {fmt_set(mg['reviewer_messages'])}")
src, cnt = np.unique(mg["coder_answer_source"], return_counts=True)
print("  coder's answer recovered as: " + ", ".join(f"{s} {c}" for s, c in zip(src, cnt)))
print(f"  tester's first positive assertion equals the coder's answer: {int(mg['tester_restates_coder'].sum())} of {n_mg}")
other = mg["tester_other_expected"] != ""
print(f"  runs where the tests also assert answers for inputs the coder's function does not accept: {int(other.sum())} "
      f"(questions {', '.join(str(i) for i in mg['question'][other])})")
print(f"  runs where a reviewer message asks for edge cases, other inputs or generality: "
      f"{int(mg['reviewer_asks_scope'].sum())} of {n_mg}")
disp = mg["reviewer_disputes_answer"] == 1
print(f"  runs where the reviewer disputes the chosen answer itself: {int(disp.sum())} (question "
      f"{', '.join(str(i) for i in mg['question'][disp])}); coder messages after the dispute: "
      f"{int(mg['messages_after_dispute_by_coder'][disp].sum())}")
w = {r: mg[f"{r}_words"].astype(float) for r in ("coder", "tester", "reviewer")}
tw = sum(x.sum() for x in w.values())
print("  words by role, pooled: " + ", ".join(f"{r} {x.sum() / tw:.0%}" for r, x in w.items()))
print(f"  wall-clock seconds per run: median {q(mg['duration_s'], .5):.0f}")

# Figure 1: where the tokens went
fig, ax = plt.subplots(figsize=(8.0, 3.8))
order = sorted(stages, key=lambda s: tok[s].sum())
vals = [tok[s].sum() / total * 100 for s in order]
ax.barh([stage_names[s] for s in order], vals, color=BLUE, height=0.62)
for y, v, s in zip(range(len(order)), vals, order):
    ax.text(v + 0.8, y, f"{v:.0f}%  (median {q(tok[s], .5) / 1000:.1f}k per run)", va="center", fontsize=9, color=INK2)
ax.set_xlim(0, 62)
ax.set_xticks([0, 10, 20, 30, 40])
ax.set_xlabel("Share of all tokens in 60 ChatDev runs on 30 MMLU questions (%)")
ax.grid(axis="y", visible=False)
fig.suptitle("Where an agent 'software company' spent its tokens on multiple-choice questions",
             x=0.01, ha="left", fontweight="bold", fontsize=11.5)
fig.text(0.01, 0.01, "Data: ChatDev traces released with MAST (Cemri et al. 2025), commit a70542e. "
         f"Median run: {q(cd['total_tokens'], .5):,.0f} tokens.", fontsize=8, color=INK2)
fig.tight_layout(rect=(0, 0.04, 1, 0.94))
fig.savefig(FIG / "fig1_chatdev_tokens.png")
plt.close(fig)

# ---------------------------------------------------------------------------------------
# 2. Separation of duties with correlated and anchored reviewers
# ---------------------------------------------------------------------------------------
header("2. Separation of duties: residual error after k reviewers")
print("Model: item difficulty U ~ N(0,1). Agent j errs on an item when lam_j*U + sqrt(1-lam_j^2)*Z_j > tau_j,")
print("Z_j independent N(0,1); tau_j = Phi^-1(1 - q_j). Two agents' errors then have tetrachoric correlation lam_j*lam_k.")
print("A reviewer stops a producer's error only if it would itself answer correctly AND it is not anchored on the")
print("producer's answer (anchoring probability a). Residual error after k reviewers:")
print("   R_k = E_U[ p_P(U) * prod_j ( a + (1 - a) p_j(U) ) ],  p_j(U) = Phi((lam_j U - tau_j) / sqrt(1 - lam_j^2))")

GH_X, GH_W = np.polynomial.hermite_e.hermegauss(160)
GH_W = GH_W / GH_W.sum()


def p_err(u, qerr, lam):
    if lam >= 1:
        return (u > stats.norm.ppf(1 - qerr)).astype(float)
    return stats.norm.cdf((lam * u - stats.norm.ppf(1 - qerr)) / np.sqrt(1 - lam ** 2))


def residual(k, qp=0.2, rho=0.5, a=0.0, qr=None, rho_r=None):
    """Residual error with a producer (qp, loading sqrt(rho)) and k identical reviewers.

    rho_r, if given, is the producer-reviewer error correlation (reviewers load sqrt(rho_r/rho) relative
    to the producer's loading); by default reviewers are copies of the producer."""
    qr = qp if qr is None else qr
    lp = np.sqrt(rho)
    lr = lp if rho_r is None else rho_r / lp
    pp, pr = p_err(GH_X, qp, lp), p_err(GH_X, qr, lr)
    return float(np.sum(GH_W * pp * (a + (1 - a) * pr) ** k))


QP = 0.2
print(f"\nProducer and reviewers all err on {QP:.0%} of items (same model, same role prompt quality).")
print("Residual error after k reviewers, a = 0 (blind review):")
rhos = [0.0, 0.25, 0.5, 0.75, 0.9]
for rho in rhos:
    row = [residual(k, QP, rho) for k in range(0, 6)]
    print(f"  rho={rho:<4}  " + "  ".join(f"k={k}: {r:7.4f}" for k, r in enumerate(row)))
print("Effective number of independent reviewers, n_eff = log(R_k)/log(q) - 1:")
for rho in rhos:
    print(f"  rho={rho:<4}  " + "  ".join(f"k={k}: {np.log(residual(k, QP, rho)) / np.log(QP) - 1:4.2f}" for k in (1, 3, 5)))
print("Anchoring: residual error after 3 reviewers")
for rho in (0.0, 0.5, 0.9):
    print(f"  rho={rho:<4} " + "  ".join(f"a={a}: {residual(3, QP, rho, a):.4f}" for a in (0.0, 0.25, 0.5, 0.75, 1.0)))

# Monte Carlo check of the quadrature
n_mc = 400_000
u = RNG.standard_normal(n_mc)
for rho, a, k in [(0.5, 0.0, 3), (0.9, 0.5, 3)]:
    lam = np.sqrt(rho)
    tau = stats.norm.ppf(1 - QP)
    e = lam * u[:, None] + np.sqrt(1 - lam ** 2) * RNG.standard_normal((n_mc, k + 1)) > tau
    anchored = RNG.random((n_mc, k)) < a
    miss = e[:, 0] & np.all(anchored | e[:, 1:], axis=1)
    print(f"  Monte Carlo check rho={rho}, a={a}, k={k}: {miss.mean():.4f} (quadrature {residual(k, QP, rho, a):.4f}, "
          f"MC s.e. {np.sqrt(miss.mean() * (1 - miss.mean()) / n_mc):.4f})")

print("\nA weaker but independent check against a stronger copy (one reviewer, producer q=0.2, rho_PP=0.8):")
copy_r = residual(1, 0.2, 0.8)
print(f"  same-model reviewer, q=0.20, error correlation with producer 0.80: residual {copy_r:.4f}")
for qr in (0.30, 0.40, 0.50):
    for rr in (0.0, 0.2):
        print(f"  independent-evidence reviewer, q={qr:.2f}, correlation {rr:.1f}: residual {residual(1, 0.2, 0.8, 0.0, qr, rr):.4f}")
# break-even: how bad can an uncorrelated reviewer be and still match the copy?
be = optimize.brentq(lambda x: residual(1, 0.2, 0.8, 0.0, x, 0.0) - copy_r, 0.2, 0.99)
print(f"  an uncorrelated reviewer matches the same-model copy at its own error rate q = {be:.3f}")
be2 = optimize.brentq(lambda x: residual(1, 0.2, 0.8, 0.0, x, 0.2) - copy_r, 0.2, 0.99)
print(f"  a reviewer with correlation 0.2 matches it at q = {be2:.3f}")
print(f"  MetaGPT's tester (oracle = the coder's own answer) is the case a = 1: residual = q = {residual(3, QP, 0.5, 1.0):.2f} "
      f"for any number of tests")

# Figure 5: a weaker independent check against a stronger copy
fig, ax = plt.subplots(figsize=(7.4, 3.9))
qrs = np.linspace(0.05, 0.8, 76)
for rr, col, lab in [(0.0, AQUA, "reviewer errors uncorrelated with the producer's"),
                     (0.2, BLUE, "error correlation 0.2 with the producer")]:
    ys = [residual(1, 0.2, 0.8, 0.0, x, rr) for x in qrs]
    ax.plot(qrs, ys, color=col, label=lab)
ax.axhline(copy_r, color=ORANGE, linewidth=2)
ax.text(0.06, copy_r + 0.005, f"copy of the producer as reviewer (q = 0.20, ρ = 0.8): {copy_r:.3f}",
        fontsize=8.5, color=INK2)
ax.axhline(0.2, color=INK2, linestyle=":", linewidth=1.2)
ax.text(0.06, 0.2 + 0.006, "no review", fontsize=8.5, color=INK2)
for x in (be, be2):
    ax.plot([x], [copy_r], marker="o", markersize=6, color=INK2)
ax.annotate(f"break-even q = {be:.2f}", (be, copy_r), xytext=(be + 0.015, copy_r - 0.035), fontsize=8.5, color=INK2)
ax.annotate(f"break-even q = {be2:.2f}", (be2, copy_r), xytext=(be2 - 0.16, copy_r + 0.022), fontsize=8.5, color=INK2)
ax.set_xlabel("The independent reviewer's own error rate (q)")
ax.set_ylabel("Producer errors that get through")
ax.set_xlim(0.05, 0.8)
ax.set_ylim(0, 0.23)
ax.legend(loc="lower right", fontsize=8.5)
fig.suptitle("One reviewer: a worse but independent check beats a copy of the producer", x=0.01, ha="left",
             fontweight="bold", fontsize=11.5)
fig.text(0.01, 0.01, "Producer errs on 20% of items. Same one-factor probit model as figure 2, exact by quadrature.",
         fontsize=8, color=INK2)
fig.tight_layout(rect=(0, 0.04, 1, 0.94))
fig.savefig(FIG / "fig5_independent_check.png")
plt.close(fig)

# Figure 2: residual error vs number of reviewers
fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.9), sharey=True)
ks = np.arange(0, 7)
for ax, a, title in [(axes[0], 0.0, "Blind review (reviewer has not seen the answer)"),
                     (axes[1], 0.5, "Review after seeing the answer (anchoring a = 0.5)")]:
    for rho, col in zip(rhos, BLUE_RAMP):
        r = [residual(k, QP, rho, a) for k in ks]
        ax.plot(ks, r, color=col, marker="o", markersize=4.5, label=f"ρ = {rho}")
        if a == 0.0:
            ax.text(ks[-1] + 0.15, r[-1], f"ρ = {rho}", va="center", fontsize=8.5, color=INK2)
    if a > 0:
        ax.legend(title="error correlation", loc="lower left", fontsize=8.5, title_fontsize=8.5)
    ax.set_yscale("log")
    ax.set_xlim(-0.2, 7.3)
    ax.set_xticks(ks)
    ax.set_xlabel("Reviewers stacked after the producer (k)")
    ax.set_title(title, loc="left", fontsize=10)
axes[0].set_ylabel("Errors that get through (log scale)")
axes[0].set_ylim(1e-5, 0.3)
fig.suptitle("Separation of duties assumes independent checkers: every agent errs on 20% of items", x=0.01,
             ha="left", fontweight="bold", fontsize=11.5)
fig.text(0.01, 0.01, "One-factor probit model of agent errors; ρ = error correlation between any two agents. "
         "Exact values by Gauss–Hermite quadrature.", fontsize=8, color=INK2)
fig.tight_layout(rect=(0, 0.04, 1, 0.95))
fig.savefig(FIG / "fig2_correlated_review.png")
plt.close(fig)

# ---------------------------------------------------------------------------------------
# 3. Deliberation with conformity, authority and correction
# ---------------------------------------------------------------------------------------
header("3. Deliberation: a lead and four advisers on four-option questions")
N_AGENTS, N_ITEMS, ROUNDS = 5, 40_000, 3
Q_D, RHO_D = 0.30, 0.5


def solve_attractor(target=0.60):
    """Probability g that an erring agent picks the item's 'attractor' wrong option, so that two erring
    agents choose the same wrong option with probability `target` (Kim et al. 2025 report 60%)."""
    f = lambda g: (g + (1 - g) / 3) ** 2 + 2 * ((1 - g) / 3) ** 2 - target
    return optimize.brentq(f, 0.0, 1.0)


G_ATTR = solve_attractor()
print(f"Agents err with probability {Q_D} (error correlation rho = {RHO_D}); an erring agent picks the item's most")
print(f"attractive wrong option with probability g = {G_ATTR:.3f}, so two erring agents agree 60% of the time.")
print("Each round, every agent sees the others' current answers. With probability c it is persuaded by evidence:")
print("if it is wrong and someone holds the right answer, it switches to it. Otherwise, with probability s it conforms:")
print("it adopts the answer with the most weight among the others (the lead counts w, advisers 1; it keeps its own")
print("answer if that answer ties for the most weight). Otherwise it keeps its answer. Three rounds. Option 0 is correct.")


def initial_answers(rng, n_items=N_ITEMS, n=N_AGENTS, qerr=Q_D, rho=RHO_D, g=G_ATTR):
    lam = np.sqrt(rho)
    u = rng.standard_normal((n_items, 1))
    err = lam * u + np.sqrt(1 - lam ** 2) * rng.standard_normal((n_items, n)) > stats.norm.ppf(1 - qerr)
    attractor = rng.integers(1, 4, size=(n_items, 1))
    other = rng.integers(1, 4, size=(n_items, n))
    wrong = np.where(rng.random((n_items, n)) < g, attractor, other)
    return np.where(err, wrong, 0)


def plurality(ans, weights, tiebreak):
    """Weighted plurality per row; ties go to the tiebreak column's answer if it is tied, else lowest option."""
    sup = np.stack([((ans == o) * weights).sum(axis=1) for o in range(4)], axis=1)
    best = sup.max(axis=1, keepdims=True)
    tied = sup == best
    tb = tiebreak[:, None] == np.arange(4)[None, :]
    pick = np.where((tied & tb).any(axis=1), tiebreak, tied.argmax(axis=1))
    return pick


def deliberate(ans0, rng, s, c, w, rounds=ROUNDS):
    ans = ans0.copy()
    n_items, n = ans.shape
    for _ in range(rounds):
        new = ans.copy()
        for i in range(n):
            weights = np.where(np.arange(n) == 0, w, 1.0).astype(float)
            weights[i] = 0.0
            sup = np.stack([((ans == o) * weights).sum(axis=1) for o in range(4)], axis=1)
            best = sup.max(axis=1)
            own_ties = sup[np.arange(n_items), ans[:, i]] >= best
            # random tie-break among the others' top options
            noise = rng.random((n_items, 4)) * 1e-6
            target = np.argmax(np.where(sup >= best[:, None], 1.0 + noise, 0.0), axis=1)
            target = np.where(own_ties, ans[:, i], target)
            someone_right = ((ans == 0) & (weights > 0)).any(axis=1)
            r = rng.random(n_items)
            persuaded = (r < c) & (ans[:, i] != 0) & someone_right
            conform = (~persuaded) & (rng.random(n_items) < s)
            new[:, i] = np.where(persuaded, 0, np.where(conform, target, ans[:, i]))
        ans = new
    return ans


ans0 = initial_answers(RNG)
lead0 = ans0[:, 0]
vote0 = plurality(ans0, np.ones(N_AGENTS), lead0)
acc_single = float(np.mean(ans0 == 0))
acc_vote0 = float(np.mean(vote0 == 0))
print(f"\nSingle agent accuracy (any one agent, before talking): {acc_single:.3f}")
print(f"Plurality vote of the five initial answers (no discussion; ties to the lead): {acc_vote0:.3f}")
print(f"Lead alone, initial answer: {np.mean(lead0 == 0):.3f}")
unan0 = (ans0 == ans0[:, :1]).all(axis=1)
print(f"Initial unanimity: {np.mean(unan0):.3f}; accuracy on questions where the five first answers agree: "
      f"{np.mean(ans0[unan0, 0] == 0):.3f}")

grid_w = [1, 1.5, 2, 3, 4, 6]
grid_s = [0.3, 0.6, 0.9]
res = {}
for s in grid_s:
    for c in (0.0, 0.1, 0.3):
        for wv in grid_w:
            fin = deliberate(ans0, np.random.default_rng(int(1000 * s + 100 * c + 10 * wv)), s, c, wv)
            lead = fin[:, 0]
            vote = plurality(fin, np.ones(N_AGENTS), lead)
            unan = (fin == fin[:, :1]).all(axis=1)
            c2i = np.mean((ans0 == 0) & (fin != 0))
            i2c = np.mean((ans0 != 0) & (fin == 0))
            res[(s, c, wv)] = (np.mean(lead == 0), np.mean(vote == 0), np.mean(unan), c2i, i2c,
                               np.mean(lead == lead0), np.mean(lead[unan] == 0) if unan.any() else np.nan)
print("\nFinal accuracy when the lead decides | plurality of final answers | unanimity | agent answers flipped right->wrong | wrong->right")
for c in (0.0, 0.1, 0.3):
    print(f"  evidence-driven correction c = {c}")
    for s in grid_s:
        for wv in (1, 2, 4, 6):
            a_lead, a_vote, un, c2i, i2c, same, acc_un = res[(s, c, wv)]
            print(f"    s={s} w={wv:<3}  lead {a_lead:.3f} | vote {a_vote:.3f} | unanimous {un:.3f} | "
                  f"right->wrong {c2i:.3f} | wrong->right {i2c:.3f} | lead kept own first answer {same:.3f} | "
                  f"accuracy when unanimous {acc_un:.3f}")
gap = {s: res[(s, 0.0, 6)][0] - acc_single for s in grid_s}
print("\nWith no correction and w = 6, the lead's final accuracy minus a single agent's: "
      + ", ".join(f"s={s}: {v:+.3f}" for s, v in gap.items()))
print("With no correction, group answer (plurality of final answers) minus the no-discussion vote, w = 1 -> w = 6: "
      + ", ".join(f"s={s}: {res[(s, 0.0, 1)][1] - acc_vote0:+.3f} -> {res[(s, 0.0, 6)][1] - acc_vote0:+.3f}" for s in grid_s))
print("Right answers abandoned (share of agent-question pairs), w = 1 -> w = 6, c = 0: "
      + ", ".join(f"s={s}: {res[(s, 0.0, 1)][3]:.3f} -> {res[(s, 0.0, 6)][3]:.3f}" for s in grid_s))

# Figure 3
fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.9))
ax = axes[0]
offsets = {0.3: -0.004, 0.6: 0.004, 0.9: 0.0}
for s, col in zip(grid_s, [BLUE, ORANGE, AQUA]):
    ys = [res[(s, 0.0, wv)][1] for wv in grid_w]
    ax.plot(grid_w, ys, color=col, marker="o", markersize=4.5, label=f"conformity s = {s}")
    ax.text(grid_w[-1] + 0.15, ys[-1] + offsets[s], f"s = {s}", va="center", fontsize=8.5, color=INK2)
ax.axhline(acc_vote0, color=INK2, linestyle="--", linewidth=1.2)
ax.text(1.6, acc_vote0 - 0.006, "vote of the five first answers, no discussion", fontsize=8, color=INK2)
ax.axhline(acc_single, color=INK2, linestyle=":", linewidth=1.2)
ax.text(1.6, acc_single + 0.003, "one agent alone", fontsize=8, color=INK2)
ax.legend(loc="center left", bbox_to_anchor=(0.0, 0.42), fontsize=8.5)
ax.set_xlim(0.8, 7.2)
ax.set_xlabel("Weight of the lead's answer in the others' eyes (w)")
ax.set_ylabel("Accuracy of the group's final answer")
ax.set_title("Deference to the lead erases the ensemble gain", loc="left", fontsize=10)
ax = axes[1]
for cc, col in zip((0.0, 0.1, 0.3), [BLUE, ORANGE, AQUA]):
    xs = [res[(s, cc, wv)][2] for s in grid_s for wv in grid_w]
    ys = [res[(s, cc, wv)][1] for s in grid_s for wv in grid_w]
    ax.scatter(xs, ys, s=34, color=col, edgecolor=SURFACE, linewidth=1.5, label=f"correction c = {cc}")
ax.axhline(acc_vote0, color=INK2, linestyle="--", linewidth=1.2)
ax.scatter([np.mean(unan0)], [acc_vote0], s=46, marker="D", color=INK2, edgecolor=SURFACE, linewidth=1.5)
ax.annotate("five first answers,\nbefore any discussion", (np.mean(unan0), acc_vote0),
            xytext=(np.mean(unan0) + 0.02, acc_vote0 + 0.015), fontsize=8, color=INK2)
ax.set_xlabel("Share of questions ending unanimous")
ax.set_ylabel("Accuracy of the group's final answer")
ax.set_title("Consensus is cheap; only correction buys accuracy", loc="left", fontsize=10)
ax.legend(loc="upper left", fontsize=8.5)
ax.set_xlim(0.3, 1.03)
fig.suptitle("Five agents, four-option questions, each agent right 70% of the time (error correlation 0.5)",
             x=0.01, ha="left", fontweight="bold", fontsize=11.5)
fig.text(0.01, 0.01, f"Simulation, {N_ITEMS:,} questions, three rounds; group answer = plurality of final answers. "
         "Left: c = 0. Right: every (s, w) cell of the grid.", fontsize=8, color=INK2)
fig.tight_layout(rect=(0, 0.04, 1, 0.95))
fig.savefig(FIG / "fig3_deliberation.png")
plt.close(fig)

# ---------------------------------------------------------------------------------------
# 4. The last human: a fixed review budget
# ---------------------------------------------------------------------------------------
header("4. The last human: one reviewer, a fixed budget, a growing team")
H_MIN = 120          # minutes a day the human spends reading agent output
WPM = READING_WPM
M_PER_AGENT = 4      # deliverables per agent per day
L_WORDS = 1_000      # words per deliverable
COORD_WORDS = 150    # words of coordination traffic per pair of agents per day that reach the human
ERR = 0.15           # share of deliverables with a material error
H_MAX = 0.8          # chance of catching an error in a deliverable read in full
V = 1.0              # value of a correct deliverable (the unit)
print(f"Human reads {WPM} words/min for {H_MIN} min/day -> {H_MIN * WPM:,} words/day.")
print(f"Each agent hands over {M_PER_AGENT} deliverables of {L_WORDS:,} words a day; each pair of agents adds "
      f"{COORD_WORDS} words of coordination the human must read (Graicunas/Brooks: pairs grow as n(n-1)/2).")
print(f"A deliverable carries a material error with probability {ERR}; read in full, an error is caught with "
      f"probability {H_MAX}; read in part, in proportion to the share read. A caught error costs the deliverable;")
print("an uncaught one costs D (in units of a correct deliverable's value V = 1).")


def day(n, D, h_min=H_MIN, m=M_PER_AGENT, L=L_WORDS, coord=COORD_WORDS, err=ERR, hmax=H_MAX):
    words = n * m * L + coord * n * (n - 1) / 2
    share = min(1.0, h_min * WPM / words)
    catch = hmax * share
    value = n * m * ((1 - err) * V - err * (1 - catch) * D)
    return value, share, words


ns = np.arange(1, 21)
Ds = [2, 5, 10, 20]
for D in Ds:
    vals = np.array([day(n, D)[0] for n in ns])
    best = ns[np.argmax(vals)]
    neg = ns[vals < 0]
    print(f"  D = {D:<3} best team size {best:>2} (net value {vals.max():.2f}/day); net value at n=12: "
          f"{vals[11]:.2f}; first n with negative net value: {neg[0] if len(neg) else 'none up to 20'}")
print("  share of output the human can read: " + ", ".join(f"n={n}: {day(n, 5)[1]:.0%}" for n in (1, 2, 4, 8, 12)))
print("  words produced per day (work 'done'): " + ", ".join(f"n={n}: {day(n, 5)[2]:,.0f}" for n in (1, 4, 12)))
# Closed form when the human cannot read everything and coordination is ignored: value is
# n*m*((1-e) - e*D) + e*D*hmax*H*WPM/L, which is linear in n with slope m*((1-e) - e*D).
for D in Ds:
    slope = M_PER_AGENT * ((1 - ERR) - ERR * D)
    print(f"  D = {D:<3} slope per extra agent once reading is saturated (no coordination): {slope:+.2f}; "
          f"break-even D = (1-e)/e = {(1 - ERR) / ERR:.2f}")
print("Sensitivity: best team size by daily reading budget (D = 10)")
for hm in (30, 60, 120, 240, 480):
    vals = np.array([day(n, 10, h_min=hm)[0] for n in ns])
    print(f"  {hm:>3} min/day -> best n = {ns[np.argmax(vals)]}, net value {vals.max():.2f}")
print("Anchor from section 1: auditing one ChatDev answer means reading a median of "
      f"{q(cd['agent_words'], .5):,.0f} words, {q(minutes, .5):.0f} minutes at {WPM} words/min; "
      f"a {H_MIN}-minute budget audits {H_MIN / q(minutes, .5):.1f} such answers a day.")

# Figure 4
fig, ax = plt.subplots(figsize=(7.4, 4.0))
for D, col in zip([2, 10, 20], [AQUA, BLUE, ORANGE]):
    vals = np.array([day(n, D)[0] for n in ns])
    ax.plot(ns, vals, color=col, marker="o", markersize=4)
    ax.text(ns[-1] + 0.3, vals[-1], f"uncaught error costs {D}×", va="center", fontsize=8.5, color=INK2)
ax.axhline(0, color=INK2, linewidth=1)
sat = ns[[day(n, 5)[1] < 1 for n in ns]][0]
ax.axvline(sat - 0.5, color=INK2, linestyle=":", linewidth=1.2)
ax.text(sat - 0.3, -95, "from here on the human\ncannot read everything", fontsize=8, color=INK2, va="bottom")
ax.set_xlim(0.5, 25)
ax.set_xticks([1, 4, 8, 12, 16, 20])
ax.set_xlabel("Agents reporting to one human (n)")
ax.set_ylabel("Net value per day (correct deliverable = 1)")
ax.set_title("More agents, more output, less value: one human with two hours a day", loc="left")
fig.text(0.01, 0.01, f"Each agent: {M_PER_AGENT} deliverables/day of {L_WORDS:,} words, {ERR:.0%} with a material error; "
         f"{COORD_WORDS} words of coordination per pair; reading at {WPM} words/min.", fontsize=8, color=INK2)
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(FIG / "fig4_review_budget.png")
plt.close(fig)

header("Figures written")
for f in sorted(FIG.glob("*.png")):
    print(f"  figures/{f.name}")
