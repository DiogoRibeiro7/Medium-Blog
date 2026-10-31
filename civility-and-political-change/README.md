# When Power Calls for Civility

Reproducible companion for the long-form essay
**[When Power Calls for Civility: How Moderation Can Contain Social Change](article.md)**.

## Scope

This is an interpretive essay that brings political history, social-movement
research, normative political theory, and a **strictly hypothetical** bargaining
illustration into conversation. Its starting point is Alex Zamalin's *Against
Civility* (Beacon Press, 2021), alongside Martin Luther King Jr.'s *Letter from
Birmingham Jail* (1963). The essay does not equate interpersonal politeness,
political compromise, nonviolence, centrism, and enforced restraint.

**No historical outcomes are fitted or estimated by the Python code.** All
quantitative parameters are explicit thought-experiment assumptions, not
estimates from any country, movement, institution, or period. No figure should
be circulated as evidence of a real political effect. Historical and empirical
claims are separately cited to their primary sources in the manuscript.

## Reproduce

Python 3.10+ with the local dependencies:

```bash
python -m pip install -r requirements.txt
python analysis.py
python -m unittest discover -s tests -v
```

From the repository root:

```bash
cd civility-and-political-change
python analysis.py --output /tmp/civility-illustrations
```

The script is deterministic and does not download anything or use random seeds.
It produces eight original PNG figures and four plain-text CSV tables. The
figure-generating code, tests, and table schemas are held in this directory.
The [model walkthrough notebook](model_walkthrough.ipynb) provides a
separate, executable demonstration of the same assumptions and calculations.
The path-scoped GitHub Actions workflow verifies the mathematical and notebook
contracts under Python 3.12.
The illustrative equations are also described directly in the article.

## Generated figures

| Figure | Meaning | Evidence category |
| --- | --- | --- |
| 1 | Midpoint versus a hypothetical rights threshold | Normative illustration |
| 2 | Politeness versus tactical disruption | Conceptual classification |
| 3 | Two hypothesized pathways from mobilisation to policy | Mechanism hypothesis, not identified |
| 4 | Equal Nash weights and unequal outside options | Deduction under assumed model |
| 5 | Nash outcome sensitivity to weight and outside option | Deduction under assumed model |
| 6 | The assumed effect of procedural restraint on leverage | Conditional model illustration |
| 7 | Present value of a constant flow of harm from reform delay | Mathematical identity under assumptions |
| 8 | Distinct kinds of evidence and their inferential limits | Epistemic classification |

Files live in [`figures/`](figures/) and the data behind figures 4–7 in
[`results/`](results/). The two conceptual figures and two mechanism/evidence
diagrams are produced from explicit layout coordinates rather than CSV data.

## Model

The model considers a unit of **transferable surplus** split between a reformer
and an incumbent, with disagreement utilities \(d_R,d_I\) and reformer weight
\(\alpha\). The assumptions require \(d_R,d_I\ge 0\),
\(d_R+d_I\le 1\), \(\alpha\in[0,1]\). The weighted Nash solution is:

\[
r^* = d_R + \alpha(1-d_R-d_I),\qquad i^*=1-r^*.
\]

The toy restraint extension postulates
\(d_R(c)=\max(0,0.35-0.30c)\), \(d_I=0.45\),
\(\alpha=0.5\). The assumption that restraint lowers the reformer's outside
option is **not an empirical finding**; the paper discusses circumstances
under which it may fail or reverse.

Waiting-cost curves show
\(L(T)=h\int_0^T e^{-\rho t}dt\) for \(h=1\) and hypothetical
discount rates \(\rho=0,0.03,0.08\), with the appropriate continuous
zero-rate limit.

The model is not a game of nontransferable rights, does not identify an ethical
optimal policy, assumes complete information, and has no causal connection to
the historical examples. Those limitations matter. A mathematically balanced
split of a surplus and a just distribution of civil rights are different
questions.

## Editorial checks

- No real-world causal inference from the toy figures.
- Separate a request for respectful interpersonal language from a demand to
  stop mobilisation or postpone reform.
- Treat the different histories of Birmingham, British suffrage, and Portugal's
  1974–76 transition as distinct, not as interchangeable illustrations.
- Report inconvenient evidence about the benefits of deliberation, the costs of
  elite incivility, and the inconsistent outcomes of protest.
- Keep short quotations within applicable copyright limits.
- Consult the direct links and DOIs in the article's reference section before
  publication; the manuscript is not a peer-reviewed systematic review.

## Publishing

The Markdown article and its images are prepared for manual publication on
Medium. A pull request in this repository does **not** itself publish anything.
