# A category-theoretic context map for LLM agents, built as a database

Companion code for the Medium article *A Category-Theoretic Context Map for LLM Agents, Built as a Database and Tested*.

A recent proposal for making LLM agents reliable keeps the model and adds a "map" of semantic contexts: contexts as the objects of a category, the meanings valid in each context as the fibres of a functor, tokens attached to contexts by a profunctor, the whole assembled by the Grothendieck construction, and three checks (declared laws, commuting squares, loops). This folder builds that map for the proposal's own example, a `deploy` request that means different things in cloud infrastructure, production and a live incident, twice: once from the category theory and once as an SQLite database. It then measures what the checks can and cannot catch.

One script prints every number in the article and draws every figure. It calls no language model and downloads nothing.

## What is built

- `fincat.py`: finitely presented categories (generating arrows and path equations), normal forms by rewriting, a confluence check, enumeration of morphisms and hom-sets.
- `cset.py`: instances (functors to sets) as tables and foreign keys; path-equation and naturality checks row by row; the category of elements as one flat table and back; set-valued profunctors and their composition (a coend, computed by union-find).
- `sql.py`: the same objects in SQLite. Each context is a table, each arrow a `NOT NULL` foreign key, each path equation a join query, hom-set emptiness a recursive query, the category of elements a single table with `CHECK` constraints, profunctor composition a join followed by connected components.
- `deploy_example.py`: the deploy schema (Cloud, Staging, Production, Incident; arrows `s`, `p`, `i`, `u`; equation `u;i = id`), a 16-row instance, a ten-turn conversation, a variant schema with one more arrow `w: Cloud -> Production`, the declared law "a release authorisation in Production can only come from one already there", and two updates (revoke everywhere, revoke inside the incident only).
- `agreement.py`: 800 random instances (a third clean, the rest corrupted), 300 random schemas and 200 random pairs of profunctors, each checked in Python and in SQL.
- `drift.py`: learned transports. Adapters fitted by least squares on noisy pairs; the loop drift, its closed form, an exact sampler that does not grow with the sample size (checked against brute force), and the loop check as a statistical test, with its power and its sensitivity to the assumed noise level.
- `grounding.py`: a calibrated (and an overconfident) context classifier in front of an exact permission database; unpermitted actions and refusals under the most-likely-context rule and an expected-cost rule; an audit that runs the decisions through SQLite.
- `figures.py`: the three figures.

## Run

From this folder:

```bash
python -m pip install -r requirements.txt
python context_map_database.py > results/output.txt
python -m pytest -q tests
```

Python 3.13, about 30 seconds. Every random draw comes from a seeded generator, so a rerun with the pinned versions reproduces `results/output.txt` exactly; CI checks that it does. `results/summary.json` holds the same numbers in machine-readable form.

## Sections of the output

| Section | Question |
|---------|----------|
| 1 | The deploy schema as a category: its 10 morphisms, and the 18 after adding one arrow. Is there a path from Staging to Production? |
| 2 | The instance as a functor; the loop `i;u` and what it moves; the flat table and which of its conditions SQLite can enforce with `CHECK`. |
| 3 | The conversation under a context-blind rule and a context-aware query, on the first schema and on two versions of the second. |
| 4 | The proposal's three checks as queries: a declared law, commuting squares (naturality of an update), a loop. |
| 5 | Python and SQL on random instances, schemas and profunctors; how much a join alone overcounts a profunctor composite. |
| 6 | Learned adapters: loop drift against the formula, and the loop check's power and false-alarm rate. |
| 7 | Grounding: unpermitted actions carried out and permitted ones refused, at context accuracies of 80%, 90%, 95% and 99%. |

## Limits

The learned-transport study uses linear adapters, Gaussian inputs and a known noise level, the most favourable case for the loop check; the closed form is first order in the noise. The grounding study uses a synthetic, two-dimensional observation model with four contexts and one permitted action per context; its point is the structure of the error (it comes from reading the context, not from the database), not the particular rates. No language model is run anywhere.

## Licence

Code: Apache-2.0.
