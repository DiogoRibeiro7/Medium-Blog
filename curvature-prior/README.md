# Curvature as a prior: flat and hyperbolic embeddings of hierarchies

Companion code for the Medium article *Flat Versus Hyperbolic Embeddings of Hierarchies: WordNet, Controls and Cross-Links from 2 to 200 Dimensions*.

The claim being tested is that hierarchical, nested context needs curved (hyperbolic) geometry, and that today's "flat" representations are the wrong shape for it. This folder embeds the same graphs in flat (Euclidean) and in hyperbolic space, with the same loss, optimiser and budget, at 2, 5, 10, 20, 50, 100 and 200 dimensions. It then measures how much curvature helps, on real hierarchies and on controls of known shape.

One script prints every number in the article and draws every figure. It calls no language model.

## Data

WordNet 3.0, as packaged in the NLTK data repository, is downloaded at run time from a pinned commit, `550b6625`, and verified against the size and SHA-256 in the data-registry contract `data_sources/princeton-wordnet-3-0.json`. The registry record is `external/princeton-wordnet-3-0` in DiogoRibeiro7/data. Nothing from WordNet is committed here. WordNet 3.0 is © 2006 Princeton University and is used under its licence. Cite Miller (1995) and Fellbaum (1998).

The graphs:

- **WordNet mammals.** The 1,182 noun synsets below `mammal`, with hypernym links as undirected edges. It is a tree plus one extra edge.
- **WordNet food.** The 1,527 synsets below `food`. It is a tree plus 16 extra edges from multiple inheritance.
- **WordNet mammals, transitive closure.** Every synset is joined to every one of its ancestors. This is the object Nickel and Kiela (2017) embed.
- **A complete tree** with branching 3 and depth 6, which has 1,093 nodes.
- **A 33 x 33 grid**, which has 1,089 nodes and is flat.
- **The dial.** The mammal tree with random cross-links added, from 0.5% to 50% of its node count, with three random placements per level.

Each graph's metric is its shortest-path distance. Tree-likeness is Gromov's four-point delta, taken as the maximum and the mean over 200,000 random quadruples. The maximum is a lower bound on the exact value.

## Method

- **Same treatment for both geometries.** Every node gets a point, and the loss is the mean squared relative distance error over all pairs, `(u * d_embedded / d_graph - 1)^2`. The scale `u` is learned. In hyperbolic space, with curvature fixed at -1, learning `u` is the same as learning the curvature.
- **Optimiser.** Adam with a cosine-decayed learning rate of 0.05, 1,500 steps of 256 rows each against all columns, and the same seed for both geometries. It runs in float64 with JAX.
- **Hyperbolic coordinates.** Points are stored in polar form, as tangent vectors at the origin. Distances use the hyperbolic law of cosines written as a sum of non-negative terms, so they stay accurate at large radii; a test checks a pair of points at radius 11. Radii are capped at 12, and no fit comes near the cap.
- **Measures, on all pairs.**
  - Average distortion: the mean of `|c * d_embedded / d_graph - 1|`, with the best scale `c`.
  - Worst-case distortion: the ratio of the largest to the smallest `d_embedded / d_graph`.
  - Mean average precision (MAP): how well each node's graph neighbours are recovered by ranking all other nodes by embedded distance, the reconstruction score of Nickel and Kiela (2017).

## Files

- `wordnet.py`: downloads the WordNet zip, verifies it against the contract, and parses `data.noun` and `index.noun` into hypernym subtrees.
- `graphs.py`: builds the graphs, shortest-path metrics, transitive closure, cross-links and the four-point delta.
- `embed.py`: the flat and hyperbolic distance functions, the fit and the evaluation.
- `curvature_prior.py`: the analysis. It prints `results/output.txt`, writes `results/summary.json` and draws the four figures.
- `check_results.py`: compares a rerun with the committed `summary.json`, within tolerances (see below).
- `figures.py` and `tests/`.

## Run

From this folder:

```bash
python -m pip install -r requirements.txt
python -m pytest -q tests
python curvature_prior.py > results/output.txt
```

Python 3.13. The full run takes about half an hour on two CPU cores.

## Reproducibility

Every random draw is seeded, and a rerun on the same machine reproduces `summary.json` exactly. Across machines, XLA's floating-point arithmetic can differ in the last bits, and a non-convex fit can carry that forward. So CI reruns everything and requires each reported number to match the committed run within a tolerance far below any difference the article relies on:

- graph statistics exactly;
- average distortion and MAP within 0.003 or 3%;
- worst-case distortion and radii within 25%.

Two checks in the output bound the optimisation's own effect. Three times the training steps, and three different initialisations, change the results the article quotes by less than the gaps it discusses.

## Limits

- **Simple ingredients.** The fits use one loss, one optimiser, a learned scale and a fixed curvature of -1. They are not the best embeddings either geometry can reach. Combinatorial constructions reach far lower hyperbolic distortion in two dimensions (Sala et al., 2018), and other losses suit ranking better.
- **Lower-bound delta.** The four-point delta is sampled, so its maximum is a lower bound.
- **Random cross-links.** The dial's cross-links are random. Real extra structure in a taxonomy (multiple inheritance, part-of links) is not random, and may hurt hyperbolic geometry less or more.
- **Precision.** Hyperbolic embeddings need more numerical precision as radii grow. These fits stay within radius 12 in float64; the precision cost of larger radii is not studied here.

## Licence

Code: Apache-2.0. WordNet 3.0: © 2006 Princeton University, WordNet licence.
