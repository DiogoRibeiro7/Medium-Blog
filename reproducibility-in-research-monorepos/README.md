# Reproducibility Is Local

**Status:** draft  
**Theme:** research engineering, reproducibility, repository design

A research repository can contain ten years of notebooks, scripts, figures, datasets, and article-supporting code without being one software project.

Treating it as one usually makes reproducibility worse.

The natural unit of reproducibility is often the **analysis or article**, not the repository root.

## The monorepo illusion

Suppose a repository contains:

- an old R analysis;
- several standalone Jupyter notebooks;
- a recent Python package;
- finance experiments;
- simulation code;
- images and datasets attached to published articles.

Adding one root `requirements.txt` does not make these artifacts reproducible together. It merely invents a dependency graph that never existed.

Historical work may have been written under different Python versions, different libraries, or entirely different languages.

The repository is a collection. The environment belongs to the project that produced the result.

## Reproducibility has a boundary

For a maintained analysis, a useful reproducibility contract answers four questions:

1. what data are required;
2. what environment executes the analysis;
3. what command or notebook regenerates the result;
4. what checks distinguish a successful reproduction from a merely completed process.

Those answers can be local:

```text
article-project/
  README.md
  pyproject.toml
  src/
  tests/
  data/
  figures/
```

Another project in the same repository may have a different structure and different runtime.

That is not inconsistency. It is honest scoping.

## Historical code needs a different promise

There is a large difference between

> this is the original code behind a published article

and

> this analysis is continuously reproducible today.

Repositories often blur those claims.

A useful classification is:

\[
\text{maintained and verified}
\]

\[
\text{reproducible historical}
\]

\[
\text{preserved historical}
\]

\[
\text{not yet audited}.
\]

That vocabulary is more informative than pretending every notebook carries the same maintenance guarantee.

## Do not modernize history cosmetically

Old folders often contain spaces, strange names, outdated APIs, or notebook conventions we would not choose today.

Renaming everything makes the tree prettier, but it can break article links and destroy provenance.

A better principle is:

\[
\boxed{
\text{classify first, document second, automate third, move last.}
}
\]

The repository catalogue can provide a clean modern navigation layer while historical paths remain intact.

## Continuous integration should respect project boundaries

A common failure mode is a workflow that attempts to install one environment and execute every notebook in the repository.

That creates meaningless failures.

Project-scoped CI is usually better. A maintained project should validate the files and environment it owns. Repository-wide CI should check only repository-wide invariants such as metadata schemas, broken catalogue paths, or accidental introduction of secrets.

In notation:

\[
\text{project CI} \neq \text{repository CI}.
\]

They answer different questions.

## A catalogue can be more valuable than a directory migration

For a large article repository, a machine-readable catalogue can record:

```text
path
status
topic
runtime
article URL
reproducibility tier
last audited
```

From this we can generate human-readable topic views without physically reorganizing everything.

That is especially useful when public URLs are already part of the historical record.

## Reproducibility is a claim

The word "reproducible" should be treated like any other technical claim. It needs evidence.

A lockfile is useful, but not sufficient by itself. A notebook that runs is useful, but not sufficient if the result silently depends on private data or a stale external API.

The strongest maintained projects make the contract explicit and test at least some of it.

## The broader lesson

A research monorepo should not aim to look as though every artifact was built yesterday.

Its job is to preserve history while making current guarantees legible.

That leads to a simpler rule:

\[
\boxed{
\text{reproducibility belongs to the smallest meaningful scientific unit.}
}
\]

Once that boundary is explicit, old work can remain old, new work can be rigorous, and the repository can be both an archive and a trustworthy research portfolio.
