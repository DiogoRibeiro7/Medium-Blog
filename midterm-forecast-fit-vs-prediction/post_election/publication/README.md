# Medium preparation — 2026 House forecast sequel

This is a **provisional** import package for the 4 November 2026 live-results
study, not a final scored post-election publication. It preserves the article's
full methodological caveats. The original manuscript is
[`../article.md`](../article.md). The generated derivative is
[`medium_ready.md`](medium_ready.md).

## What this package does

1. Converts eight `figures/*.png` Markdown references to full public GitHub
   raw image URLs pinned to commit `30e41b905990e321826bc0746d2ef8b09e239d7d`.
2. Verifies each image's SHA-256 against
   [`figure_manifest.json`](figure_manifest.json) before including its URL.
3. Converts relative repository-companion references to pinned GitHub links;
   external source URLs remain unchanged.
4. Checks that the manuscript is at least 8,000 words, is marked as a
   research draft, reports the 68 uncalled races, retains both pre-election
   dates, and contains no unsupported Markdown pipe tables or LaTeX.
5. Leaves the 31 October forecast, 1 November lock, source analysis and
   archived 4 November AP snapshot untouched.

## Reproduce

From the original study directory:

```bash
python post_election/prepare_medium_publication.py --write
python post_election/prepare_medium_publication.py --check
```

The script uses only the Python standard library. CI runs `--check`, not
`--write`; no publication or external account operation is performed.

## Medium import checklist

- Use `medium_ready.md` as the copy source, not the raw `article.md`.
- Check that all eight figure images import correctly and keep their captions.
- If Medium does not preserve a linked image, upload the corresponding PNG
  manually from `../figures/`; the image name is in the manifest.
- Keep the opening provisional-results disclaimer and distinguish AP leads
  from called races; do not present the 172-seat scenario as official.
- Do not call this a final outcome study until a House Historian / American
  Presidency Project aligned initial-election result has been recorded and
  scored separately through the pre-registered scorer.
- Review citations and link behavior in the Medium editor before publication.

The article uses plain language and Unicode notation; there are no raw
LaTeX display equations or Markdown data tables to convert. When the final
result is published, regenerate a *new dated edition* and update the pinned
figure reference and checksums if any figures change. Do not silently edit
the archived provisional snapshot or overwrite the original forecast lock.
