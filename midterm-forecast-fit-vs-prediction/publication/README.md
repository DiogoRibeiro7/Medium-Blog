# Medium publication handoff — 1 November 2026

**Status: final pre-election manuscript; ready for the author's manual Medium publication.**
The statistical study is complete today. The 3 November election is **not** a
dependency for publishing an analysis that explicitly precedes it. The
post-election scoring utility remains an optional later retrospective.

## Article metadata

| Field | Publication value |
| --- | --- |
| Title | **The Forecast Is Not the Fit: Rebuilding an 89% Midterm Election Model** |
| Subtitle | *What an R² of 0.89 tells us about a 33-seat forecast—and what it cannot tell us.* |
| Short description | A reproducible audit of an election model: historical fit, genuine forward tests, uncertainty, and a forecast recorded before voting. |
| Suggested topics | Data Science, Statistics, Forecasting, Elections, Political Science |
| Recommended lead image | [Figure 6: original 2026 forecast intervals](../figures/fig6_2026_prediction_intervals.png) |
| Manuscript | [`medium_ready.md`](medium_ready.md) |

The final copy contains **over 16,000 words, 19 references, 13 figures and 15
equations rendered as PNG images**. A reader-facing note explains immediately
that **−32.32** is a conditional Republican House seat-change *forecast*, not
the actual 2026 outcome or an 89% probability of an election result. The article
and numerical sources maintain their original 31 October information cut-off;
the forecast was separately registered on 1 November.

## Publishing on Medium

1. Open [`medium_ready.md`](medium_ready.md) and create a story in the author's
   Medium account using the title and subtitle above. This repository does not
   claim to have published the story or accessed the author's account.
2. Transfer the text and verify all **13 numbered scientific figures**. The
   figure URLs in the publication copy point to a permanent pre-election
   GitHub revision; each figure also exists in [`../figures/`](../figures/).
3. Insert the **15 display equations** as the numbered files in
   [`equations/`](equations/), matching their positions in the article. Inline
   mathematical notation in this release is rendered as readable Unicode.
   Medium's native editor does not typeset LaTeX, so the source mathematics
   remains in [`../article.md`](../article.md) for exact auditing.
4. Check image captions, the two date labels, bibliography hyperlinks and the
   ending registration note in the Medium preview. Use Figure 6 as the preview
   image if the layout suits the publication.
5. Apply the suggested topics, choose any Partner Program settings, and publish
   when ready. Do **not** update the frozen input scenario with later polls or
   describe an eventual election outcome as though it were already observed.

## Release reproducibility

From the project directory:

```bash
python prepare_medium_publication.py --check
python -m unittest discover -s tests -v
```

To regenerate the publication Markdown **and** all equation PNGs on a computer
with `pdflatex` (LaTeX packages `standalone`, `amsmath`, `amssymb`) and
`pdftoppm` installed:

```bash
python prepare_medium_publication.py --build
```

The builder does not modify historical data, the model, the frozen article's
scientific results, the registered forecast, or its SHA-256 manifest. The
publication check runs in CI without requiring TeX.

This handoff is the project's **pre-election release**, not a plan for another
round of model additions. A later outcome audit is a separately scoped study.
