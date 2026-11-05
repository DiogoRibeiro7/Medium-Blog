# Medium preparation utility — 2026 House forecast sequel

The **article source and Medium-ready derivative are private editorial files**
in [`research-articles-medium`](https://github.com/DiogoRibeiro7/research-articles-medium/tree/main/midterm-forecast-fit-vs-prediction/post_election)
(owner access required). They are deliberately not stored in this public
research repository. The historical forecast, reproducibility code, dated
snapshots, eight figures and research tests remain here.

This folder retains [`figure_manifest.json`](figure_manifest.json), which
pins the exact SHA-256 digest of each of the eight public figures and the
immutable figure commit `b83108c28d11a31e9dba6c34304730579df0cfd3`.
The independent research CI verifies these bytes without reading any private
files or requiring a private-repository token.

## Rebuild the Medium derivative locally

Clone both repositories as siblings. From the **Medium-Blog repository root**:

```bash
python midterm-forecast-fit-vs-prediction/post_election/prepare_medium_publication.py \
  --write \
  --source ../research-articles-medium/midterm-forecast-fit-vs-prediction/post_election/article.md \
  --output ../research-articles-medium/midterm-forecast-fit-vs-prediction/post_election/publication/medium_ready.md

python midterm-forecast-fit-vs-prediction/post_election/prepare_medium_publication.py \
  --check \
  --source ../research-articles-medium/midterm-forecast-fit-vs-prediction/post_election/article.md \
  --output ../research-articles-medium/midterm-forecast-fit-vs-prediction/post_election/publication/medium_ready.md
```

The converter (standard library only) checks the provisional-result
disclaimers, source dates, eight figure placements and Medium-compatible
format. It embeds immutable public image links after verifying each figure's
checksum. It never modifies model inputs or computes an official election
score. Its tests use a clearly synthetic manuscript fixture, so public CI
does not need access to a private editorial document.

The 4 and 5 November AP snapshots are **live, provisional observations**.
Neither snapshot is a certified initial-election House division. The final
score must follow [`../FINALIZATION_PROTOCOL.md`](../FINALIZATION_PROTOCOL.md).
