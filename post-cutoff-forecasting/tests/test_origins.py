import pandas as pd

from postcutoff.data import prepare
from postcutoff.origins import build_origins, contexts, targets, window_status
from postcutoff.synth import synthetic_series


def test_origins_alignment_and_exclusions(cfg):
    series = synthetic_series(cfg, end="2026-10-08")
    prep = prepare(series, cfg)
    o = build_origins(prep, cfg)
    assert set(o["window"]) == {w.name for w in cfg.windows}
    ex = o[o["status"] == "excluded"]
    # The blackout on 28 April 2025 removes the seven origins whose week contains it.
    assert list(ex["origin"].dt.strftime("%Y-%m-%d")) == [f"2025-04-{d}" for d in range(22, 29)]
    scored = o[o["status"] == "scored"].head(3)
    X = contexts(prep, scored["pos"], cfg.context_length)
    Y = targets(prep, scored["pos"], cfg.horizon)
    first = scored.iloc[0]
    assert prep.index[first["pos"]] == first["origin"]
    assert X[0, -1] == prep.inputs[first["pos"] - 1]
    assert Y[0, 0] == prep.observed[first["pos"]]
    assert X.shape == (3, cfg.context_length) and Y.shape == (3, cfg.horizon)


def test_short_snapshot_excludes_late_origins(cfg):
    series = synthetic_series(cfg, end="2026-03-01")
    o = build_origins(prepare(series, cfg), cfg)
    late = o[o["origin"] >= pd.Timestamp("2026-02-23T00:00Z")]
    assert (late["status"] == "excluded").all()


def test_window_status(cfg):
    st = window_status(cfg).set_index(["window", "model"])["status"]
    assert st[("2025-26", "chronos2")] == "after release"
    assert st[("2023-24", "tirex")] == "before release"
    assert st[("2024-25", "tirex")] == "spans release"
    assert st[("2024-25", "chronos2")] == "before release"


def test_window_status_uses_the_later_weights_commit(cfg):
    st = window_status(cfg, {"chronos2": "2025-11-05T00:00:00+00:00"}).set_index(["window", "model"])
    assert st.loc[("2025-26", "chronos2"), "status"] == "spans release"
    assert st.loc[("2025-26", "tirex"), "status"] == "after release"
