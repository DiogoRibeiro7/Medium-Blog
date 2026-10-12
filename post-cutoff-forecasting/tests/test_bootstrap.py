import numpy as np

from postcutoff.summarize import block_counts


def test_block_counts_draw_n_origins_each_time():
    c = block_counts(50, 7, 300, np.random.default_rng(0))
    assert c.shape == (300, 50)
    assert (c.sum(axis=1) == 50).all()
    # Each origin is drawn about once on average.
    assert abs(c.mean() - 1.0) < 0.05


def test_block_counts_keep_neighbours_together():
    c = block_counts(30, 30, 50, np.random.default_rng(1))
    # One block covering the whole circle: every origin exactly once.
    assert (c == 1).all()
