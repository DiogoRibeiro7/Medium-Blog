from ..bhattacharyya_metrics import KLDivergence

def test_kl_divergence():
    p = [0.4, 0.3, 0.2, 0.1]
    q = [0.3, 0.3, 0.2, 0.2]

    kl_calculator = KLDivergence(p, q)
    kl_divergence = kl_calculator.kl_divergence()

    # Add your assertion here based on expected value
    assert np.isclose(kl_divergence, expected_value, atol=1e-6)
