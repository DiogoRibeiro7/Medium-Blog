from ..bhattacharyya_metrics import BhattacharyyaMetrics
import numpy as np

def test_battacharyya_distance():
    mean1 = [1, 2]
    cov1 = [[1, 0.5], [0.5, 2]]
    mean2 = [3, 4]
    cov2 = [[2, -0.5], [-0.5, 1]]

    bm = BhattacharyyaMetrics(mean1, cov1, mean2, cov2)
    b_distance = bm.bhattacharyya_distance()

    # Add your assertion here based on expected value
    assert np.isclose(b_distance, expected_value, atol=1e-6)

def test_battacharyya_coefficient():
    # Similar to above, add test cases for bhattacharyya_coefficient
    pass
