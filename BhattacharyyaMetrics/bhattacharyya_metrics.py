import numpy as np
from typing import List, Union


class BhattacharyyaMetrics:
    """
    A class for calculating Bhattacharyya coefficient and distance between two distributions.

    Attributes:
        mean1 (List[float]): Mean of the first distribution.
        cov1 (Union[List[List[float]], np.ndarray]): Covariance matrix of the first distribution.
        mean2 (List[float]): Mean of the second distribution.
        cov2 (Union[List[List[float]], np.ndarray]): Covariance matrix of the second distribution.
    """

    def __init__(self, mean1: List[float], cov1: Union[List[List[float]], np.ndarray],
                 mean2: List[float], cov2: Union[List[List[float]], np.ndarray]):
        """
        Initialize the BhattacharyyaMetrics instance.

        Args:
            mean1 (List[float]): Mean of the first distribution.
            cov1 (Union[List[List[float]], np.ndarray]): Covariance matrix of the first distribution.
            mean2 (List[float]): Mean of the second distribution.
            cov2 (Union[List[List[float]], np.ndarray]): Covariance matrix of the second distribution.
        """
        self.mean1 = np.array(mean1, dtype=float)
        self.cov1 = np.array(cov1, dtype=float)
        self.mean2 = np.array(mean2, dtype=float)
        self.cov2 = np.array(cov2, dtype=float)

        self._validate_inputs()

    def _validate_inputs(self):
        """Validate input dimensions and types."""
        if self.mean1.shape != self.mean2.shape:
            raise ValueError("Mean vectors must have the same shape.")

        if self.cov1.shape != self.cov2.shape:
            raise ValueError("Covariance matrices must have the same shape.")

    def bhattacharyya_coefficient(self) -> float:
        """
        Calculate the Bhattacharyya coefficient between two distributions.

        Returns:
            float: Bhattacharyya coefficient.
        """
        det1 = np.linalg.det(self.cov1)
        det2 = np.linalg.det(self.cov2)
        det_avg = np.sqrt(det1 * det2)

        cov_avg = 0.5 * (self.cov1 + self.cov2)
        coeff = 0.125 * np.trace(cov_avg @ np.linalg.inv(self.cov1) + cov_avg @
                                 np.linalg.inv(self.cov2) - 2 * np.eye(self.cov1.shape[0]))

        return coeff / np.sqrt(det_avg)

    def bhattacharyya_distance(self) -> float:
        """
        Calculate the Bhattacharyya distance between two distributions.

        Returns:
            float: Bhattacharyya distance.
        """
        dist = 0.125 * (self.mean2 - self.mean1).T @ np.linalg.inv(0.5 *
                                                                   (self.cov1 + self.cov2)) @ (self.mean2 - self.mean1)
        return dist


# Example usage
mean1 = [1.0, 2.0]
cov1 = [[1.0, 0.5], [0.5, 2.0]]

mean2 = [3.0, 4.0]
cov2 = [[2.0, -0.5], [-0.5, 1.0]]

bhattacharyya_metrics = BhattacharyyaMetrics(mean1, cov1, mean2, cov2)

bcoefficient = bhattacharyya_metrics.bhattacharyya_coefficient()
print("Bhattacharyya Coefficient:", bcoefficient)

bdistance = bhattacharyya_metrics.bhattacharyya_distance()
print("Bhattacharyya Distance:", bdistance)


class KLDivergence:
    """
    A class for calculating Kullback-Leibler (KL) Divergence between two distributions.

    Attributes:
        p (List[float]): Probability distribution p.
        q (List[float]): Probability distribution q.
    """

    def __init__(self, p: List[float], q: List[float]):
        """
        Initialize the KLDivergence instance.

        Args:
            p (List[float]): Probability distribution p.
            q (List[float]): Probability distribution q.
        """
        self.p = np.array(p, dtype=float)
        self.q = np.array(q, dtype=float)

        self._validate_inputs()

    def _validate_inputs(self):
        """Validate input dimensions and types."""
        if self.p.shape != self.q.shape:
            raise ValueError(
                "Probability distributions must have the same shape.")

    def kl_divergence(self) -> float:
        """
        Calculate the Kullback-Leibler (KL) Divergence between two distributions.

        Returns:
            float: KL Divergence.
        """
        kl_div = np.sum(np.where(self.p != 0, self.p *
                        np.log(self.p / self.q), 0))
        return kl_div


# Example usage
p = [0.4, 0.3, 0.2, 0.1]
q = [0.3, 0.3, 0.2, 0.2]

kl_divergence_calculator = KLDivergence(p, q)

kl_divergence = kl_divergence_calculator.kl_divergence()
print("Kullback-Leibler Divergence:", kl_divergence)
