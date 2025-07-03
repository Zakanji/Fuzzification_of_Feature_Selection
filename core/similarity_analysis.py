import numpy as np
from typing import Tuple
from .config import Config  # Adjust import path to your setup

class FuzzySimilarity:
    def __init__(self, config: Config):
        self.config = config

    def compute_similarity_1d(self, v1: np.ndarray, v2: np.ndarray, method: str) -> float:
        """
        Compute 1D fuzzy similarity between two fuzzy vectors using the specified method.
        """
        if method == "max-min":
            return np.max(np.minimum(v1, v2))
        elif method == "sum-product":
            return np.sum(v1 * v2)
        elif method == "kleene-dienes":
            return np.min(np.maximum(1 - v1, v2))
        else:
            raise ValueError(f"Similarity: Unknown similarity method: {method}")

    def compute_pairwise_similarity(self, fuzzy_matrix: np.ndarray) -> np.ndarray:
        """
        Compute full pairwise feature similarity matrix.
        """
        n = fuzzy_matrix.shape[0]
        sim_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(i, n):
                sim = self.compute_similarity_1d(
                    fuzzy_matrix[i], fuzzy_matrix[j], self.config.similarity_method
                )
                sim_matrix[i, j] = sim_matrix[j, i] = sim

        return sim_matrix

    def aggregate_all(self, similarities: np.ndarray) -> float:
        """
        Aggregate feature-wise similarity scores into one scalar value using config.
        """
        agg = self.config.aggregation_method
        if agg == "min":
            return np.min(similarities)
        elif agg == "max":
            return np.max(similarities)
        elif agg == "i-or":
            if 0 in similarities and 1 in similarities:
                return 0.0
            prod_sim = np.prod(similarities)
            prod_compl = np.prod(1 - similarities)
            return prod_sim / (prod_sim + prod_compl + 1e-10)
        elif agg == "owa":
            weights = np.array(self.config.owa_weights)
            sorted_sim = np.sort(similarities)[::-1]
            return np.sum(weights * sorted_sim[: len(weights)])
        else:
            raise ValueError(f"Similarity: Unknown aggregation method: {agg}")

    def empirical_validation(self, fuzzy_matrix: np.ndarray, tolerance: float = 1/8) -> Tuple[bool, np.ndarray]:
        """
        Empirically validate whether the fuzzy similarity satisfies NC and 
        the difference between max-min and sum-product ≤ 1/8.
        Returns: (is_valid, difference_matrix)
        """
        n = fuzzy_matrix.shape[0]
        differences = np.zeros((n, n))
        is_valid = True

        for i in range(n):
            for j in range(i, n):
                d_max_min = self.compute_similarity_1d(fuzzy_matrix[i], fuzzy_matrix[j], "max-min")
                d_sum_prod = self.compute_similarity_1d(fuzzy_matrix[i], fuzzy_matrix[j], "sum-product")
                diff = abs(d_max_min - d_sum_prod)
                differences[i, j] = differences[j, i] = diff
                if diff > tolerance:
                    is_valid = False

        return is_valid, differences
