import numpy as np
from typing import Tuple
from .similarity_analysis import FuzzySimilarity
from .clusterer import FuzzyClusterer
from .fuzzifier import FuzzyMembership
from .config import Config
from .evaluator import FeatureEvaluator

class FeaturesFuzzyInterface:
    def __init__(self, config: Config):
        self.config = config
        self.similarity = FuzzySimilarity(config)
        self.clusterer = FuzzyClusterer(config)
        self.evaluator = FeatureEvaluator(config)
        self.membership = FuzzyMembership(config)

    def select_features(self, scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Full fuzzy interface system for feature selection.

        Returns:
        --------
        selected_indices: np.ndarray
            Indices of selected features
        fuzzy_matrix: np.ndarray
            Fuzzy values for all features (low, med, high)
        sim_matrix: np.ndarray
            Feature-to-feature similarity matrix
        """

        # Step 1: Normalize and fuzzify scores
        fuzzy_matrix, norm_scores = self.membership.fuzzify(scores)

        # Step 2: Similarity between features
        sim_matrix = self.similarity.compute_pairwise_similarity(fuzzy_matrix)

        # Step 3: Cluster fuzzy vectors of features
        centers, membership_matrix = self.clusterer.cmeans(fuzzy_matrix)

        # Step 5: Apply fuzzy rules for feature selection
        selected_indices = []
        for i, row in enumerate(fuzzy_matrix):
            high_value = row[2]  # Fuzzy membership for 'high' importance
            similarity = np.mean(sim_matrix[i])  # Average similarity with other features
            cluster_strength = np.max(membership_matrix[i])  # Strongest cluster membership

            # --- Fuzzy Rules ---

            # IF feature is highly important
            high_condition = high_value >= self.config.selection_threshold

            # AND IF it's not redundant (low similarity to others)
            similarity_condition = similarity < self.config.similarity_threshold

            # AND IF it's well clustered (high membership strength)
            cluster_condition = cluster_strength > (1.0 / self.config.n_clusters)


            # THEN select the feature
            if high_condition and similarity_condition and cluster_condition :
                selected_indices.append(i)

        return np.array(selected_indices), fuzzy_matrix, sim_matrix
