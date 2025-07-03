import numpy as np
from fcmeans import FCM
from typing import Tuple
from .config import Config
import skfuzzy as fuzz
from .evaluator import FeatureEvaluator  # Assuming you have an evaluator module for feature evaluation  

class FuzzyClusterer:
    def __init__(self, config: Config):  # Use Config for clarity
        """
        Initializes the fuzzy clusterer using Fuzzy C-Means.

        Parameters:
        -----------
        config : Config
            Configuration object with clustering parameters.
        """
        self.config = config



    def cmeans(self, X: np.ndarray, n_clusters: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply Fuzzy C-Means clustering to input data.

        Parameters:
        -----------
        X : np.ndarray
            Input feature matrix (n_samples x n_features)

        Returns:
        --------
        centers : np.ndarray
            Cluster centers (n_clusters x n_features)
        membership : np.ndarray
            Membership matrix (n_samples x n_clusters)
        """
        print("Starting Fuzzy C-Means clustering...")
        if n_clusters is None:
            n_clusters = self.config.n_clusters
        
        fcm = FCM(
            n_clusters= n_clusters,
            m=self.config.fuzzy_m,
            max_iter=self.config.clustering_max_iter,
            error=self.config.clustering_error,
            random_state=42
        )
        fcm.fit(X)

        centers = fcm.centers
        membership = fcm.u
        return centers, membership

    def optimal_n(self, X: np.ndarray) -> int:
        """
        Determine the optimal number of clusters using the Silhouette method.

        Parameters:
        -----------
        X : np.ndarray
            Input feature matrix (n_samples x n_features)

        Returns:
        --------
        optimal_n : int
            Optimal number of clusters
        """
        evaluator = FeatureEvaluator(X)
        best_n_clusters = 10
        best_index = 1000      
        for i in range(self.config.min_clusters, self.config.max_clusters + 1):
            centers, membership = self.cmeans(X, n_clusters=i)
            xie_beni_index = evaluator.xie_beni_index(X, centers, membership, self.config.fuzzy_m)
            print("FuzzyClusterer.optimal_n: iteration number:", i, xie_beni_index)
            if xie_beni_index <= best_index:
                print("FuzzyClusterer.optimal_n: Hi I have a xiebeni score lower than the best, my index is:'", i, "' My xiebeni score is:", xie_beni_index, "which is lower than the best index:", best_index)
                best_index = xie_beni_index
                best_n_clusters = i
        best_xb_index = best_index
        return best_n_clusters, best_xb_index