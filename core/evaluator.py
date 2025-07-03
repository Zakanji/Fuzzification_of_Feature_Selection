import numpy as np
from skrebate import ReliefF
from scipy.spatial.distance import cdist
from typing import Tuple

class FeatureEvaluator:
    def __init__(self, config):
        self.config = config
        
    def compute_relieff(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute ReliefF feature importance scores"""
        n_features = X.shape[1] if self.config.relief_n_features is None else min(
            self.config.relief_n_features, X.shape[1]
        )
        
        fs = ReliefF(
            n_neighbors=self.config.relief_n_neighbors,
            n_features_to_select=n_features
        )
        fs.fit(X, y)
        return fs.feature_importances_
    
    @staticmethod
    def xie_beni_index(
        X: np.ndarray, 
        centers: np.ndarray, 
        membership: np.ndarray, 
        m: float = 2.0
    ) -> float:
        """Compute Xie-Beni cluster validity index"""
        n = X.shape[0]
        dist = cdist(X, centers)
        
        # Numerator: sum of (membership^m * distance^2)
        numerator = np.sum((membership ** m) * (dist ** 2))
        
        # Denominator: n * (minimum squared distance between centers)
        center_dists = cdist(centers, centers)
        np.fill_diagonal(center_dists, np.inf)  # Ignore self-distances
        min_dist_sq = np.min(center_dists) ** 2
        
        denominator = n * min_dist_sq
        return numerator / denominator