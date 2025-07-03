from dataclasses import dataclass, field
from typing import Optional, Dict, List


@dataclass
class Config:
    # ReliefF configuration
    relief_n_neighbors: int = 10
    relief_n_features: Optional[int] = None
    
    # Fuzzy membership configuration
    mf_type: str = 'trapmf'  # Options: 'trapmf', 'trimf', 'sigmf', 'gaussmf', etc.
    mf_params: Dict[str, dict] = field(default_factory=dict)
    
    # Feature selection
    selection_threshold: float = 0.6
    
    # Clustering configuration
    n_clusters: int = 3
    min_clusters: int = 2
    max_clusters: int = 10
    fuzzy_m: float = 2.0
    clustering_max_iter: int = 1000
    clustering_error: float = 0.005


    # Similarity analysis configuration
    similarity_method: str = 'max-min'  # 'max-min', 'sum-product', 'kleene-dienes'
    aggregation_method: str = 'min'  # 'min', 'max', 'i-or', 'owa'
    owa_weights: Optional[List[float]] = None
    similarity_threshold: float = 0.7  # For empirical validation

    
    def __post_init__(self):
        default_params = {
            'trapmf': {
                'low': [0.0, 0.0, 0.2, 0.4],
                'med': [0.2, 0.4, 0.6, 0.8],
                'high': [0.6, 0.8, 1.0, 1.0]
            },
            'trimf': {
                'low': [0.0, 0.0, 0.5],
                'med': [0.0, 0.5, 1.0],
                'high': [0.5, 1.0, 1.0]
            },
            'gaussmf': {
                'low': [0.0, 0.2, 0.1],
                'med': [0.5, 0.5, 0.1],
                'high': [0.8, 1.0, 0.1]
            },
            'sigmf': {
                'low': [-10, 0.3],
                'med': [0, 0.5],
                'high': [10, 0.7]
            },
            'bellmf': {
                'low': [0.2, 0.1, 2],
                'med': [0.5, 0.1, 2],
                'high': [0.8, 0.1, 2]
            },
            'zmf': {
                'low': [0.0, 0.4],
                'med': [0.3, 0.5, 0.7],
                'high': [0.6, 1.0]
            },
            'smf': {
                'low': [0.0, 0.4],
                'med': [0.3, 0.5, 0.7],
                'high': [0.6, 1.0]
            },
            'pimf': {
                'low': [0.0, 0.4],
                'med': [0.2, 0.4, 0.6, 0.8],
                'high': [0.6, 1.0]
            }
        }
        
        # Set default OWA weights if not provided
        if self.owa_weights is None and self.aggregation_method == 'owa':
            self.owa_weights = [0.5, 0.3, 0.2]  # Default weights
            
        for key, value in default_params.items():
            if key not in self.mf_params:
                self.mf_params[key] = value
