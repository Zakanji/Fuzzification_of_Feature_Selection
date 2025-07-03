import numpy as np
from typing import Tuple
from .config import Config

class FuzzyMembership:
    def __init__(self, config: Config):
        self.config = config

    def normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        min_val = np.min(scores)
        max_val = np.max(scores)
        return (scores - min_val) / (max_val - min_val + 1e-10)

    def triangular_mf(self, score: float, params: dict) -> Tuple[float, float, float]:
        """Triangular membership function"""
        low_params = params['low']
        med_params = params['med']
        high_params = params['high']
        
        low = max(0, min(1, 1 - score * (1/low_params[2])))
        medium = max(0, med_params[2] - abs((med_params[2]/med_params[1]) * score - med_params[2]))
        high = max(0, min(1, (score - med_params[1]) * (high_params[1]/med_params[1])))
        return low, medium, high
    
    def trapezoidal_mf(self, score: float, params: dict) -> Tuple[float, float, float]:
        """Trapezoidal membership function"""
        low_params = params['low']
        med_params = params['med']
        high_params = params['high']
        
        low = max(0, min(1, (low_params[3] - score) * (1/(low_params[3]-low_params[2]))))
        medium = max(0, min(1, 
            (score - med_params[0]) * (1/(med_params[1]-med_params[0])), 
            (med_params[3] - score) * (1/(med_params[3]-med_params[2]))
        ))
        high = max(0, min(1, (score - high_params[0]) * (1/(high_params[1]-high_params[0]))))
        return low, medium, high

    def gaussian_mf(self, score: float, params: dict) -> Tuple[float, float, float]:
        return tuple(np.exp(-((score - p[1]) ** 2) / (2 * p[2] ** 2)) for p in [params['low'], params['med'], params['high']])

    def sigmoid_mf(self, score: float, params: dict) -> Tuple[float, float, float]:
        return tuple(1 / (1 + np.exp(p[0] * (score - p[1]))) for p in [params['low'], params['med'], params['high']])

    def bell_shaped_mf(self, score: float, params: dict) -> Tuple[float, float, float]:
        def bell(x, a, b, c):
            return 1 / (1 + abs((x - a) / b) ** (2 * c))
        return tuple(bell(score, *params[key]) for key in ['low', 'med', 'high'])

    def z_shaped_mf(self, score: float, params: dict) -> Tuple[float, float, float]:
        def z_shaped(x, a, b):
            if x <= a:
                return 1.0
            elif x >= b:
                return 0.0
            else:
                return 1 - (x - a) / (b - a)

        low = z_shaped(score, *params['low'])
        med = self.triangular_mf(score, {'low': params['med'], 'med': params['med'], 'high': params['med']})[1]
        high = 1 - z_shaped(score, *params['high'])
        return low, med, high

    def s_shaped_mf(self, score: float, params: dict) -> Tuple[float, float, float]:
        def s_shaped(x, a, b):
            if x <= a:
                return 0.0
            elif x >= b:
                return 1.0
            else:
                return (x - a) / (b - a)

        low = 1 - s_shaped(score, *params['low'])
        med = self.triangular_mf(score, {'low': params['med'], 'med': params['med'], 'high': params['med']})[1]
        high = s_shaped(score, *params['high'])
        return low, med, high

    def pi_shaped_mf(self, score: float, params: dict) -> Tuple[float, float, float]:
        def pi(x, a, b, c, d):
            if x <= a:
                return 0.0
            elif x <= b:
                return (x - a) / (b - a)
            elif x <= c:
                return 1.0
            elif x <= d:
                return 1 - (x - c) / (d - c)
            else:
                return 0.0

        low = self.z_shaped_mf(score, {'low': params['low'], 'med': params['med'], 'high': params['high']})[0]
        med = pi(score, *params['med'])
        high = self.s_shaped_mf(score, {'low': params['low'], 'med': params['med'], 'high': params['high']})[2]
        return low, med, high

    def fuzzify(self, scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        norm_scores = self.normalize_scores(scores)
        mf_type = self.config.mf_type
        if not mf_type:
            raise ValueError("Fuzzifier: Membership function type is not set in the configuration")
        params = self.config.mf_params.get(mf_type)
        if not params:
            raise ValueError(f"Fuzzifier: No parameters found for membership function type '{mf_type}'")

        mf_methods = {
            'trimf': self.triangular_mf,
            'trapmf': self.trapezoidal_mf,
            'gaussmf': self.gaussian_mf,
            'sigmf': self.sigmoid_mf,
            'bellmf': self.bell_shaped_mf,
            'zmf': self.z_shaped_mf,
            'smf': self.s_shaped_mf,
            'pimf': self.pi_shaped_mf,
        }

        if mf_type not in mf_methods:
            raise ValueError(f"Fuzzifier: Unknown membership function type: {mf_type}")

        fuzzy_func = mf_methods[mf_type]
        fuzzy_matrix = np.array([fuzzy_func(score, params) for score in norm_scores])
        return fuzzy_matrix, norm_scores
