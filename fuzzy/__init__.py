"""
Fuzzy - A  Python package for using fuzzy logic in feature selection.
"""

from .fuzzy_selector import FuzzySelector
from .membership import triangular_membership, trapezoidal_membership

__version__ = '0.1.0'
__author__ = 'NAJI & OUHADDA'

__all__ = [
    'FuzzySelector',
    'triangular_membership',
    'trapezoidal_membership',
]
