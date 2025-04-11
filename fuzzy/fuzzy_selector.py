import numpy as np
from typing import Callable

class FuzzySelector:
    def __init__(self, membership_function: Callable[[np.ndarray, np.ndarray], np.ndarray]):
        self.membership_function = membership_function

    def setStrategy(self, membership_function: Callable[[np.ndarray, np.ndarray], np.ndarray]):
        self.membership_function = membership_function

    def computeMembership(self, x: np.ndarray, abcd: np.ndarray) -> np.ndarray:
        return self.membership_function(x, abcd)
