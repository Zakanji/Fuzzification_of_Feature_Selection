import numpy as np
from typing import Callable, Dict
import operator
import inspect
from fuzzy.membership import *
import functools

class FuzzySelector:
    def __init__(self, membership_function: Callable[[np.ndarray, np.ndarray], np.ndarray], name: str = "Generic Fuzzy Selector"):
        self.membership_function = membership_function
        self.name = name

    def setStrategy(self, membership_function: Callable[[np.ndarray, np.ndarray], np.ndarray]):
        self.membership_function = membership_function

    def computeMembership(self, x: np.ndarray, abcd: np.ndarray) -> np.ndarray:
        """Compute the membership for the given feature values."""
        return self.membership_function(x, abcd)

    @staticmethod
    def availableFunctions() -> Dict[str, Callable[[np.ndarray, np.ndarray], np.ndarray]]:
        """Dynamically fetch and return all available membership functions."""
        functions = inspect.getmembers(fuzzy.membership, inspect.isfunction)
        membership_functions = {
            name: func
            for name, func in functions
            if name.endswith("_membership")
        }
        return membership_functions

    def __str__(self):
        """String representation for user-friendly output."""
        return f"{self.name} using {self.membership_function.__name__} membership function"

    def __repr__(self):
        """Detailed string representation for debugging."""
        return f"FuzzySelector(membership_function={self.membership_function.__name__}, name={self.name})"

    def __call__(self, x: np.ndarray, abcd: np.ndarray) -> np.ndarray:
        """Makes the instance callable like a function."""
        return self.computeMembership(x, abcd)

    # Fuzzy "and" operator (min of two memberships)
    def __and__(self, other: "FuzzySelector") -> "FuzzySelector":
        def and_func(x: np.ndarray, abcd: np.ndarray) -> np.ndarray:
            return np.minimum(self.computeMembership(x, abcd), other.computeMembership(x, abcd))
        return FuzzySelector(and_func, name=f"{self.name} AND {other.name}")

    # Fuzzy "or" operator (max of two memberships)
    def __or__(self, other: "FuzzySelector") -> "FuzzySelector":
        def or_func(x: np.ndarray, abcd: np.ndarray) -> np.ndarray:
            return np.maximum(self.computeMembership(x, abcd), other.computeMembership(x, abcd))
        return FuzzySelector(or_func, name=f"{self.name} OR {other.name}")

    # Fuzzy "not" operator (1 - membership)
    def __invert__(self) -> "FuzzySelector":
        def not_func(x: np.ndarray, abcd: np.ndarray) -> np.ndarray:
            return 1 - self.computeMembership(x, abcd)
        return FuzzySelector(not_func, name=f"NOT {self.name}")

    def apply_to_data(self, data: np.ndarray, abcd: np.ndarray) -> np.ndarray:
        """Apply the fuzzy selector to each feature in a dataset."""
        return np.apply_along_axis(self.computeMembership, 0, data, abcd)

    def describe(self) -> str:
        """Return a descriptive string about the selector's function."""
        return f"FuzzySelector: {self.name}, Membership Function: {self.membership_function.__name__}"

    # Aggregation methods (can be extended for more complex fuzzy operations)
    def aggregate(self, data: np.ndarray, abcd: np.ndarray, agg_func: Callable[[np.ndarray], float] = np.mean) -> np.ndarray:
        """Aggregate the fuzzy membership values for a dataset feature-wise."""
        memberships = self.apply_to_data(data, abcd)
        return np.apply_along_axis(agg_func, 0, memberships)

    def compare_to(self, other: "FuzzySelector", data: np.ndarray, abcd: np.ndarray) -> np.ndarray:
        """Compare the membership values between two selectors for the same data."""
        self_memberships = self.apply_to_data(data, abcd)
        other_memberships = other.apply_to_data(data, abcd)
        return np.abs(self_memberships - other_memberships)

    def select(self, data: pd.DataFrame, abcd: np.ndarray, selection_threshold: float = 0.5) -> List[str]:
        """
        Select features based on the computed fuzzy membership values.
        
        Parameters
        ----------
        data : pd.DataFrame
            The dataset containing features to apply the fuzzy selection on.
        abcd : np.ndarray
            The parameters (a, b, c, d) for the selected membership function.
        selection_threshold : float, optional
            The threshold for selecting features based on their membership values. 
            Default is 0.5.
        
        Returns
        -------
        List[str]
            List of selected feature names.
        """
        selected_features = []
        
        for feature_name in data.columns:
            feature_data = data[feature_name].values
            
            # Compute membership values for the current feature
            membership_values = self.computeMembership(feature_data, abcd)
            
            # Select the feature if the maximum membership value meets the threshold
            if np.max(membership_values) >= selection_threshold:
                selected_features.append(feature_name)
        
        return selected_features
