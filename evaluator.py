import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from fuzzy.fuzzy_selector import FuzzySelector
from typing import Tuple

class Evaluator:
    def __init__(self, selector: FuzzySelector, classifier=None):
        """
        Initialize the Evaluator with a FuzzySelector and optional classifier.

        Parameters
        ----------
        selector : FuzzySelector
            The fuzzy selector to be used for feature selection.
        classifier : sklearn classifier, optional
            The classifier to be used for evaluation. If not provided, RandomForestClassifier is used.
        """
        self.selector = selector
        self.classifier = classifier or RandomForestClassifier()

    def evaluate(self, data: pd.DataFrame, target: pd.Series, abcd: np.ndarray, selection_threshold: float = 0.5) -> dict:
        """
        Evaluate the fuzzy selector on a given dataset by selecting features and training a classifier.
        
        Parameters
        ----------
        data : pd.DataFrame
            The dataset to evaluate.
        target : pd.Series
            The target labels (dependent variable).
        abcd : np.ndarray
            The parameters (a, b, c, d) for the selected membership function.
        selection_threshold : float, optional
            The threshold for feature selection. Default is 0.5.
        
        Returns
        -------
        dict
            A dictionary containing evaluation metrics (accuracy, precision, recall, F1 score).
        """
        # Step 1: Select the relevant features using the fuzzy selector
        selected_features = self.selector.select(data, abcd, selection_threshold)

        # If no features are selected, raise an error
        if not selected_features:
            raise ValueError("No features selected by the fuzzy selector.")

        # Step 2: Prepare the data with selected features
        selected_data = data[selected_features]
        
        # Step 3: Scale the features (important for many machine learning models)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(selected_data)
        
        # Step 4: Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, target, test_size=0.3, random_state=42)

        # Step 5: Train the classifier
        self.classifier.fit(X_train, y_train)
        
        # Step 6: Make predictions on the test set
        y_pred = self.classifier.predict(X_test)
        
        # Step 7: Calculate evaluation metrics
        metrics = self._get_metrics(y_test, y_pred)
        
        return metrics

    def _get_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """
        Calculate evaluation metrics.
        
        Parameters
        ----------
        y_true : np.ndarray
            The true target values.
        y_pred : np.ndarray
            The predicted target values.
        
        Returns
        -------
        dict
            A dictionary containing accuracy, precision, recall, and F1 score.
        """
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average='weighted', zero_division=1),
            "recall": recall_score(y_true, y_pred, average='weighted', zero_division=1),
            "f1_score": f1_score(y_true, y_pred, average='weighted', zero_division=1)
        }
        return metrics
