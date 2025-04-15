import shap
import lime
import lime.lime_tabular
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from typing import Callable


class Explainer:
    def __init__(self, fuzzy_selector: "FuzzySelector"):
        """
        Initializes the Explainer class with a given FuzzySelector instance.
        
        Parameters:
        fuzzy_selector (FuzzySelector): The FuzzySelector instance to explain.
        """
        self.fuzzy_selector = fuzzy_selector

    def explain_with_shap(self, data: np.ndarray, model=None) -> shap.Explanation:
        """
        Explains the feature selection process using SHAP.
        
        Parameters:
        data (np.ndarray): The data to be explained.
        model: (Optional) A pre-trained model to explain (default: RandomForest).
        
        Returns:
        shap.Explanation: The SHAP explanation for the data.
        """
        # If no model is provided, use a default RandomForestClassifier
        if model is None:
            model = RandomForestClassifier()

        # Fit the model (this assumes the target is binary for simplicity)
        model.fit(data, np.ones(data.shape[0]))

        # Compute SHAP values using KernelExplainer
        explainer = shap.KernelExplainer(model.predict, data)
        shap_values = explainer.shap_values(data)

        return shap_values

    def explain_with_lime(self, data: np.ndarray, model=None) -> lime.lime_tabular.LimeTabularExplainer:
        """
        Explains the feature selection process using LIME.
        
        Parameters:
        data (np.ndarray): The data to be explained.
        model: (Optional) A pre-trained model to explain (default: RandomForest).
        
        Returns:
        lime.lime_tabular.LimeTabularExplainer: The LIME explanation for the data.
        """
        # If no model is provided, use a default RandomForestClassifier
        if model is None:
            model = RandomForestClassifier()

        # Fit the model (this assumes the target is binary for simplicity)
        model.fit(data, np.ones(data.shape[0]))

        # Create a LimeTabularExplainer for the data
        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=data,
            mode="classification",
            training_labels=np.ones(data.shape[0]),
            class_names=["selected", "not selected"]
        )

        # Select an instance from the data to explain
        instance = data[0]  # Just take the first sample for simplicity
        explanation = explainer.explain_instance(instance, model.predict)

        return explanation
