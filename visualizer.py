import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Optional, List
from mpl_toolkits.mplot3d import Axes3D

class Visualizer:
    def __init__(self, data_handler):
        """
        Initialize the Visualizer class.

        Parameters
        ----------
        data_handler : DataHandler
            The DataHandler object containing the dataset to visualize.
        """
        self.data_handler = data_handler
        self.data = self.data_handler.get_data()
        self.target = self.data_handler.get_target()
        self.feature_names = self.data_handler.get_feature_names()

    def plot_scatter(self, x_feature: str, y_feature: str, target_labels: Optional[List[str]] = None) -> None:
        """
        Plot a 2D scatter plot of two features in the dataset.

        Parameters
        ----------
        x_feature : str
            The feature to plot on the x-axis.
        y_feature : str
            The feature to plot on the y-axis.
        target_labels : Optional[List[str]]
            If provided, color the points based on the target labels.
        """
        plt.figure(figsize=(8, 6))
        plt.scatter(self.data[x_feature], self.data[y_feature], c=self.target if target_labels is None else target_labels, cmap='viridis')
        plt.xlabel(x_feature)
        plt.ylabel(y_feature)
        plt.title(f'Scatter plot of {x_feature} vs {y_feature}')
        plt.colorbar(label='Target')
        plt.show()

    def plot_3d_scatter(self, x_feature: str, y_feature: str, z_feature: str, target_labels: Optional[List[str]] = None) -> None:
        """
        Plot a 3D scatter plot of three features in the dataset.

        Parameters
        ----------
        x_feature : str
            The feature to plot on the x-axis.
        y_feature : str
            The feature to plot on the y-axis.
        z_feature : str
            The feature to plot on the z-axis.
        target_labels : Optional[List[str]]
            If provided, color the points based on the target labels.
        """
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.data[x_feature], self.data[y_feature], self.data[z_feature], c=self.target if target_labels is None else target_labels, cmap='viridis')
        ax.set_xlabel(x_feature)
        ax.set_ylabel(y_feature)
        ax.set_zlabel(z_feature)
        ax.set_title(f'3D Scatter plot of {x_feature}, {y_feature}, and {z_feature}')
        plt.show()

    def plot_feature_importance(self, feature_importances: np.ndarray) -> None:
        """
        Plot the importance of features based on fuzzy logic or model results.

        Parameters
        ----------
        feature_importances : np.ndarray
            An array of feature importances to plot.
        """
        plt.figure(figsize=(10, 6))
        sns.barplot(x=self.feature_names, y=feature_importances)
        plt.xlabel('Feature')
        plt.ylabel('Importance')
        plt.title('Feature Importance')
        plt.xticks(rotation=45)
        plt.show()

    def plot_heatmap(self, corr_matrix: Optional[pd.DataFrame] = None) -> None:
        """
        Plot a heatmap to visualize the correlation between features.

        Parameters
        ----------
        corr_matrix : Optional[pd.DataFrame]
            The correlation matrix to plot. If not provided, it will be computed from the data.
        """
        if corr_matrix is None:
            corr_matrix = self.data.corr()

        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Correlation Heatmap of Features')
        plt.show()

    def plot_histogram(self, feature: str, bins: int = 20) -> None:
        """
        Plot a histogram of a given feature in the dataset.

        Parameters
        ----------
        feature : str
            The feature to plot.
        bins : int, optional
            Number of bins for the histogram. Default is 20.
        """
        plt.figure(figsize=(8, 6))
        plt.hist(self.data[feature], bins=bins, color='skyblue', edgecolor='black')
        plt.xlabel(feature)
        plt.ylabel('Frequency')
        plt.title(f'Histogram of {feature}')
        plt.show()

    def plot_membership_functions(self, x: np.ndarray, y: np.ndarray, function_name: str) -> None:
        """
        Plot the membership function results over the range of `x` values.

        Parameters
        ----------
        x : np.ndarray
            The range of input values for the membership function.
        y : np.ndarray
            The resulting membership values.
        function_name : str
            The name of the membership function.
        """
        plt.figure(figsize=(8, 6))
        plt.plot(x, y, label=function_name)
        plt.xlabel('Input Value (x)')
        plt.ylabel('Membership Value')
        plt.title(f'Membership Function: {function_name}')
        plt.legend()
        plt.show()

    def describe_data(self) -> None:
        """
        Print a brief description of the dataset (mean, std, etc.).
        """
        print("Data Description:")
        print(self.data.describe())

    def show_target_distribution(self) -> None:
        """
        Show the distribution of the target variable in the dataset.
        """
        plt.figure(figsize=(8, 6))
        sns.countplot(x=self.target)
        plt.title('Target Variable Distribution')
        plt.xlabel('Target Value')
        plt.ylabel('Frequency')
        plt.show()
