import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA


class Visualizer:
    def __init__(self, config):
        self.config = config
        from .fuzzifier import FuzzyMembership
        self.fuzzy = FuzzyMembership(config)

    def plot_results(self, relieff_scores, norm_scores, fuzzy_scores, selected_features, xb_score):

        plt.figure(figsize=(10, 6))
        plt.bar(range(len(relieff_scores)), relieff_scores, alpha=0.6, label="ReliefF Scores")
        plt.plot(range(len(norm_scores)), norm_scores, color='red', marker='o', label="Normalized Scores")
        plt.title("ReliefF Scores and Normalized Scores")
        plt.xlabel("Feature Index")
        plt.ylabel("Score")
        plt.legend()
        plt.tight_layout()
        plt.show()



    def plot_fuzzy_sets(self, fuzzy_matrix: np.ndarray, norm_scores: np.ndarray):
        """
        Plot the fuzzy membership functions using FuzzyMembership class,
        and overlay feature points from the fuzzy_matrix.
        """
        mf_type = self.config.mf_type
        # Mapping dictionary for mf_type normalization
        mf_type_map = {
            "trapmf": "trapezoidal",
            "trimf": "triangular",
            "gaussmf": "gaussian",
            "sigmf": "sigmoid",
            "bellmf": "bell_shaped",
            "zmf": "z_shaped",
            "smf": "s_shaped",
            "pimf": "pi_shaped",
            # Add more mappings as needed
        }
        params = self.config.mf_params.get(mf_type)
        if not params:
            raise ValueError(f"Visualizer: No parameters found for '{mf_type}'")

        # Get the method
        if not hasattr(self.fuzzy, f"{mf_type_map.get(mf_type)}_mf"):
            raise ValueError(f"Visualizer: FuzzyMembership does not implement '{mf_type}_mf'")

        fuzzy_method = getattr(self.fuzzy, f"{mf_type_map.get(mf_type)}_mf")

        x = np.linspace(0, 1, 500)
        y_low, y_med, y_high = [], [], []

        for xi in x:
            l, m, h = fuzzy_method(xi, params)
            y_low.append(l)
            y_med.append(m)
            y_high.append(h)

        plt.figure(figsize=(10, 6))
        plt.plot(x, y_low, label="Low", color="blue")
        plt.plot(x, y_med, label="Medium", color="orange")
        plt.plot(x, y_high, label="High", color="green")

        for i, s in enumerate(norm_scores):
            l, m, h = fuzzy_matrix[i]
            plt.scatter([s], [l], color="blue", marker='o')
            plt.scatter([s], [m], color="orange", marker='x')
            plt.scatter([s], [h], color="green", marker='^')

        plt.title(f"Fuzzy Membership Functions ({mf_type})")
        plt.xlabel("Normalized Score")
        plt.ylabel("Membership Degree")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


    def plot_similarity_matrix(self, sim_matrix: np.ndarray, feature_names=None):
        """
        Plot a heatmap of the fuzzy similarity matrix.
        """
        plt.figure(figsize=(10, 8))
        if feature_names is None:
            feature_names = [f"F{i}" for i in range(sim_matrix.shape[0])]
        sns.heatmap(sim_matrix, annot=True, fmt=".2f", cmap="viridis",
                    xticklabels=feature_names, yticklabels=feature_names,
                    square=True, linewidths=0.5)
        plt.title("Fuzzy Feature Similarity Matrix")
        plt.tight_layout()
        plt.show()

    def plot_clustering(self, fuzzy_matrix: np.ndarray, membership: np.ndarray, centers: np.ndarray):
        """
        Plot clustering results using PCA to reduce fuzzy vectors to 2D.
        Points are colored by their strongest cluster membership.
        """
        pca = PCA(n_components=2)
        reduced_data = pca.fit_transform(fuzzy_matrix)
        reduced_centers = pca.transform(centers)

        cluster_labels = np.argmax(membership, axis=1)

        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=cluster_labels, cmap='tab10', s=60)
        plt.scatter(reduced_centers[:, 0], reduced_centers[:, 1], color='black', marker='X', s=200, label="Centers")
        plt.title("Fuzzy C-Means Clustering of Features")
        plt.xlabel("PCA 1")
        plt.ylabel("PCA 2")
        plt.legend(*scatter.legend_elements(), title="Clusters")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
