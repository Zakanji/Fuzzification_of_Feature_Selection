import sys
import csv
import pandas as pd
import numpy as np
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import QMessageBox
from PyQt5.uic import loadUi
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QMainWindow, QApplication

from typing import Optional, Tuple, List
from sklearn.impute import SimpleImputer, KNNImputer
from enum import Enum

from core import Config, DataPreprocessor, FeatureEvaluator, FuzzyMembership, FeaturesFuzzyInterface, FuzzyClusterer, visualizer, FuzzySimilarity

class ScalingMethod(Enum):
    MINMAX = 'minmax'
    STANDARD = 'standard'
    ROBUST = 'robust'
    NONE = None

class ImputationMethod(Enum):
    MEAN = 'mean'
    MEDIAN = 'median'
    MOST_FREQUENT = 'most_frequent'
    KNN = 'knn'
    NONE = None

class MembershipFunction(Enum):
    TRAP = 'trapezoidal_mf'
    SIG = 'sigmoid_mf'
    TRI = 'triangular_mf'
    GAUSS = 'gaussian_mf'
    BELL = 'bell_shaped_mf'



class App(QtWidgets.QMainWindow):
    
    def __init__(self):
        super().__init__()
        loadUi("app.ui", self)

        self.summaryBox.hide()
        self.feature_scaling = ScalingMethod.MINMAX
        self.target_scaling = False
        self.feature_range = (0, 1)
        self.imputation_strategy = ImputationMethod.NONE
        self.knn_neighbors = 5
        self.missing_values = np.nan
        

        self._config = Config()

        self._setup_connections()

        
    def _setup_connections(self):

        # Actions Menu
        self.actionLoad.triggered.connect(self.load_csv)

        # Feature scaling
        self.comboFeatureScaling.currentTextChanged.connect(self._on_feature_scaling_changed)
        self.spinMinRange.valueChanged.connect(self._on_range_changed)
        self.spinMaxRange.valueChanged.connect(self._on_range_changed)
        
        # Target scaling
        self.checkTargetScaling.stateChanged.connect(self._on_target_scaling_changed)
        
        # Imputation
        self.comboImputationStrategy.currentTextChanged.connect(self._on_imputation_changed)
        self.spinKNNNeighbors.valueChanged.connect(self._on_knn_neighbors_changed)
        
        # Class Selection
        self.comboClassName.currentTextChanged.connect(self._on_class_changed)

        # Feature Selection
        self.spinReliefNNeighbors.valueChanged.connect(self._on_relief_n_neighbors_changed)
        self.spinReliefNFeatures.valueChanged.connect(self._on_relief_n_features_changed)

        # Fuzzy Membership
        self.comboMF.currentTextChanged.connect(self._on_mf_changed)
        ## MF Custom para
        #self.editMFCustonParameters.currentTextChanged.connect(self._on_mf_custom_parameters)
        self.spinSelectionThreshold.valueChanged.connect(self._on_spin_selection_threshold_changed)
        self.sliderSelectionThreshold.valueChanged.connect(self._on_slider_selection_threshold_changed)
        self.spinClustersNumber.valueChanged.connect(self._on_clusters_number_changed)

        # Action buttons
        self.btnLoadDataset.clicked.connect(self.load_csv)
        self.btnPreprocess.clicked.connect(self.apply_preprocessing)
        self.btnReset.clicked.connect(self.reset_defaults)
        #self.btnLoadMFCustonParameters.clicked.connect()
        #self.btnSaveConfig.clicked.connect()
        #self.btnLoadConfig.clicked.connect()
        self.btnFuzzify.clicked.connect(self.apply_fuzzification)

     # Signal handlers
    def _on_feature_scaling_changed(self, text: str):
        scaling_map = {
            "MinMax": ScalingMethod.MINMAX,
            "Standard": ScalingMethod.STANDARD,
            "Robust": ScalingMethod.ROBUST,
            "None": ScalingMethod.NONE
        }
        self.feature_scaling = scaling_map.get(text, ScalingMethod.MINMAX)
        
    def _on_target_scaling_changed(self, state: int):
        self.target_scaling = state == 2  # 2 is Qt.Checked
        
    def _on_range_changed(self):
        min_val = self.spinMinRange.value()
        max_val = self.spinMaxRange.value()
        self.feature_range = (float(min_val), float(max_val))
        
    def _on_imputation_changed(self, text: str):
        imputation_map = {
            "Mean": ImputationMethod.MEAN,
            "Median": ImputationMethod.MEDIAN,
            "Most Frequent": ImputationMethod.MOST_FREQUENT,
            "KNN": ImputationMethod.KNN,
            "None": ImputationMethod.NONE
        }
        self.imputation_strategy = imputation_map.get(text, ImputationMethod.NONE)
        self.spinKNNNeighbors.setEnabled(self.imputation_strategy == ImputationMethod.KNN)
        
    def _on_knn_neighbors_changed(self, value: int):
        self.knn_neighbors = value
    
    def _on_class_changed(self, text: str):
        self.class_name = text
    # Main processing function
    def apply_preprocessing(self):

        params = {
            'feature_scaling': self.feature_scaling.value,
            'target_scaling': self.target_scaling,
            'feature_range': self.feature_range,
            'imputation_strategy': self.imputation_strategy.value,
            'knn_neighbors': self.knn_neighbors,
            'missing_values': self.missing_values
        }
        
        self.X = self.df.drop(self.class_name, axis=1)
        self.y = self.df[self.class_name]
        self._process_data(params)
    
    def reset_defaults(self):
        """Reset all controls to default values"""
        self.comboFeatureScaling.setCurrentText("MinMax Scaling")
        self.checkTargetScaling.setChecked(False)
        self.spinMinRange.setValue(0)
        self.spinMaxRange.setValue(1)
        self.comboImputationStrategy.setCurrentText("None")
        self.spinKNNNeighbors.setValue(5)


    def _process_data(self, params: dict):

        try:
            preprocessor = DataPreprocessor(**params)
            self.X, self.y, self.feature_names = preprocessor.process(self.X, self.y, np.array(self.df.columns.drop(self.class_name)))
            self.processed_view(self.X, self.y, self.feature_names)


        except Exception as e:
            QMessageBox.critical(self, "Preprocessing Error", f"Processing failed: {str(e)}")
            return None


        
    def _on_relief_n_neighbors_changed(self):
        self._config.relief_n_neighbors = self.spinReliefNNeighbors.value()

    def _on_relief_n_features_changed(self):
        self._config_relief_n_features = self.spinReliefNFeatures.value()
    
    def _on_mf_changed(self, text: str):
        membershipfunction_map = {
            "Trapezoidal": "trapmf",
            "Triangular": "trimf",
            "Sigmoid": "sigmf",
            "Gaussian": "gaussmf",
            "Bell-shaped": "bellmf",
            "Z-shaped": "zmf",
            "S-shaped": "smf",
            "Pi-shaped": "pimf",
        }
        self._config.mf_type = membershipfunction_map.get(text, "trapmf")

    def _on_mf_custom_parameters(self, text: str):
        pass

    def _on_spin_selection_threshold_changed(self, value:int):
        self.sliderSelectionThreshold.setValue(int(value*100))
        self._config.selection_threshold = value

    def _on_slider_selection_threshold_changed(self, value:int):
        self.spinSelectionThreshold.setValue(value/100)
        self._config.selection_threshold = value


    def _on_clusters_number_changed(self, value: int):
        self._config.n_clusters = value


    def apply_fuzzification(self):
        #X_selected = self.X
        try:
            # Step 1: Compute ReliefF scores
            evaluator = FeatureEvaluator(self._config)
            relieff_scores = evaluator.compute_relieff(self.X, self.y)

            # Step 2: Fuzzify ReliefF scores
            fuzzifier = FuzzyMembership(self._config)
            fuzzy_matrix, norm_scores = fuzzifier.fuzzify(relieff_scores)


            # Step 3: Fuzzy Rule-Based Selection
            selector = FeaturesFuzzyInterface(self._config)
            selected_indices, _, _ = selector.select_features(relieff_scores)
            selected_features = [self.feature_names[i] for i in selected_indices]
            X_selected = self.X[:, selected_indices]  # Select only the features that passed the fuzzy rules
            selected_fuzzy_matrix = fuzzy_matrix[selected_indices, :]  # Select corresponding fuzzy values

            # Step 4: Similarity matrix
            similarity = FuzzySimilarity(self._config)
            sim_matrix = similarity.compute_pairwise_similarity(fuzzy_matrix)

            # Step 5: Clustering
            clusterer = FuzzyClusterer(self._config)

            best_n_clusters, best_xb_index = clusterer.optimal_n(X_selected)

            centers, membership = clusterer.cmeans(X_selected, n_clusters=best_n_clusters)
            xb_index = evaluator.xie_beni_index(X_selected, centers, membership, self._config.fuzzy_m)

            if best_xb_index == xb_index:
                print("Xie-Beni Index is indeed optimal. Its value is:", xb_index)

            centers, membership = clusterer.cmeans(fuzzy_matrix)

            # Step 6: Visualization
            vis = visualizer.Visualizer(self._config)
            vis.plot_fuzzy_sets(fuzzy_matrix, norm_scores)
            vis.plot_similarity_matrix(sim_matrix, self.feature_names)
            vis.plot_clustering(fuzzy_matrix, membership, centers)
            vis.plot_results(
                relieff_scores, norm_scores, selected_fuzzy_matrix, 
                selected_features, xb_index
            )

            # Step 7:Display results
            result = f"""-\n\nReliefF Scores: \n {relieff_scores} \n\nNormalized Scores: \n {norm_scores} \n\nFuzzy Scores: \n{fuzzy_matrix} \n\n Best Features: \n{selected_features} \n\n Best n_clusters: \n{best_n_clusters} \n\n Xie Beni Score: \n {xb_index} \n\n--------------------------------------------------------------
            """.strip()
            print(result)
            self.log.insertPlainText(result)

            # Step 8: Display selected features
            feature_str = "\n".join(selected_features)
            QMessageBox.information(self, "Selected Features", f"Selected Features:\n{feature_str}")

        except Exception as e:
            QMessageBox.critical(self, "Fuzzification Error", f"Fuzzification failed:\nFailed at exception:\n{str(e)}")



    def load_csv(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open CSV", "data", "CSV files (*.csv);;",
        )
        if not path:
            return

        self.datasetLabel.setText("The File: " + QtCore.QFileInfo(path).fileName())

        try:
            with open(path, 'r', encoding='utf-8') as f:
                sample = f.read(1024)
                f.seek(0)
                dialect = csv.Sniffer().sniff(sample)
                delimiter = dialect.delimiter

            # Load CSV with robust error handling
            self.comboClassName.clear()
            self.df = pd.read_csv(path, delimiter=delimiter)
            self.summary(self.df)
            

        except Exception as e:
            QMessageBox.critical(self, "CSV Load Error", f"Failed to read CSV:\n{str(e)}")
            return

    def summary(self, df):

        # Show CSV in QTableWidget
        self.summaryBox.show()
        self.tableWidget.setRowCount(len(df))
        self.tableWidget.setColumnCount(len(df.columns))
        self.tableWidget.setHorizontalHeaderLabels(df.columns)

        for i, row in df.iterrows():
            for j, value in enumerate(row):
                self.tableWidget.setItem(i, j, QtWidgets.QTableWidgetItem(str(value)))

        # Summary of dataset
        num_rows, num_cols = df.shape
        missing_per_column = df.isnull().sum()
        total_missing = missing_per_column.sum()

        summary = f"""
>>> CSV Summary:
Rows: {num_rows}
Columns: {num_cols}
Total Missing Values: {total_missing}

>>> Missing by Column:
{missing_per_column.to_string()}

>>> Extra:
{df.describe()}
"""
        self.summaryText.setPlainText(summary.strip())
        self.comboClassName.addItems(df.columns)
    
    def processed_view(self, X, y, feature_names):
        self.xView.setRowCount(len(X))
        self.xView.setColumnCount(len(X[0]))
        self.xView.setHorizontalHeaderLabels(feature_names)

        for i in range(len(X)):
            for j in range(len(X[i])):
                self.xView.setItem(i, j, QtWidgets.QTableWidgetItem(str(X[i, j])))

        self.yView.setRowCount(len(y))
        self.yView.setColumnCount(1)
        self.yView.setHorizontalHeaderLabels([self.class_name])
        for i in range(len(y)):
            self.yView.setItem(i, 0, QtWidgets.QTableWidgetItem(str(y[i])))

        self.tabProcess.setCurrentIndex(1)



if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = App()
    window.setWindowTitle("Fuzzification of Feature Selection Method")
    window.show()
    sys.exit(app.exec_())
