import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, LabelEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from typing import Tuple, Optional, Union, Literal

class DataPreprocessor:
    def __init__(
        self,
        feature_scaling: str = 'minmax', # 'minmax', 'standard', 'robust', or None
        target_scaling: bool = False,
        feature_range: Tuple[float, float] = (0, 1),
        imputation_strategy: Literal['mean', 'median', 'most_frequent', 'knn', None] = None,
        knn_neighbors: int = 5,
        missing_values: Union[str, float] = np.nan,
        encode_categoricals: bool = True
    ):

        """
        Initialize the preprocessor for regression tasks.
        
        Parameters:
        -----------
        feature_scaling : str
            Type of scaling to apply to features ('minmax', 'standard', 'robust', or None)
        target_scaling : bool
            Whether to scale the target variable
        feature_range : Tuple[float, float]
            Desired range for min-max scaling (if selected)
        imputation_strategy : str
            Strategy for handling missing values ('mean', 'median', 'most_frequent', 'knn', or None)
        knn_neighbors : int
            Number of neighbors for KNN imputation (if strategy='knn')
        missing_values : Union[str, float]
            Placeholder for missing values (default np.nan)
        """

        self.feature_scaling = feature_scaling
        self.target_scaling = target_scaling
        self.feature_range = feature_range
        self.encode_categoricals = encode_categoricals

        self.feature_scaler = None
        self.target_scaler = None
        self.label_encoders = {}

        if feature_scaling == 'minmax':
            self.feature_scaler = MinMaxScaler(feature_range=feature_range)
        elif feature_scaling == 'standard':
            self.feature_scaler = StandardScaler()
        elif feature_scaling == 'robust':
            self.feature_scaler = RobustScaler()

        if target_scaling:
            self.target_scaler = MinMaxScaler(feature_range=feature_range)


        self.imputation_strategy = imputation_strategy
        self.knn_neighbors = knn_neighbors
        self.missing_values = missing_values
        self.imputer = None
        
        if imputation_strategy == 'knn':
            self.imputer = KNNImputer(n_neighbors=knn_neighbors, missing_values=missing_values)
        elif imputation_strategy in ['mean', 'median', 'most_frequent']:
            self.imputer = SimpleImputer(strategy=imputation_strategy, missing_values=missing_values)
    def process(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None,
        feature_names: Optional[list] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:



        # Label encoding
        if isinstance(X, pd.DataFrame):
            if self.encode_categoricals:
                for col in X.select_dtypes(include='object').columns:
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col].astype(str))
                    self.label_encoders[col] = le


        X = np.array(X) if not isinstance(X, np.ndarray) else X

        if y is not None:
            y = np.array(y) if not isinstance(y, np.ndarray) else y
            y = y.reshape(-1, 1)
        

        # Handle missing values
        if self.imputer is not None:
            X = self.imputer.fit_transform(X)
            if y is not None and np.any(np.isnan(y)):
                y_imputer = SimpleImputer(strategy=self.imputation_strategy if self.imputation_strategy != 'knn' else 'mean')
                y = y_imputer.fit_transform(y.reshape(-1, 1)).flatten()

        # Handle feature names        
        if feature_names is None:
            if isinstance(X, pd.DataFrame):
                feature_names = np.array(X.columns)
            else:
                feature_names = np.array([f"feature_{i}" for i in range(X.shape[1])])

        # Remove constant columns (zero variance), which don't help regression
        variance = np.var(X, axis=0)
        non_constant_indices = np.where(variance > 1e-8)[0]
        X = X[:, non_constant_indices]
        feature_names = feature_names[non_constant_indices]

        # Scale features if required
        if self.feature_scaler is not None:
            X = self.feature_scaler.fit_transform(X)

        # Scale target if required
        if y is not None and self.target_scaler is not None:
            y = self.target_scaler.fit_transform(y).flatten()
        elif y is not None:
            y = y.flatten()

        return X, y, feature_names
    

    def inverse_transform_target(self, y: np.ndarray) -> np.ndarray:
        if self.target_scaler is None:
            return y
        return self.target_scaler.inverse_transform(y.reshape(-1, 1)).flatten()










    