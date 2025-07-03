# Fuzzification of Feature Selection Techniques (Featuring ReliefF)

## Overview

This project implements a fuzzy logic approach to feature selection, specifically focusing on enhancing the ReliefF algorithm through fuzzification. The application provides a graphical interface for preprocessing data, performing fuzzy feature selection, and visualizing the results.

## Features

- **Data Preprocessing**:
  - Multiple scaling methods (MinMax, Standard, Robust)
  - Missing value imputation (Mean, Median, Most Frequent, KNN)
  - Target variable scaling option

- **Fuzzy Feature Selection**:
  - ReliefF algorithm implementation
  - Fuzzy membership functions (Trapezoidal, Triangular, Sigmoid, Gaussian, Bell-shaped)
  - Customizable selection thresholds

- **Clustering & Visualization**:
  - Fuzzy C-means clustering
  - Xie-Beni index for optimal cluster determination
  - Visualization of fuzzy sets, similarity matrices, and clustering results

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Zakanji/Fuzzification_of_Feature_Selection
   cd Fuzzification_of_Feature_Selection
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the application:
   ```bash
   python app.py
   ```

2. Load your dataset (CSV format) through the GUI.

3. Configure preprocessing options:
   - Feature scaling method
   - Target scaling
   - Missing value imputation strategy

4. Set fuzzy parameters:
   - Membership function type
   - Selection threshold
   - Number of clusters

5. Click "Preprocess" to prepare your data.

6. Click "Fuzzify" to perform fuzzy feature selection and clustering. Its that simple

## Results Interpretation

The application provides several visual outputs:

- **Fuzzy Sets Plot**: Shows how ReliefF scores are fuzzified
- **Similarity Matrix**: Displays pairwise feature similarities
- **Clustering Results**: Visualizes feature clusters
- **Selected Features**: Lists the most important features based on fuzzy rules

The console output includes:
- ReliefF scores
- Normalized scores
- Fuzzy membership values
- Selected feature indices
- Optimal cluster count
- Xie-Beni index value

## Dependencies

- Python 3.7+
- PyQt5
- pandas
- numpy
- scikit-learn
- scikit-fuzzy
- matplotlib

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your changes.

