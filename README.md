# Machine Learning Algorithms Implementation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)

A dual-implementation machine learning library: classic algorithms from scratch (NumPy/Python) and with scikit-learn, for learning and benchmarking.

## Table of Contents

- [Project Overview](#project-overview)
- [Directory Structure](#directory-structure)
- [Installation](#installation)
- [Dataset Acquisition](#dataset-acquisition)
- [Usage](#usage)
- [API Reference](#api-reference)
- [Development & Testing](#development--testing)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Project Overview

This repository contains implementations of various machine learning algorithms, each with two versions:

- **From Scratch:**
  - Linear Regression
  - Logistic Regression
  - Support Vector Machine (SVM)
  - K-Means Clustering
- **Scikit-learn Implementations:**
  - Parallel implementations for each algorithm using scikit-learn for comparison and validation.

## Directory Structure

```text
ML_algos/
├── src/
│   ├── scratch/
│   │   ├── models/
│   │   │   ├── base_model.py
│   │   │   ├── linear_regression.py
│   │   │   ├── logistic_regression.py
│   │   │   ├── svm.py
│   │   │   ├── K_Means.py
│   │   └── utils/
│   │       ├── data_utils.py
│   │       ├── math_utils.py
│   │       ├── metrics.py
│   │       ├── smo_utils.py
│   │       ├── training_utils.py
│   │       └── viz_utils.py
│   └── sklearn_impl/
│       ├── linear_regression_sk.py
│       ├── logistic_regression_sk.py
│       ├── svm_sk.py
│       └── k_means_sk.py
├── data/
│   ├── raw/
│   ├── processed/
│   └── examples/
├── notebooks/
│   ├── data_preprocessing.ipynb
│   ├── experiments/
│   │   ├── linear_regression/
│   │   ├── logistic_regression/
│   │   ├── SVM/
│   │   └── K_means/
│   └── comparisons/
├── tests/
│   ├── test_models/
│   └── test_utils/
├── requirements.txt
├── environment.yml
├── README.md
└── .gitignore
```

## Installation

1. **Clone the repository:**

   ```powershell
   git clone https://github.com/moaz-loaie/ML_algos.git
   cd ML_algos
   ```

2. **Create and activate a virtual environment (recommended):**

   - Using conda:

     ```powershell
     conda env create -p .env -f environment.yml
     conda activate ./.env
     ```

   - Using venv:

     ```powershell
     python -m venv .env
     .env\Scripts\Activate.ps1
     ```

3. **Install dependencies:**

   ```powershell
   pip install -r requirements.txt
   ```

## Dataset Acquisition

Datasets are from Kaggle. Download and extract them into `data/raw/` as described below.

- [Student Performance - Multiple Linear Regression](https://www.kaggle.com/datasets/nikhil7280/student-performance-multiple-linear-regression)
- [Breast Cancer Dataset](https://www.kaggle.com/datasets/yasserh/breast-cancer-dataset)
- [Clustering Penguins Species](https://www.kaggle.com/datasets/youssefaboelwafa/clustering-penguins-species)

### Download with Kaggle API

1. **Install the Kaggle API if not already installed:**

   - Using conda:

     ```bash
     conda install -c conda-forge kaggle
     ```

   - Using pip:

     ```bash
     pip install kaggle
     ```

2. **Authenticate:**
   - Place your `kaggle.json` API token in `~/.kaggle/` (Linux/macOS) or `%USERPROFILE%\.kaggle\` (Windows). See [Kaggle API documentation](https://www.kaggle.com/docs/api).

3. **Download the datasets:**

   **Windows (PowerShell) or Linux/macOS (Bash):**

   ```sh
   kaggle datasets download -d nikhil7280/student-performance-multiple-linear-regression -p data/raw/Regression_Dataset
   kaggle datasets download -d yasserh/breast-cancer-dataset -p data/raw/Classification_Dataset
   kaggle datasets download -d youssefaboelwafa/clustering-penguins-species -p data/raw/K_Means_Dataset
   ```

### Manual Download (Alternative)

1. Visit the dataset URLs above and download the files manually.
2. Place the downloaded files in the appropriate subfolders under `data/raw/`:
   - `data/raw/Regression_Dataset/`
   - `data/raw/Classification_Dataset/`
   - `data/raw/K_Means_Dataset/`
3. Extract the files:

   **Windows (PowerShell):**

   ```powershell
   Expand-Archive data/raw/Regression_Dataset/*.zip -DestinationPath data/raw/Regression_Dataset
   Expand-Archive data/raw/Classification_Dataset/*.zip -DestinationPath data/raw/Classification_Dataset
   Expand-Archive data/raw/K_Means_Dataset/*.zip -DestinationPath data/raw/K_Means_Dataset
   ```

   **Linux/macOS (Bash):**

   ```bash
   unzip -o 'data/raw/Regression_Dataset/'*.zip -d data/raw/Regression_Dataset
   unzip -o 'data/raw/Classification_Dataset/'*.zip -d data/raw/Classification_Dataset
   unzip -o 'data/raw/K_Means_Dataset/'*.zip -d data/raw/K_Means_Dataset
   ```

### Preprocessing

- Preprocess the data using [`notebooks/data_preprocessing.ipynb`](notebooks/data_preprocessing.ipynb). This notebook reads from `data/raw/` and writes processed files to `data/processed/`.
- Both `data/raw/` and `data/processed/` are not tracked by Git. Manage these directories manually.

## Usage

1. **Preprocess the data:**

   - Run [`notebooks/data_preprocessing.ipynb`](notebooks/data_preprocessing.ipynb) to generate processed datasets.

2. **Experiment and train models:**

   - Use notebooks in `notebooks/experiments/` for each algorithm (from scratch and sklearn).
   - Compare results in `notebooks/comparisons/`.

## API Reference

### Models (from scratch)

- `LinearRegression` ([src/scratch/models/linear_regression.py](src/scratch/models/linear_regression.py))
- `LogisticRegression` ([src/scratch/models/logistic_regression.py](src/scratch/models/logistic_regression.py))
- `SVM` ([src/scratch/models/svm.py](src/scratch/models/svm.py))
- `KMeans` ([src/scratch/models/K_Means.py](src/scratch/models/K_Means.py))

### Scikit-learn Wrappers

- `LinearRegressionSK` ([src/sklearn_impl/linear_regression_sk.py](src/sklearn_impl/linear_regression_sk.py))
- `LogisticRegressionSK` ([src/sklearn_impl/logistic_regression_sk.py](src/sklearn_impl/logistic_regression_sk.py))
- `SVM_SK` ([src/sklearn_impl/svm_sk.py](src/sklearn_impl/svm_sk.py))
- `KMeansSK` ([src/sklearn_impl/k_means_sk.py](src/sklearn_impl/k_means_sk.py))

### Utilities

- Data: `data_utils.py`
- Math: `math_utils.py`
- Metrics: `metrics.py`
- Visualization: `viz_utils.py`
- SVM helpers: `smo_utils.py`
- Training helpers: `training_utils.py`

## Development & Testing

- **Python version:** 3.8+
- **Run tests:**

  ```powershell
  # (Assuming pytest is installed)
  pytest tests/
  ```

- **Supported OS:** Windows, Linux, macOS

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository and create a new branch.
2. Make your changes with clear commit messages.
3. Run tests and ensure all pass.
4. Open a pull request with a description of your changes.

For major changes, please open an issue first to discuss your ideas.

## License

[MIT License](LICENSE)

## Acknowledgements

- [scikit-learn](https://scikit-learn.org/)
- [NumPy](https://numpy.org/)
- [Matplotlib](https://matplotlib.org/)
- All contributors and the open-source community.
