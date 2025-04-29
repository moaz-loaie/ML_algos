# Machine Learning Algorithms Implementation

This project provides dual implementations of common machine learning algorithms: one from scratch using pure Python/NumPy and another using scikit-learn. The goal is to understand the underlying mechanics of machine learning algorithms while comparing them with industry-standard implementations.

## Table of Contents

- [Project Overview](#project-overview)
- [Directory Structure](#directory-structure)
- [Installation](#installation)
- [Dataset Acquisition](#dataset-acquisition)
- [Usage](#usage)
- [Project Components](#project-components)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Project Overview

This repository contains implementations of various machine learning algorithms:

### From Scratch Implementations

- Linear Regression
- Logistic Regression
- Decision Trees (coming soon)
- K-Means Clustering (coming soon)
- Neural Networks (coming soon)

### Scikit-learn Implementations

Parallel implementations using scikit-learn for performance comparison and validation.

## Directory Structure

### Here is a proposal for working tree structure (not all files or directories shall be present in the final project)

```text
ml_algos/
├── src/
│   ├── scratch/
|   |   ├── train_model.py
│   │   ├── models/
│   │   │   ├── **__init__**.py
│   │   │   ├── base_model.py # Abstract base class for all models
│   │   │   ├── linear_regression.py
│   │   │   ├── logistic_regression.py
│   │   │   ├── svm.py
│   │   │   ├── K_Means.py
│   │   │   ├── tree_models.py # Decision trees (coming soon)
│   │   │   └── neural_network.py # Neural network implementation (coming soon)
│   │   │   └── ... (other models)
│   │   └── utils/
│   │       ├── **__init__**.py
│   │       ├── data_utils.py # Data processing utilities
│   │       ├── math_utils.py # Mathematical operations
│   │       ├── metrics.py # Evaluation metrics
│   │       ├── smo_utils.py # Helper functions specific to the SMO algorithm (e.g., compute error cache, clipping functions)
│   │       ├── training_utils.py # Learning rate schedules, early stopping helpers, logging functions, etc.
│   │       └── viz_utils.py # Visualization tools
│   └── sklearn_impl/ # Parallel sklearn implementations
├── data/
│   ├── raw/ # Original datasets **(not tracked)**
│   ├── processed/ # Preprocessed datasets **(not tracked)**
│   └── examples/ # Sample input/output for testing
├── notebooks/
│   ├── data_preprocessing.ipynb # preprocessing notebook (run it to get the preprocessed data)
│   ├── experiments/ # Model experimentation
│   │   ├── linear_regression/
│   │   │   ├── sgd_from_scratch.ipynb # SGD for Linear Regression from scratch
│   │   │   ├── batch_gradient_descent_from_scratch.ipynb # Batch Gradient Descent for Linear Regression from scratch
│   │   │   ├── normal_equation_from_scratch.ipynb # Normal Equation for Linear Regression from scratch
│   │   │   ├── sgd_sklearn.ipynb # SGD for Linear Regression with scikit-learn
│   │   │   ├── batch_gradient_descent_sklearn.ipynb # Batch Gradient Descent for Linear Regression with scikit-learn (if applicable)
│   │   │   └── normal_equation_sklearn.ipynb # Normal Equation for Linear Regression with scikit-learn (if applicable)
│       └── logistic_regression/
│   │       ├── sgd_from_scratch.ipynb # SGD for Logistic Regression from scratch
│   │       ├── batch_gradient_descent_from_scratch.ipynb # Batch Gradient Descent for Logistic Regression from scratch
│   │       ├── sgd_sklearn.ipynb # SGD for Logistic Regression with scikit-learn
│   │       └── batch_gradient_descent_sklearn.ipynb # Batch Gradient Descent for Logistic Regression with scikit-learn (if applicable)
│   └── comparisons/ # Comparing implementations
│       ├── linear_regression_comparisons/
│       │   ├── sgd_comparisons.ipynb # Compare SGD for Linear Regression from scratch vs scikit-learn
│       │   ├── batch_gradient_descent_comparisons.ipynb # Compare Batch Gradient Descent for Linear Regression from scratch vs scikit-learn
│       │   └── normal_equation_comparisons.ipynb # Compare Normal Equation for Linear Regression from scratch vs scikit-learn
│       └── logistic_regression_comparisons/
│           ├── sgd_comparisons.ipynb # Compare SGD for Logistic Regression from scratch vs scikit-learn
│           └── batch_gradient_descent_comparisons.ipynb # Compare Batch Gradient Descent for Logistic Regression from scratch vs scikit-learn
├── tests/
│   ├── test_models/
│   └── test_utils/
├── requirements.txt
├── environment.yml
├── README.mdر
└── .gitignore
```

This structure emphasizes separation of concerns, ensuring that data, code, tests, and documentation are organized neatly for ease of development and scalability.

## Installation

Follow these steps to set up the project on your local machine:

1. **Clone the repository:**

   ```bash
   git clone https://github.com/moaz-loaie/ML_algos.git
   cd ML_algos
   ```

2. **Create and activate a virtual environment (optional but recommended):**

   - using conda:

     ```bash
     conda env create -p .env -f environment.yml  # in the same project working directory
     conda activate ./.env
     ```

   - using venv:

     ```bash
     python -m venv .env  # in the same project working directory
     source .env/bin/activate  # On Windows: .env\Scripts\activate
     ```

3. **Install the required dependencies(using venv):**

   ```bash
   pip install -r requirements.txt
   ```

## Dataset Acquisition

### Find the Datasets on Kaggle

- **Visit Kaggle Datasets and copy the URLs for the datasets:**
  - [Student Performance - Multiple Linear Regression](https://www.kaggle.com/datasets/nikhil7280/student-performance-multiple-linear-regression)
  - [Breast Cancer Dataset](https://www.kaggle.com/datasets/yasserh/breast-cancer-dataset)
  - [clustering penguins species](https://www.kaggle.com/datasets/youssefaboelwafa/clustering-penguins-species)

### Download the Datasets Using the Kaggle API

1. **Install the Kaggle API if not already installed:**

   - using conda:

     ```bash
     conda install -c conda-forge kaggle
     ```

   - If you prefer Pip, install the packages:

     ```bash
     pip install kaggle
     ```

2. **Authenticate by placing your `kaggle.json` API token in `~/.kaggle/` (see [Kaggle API documentation](https://www.kaggle.com/docs/api)).**
3. **Download the datasets to `data/raw/`:**

   ```bash
   kaggle datasets download -d nikhil7280/student-performance-multiple-linear-regression -p data/raw/Regression_Dataset
   kaggle datasets download -d yasserh/breast-cancer-dataset -p data/raw/Classification_Dataset
   kaggle datasets download -d youssefaboelwafa/clustering-penguins-species -p data/raw/K_Means_Dataset
   ```

### Manually Download (Alternative)

1. **Visit the dataset URLs:**
   - [Student Performance - Multiple Linear Regression](https://www.kaggle.com/datasets/nikhil7280/student-performance-multiple-linear-regression)
   - [Breast Cancer Dataset](https://www.kaggle.com/datasets/yasserh/breast-cancer-dataset)
   - [clustering penguins species](https://www.kaggle.com/datasets/youssefaboelwafa/clustering-penguins-species)
2. **Download the files and place them in `data/raw/`.**
3. **Extract the Datasets**

   - If the downloaded files are zip archives, extract them:

     ```bash
     unzip data/raw/student-performance-multiple-linear-regression.zip -d data/raw/Regression_Dataset
     unzip data/raw/breast-cancer-dataset.zip -d data/raw/Classification_Dataset
     unzip data/raw/clustering-penguins-species.zip -d data/raw/K_Means_Dataset
     ```

### Modify the Working Tree

Since `data/raw/` and `data/processed/` are not tracked by Git, manually manage these directories.

- Preprocess the data (e.g., using a script or notebook) and save the processed files to `data/processed/`. Example:

  ```python
  import pandas as pd

  # Student Performance Dataset
  student_data = pd.read_csv('data/raw/student-performance.csv')
  # Preprocess student data here
  student_data.to_csv('data/processed/student_processed.csv', index=False)

  # Breast Cancer Dataset
  cancer_data = pd.read_csv('data/raw/breast-cancer.csv')
  # Preprocess breast cancer data here
  cancer_data.to_csv('data/processed/cancer_processed.csv', index=False)
  ```

## Usage

### 1. Preprocessing the Data

Before you can train a model, the raw data needs to be preprocessed. To do this, run the following [notebook](notebooks/data_preprocessing.ipynb):

This notebook takes the raw datasets from **`data/raw/`** and processes them, saving the results to **`data/processed/`**. This step is essential, as all models depend on the preprocessed data.

### 2. Training Your Model

The project does not include a standalone **`train_model.py`** script. Instead, you can train your own model by working with the notebooks in the [notebooks](notebooks/) folder. These notebooks are organized by algorithm and implementation (e.g., from scratch or using libraries like scikit-learn).

You have two options:

- **Modify an Existing Notebook:** Open one of the notebooks in notebooks/experiments/, tweak the parameters, or adjust the model to suit your needs.
- **Write Your Own Notebook:** Create a new notebook in the same folder for a fully custom training setup.
  The notebooks are designed to be flexible and interactive, allowing you to experiment with different models and configurations easily.

## Project Components

### Data Handling

The `data/` directory is organized to manage all datasets effectively. It includes:

- **`raw/`**: Contains original datasets in their unaltered form, ensuring access to the original data for reference or reprocessing **(not tracked)**.
- **`processed/`**: Holds preprocessed datasets that are ready for model training, allowing for efficient data handling without the need to preprocess each time **(not tracked)**.
- **`examples/`**: Provides sample input/output files for testing and demonstration purposes, facilitating the validation of implementations.

### Model Implementations

The `src/scratch/models/` directory is where machine learning algorithms are implemented from scratch. Each model is encapsulated in its own file, allowing for clear organization and separation of concerns. Key components include:

- **`base_model.py`**: An abstract base class that defines the interface for all models, ensuring consistency across different implementations.
- **`linear_models.py`**: Contains implementations for linear regression and logistic regression, focusing on the mathematical foundations and algorithmic details.
- **`tree_models.py`**: Implements decision tree algorithms, providing insights into tree-based learning methods.
- **`neural_network.py`**: Contains the implementation of neural networks, exploring the architecture and training processes involved.

### Utilities

The `src/scratch/utils/` directory houses reusable helper functions that support various aspects of the project. These utilities include:

- **`data_utils.py`**: Functions for data preprocessing, such as normalization, encoding, and splitting datasets, streamlining the data handling process.
- **`math_utils.py`**: Core mathematical operations essential for implementing algorithms, including functions for calculating gradients and cost functions.
- **`metrics.py`**: Evaluation metrics that help assess model performance, such as accuracy, precision, and recall, ensuring effective measurement of model success.
- **`viz_utils.py`**: Visualization tools that assist in plotting results and understanding model behavior, making it easier to interpret outcomes and share findings.

### Testing

The `tests/` directory is dedicated to maintaining the reliability of implementations. It includes:

- **`test_models/`**: Contains unit tests for various model implementations, ensuring that each model behaves as expected and meets performance criteria.
- **`test_utils/`**: Houses tests for utility functions, validating that data processing and mathematical operations function correctly.

## Contributing

Contributions are welcome! If you have suggestions, bug fixes, or new ideas (such as additional models or optimizations), please follow these steps:

- Fork this repository.
- Create a new branch for your feature or fix.
- Commit your changes with clear messages.
- Submit a pull request with a detailed explanation of your modifications.

For major changes, please open an issue first to discuss your ideas.

## License

MIT License

Copyright (c) 2025 Moaz-Loaie

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Acknowledgements

- Thanks to the [scikit-learn](https://scikit-learn.org/) team for their excellent library
- Special thanks to the developers of [NumPy](https://numpy.org/) and [Matplotlib](https://matplotlib.org/) for providing the backbone tools that make numerical computing and visualization accessible.
- All contributors and maintainers and the open-source community.
