import numpy as np
import pandas as pd


def load_data(file_path):
    """Load a CSV dataset into a pandas DataFrame."""
    return pd.read_csv(file_path)


def split_data(X, y, test_size=0.2):
    """Split data into training and test sets."""
    indices = np.random.permutation(len(X))
    test_set_size = int(len(X) * test_size)
    test_indices = indices[:test_set_size]
    train_indices = indices[test_set_size:]
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]


def normalize(X):
    """Normalize features to have zero mean and unit variance."""
    return (X - X.mean(axis=0)) / X.std(axis=0)
