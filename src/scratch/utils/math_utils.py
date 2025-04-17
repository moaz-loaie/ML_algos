import numpy as np


def add_bias_term(X):
    """Add a column of ones to the input matrix X for the bias term."""
    return np.c_[np.ones(X.shape[0]), X]


def matrix_inverse(A):
    """Compute the inverse of a matrix using NumPy."""
    return np.linalg.inv(A)


def sigmoid(z):
    """Compute the sigmoid function."""
    return 1 / (1 + np.exp(-z))
