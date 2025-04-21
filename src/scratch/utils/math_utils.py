import numpy as np


def add_bias_term(X):
    """
    Add a column of ones to the input matrix X for the bias term.

    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Input data.

    Returns:
    --------
    array-like, shape (n_samples, n_features + 1)
        Input data with an additional bias column.
    """
    return np.c_[np.ones(X.shape[0]), X]


def matrix_inverse(A):
    """
    Compute the inverse of a matrix using NumPy.

    Parameters:
    -----------
    A : array-like, shape (n, n)
        Square matrix to invert.

    Returns:
    --------
    array-like, shape (n, n)
        Inverse of the matrix A.
    """
    return np.linalg.inv(A)


def sigmoid(z):
    """
    Compute the sigmoid function.

    Parameters:
    -----------
    z : array-like
        Input to the sigmoid function.

    Returns:
    --------
    array-like
        Sigmoid of z.
    """
    return 1 / (1 + np.exp(-z))
