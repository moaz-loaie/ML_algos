import numpy as np


def add_bias_term(X):
    """
    Add a column of ones to the input matrix X for the bias term.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Input data.

    Returns
    -------
    array-like, shape (n_samples, n_features + 1)
        Input data with an additional bias column.
    """
    return np.c_[np.ones(X.shape[0]), X]


def matrix_inverse(A):
    """
    Compute the inverse of a matrix using NumPy.

    Parameters
    ----------
    A : array-like, shape (n, n)
        Square matrix to invert.

    Returns
    -------
    array-like, shape (n, n)
        Inverse of the matrix A.
    """
    return np.linalg.inv(A)


def sigmoid(z):
    """
    Compute the sigmoid function.

    Parameters
    ----------
    z : array-like
        Input to the sigmoid function.

    Returns
    -------
    array-like
        Sigmoid of z.
    """
    return 1 / (1 + np.exp(-z))


def linear_kernel(x1, x2):
    """
    Compute the linear kernel between two vectors.

    Parameters
    ----------
    x1 : array-like
        First input vector.
    x2 : array-like
        Second input vector.

    Returns
    -------
    float
        Dot product of x1 and x2.
    """
    return np.dot(x1, x2)


def polynomial_kernel(x1, x2, degree=3, coef0=1):
    """
    Compute the polynomial kernel between two vectors.

    Parameters
    ----------
    x1 : array-like
        First input vector.
    x2 : array-like
        Second input vector.
    degree : int, default=3
        Degree of the polynomial.
    coef0 : float, default=1
        Constant term in the polynomial.

    Returns
    -------
    float
        Polynomial kernel value.
    """
    return (np.dot(x1, x2) + coef0) ** degree


def rbf_kernel(x1, x2, gamma=0.1):
    """
    Compute the RBF (Gaussian) kernel between two vectors.

    Parameters
    ----------
    x1 : array-like
        First input vector.
    x2 : array-like
        Second input vector.
    gamma : float, default=0.1
        Kernel coefficient.

    Returns
    -------
    float
        RBF kernel value.
    """
    diff = x1 - x2
    return np.exp(-gamma * np.dot(diff, diff))
