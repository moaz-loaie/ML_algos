import numpy as np


def accuracy(y_true, y_pred):
    """Calculate classification accuracy."""
    return np.mean(y_true == y_pred)


def mean_squared_error(y_true, y_pred):
    """Calculate mean squared error for regression."""
    return np.mean((y_true - y_pred) ** 2)
