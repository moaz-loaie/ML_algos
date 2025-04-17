import numpy as np

from ..utils.math_utils import add_bias_term, matrix_inverse
from .base_model import BaseModel


class LinearRegressionNormal(BaseModel):
    def __init__(self):
        self.weights = None

    def fit(self, X, y):
        """Fit the model using the normal equation: (X^T X)^(-1) X^T y"""
        X_b = add_bias_term(X)  # Add bias term
        self.weights = matrix_inverse(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

    def predict(self, X):
        """Predict using the learned weights."""
        X_b = add_bias_term(X)
        return X_b.dot(self.weights)
