import numpy as np
from ..utils.math_utils import add_bias_term
from ..utils.metrics import mean_squared_error
from .base_model import BaseModel

class LinearRegressionBGD(BaseModel):
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.loss_history = []  # To store MSE at each iteration

    def fit(self, X, y):
        """
        Train the model using batch gradient descent.
        """
        # Add bias term to X
        X_b = add_bias_term(X)  # Shape: (n_samples, n_features + 1)
        
        # Initialize weights with zeros
        n_features = X_b.shape[1]
        self.weights = np.zeros(n_features)
        
        # Batch gradient descent
        for _ in range(self.n_iterations):
            # Compute predictions
            y_pred = X_b.dot(self.weights)
            
            # Compute gradients
            gradients = (1 / X_b.shape[0]) * X_b.T.dot(y_pred - y)
            
            # Update weights
            self.weights -= self.learning_rate * gradients
            
            # Compute and store MSE
            mse = mean_squared_error(y, y_pred)
            self.loss_history.append(mse)

    def predict(self, X):
        """
        Predict using the learned weights.
        Returns:
            np.ndarray: Predicted values of shape (n_samples,).
        """
        X_b = add_bias_term(X)
        return X_b.dot(self.weights)