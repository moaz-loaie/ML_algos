import numpy as np
from ..utils.math_utils import add_bias_term
from ..utils.metrics import mean_squared_error
from .base_model import BaseModel

class LinearRegressionSGD(BaseModel):
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.loss_history = []  # To store MSE at each iteration

    def fit(self, X, y):
        """
        Train the model using stochastic gradient descent.
        """
        # Add bias term to X
        X_b = add_bias_term(X)  # Shape: (n_samples, n_features + 1)
        
        # Initialize weights with zeros
        n_samples, n_features = X_b.shape
        self.weights = np.zeros(n_features)
        
        # Stochastic gradient descent
        for _ in range(self.n_iterations):
            # Randomly select one sample
            idx = np.random.randint(0, n_samples)
            x_i = X_b[idx:idx+1]  # Shape: (1, n_features)
            y_i = y[idx:idx+1]   # Shape: (1,)
            
            # Compute prediction for the sample
            y_pred_i = x_i.dot(self.weights)
            
            # Compute gradient for the sample
            gradient = x_i.T.dot(y_pred_i - y_i)  # No averaging since it's one sample
            
            # Update weights
            self.weights -= self.learning_rate * gradient
            
            # Compute and store MSE on full dataset
            y_pred = X_b.dot(self.weights)
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