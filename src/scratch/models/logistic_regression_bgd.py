import numpy as np
from ..utils.math_utils import add_bias_term, sigmoid
from .base_model import BaseModel

class LogisticRegression(BaseModel):
    def __init__(self, method="batch_gd", learning_rate=0.01, n_iterations=1000):
        self.method = method
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.loss_history = [] 

    def fit(self, X, y):
        X_b = add_bias_term(X)
        self.weights = np.zeros(X_b.shape[1])

        if self.method == "batch_gd":
            self._fit_batch_gd(X_b, y)
        elif self.method == "stochastic_gd":
            self._fit_stochastic_gd(X_b, y)
        else:
            raise ValueError("Invalid method specified")
        return self

    def _fit_batch_gd(self, X_b, y):
        n_samples = len(y)
        for _ in range(self.n_iterations):
            linear_model = np.dot(X_b, self.weights)
            y_predicted = sigmoid(linear_model)
            errors = y_predicted - y
            gradient = np.dot(X_b.T, errors) / n_samples
            self.weights -= self.learning_rate * gradient
            loss = -np.mean(y * np.log(y_predicted + 1e-15) + (1 - y) * np.log(1 - y_predicted + 1e-15))
            self.loss_history.append(loss)

    def predict(self, X):
        X_b = add_bias_term(X)
        probabilities = sigmoid(np.dot(X_b, self.weights))
        return (probabilities >= 0.5).astype(int)

    def get_loss_history(self):
        return self.loss_history