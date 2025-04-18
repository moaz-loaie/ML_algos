import numpy as np

from ..utils.math_utils import add_bias_term, sigmoid
from .base_model import BaseModel


class LogisticRegression(BaseModel):
    """
    Logistic Regression model with gradient descent methods.

    This class implements logistic regression using:
    - 'batch_gd': Batch Gradient Descent
    - 'stochastic_gd': Stochastic Gradient Descent

    Parameters:
    -----------
    method : str, default='batch_gd'
        The method to use for fitting the model.
    learning_rate : float, default=0.01
        Learning rate for gradient descent.
    n_iterations : int, default=1000
        Number of iterations for gradient descent.

    Attributes:
    -----------
    weights : array-like
        The learned weights of the model.
    loss_history : list
        History of loss values during training.
    """

    def __init__(self, method="batch_gd", learning_rate=0.01, n_iterations=1000):
        """
        Initialize the LogisticRegression model.

        Parameters:
        -----------
        method : str, default='batch_gd'
            The method to use for fitting the model.
        learning_rate : float, default=0.01
            Learning rate for gradient descent.
        n_iterations : int, default=1000
            Number of iterations for gradient descent.
        """
        self.method = method
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.loss_history = []

    def fit(self, X, y):
        """
        Fit the model to the training data.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data.
        y : array-like, shape (n_samples,)
            Target values (binary).

        Returns:
        --------
        self : object
            Returns the instance itself.
        """
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
        """
        Fit the model using Batch Gradient Descent.

        Parameters:
        -----------
        X_b : array-like, shape (n_samples, n_features + 1)
            Training data with bias term.
        y : array-like, shape (n_samples,)
            Target values.
        """
        for _ in range(self.n_iterations):
            predictions = sigmoid(X_b.dot(self.weights))
            error = predictions - y
            loss = -np.mean(
                y * np.log(predictions + 1e-15)
                + (1 - y) * np.log(1 - predictions + 1e-15)
            )
            self.loss_history.append(loss)
            gradient = X_b.T.dot(error) / len(y)
            self.weights -= self.learning_rate * gradient

    def _fit_stochastic_gd(self, X_b, y):
        """
        Fit the model using Stochastic Gradient Descent.

        Parameters:
        -----------
        X_b : array-like, shape (n_samples, n_features + 1)
            Training data with bias term.
        y : array-like, shape (n_samples,)
            Target values.
        """
        for _ in range(self.n_iterations):
            total_loss = 0
            indices = np.random.permutation(len(y))
            for i in indices:
                xi = X_b[i : i + 1]
                yi = y[i : i + 1]
                prediction = sigmoid(xi.dot(self.weights))
                error = prediction - yi
                total_loss -= yi * np.log(prediction + 1e-15) + (1 - yi) * np.log(
                    1 - prediction + 1e-15
                )
                gradient = xi.T.dot(error)
                self.weights -= self.learning_rate * gradient
            self.loss_history.append(total_loss / len(y))

    def predict(self, X):
        """
        Predict binary classes for given data.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Data to predict.

        Returns:
        --------
        array-like, shape (n_samples,)
            Predicted class labels (0 or 1).
        """
        X_b = add_bias_term(X)
        probabilities = sigmoid(X_b.dot(self.weights))
        return (probabilities >= 0.5).astype(int)

    def get_loss_history(self):
        """
        Get the history of loss values during training.

        Returns:
        --------
        list
            List of loss values.
        """
        return self.loss_history
