import numpy as np

from ..utils.math_utils import add_bias_term, matrix_inverse
from .base_model import BaseModel


class LinearRegression(BaseModel):
    """
    Linear Regression model with multiple fitting methods.

    This class implements linear regression using different methods:
    - 'normal': Normal Equation
    - 'batch_gd': Batch Gradient Descent
    - 'stochastic_gd': Stochastic Gradient Descent
    - 'ridge': Ridge Regression (Regularized Linear Regression)

    Parameters:
    -----------
    method : str, default='normal'
        The method to use for fitting the model.
    learning_rate : float, default=0.01
        Learning rate for gradient descent methods.
    n_iterations : int, default=1000
        Number of iterations for gradient descent methods.
    alpha : float, default=0.01
        Regularization strength for Ridge Regression.
    verbose : bool, default=False
        If True, print training progress during iterative methods.

    Attributes:
    -----------
    weights : array-like
        The learned weights of the model.
    loss_history : list
        History of loss values during training (for GD methods).
    """

    def __init__(
        self,
        method="normal",
        learning_rate=0.01,
        n_iterations=1000,
        alpha=0.01,
        verbose=False,
    ):
        """
        Initialize the LinearRegression model.

        Parameters:
        -----------
        method : str, default='normal'
            The method to use for fitting the model.
        learning_rate : float, default=0.01
            Learning rate for gradient descent methods.
        n_iterations : int, default=1000
            Number of iterations for gradient descent methods.
        alpha : float, default=0.01
            Regularization strength for Ridge Regression.
        verbose : bool, default=False
            If True, print training progress during iterative methods.
        """
        self.method = method
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.alpha = alpha
        self.verbose = verbose  # Controls whether training progress is printed
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
            Target values.

        Returns:
        --------
        self : object
            Returns the instance itself.
        """
        X_b = add_bias_term(X)
        if self.method == "normal":
            self._fit_normal(X_b, y)
        elif self.method == "batch_gd":
            self._fit_batch_gd(X_b, y)
        elif self.method == "stochastic_gd":
            self._fit_stochastic_gd(X_b, y)
        elif self.method == "ridge":
            self._fit_ridge(X_b, y)
        else:
            raise ValueError("Invalid method specified")
        return self

    def _fit_normal(self, X_b, y):
        """
        Fit the model using the Normal Equation.

        Parameters:
        -----------
        X_b : array-like, shape (n_samples, n_features + 1)
            Training data with bias term.
        y : array-like, shape (n_samples,)
            Target values.
        """
        self.weights = matrix_inverse(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

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
        self.weights = np.zeros(X_b.shape[1])
        self.loss_history = []
        for i in range(self.n_iterations):
            predictions = X_b.dot(self.weights)
            error = predictions - y
            loss = np.mean(error**2)
            self.loss_history.append(loss)
            # Print training progress if verbose is enabled
            if self.verbose and (i % 100 == 0 or i == self.n_iterations - 1):
                print(f"Iteration {i}: Loss = {loss:.6f}")
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
        self.weights = np.zeros(X_b.shape[1])
        self.loss_history = []
        for i in range(self.n_iterations):
            total_loss = 0
            indices = np.random.permutation(len(y))
            for j in indices:
                xi = X_b[j : j + 1]
                yi = y[j : j + 1]
                prediction = xi.dot(self.weights)
                error = prediction - yi
                total_loss += error**2
                gradient = xi.T.dot(error)
                self.weights -= self.learning_rate * gradient
            avg_loss = total_loss / len(y)
            self.loss_history.append(avg_loss)
            # Print training progress if verbose is enabled
            if self.verbose and (i % 10 == 0 or i == self.n_iterations - 1):
                print(f"Epoch {i}: Average Loss = {avg_loss:.6f}")

    def _fit_ridge(self, X_b, y):
        """
        Fit the model using Ridge Regression.

        Parameters:
        -----------
        X_b : array-like, shape (n_samples, n_features + 1)
            Training data with bias term.
        y : array-like, shape (n_samples,)
            Target values.
        """
        identity = np.eye(X_b.shape[1])
        identity[0, 0] = 0  # Don't regularize bias term
        self.weights = (
            matrix_inverse(X_b.T.dot(X_b) + self.alpha * identity).dot(X_b.T).dot(y)
        )

    def predict(self, X):
        """
        Predict target values for given data.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Data to predict.

        Returns:
        --------
        array-like, shape (n_samples,)
            Predicted values.
        """
        X_b = add_bias_term(X)
        return X_b.dot(self.weights)

    def get_loss_history(self):
        """
        Get the history of loss values during training.

        Returns:
        --------
        list
            List of loss values.
        """
        return self.loss_history
