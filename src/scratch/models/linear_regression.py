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
    lr_decay : float, default=0.0
        The decay rate for learning rate schedule. Default is 0 (no decay).
    batch_size : int, default=1
        The number of samples per mini-batch. Default is 1 (pure stochastic gradient descent).
    early_stopping : bool, default=False
        Whether to use early stopping for iterative methods ('batch_gd', 'stochastic_gd').
    n_iter_no_change : int, default=10
        Number of iterations with no improvement to wait before stopping.
    tol : float, default=1e-4
        Minimum improvement in loss required to reset the early stopping counter.

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
        lr_decay=0.0,
        batch_size=1,
        early_stopping=False,
        n_iter_no_change=10,
        tol=1e-4,
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
        lr_decay : float, default=0.0
            The decay rate for learning rate schedule. Default is 0 (no decay).
        batch_size : int, default=1
            The number of samples per mini-batch. Default is 1 (pure stochastic gradient descent).
        early_stopping : bool, default=False
            Whether to use early stopping for iterative methods ('batch_gd', 'stochastic_gd').
        n_iter_no_change : int, default=10
            Number of iterations with no improvement to wait before stopping.
        tol : float, default=1e-4
            Minimum improvement in loss required to reset the early stopping counter.
        """
        self.method = method.lower()
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.alpha = alpha
        self.verbose = verbose
        self.lr_decay = lr_decay
        self.batch_size = batch_size
        self.early_stopping = early_stopping
        self.n_iter_no_change = n_iter_no_change
        self.tol = tol
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
        Fit the model using Batch Gradient Descent with optional early stopping.

        Parameters:
        -----------
        X_b : array-like, shape (n_samples, n_features + 1)
            Training data with bias term.
        y : array-like, shape (n_samples,)
            Target values.
        """
        # Store initial learning rate to apply decay over epochs.
        init_lr = self.learning_rate
        self.weights = np.zeros(X_b.shape[1])
        self.loss_history = []
        if self.early_stopping:
            best_loss = float("inf")
            counter = 0
        for i in range(self.n_iterations):
            # Apply decay to the learning rate, e.g., new_lr = initial_lr / (1 + decay_rate * epoch)
            current_lr = (
                init_lr / (1 + self.lr_decay * i) if self.lr_decay > 0 else init_lr
            )
            predictions = X_b.dot(self.weights)
            error = predictions - y
            loss = np.mean(error**2)
            self.loss_history.append(loss)
            # Print training progress if verbose is enabled
            if self.verbose and (i % 100 == 0 or i == self.n_iterations - 1):
                print(
                    f"Epoch {i + 1}/{self.n_iterations}: Loss = {loss:.6f} | Learning Rate = {current_lr:.6f}"
                )
            if self.early_stopping:
                if loss < best_loss - self.tol:
                    best_loss = loss
                    counter = 0
                else:
                    counter += 1
                    if counter >= self.n_iter_no_change:
                        if self.verbose:
                            print(f"Early stopping at iteration {i}: Loss = {loss:.6f}")
                        break
            gradient = X_b.T.dot(error) / len(y)
            self.weights -= current_lr * gradient

    def _fit_stochastic_gd(self, X_b, y):
        """
        Fit the model using Stochastic Gradient Descent with optional early stopping.

        Parameters:
        -----------
        X_b : array-like, shape (n_samples, n_features + 1)
            Training data with bias term.
        y : array-like, shape (n_samples,)
            Target values.
        """
        # Store initial learning rate to apply decay over epochs.
        init_lr = self.learning_rate
        self.weights = np.zeros(X_b.shape[1])
        self.loss_history = []
        n_samples = len(y)
        # If batch_size is not set, default to pure SGD (batch_size = 1).
        batch_size = self.batch_size if self.batch_size is not None else 1
        if self.early_stopping:
            best_loss = float("inf")
            counter = 0
        for i in range(self.n_iterations):
            # Apply decay to the learning rate, e.g., new_lr = initial_lr / (1 + decay_rate * epoch)
            current_lr = (
                init_lr / (1 + self.lr_decay * i) if self.lr_decay > 0 else init_lr
            )
            total_loss = 0
            # Shuffle the data indices for each epoch
            indices = np.random.permutation(n_samples)
            # Process data in mini-batches
            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                batch_indices = indices[start:end]
                xi = X_b[batch_indices]
                yi = y[batch_indices]
                prediction = xi.dot(self.weights)
                error = prediction - yi
                # Accumulate squared error (sum over mini-batch)
                total_loss += np.sum(error**2)
                gradient = xi.T.dot(error) / len(batch_indices)
                self.weights -= current_lr * gradient
            # Compute average loss over the entire dataset
            avg_loss = total_loss / n_samples
            # Convert avg_loss to a float using .item() if it's a single-element array.
            avg_loss = avg_loss.item() if isinstance(avg_loss, np.ndarray) else avg_loss
            self.loss_history.append(avg_loss)
            # Print the training progress along with the current decayed learning rate if verbose is enabled
            if self.verbose and (i % 10 == 0 or i == self.n_iterations - 1):
                print(
                    f"Epoch {i + 1}/{self.n_iterations}: Loss = {avg_loss:.6f} | Learning Rate = {current_lr:.6f}"
                )
            if self.early_stopping:
                if avg_loss < best_loss - self.tol:
                    best_loss = avg_loss
                    counter = 0
                else:
                    counter += 1
                    if counter >= self.n_iter_no_change:
                        if self.verbose:
                            print(f"Early stopping at epoch {i}: Loss = {avg_loss:.6f}")
                        break

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
