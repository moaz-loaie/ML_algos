import numpy as np

from ..utils.math_utils import add_bias_term, sigmoid
from ..utils.training_utils import check_early_stopping, get_decayed_lr, log_progress
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
    verbose : bool, default=False
        If True, print training progress during iterative methods.
    interval : int, default=100
        Frequency (in epochs) to output a log message when verbose is True.
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
        History of loss values during training.
    """

    def __init__(
        self,
        method="batch_gd",
        learning_rate=0.01,
        n_iterations=1000,
        verbose=False,
        interval=100,
        lr_decay=0.0,
        batch_size=1,
        early_stopping=False,
        n_iter_no_change=10,
        tol=1e-4,
    ):
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
        verbose : bool, default=False
            If True, print training progress during iterative methods.
        interval : int, default=100
            Frequency (in epochs) to output a log message when verbose is True.
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
        self.verbose = verbose
        self.interval = interval
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
            Target values (binary).

        Returns:
        --------
        self : object
            Returns the instance itself.
        """
        X_b = add_bias_term(X)
        self.weights = np.zeros(X_b.shape[1])
        methods = {
            "batch_gd": self._fit_batch_gd,
            "stochastic_gd": self._fit_stochastic_gd,
        }

        try:
            methods[self.method](X_b, y)
        except KeyError:
            raise ValueError(f"Invalid method specified: {self.method}")
        return self

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
        self.loss_history = []
        if self.early_stopping:
            best_loss = float("inf")
            counter = 0
        for epoch in range(self.n_iterations):
            # Apply decay to the learning rate, e.g., new_lr = initial_lr / (1 + decay_rate * epoch)
            current_lr = get_decayed_lr(
                initial_lr=init_lr, lr_decay=self.lr_decay, epoch=epoch
            )
            predictions = sigmoid(X_b.dot(self.weights))
            error = predictions - y
            loss = -np.mean(
                y * np.log(predictions + 1e-15)
                + (1 - y) * np.log(1 - predictions + 1e-15)
            )
            self.loss_history.append(loss)
            # Print training progress if verbose is enabled
            log_progress(
                epoch, self.n_iterations, loss, current_lr, self.verbose, self.interval
            )
            if self.early_stopping:
                best_loss, counter, stop = check_early_stopping(
                    loss,
                    best_loss,
                    self.tol,
                    counter,
                    self.n_iter_no_change,
                    self.verbose,
                    epoch,
                )
                if stop:
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
        self.loss_history = []
        n_samples = len(y)
        # If batch_size is not set, default to pure SGD (batch_size = 1).
        batch_size = self.batch_size if self.batch_size is not None else 1
        if self.early_stopping:
            best_loss = float("inf")
            counter = 0
        for epoch in range(self.n_iterations):
            # Apply decay to the learning rate, e.g., new_lr = initial_lr / (1 + decay_rate * epoch)
            current_lr = get_decayed_lr(init_lr, self.lr_decay, epoch)
            total_loss = 0
            indices = np.random.permutation(len(y))
            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                batch_indices = indices[start:end]
                xi = X_b[batch_indices]
                yi = y[batch_indices]
                prediction = sigmoid(xi.dot(self.weights))
                error = prediction - yi
                total_loss -= yi * np.log(prediction + 1e-15) + (1 - yi) * np.log(
                    1 - prediction + 1e-15
                )
                gradient = xi.T.dot(error)
                self.weights -= current_lr * gradient
            avg_loss = total_loss / len(y)
            # Convert avg_loss to a float using .item() if it's a single-element array.
            avg_loss = avg_loss.item() if isinstance(avg_loss, np.ndarray) else avg_loss
            self.loss_history.append(avg_loss)
            # Print the training progress along with the current decayed learning rate if verbose is enabled
            log_progress(
                epoch,
                self.n_iterations,
                avg_loss,
                current_lr,
                self.verbose,
                self.interval,
            )
            if self.early_stopping:
                best_loss, counter, stop = check_early_stopping(
                    avg_loss,
                    best_loss,
                    self.tol,
                    counter,
                    self.n_iter_no_change,
                    self.verbose,
                    epoch,
                )
                if stop:
                    break

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
