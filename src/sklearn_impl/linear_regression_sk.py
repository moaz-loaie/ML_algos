from typing import Literal, Optional

from sklearn.linear_model import LinearRegression as SKLinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error
from sklearn.utils import shuffle as sk_shuffle

from ..scratch.models.base_model import BaseModel
from ..scratch.utils.training_utils import check_early_stopping, log_progress


class LinearRegressionSK(BaseModel):
    """
    Scikit-learn wrapper for Linear Regression with multiple optimization methods, manual loss tracking,
    and optional early stopping for iterative methods.

    Supports:
    - 'normal': Closed-form solution using LinearRegression.
    - 'sgd': Stochastic Gradient Descent using SGDRegressor.
    - 'batch': Full-batch Gradient Descent simulated via SGDRegressor.

    For iterative methods ('sgd' and 'batch'), the loss (MSE) is computed after each epoch, and early
    stopping can be enabled to halt training if the loss does not improve over a specified number of epochs.

    Parameters
    ----------
    method : Literal['normal', 'sgd', 'batch'], default='normal'
        Optimization method to use.
    learning_rate : float, default=0.01
        Learning rate for SGD-based methods.
    n_iterations : int, default=1000
        Maximum number of epochs for iterative methods.
    early_stopping : bool, default=False
        If True, enables early stopping for iterative methods.
    n_iter_no_change : int, default=10
        Number of epochs to wait for improvement before stopping early.
    tol : float, default=1e-4
        Minimum improvement in loss required to reset the early stopping counter.
    verbose : bool, default=False
        If True, prints loss every 100 epochs.
    interval : int, default=100
        Frequency (in epochs) to output a log message when verbose is True.
    **kwargs : dict
        Additional keyword arguments passed to the scikit-learn estimator.

    Attributes
    ----------
    model : object
        The underlying scikit-learn estimator.
    loss_history_ : list
        Manually tracked loss history for iterative methods.
    """

    def __init__(
        self,
        method: Literal["normal", "sgd", "batch"] = "normal",
        learning_rate: float = 0.01,
        n_iterations: int = 1000,
        early_stopping: bool = False,
        n_iter_no_change: int = 10,
        tol: float = 1e-4,
        verbose: bool = False,
        interval=100,
        **kwargs,
    ):
        self.method = method.lower()
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.early_stopping = early_stopping
        self.n_iter_no_change = n_iter_no_change
        self.tol = tol
        self.verbose = verbose
        self.interval = interval
        self.kwargs = kwargs
        self.loss_history_ = []

        if self.method == "normal":
            # Closed-form solution; loss tracking (beyond computing one final loss) and early stopping are not applicable.
            self.model = SKLinearRegression(**kwargs)
        elif self.method in ["sgd", "batch"]:
            self.model = SGDRegressor(
                learning_rate="constant",
                eta0=self.learning_rate,
                warm_start=True,
                **kwargs,
            )
        else:
            raise ValueError("Method must be 'normal', 'sgd', or 'batch'.")

    def fit(self, X, y):
        """
        Fit the model to the training data with optional early stopping for iterative methods.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : LinearRegressionSK
            The fitted model instance.
        """
        if self.method == "normal":
            self.model.fit(X, y)
            final_loss = mean_squared_error(y, self.model.predict(X))
            self.loss_history_.append(final_loss)
            if self.verbose:
                print(f"Closed-form solution completed. Final Loss = {final_loss:.6f}")
        elif self.method in ["sgd", "batch"]:
            if self.early_stopping:
                best_loss = float("inf")
                counter = 0
            self.model.fit(X, y)  # Initial fit
            for epoch in range(self.n_iterations):
                if self.method == "sgd":
                    X_epoch, y_epoch = sk_shuffle(X, y, random_state=None)
                else:
                    X_epoch, y_epoch = X, y
                self.model.partial_fit(X_epoch, y_epoch)
                current_loss = mean_squared_error(y, self.model.predict(X))
                self.loss_history_.append(current_loss)
                log_progress(
                    epoch,
                    self.n_iterations,
                    current_loss,
                    self.learning_rate,
                    self.verbose,
                    self.interval,
                )
                if self.early_stopping:
                    best_loss, counter, stop = check_early_stopping(
                        current_loss,
                        best_loss,
                        self.tol,
                        counter,
                        self.n_iter_no_change,
                        self.verbose,
                        epoch,
                    )
                    if stop:
                        break
        return self

    def predict(self, X):
        """
        Predict using the fitted model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to predict.

        Returns
        -------
        array-like of shape (n_samples,)
            Predicted values.
        """
        return self.model.predict(X)

    def score(self, X, y) -> float:
        """
        Compute the coefficient of determination R^2 for the prediction.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data for evaluation.
        y : array-like of shape (n_samples,)
            True target values.

        Returns
        -------
        float
            R^2 score as computed by the underlying estimator.
        """
        return self.model.score(X, y)

    def get_loss_history(self):
        """
        Retrieve the training loss history.

        Returns
        -------
        list
            Loss history for iterative methods, or a single loss value for 'normal'.
        """
        return self.loss_history_
