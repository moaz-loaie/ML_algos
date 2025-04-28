from typing import Literal, Optional

from sklearn.linear_model import LogisticRegression as SKLogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import log_loss
from sklearn.utils import shuffle as sk_shuffle

from ..scratch.models.base_model import BaseModel
from ..scratch.utils.training_utils import check_early_stopping, log_progress


class LogisticRegressionSK(BaseModel):
    """
    Scikit-learn wrapper for Logistic Regression with multiple optimization methods, manual loss tracking,
    and optional early stopping for iterative methods.

    Supports:
    - 'batch_gd': Full-batch optimization using LogisticRegression.
    - 'sgd': Stochastic Gradient Descent using SGDClassifier.

    For 'sgd', the loss (log loss) is computed after each epoch, and early stopping can be enabled.

    Parameters
    ----------
    method : Literal['batch_gd', 'sgd'], default='batch_gd'
        Optimization method to use.
    learning_rate : float, default=0.01
        Learning rate for 'sgd' method.
    n_iterations : int, default=1000
        Maximum number of epochs for 'sgd'.
    early_stopping : bool, default=False
        If True, enables early stopping for 'sgd'.
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
        Manually tracked loss history for 'sgd'.
    """

    def __init__(
        self,
        method: Literal["batch_gd", "sgd"] = "batch_gd",
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

        if self.method == "batch_gd":
            # Using the full-batch optimization via LogisticRegression.
            self.model = SKLogisticRegression(solver="lbfgs", **kwargs)
        elif self.method == "sgd":
            # Using SGDClassifier with log loss to mimic logistic regression.
            # Early stopping parameters are passed to enable built-in early stopping.
            self.model = SGDClassifier(
                loss="log_loss",
                learning_rate="constant",
                eta0=learning_rate,
                warm_start=True,
                **kwargs,
            )
        else:
            raise ValueError("Method must be 'batch_gd' or 'sgd'.")

    def fit(self, X, y):
        """
        Fit the model to the training data with optional early stopping for 'sgd'.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : LogisticRegressionSK
            The fitted model instance.
        """
        if self.method == "batch_gd":
            self.model.fit(X, y)
            final_loss = log_loss(y, self.model.predict_proba(X))
            self.loss_history_.append(final_loss)
            if self.verbose:
                print(f"Batch GD completed. Final Loss = {final_loss:.6f}")
        elif self.method == "sgd":
            if self.early_stopping:
                best_loss = float("inf")
                counter = 0
            self.model.fit(X, y)  # Initial fit
            for epoch in range(self.n_iterations):
                X_epoch, y_epoch = sk_shuffle(X, y, random_state=None)
                self.model.partial_fit(X_epoch, y_epoch, classes=[0, 1])
                current_loss = log_loss(y, self.model.predict_proba(X))
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

    def predict_proba(self, X):
        """
        Predict probabilities using the fitted model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to predict.

        Returns
        -------
        array-like of shape (n_samples, n_classes)
            Probability estimates for each class.

        Raises
        ------
        AttributeError
            If the underlying model does not support probability estimates.
        """
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        else:
            raise AttributeError(
                "The underlying model does not support probability estimates."
            )

    def score(self, X, y) -> float:
        """
        Compute the accuracy of the model on the provided test data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data for evaluation.
        y : array-like of shape (n_samples,)
            True target labels.

        Returns
        -------
        float
            Accuracy score.
        """
        return self.model.score(X, y)

    def get_loss_history(self):
        """
        Retrieve the training loss history.

        Returns
        -------
        list
            Loss history for 'sgd', or a single loss value for 'batch_gd'.
        """
        return self.loss_history_
