from typing import Literal, Optional

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, hinge_loss
from sklearn.svm import SVC
from sklearn.utils import shuffle as sk_shuffle
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


class SVM_SK(BaseEstimator, ClassifierMixin):
    """
    Scikit-learn wrapper for an SVM that is entirely implemented using scikit-learn tools.

    This wrapper supports two training methods:

    - 'svc': Uses SVC (libsvm-based) for a one-shot SVM training.
    - 'sgd': Uses SGDClassifier configured with hinge loss for an iterative SVM-like training,
             with optional early stopping and loss history tracking.

    Parameters
    ----------
    method : Literal['svc', 'sgd'], default='svc'
        The underlying training method. 'svc' uses SVC whereas 'sgd' uses SGDClassifier.
    C : float, default=1.0
        Regularization parameter. For 'sgd', the inverse relationship with alpha is approximated via alpha=1/C.
    tol : float, default=1e-3
        Tolerance for stopping criteria. For 'sgd', used to assess change in hinge loss.
    max_iter : int, default=1000
        Maximum number of training epochs for the iterative method ('sgd').
    early_stopping : bool, default=False
        If True, training stops early if the hinge loss does not improve by at least tol
        for a number of consecutive epochs.
    n_iter_no_change : int, default=10
        Number of consecutive epochs with no significant improvement in loss required to trigger early stopping.
    verbose : bool, default=False
        If True, print progress messages during training.
    interval : int, default=100
        Frequency (in epochs) to log a training progress message when verbose is True.
    random_state : Optional[int], default=None
        Random seed for reproducibility.
    **kwargs : dict
        Additional keyword arguments to pass to the underlying scikit-learn estimator.

    Attributes
    ----------
    model_ : object
        The underlying scikit-learn estimator (either SVC or SGDClassifier).
    loss_history_ : list
        Recorded hinge loss values (only available if method='sgd').
    """

    def __init__(
        self,
        method: Literal["svc", "sgd"] = "svc",
        C: float = 1.0,
        tol: float = 1e-3,
        max_iter: int = 1000,
        early_stopping: bool = False,
        n_iter_no_change: int = 10,
        verbose: bool = False,
        interval: int = 100,
        random_state: Optional[int] = None,
        **kwargs,
    ):
        self.method = method.lower()
        self.C = C
        self.tol = tol
        self.max_iter = max_iter
        self.early_stopping = early_stopping
        self.n_iter_no_change = n_iter_no_change
        self.verbose = verbose
        self.interval = interval
        self.random_state = random_state
        self.kwargs = kwargs

        self.loss_history_ = []

        if self.method == "svc":
            # SVC trains in one go; early stopping and iterative loss tracking are not applicable.
            self.model_ = SVC(
                C=self.C, tol=self.tol, random_state=self.random_state, **self.kwargs
            )
        elif self.method == "sgd":
            # SGDClassifier configured with hinge loss performs an iterative training.
            self.model_ = SGDClassifier(
                loss="hinge",
                alpha=1.0
                / self.C,  # Typically, alpha = 1 / (C * n_samples); here we approximate with 1/C.
                tol=None,  # We'll handle stopping via our own early stopping logic.
                max_iter=1,  # We use partial_fit for manual control of epochs.
                warm_start=True,
                random_state=self.random_state,
                **self.kwargs,
            )
        else:
            raise ValueError("Method must be either 'svc' or 'sgd'.")

    def fit(self, X, y):
        """
        Fit the SVM model to the training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target labels, which must be exactly -1 and 1.

        Returns
        -------
        self : SVM_SK
            The fitted model.
        """
        X, y = check_X_y(X, y)
        # Ensure the labels are exactly -1 and 1.
        unique_labels = np.unique(y)
        if set(unique_labels) != {-1, 1}:
            raise ValueError("Labels must be exactly -1 and 1.")

        if self.method == "svc":
            # SVC trains directly in one call.
            self.model_.fit(X, y)
            # Record a single final loss value for compatibility.
            decision = self.model_.decision_function(X)
            final_loss = hinge_loss(y, decision)
            self.loss_history_.append(final_loss)
        elif self.method == "sgd":
            classes = np.unique(y)
            best_loss = np.inf
            no_improve_count = 0
            # Iterative training loop.
            for epoch in range(self.max_iter):
                # Shuffle the data for each epoch.
                X_shuffled, y_shuffled = sk_shuffle(
                    X, y, random_state=self.random_state
                )
                if epoch == 0:
                    self.model_.partial_fit(X_shuffled, y_shuffled, classes=classes)
                else:
                    self.model_.partial_fit(X_shuffled, y_shuffled)
                # Compute current hinge loss on the whole training set.
                decision = self.model_.decision_function(X)
                current_loss = hinge_loss(y, decision)
                self.loss_history_.append(current_loss)

                if self.verbose and epoch % self.interval == 0:
                    print(f"Epoch {epoch}: hinge loss = {current_loss:.6f}")

                if self.early_stopping:
                    # Check for improvement.
                    if current_loss < best_loss - self.tol:
                        best_loss = current_loss
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                    if no_improve_count >= self.n_iter_no_change:
                        if self.verbose:
                            print(f"Early stopping triggered at epoch {epoch}.")
                        break
        return self

    def predict(self, X):
        """
        Predict class labels for samples in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels.
        """
        X = check_array(X)
        # For SVC, check for a fitted attribute typically set during fit.
        if self.method == "svc":
            if not hasattr(self.model_, "support_"):
                raise ValueError("Model is not fitted yet.")
        else:
            # For SGDClassifier, use scikit-learn's check.
            check_is_fitted(self.model_)
        return self.model_.predict(X)

    def score(self, X, y) -> float:
        """
        Compute the accuracy of the model on the test data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test data.
        y : array-like of shape (n_samples,)
            True labels.

        Returns
        -------
        score : float
            Accuracy score.
        """
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)

    def get_loss_history(self):
        """
        Retrieve the training loss history (if available).

        Returns
        -------
        list
            A list of hinge loss values recorded per epoch (only for method='sgd', or a single value for 'svc').
        """
        return self.loss_history_
