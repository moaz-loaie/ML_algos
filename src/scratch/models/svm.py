import numpy as np

from ..utils.math_utils import linear_kernel, polynomial_kernel, rbf_kernel
from ..utils.smo_utils import (
    clip_alpha_j,
    compute_error,
    compute_L_H,
    select_second_alpha,
)
from .base_model import BaseModel


class SVM(BaseModel):
    """
    Support Vector Machine classifier using Sequential Minimal Optimization (SMO).

    This class implements a binary SVM with support for various kernels, trained using
    the SMO algorithm to solve the quadratic programming problem efficiently.

    Parameters
    ----------
    C : float, default=1.0
        Regularization parameter controlling the trade-off between margin maximization
        and classification error.
    tol : float, default=1e-3
        Tolerance for stopping criteria (KKT conditions).
    max_passes : int, default=5
        Maximum number of passes over the data without alpha changes before stopping.
    kernel : str, default='linear'
        Kernel type to use ('linear', 'poly', 'rbf').
    gamma : float, default=0.1
        Kernel coefficient for 'rbf' kernel.
    degree : int, default=3
        Degree of the polynomial kernel function.
    coef0 : float, default=1
        Independent term in polynomial kernel.
    verbose : bool, default=False
        If True, print progress during training.

    Attributes
    ----------
    alphas : ndarray
        Lagrange multipliers for the support vectors.
    b : float
        Bias term in the decision function.
    X : ndarray
        Training feature data (stored for prediction).
    y : ndarray
        Training labels (stored for prediction).
    error_cache : ndarray
        Cache of errors for each training example.
    support_vectors_ : ndarray
        Support vectors (subset of X where alphas > 0).
    support_alphas_ : ndarray
        Corresponding alphas for support vectors.
    support_y_ : ndarray
        Corresponding labels for support vectors.
    """

    def __init__(
        self,
        C=1.0,
        tol=1e-3,
        max_passes=5,
        kernel="linear",
        gamma=0.1,
        degree=3,
        coef0=1,
        verbose=False,
    ):
        self.C = C
        self.tol = tol
        self.max_passes = max_passes
        self.kernel = kernel.lower()
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.verbose = verbose
        self.alphas = None
        self.b = 0
        self.X = None
        self.y = None
        self.error_cache = None
        self.support_vectors_ = None
        self.support_alphas_ = None
        self.support_y_ = None

    def _kernel(self, x1, x2):
        """Compute the kernel function between two vectors."""
        if self.kernel == "linear":
            return linear_kernel(x1, x2)
        elif self.kernel == "rbf":
            return rbf_kernel(x1, x2, self.gamma)
        elif self.kernel == "poly":
            return polynomial_kernel(x1, x2, self.degree, self.coef0)
        else:
            raise ValueError(f"Unsupported kernel type: {self.kernel}")

    def fit(self, X, y):
        """
        Fit the SVM model using the SMO algorithm.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training feature data.
        y : array-like, shape (n_samples,)
            Target labels (must be -1 or 1).

        Returns
        -------
        self : SVM
            The fitted model instance.

        Raises
        ------
        ValueError
            If labels are not {-1, 1} or if input shapes are inconsistent.
        """
        X = np.asarray(X)
        y = np.asarray(y)
        if X.shape[0] != y.shape[0]:
            raise ValueError("Number of samples in X and y must match")
        if set(y) != {-1, 1}:
            raise ValueError("Labels must be -1 and 1")

        self.X = X
        self.y = y
        m, n = X.shape
        self.alphas = np.zeros(m)
        self.b = 0
        self.error_cache = np.zeros(m)
        passes = 0

        while passes < self.max_passes:
            num_changed_alphas = 0
            for i in range(m):
                E_i = compute_error(self, X, y, i)
                self.error_cache[i] = E_i
                # Check KKT conditions
                if (y[i] * E_i < -self.tol and self.alphas[i] < self.C) or (
                    y[i] * E_i > self.tol and self.alphas[i] > 0
                ):
                    j = select_second_alpha(self, i, E_i)
                    if j == -1:
                        continue
                    E_j = compute_error(self, X, y, j)
                    alpha_i_old, alpha_j_old = self.alphas[i], self.alphas[j]

                    # Compute L and H bounds
                    L, H = compute_L_H(alpha_i_old, alpha_j_old, y[i], y[j], self.C)
                    if L == H:
                        continue

                    # Compute eta
                    eta = (
                        2 * self._kernel(X[i], X[j])
                        - self._kernel(X[i], X[i])
                        - self._kernel(X[j], X[j])
                    )
                    if eta >= 0:
                        continue

                    # Update alpha_j
                    alpha_j_new = alpha_j_old - y[j] * (E_i - E_j) / eta
                    alpha_j_new = clip_alpha_j(alpha_j_new, L, H)
                    if abs(alpha_j_new - alpha_j_old) < 1e-5:
                        continue

                    # Update alpha_i
                    alpha_i_new = alpha_i_old + y[i] * y[j] * (
                        alpha_j_old - alpha_j_new
                    )

                    # Update bias b
                    b1 = (
                        self.b
                        - E_i
                        - y[i] * (alpha_i_new - alpha_i_old) * self._kernel(X[i], X[i])
                        - y[j] * (alpha_j_new - alpha_j_old) * self._kernel(X[i], X[j])
                    )
                    b2 = (
                        self.b
                        - E_j
                        - y[i] * (alpha_i_new - alpha_i_old) * self._kernel(X[i], X[j])
                        - y[j] * (alpha_j_new - alpha_j_old) * self._kernel(X[j], X[j])
                    )
                    if 0 < alpha_i_new < self.C:
                        self.b = b1
                    elif 0 < alpha_j_new < self.C:
                        self.b = b2
                    else:
                        self.b = (b1 + b2) / 2

                    # Apply updates
                    self.alphas[i] = alpha_i_new
                    self.alphas[j] = alpha_j_new
                    num_changed_alphas += 1
                    if self.verbose:
                        print(f"Epoch {passes + 1}: Updated alphas at i={i}, j={j}")

            if num_changed_alphas == 0:
                passes += 1
            else:
                passes = 0
            if self.verbose:
                print(
                    f"Pass {passes + 1}/{self.max_passes}: "
                    f"Changed {num_changed_alphas} alphas"
                )

        # Store support vectors
        sv_idx = self.alphas > 0
        self.support_vectors_ = X[sv_idx]
        self.support_alphas_ = self.alphas[sv_idx]
        self.support_y_ = y[sv_idx]
        if self.verbose:
            print(
                f"Training complete. Found {len(self.support_vectors_)} support vectors"
            )
        return self

    def predict(self, X):
        """
        Predict class labels for given data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Data to predict.

        Returns
        -------
        ndarray, shape (n_samples,)
            Predicted labels (-1 or 1).

        Raises
        ------
        ValueError
            If the model is not fitted or if X has incorrect feature dimension.
        """
        if self.support_vectors_ is None:
            raise ValueError("Model must be fitted before prediction")
        X = np.asarray(X)
        if X.shape[1] != self.X.shape[1]:
            raise ValueError("Feature dimension of X must match training data")

        y_pred = []
        for x in X:
            decision = (
                sum(
                    a * y_sv * self._kernel(x_sv, x)
                    for a, y_sv, x_sv in zip(
                        self.support_alphas_, self.support_y_, self.support_vectors_
                    )
                )
                + self.b
            )
            y_pred.append(np.sign(decision))
        return np.array(y_pred)
