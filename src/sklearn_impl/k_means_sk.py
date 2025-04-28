from typing import Optional
import numpy as np
from sklearn.cluster import KMeans as SKKMeans
from sklearn.utils import shuffle as sk_shuffle

from ..scratch.models.base_model import BaseModel


class KMeansSK(BaseModel):
    """
    Scikit-learn wrapper for KMeans clustering with manual inertia (loss) tracking
    and optional early stopping based on inertia improvement.

    Parameters
    ----------
    n_clusters : int, default=8
        The number of clusters to form.
    init : {'k-means++', 'random'}, default='k-means++'
        Method for initialization.
    n_init : int, default=10
        Number of time the k-means algorithm will be run with different centroid seeds.
    max_iter : int, default=300
        Maximum number of iterations of the k-means algorithm for a single run.
    early_stopping : bool, default=False
        If True, enables early stopping based on inertia improvement.
    n_iter_no_change : int, default=10
        Number of iterations to wait for improvement before stopping early.
    tol : float, default=1e-4
        Minimum improvement in inertia required to reset the early stopping counter.
    verbose : bool, default=False
        If True, prints inertia every 10 iterations.
    **kwargs : dict
        Additional keyword arguments passed to the scikit-learn KMeans.

    Attributes
    ----------
    model : object
        The underlying scikit-learn estimator.
    inertia_history_ : list
        Manually tracked inertia history.
    """

    def __init__(
        self,
        n_clusters: int = 8,
        init: str = "k-means++",
        n_init: int = 10,
        max_iter: int = 300,
        early_stopping: bool = False,
        n_iter_no_change: int = 10,
        tol: float = 1e-4,
        verbose: bool = False,
        **kwargs,
    ):
        self.n_clusters = n_clusters
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.early_stopping = early_stopping
        self.n_iter_no_change = n_iter_no_change
        self.tol = tol
        self.verbose = verbose
        self.kwargs = kwargs
        self.inertia_history_ = []

        self.model = SKKMeans(
            n_clusters=self.n_clusters,
            init=self.init,
            n_init=1,  # We'll handle multiple initializations manually if needed
            max_iter=1,  # Force 1 iter per fit to control manually
            warm_start=True,
            **self.kwargs,
        )

    def fit(self, X):
        """
        Fit the KMeans model to the data with optional early stopping.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        Returns
        -------
        self : KMeansSK
            The fitted model instance.
        """
        best_inertia = float("inf")
        wait = 0
        n_samples = X.shape[0]

        # Initialize cluster centers
        self.model._check_params_vs_input(X)
        self.model._init_centroids(X, x_squared_norms=np.sum(X**2, axis=1), init=self.init)

        for iteration in range(self.max_iter):
            X_epoch = sk_shuffle(X, random_state=None)

            self.model.partial_fit(X_epoch)
            current_inertia = self.model.inertia_
            self.inertia_history_.append(current_inertia)

            if self.verbose and (iteration % 10 == 0 or iteration == self.max_iter - 1):
                print(
                    f"Iteration {iteration + 1}/{self.max_iter}: Inertia = {current_inertia:.6f}"
                )

            if self.early_stopping:
                if current_inertia < best_inertia - self.tol:
                    best_inertia = current_inertia
                    wait = 0
                else:
                    wait += 1
                    if wait >= self.n_iter_no_change:
                        if self.verbose:
                            print(
                                f"Early stopping at iteration {iteration}: Inertia = {current_inertia:.6f}"
                            )
                        break
        return self

    def predict(self, X):
        """
        Predict the closest cluster each sample in X belongs to.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            New data to predict.

        Returns
        -------
        labels : array of shape (n_samples,)
            Index of the cluster each sample belongs to.
        """
        return self.model.predict(X)

    def score(self, X, y=None) -> float:
        """
        Return the negative inertia of the model on the given data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            New data.

        Returns
        -------
        float
            Negative inertia (the higher, the better).
        """
        return -self.model.inertia_

    def get_inertia_history(self):
        """
        Retrieve the training inertia history.

        Returns
        -------
        list
            Inertia history for monitoring convergence.
        """
        return self.inertia_history_

