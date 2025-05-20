from typing import Literal, Optional

import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans

from ..scratch.models.base_model import BaseModel


class KMeansSK(BaseModel):
    """
    Scikit-learn wrapper for K-Means clustering with support for standard KMeans and MiniBatchKMeans.

    Supports:
    - 'kmeans': Standard KMeans algorithm.
    - 'minibatch': MiniBatchKMeans for large datasets or online learning.

    For 'minibatch', the inertia is tracked after each batch.

    Parameters
    ----------
    method : {'kmeans', 'minibatch'}, default='kmeans'
        The method to use for clustering.
    n_clusters : int, default=8
        The number of clusters to form.
    init : {'k-means++', 'random'}, default='k-means++'
        Method for initialization.
    n_init : int, default=10
        Number of times the k-means algorithm will be run with different centroid seeds.
    max_iter : int, default=300
        For 'kmeans', maximum number of iterations of the k-means algorithm for a single run.
        For 'minibatch', number of batches to process.
    batch_size : int, default=100
        Size of the mini-batches for 'minibatch' method.
    verbose : bool, default=False
        If True, prints progress during training for 'minibatch'.
    interval : int, default=10
        Frequency (in iterations) to print progress when verbose is True for 'minibatch'.
    random_state : int, optional
        Random seed for reproducibility.
    **kwargs : dict
        Additional keyword arguments passed to the scikit-learn estimator.

    Attributes
    ----------
    model : object
        The underlying scikit-learn estimator.
    inertia_history_ : list
        For 'kmeans', contains the final inertia.
        For 'minibatch', contains inertia after each batch.
    """

    def __init__(
        self,
        method: Literal["kmeans", "minibatch"] = "kmeans",
        n_clusters: int = 8,
        init: str = "k-means++",
        n_init: int = 10,
        max_iter: int = 300,
        batch_size: int = 100,
        verbose: bool = False,
        interval: int = 10,
        random_state: Optional[int] = None,
        **kwargs,
    ):
        self.method = method.lower()
        if self.method not in ["kmeans", "minibatch"]:
            raise ValueError("Method must be 'kmeans' or 'minibatch'")
        if n_clusters < 1:
            raise ValueError("n_clusters must be at least 1")
        if n_init < 1:
            raise ValueError("n_init must be at least 1")
        if max_iter < 1:
            raise ValueError("max_iter must be at least 1")
        if method == "minibatch" and batch_size < 1:
            raise ValueError("batch_size must be at least 1")

        self.n_clusters = n_clusters
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.batch_size = batch_size if self.method == "minibatch" else None
        self.verbose = verbose
        self.interval = interval
        self.random_state = random_state
        self.kwargs = kwargs
        self.inertia_history_ = []

        if self.method == "kmeans":
            self.model = KMeans(
                n_clusters=n_clusters,
                init=init,
                n_init=n_init,
                max_iter=max_iter,
                random_state=random_state,
                **kwargs,
            )
        elif self.method == "minibatch":
            self.model = MiniBatchKMeans(
                n_clusters=n_clusters,
                init=init,
                n_init=n_init,
                batch_size=batch_size,
                random_state=random_state,
                **kwargs,
            )

    def fit(self, X):
        """
        Fit the clustering model to the data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        Returns
        -------
        self : KMeansSK
            The fitted model instance.

        Raises
        ------
        ValueError
            If X is empty or has invalid shape.
        """
        X = np.asarray(X)
        if X.shape[0] == 0:
            raise ValueError("Input data X cannot be empty")
        if X.ndim != 2:
            raise ValueError("Input data X must be 2-dimensional")

        if self.method == "kmeans":
            self.model.fit(X)
            self.inertia_history_ = [self.model.inertia_]
            if self.verbose:
                print(f"KMeans completed. Final Inertia = {self.model.inertia_:.6f}")
        elif self.method == "minibatch":
            n_samples = X.shape[0]
            batch_size = min(self.batch_size, n_samples)
            if batch_size != self.batch_size:
                print(
                    f"Warning: batch_size ({self.batch_size}) is larger than n_samples ({n_samples}). Using batch_size={batch_size}"
                )
            self.inertia_history_ = []
            for iteration in range(self.max_iter):
                indices = np.random.choice(n_samples, batch_size, replace=False)
                X_batch = X[indices]
                self.model.partial_fit(X_batch)
                current_inertia = -self.model.score(X)
                self.inertia_history_.append(current_inertia)
                if self.verbose and (
                    iteration % self.interval == 0 or iteration == self.max_iter - 1
                ):
                    print(
                        f"Iteration {iteration + 1}/{self.max_iter}: Inertia = {current_inertia:.6f}"
                    )
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

        Raises
        ------
        ValueError
            If X has invalid shape.
        """
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("Input data X must be 2-dimensional")
        return self.model.predict(X)

    def score(self, X):
        """
        Compute the opposite of the value of X on the K-means objective.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            New data.

        Returns
        -------
        float
            Opposite of the K-means objective (negative inertia).

        Raises
        ------
        ValueError
            If X has invalid shape.
        """
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("Input data X must be 2-dimensional")
        return self.model.score(X)

    def get_inertia_history(self):
        """
        Retrieve the training inertia history.

        Returns
        -------
        list
            For 'kmeans', contains the final inertia.
            For 'minibatch', contains inertia after each batch.
        """
        return self.inertia_history_

    def compute_inertias(self, X, k_range):
        """
        Compute inertias for a range of k values using standard KMeans.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data.
        k_range : iterable
            Range of k values to compute inertias for.

        Returns
        -------
        inertias : list
            List of inertias corresponding to each k in k_range.

        Raises
        ------
        ValueError
            If X has invalid shape or k_range contains invalid values.
        """
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("Input data X must be 2-dimensional")
        if not all(isinstance(k, int) and k >= 1 for k in k_range):
            raise ValueError("All values in k_range must be integers >= 1")
        inertias = []
        for k in k_range:
            model = KMeans(n_clusters=k, n_init=10, random_state=self.random_state)
            model.fit(X)
            inertias.append(model.inertia_)
        return inertias
