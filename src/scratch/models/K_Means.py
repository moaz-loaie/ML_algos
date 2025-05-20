import numpy as np

from .base_model import BaseModel


class KMeans(BaseModel):
    """
    K-Means clustering algorithm.

    Parameters:
    -----------
    n_clusters : int, default=8
        The number of clusters to form.
    n_init : int, default=10
        Number of time the k-means algorithm will be run with different centroid seeds.
    max_iter : int, default=300
        Maximum number of iterations for a single run.
    tol : float, default=1e-4
        Tolerance to declare convergence (based on centroid movement).
    verbose : bool, default=False
        Whether to print progress during iterations.
    random_state : int, optional
        Random seed for reproducibility.

    Attributes:
    -----------
    centroids_ : array-like, shape (n_clusters, n_features)
        Coordinates of cluster centers.
    labels_ : array, shape (n_samples,)
        Labels of each point.
    inertia_ : float
        Sum of squared distances of samples to their closest cluster center.
    loss_history : list
        History of inertia (loss) over iterations for the best run.
    """

    def __init__(
        self,
        n_clusters=8,
        n_init=10,
        max_iter=300,
        tol=1e-4,
        verbose=False,
        random_state=None,
    ):
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.random_state = random_state
        self.centroids_ = None
        self.labels_ = None
        self.inertia_ = None
        self.loss_history = []

    def fit(self, X, y=None):
        """
        Compute k-means clustering.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data.

        Returns:
        --------
        self : object
            Fitted estimator.
        """
        X = np.asarray(X)
        best_inertia = np.inf
        best_centroids = None
        best_labels = None
        best_loss_history = []

        rng = np.random.default_rng(self.random_state)

        for init_no in range(self.n_init):
            if self.verbose:
                print(f"Initialization {init_no + 1}/{self.n_init}")
            # Random initialization
            centroids = X[rng.choice(X.shape[0], self.n_clusters, replace=False)]
            loss_history = []
            for i in range(self.max_iter):
                # Assignment step
                distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
                labels = np.argmin(distances, axis=1)
                inertia = np.sum((X - centroids[labels]) ** 2)
                loss_history.append(inertia)

                if self.verbose and (i % 10 == 0 or i == self.max_iter - 1):
                    print(f"Iteration {i + 1}/{self.max_iter}: Inertia = {inertia:.6f}")

                # Update step
                new_centroids = np.array(
                    [
                        (
                            X[labels == k].mean(axis=0)
                            if np.any(labels == k)
                            else centroids[k]
                        )
                        for k in range(self.n_clusters)
                    ]
                )

                # Check for convergence
                centroid_shifts = np.linalg.norm(new_centroids - centroids, axis=1)
                if np.all(centroid_shifts <= self.tol):
                    if self.verbose:
                        print(f"Convergence reached at iteration {i + 1}")
                    break

                centroids = new_centroids

            # Check if this initialization is better
            if inertia < best_inertia:
                best_inertia = inertia
                best_centroids = centroids
                best_labels = labels
                best_loss_history = loss_history

        self.centroids_ = best_centroids
        self.labels_ = best_labels
        self.inertia_ = best_inertia
        self.loss_history = best_loss_history

        return self

    def predict(self, X):
        """
        Predict the closest cluster each sample in X belongs to.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            New data to predict.

        Returns:
        --------
        labels : array, shape (n_samples,)
            Index of the cluster each sample belongs to.
        """
        X = np.asarray(X)
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids_, axis=2)
        return np.argmin(distances, axis=1)

    def get_loss_history(self):
        """
        Get the history of inertia values during training for the best run.

        Returns:
        --------
        list
            List of inertia values.
        """
        return self.loss_history

    def compute_inertias(self, X, k_range):
        """
        Compute inertias for a range of k values to facilitate the elbow method.

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
        """
        inertias = []
        for k in k_range:
            model = self.__class__(
                n_clusters=k,
                n_init=self.n_init,
                max_iter=self.max_iter,
                tol=self.tol,
                verbose=False,
                random_state=self.random_state,
            )
            model.fit(X)
            inertias.append(model.inertia_)
        return inertias
