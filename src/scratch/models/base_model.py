from abc import ABC, abstractmethod


class BaseModel(ABC):
    """
    Abstract base class for all machine learning models.

    This class defines the interface that all models must implement, using
    Python's ABC module to enforce method implementation in subclasses.
    """

    @abstractmethod
    def fit(self, X, y):
        """
        Fit the model to the training data.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data.
        y : array-like, shape (n_samples,)
            Target values.

        Raises:
        -------
        NotImplementedError
            If the subclass does not implement this method.
        """
        pass

    @abstractmethod
    def predict(self, X):
        """
        Predict target values for given data.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Data to predict.

        Returns:
        --------
        array-like
            Predicted values.

        Raises:
        -------
        NotImplementedError
            If the subclass does not implement this method.
        """
        pass
