from abc import ABC, abstractmethod


class BaseModel(ABC):
    """Abstract base class for all machine learning models."""

    @abstractmethod
    def fit(self, X, y):
        """Train the model with input features X and target y."""
        pass

    @abstractmethod
    def predict(self, X):
        """Make predictions on input features X."""
        pass
