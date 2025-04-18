import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_learning_curve(
    loss_history, title="Learning Curve", filename="learning_curve.png"
):
    plt.figure(figsize=(8, 6))
    plt.plot(loss_history, label="Training Loss", color="blue")
    plt.title(title)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()


def plot_decision_boundary(
    model, X, y, title="Decision Boundary", filename="decision_boundary.png"
):
    if X.shape[1] != 2:
        raise ValueError("Decision boundary plotting requires 2D feature space")
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.4, cmap="coolwarm")
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors="k", marker="o", cmap="coolwarm")
    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.savefig(filename)
    plt.close()


def plot_regression_line(
    model, X, y, title="Regression Line", filename="regression_line.png"
):
    if X.shape[1] != 1:
        raise ValueError("Regression line plotting requires 1D feature space")
    plt.figure(figsize=(8, 6))
    plt.scatter(X, y, color="blue", label="Data")
    X_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    plt.plot(X_range, model.predict(X_range), color="red", label="Regression Line")
    plt.title(title)
    plt.xlabel("Feature")
    plt.ylabel("Target")
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()


def plot_confusion_matrix(
    y_true, y_pred, title="Confusion Matrix", filename="confusion_matrix.png"
):
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(filename)
    plt.close()


def plot_prediction_errors(
    model, X, y, title="Prediction Errors", filename="prediction_errors.png"
):
    predictions = model.predict(X)
    errors = predictions - y
    plt.figure(figsize=(8, 6))
    plt.hist(errors, bins=30, color="purple", alpha=0.7)
    plt.title(title)
    plt.xlabel("Error")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.savefig(filename)
    plt.close()


def plot_feature_importance(
    model, feature_names, title="Feature Importance", filename="feature_importance.png"
):
    weights = model.weights[1:]  # Exclude bias term
    plt.figure(figsize=(10, 6))
    plt.bar(feature_names, np.abs(weights), color="green", alpha=0.7)
    plt.title(title)
    plt.xlabel("Features")
    plt.ylabel("Absolute Weight")
    plt.xticks(rotation=45)
    plt.savefig(filename)
    plt.close()
