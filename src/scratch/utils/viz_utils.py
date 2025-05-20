import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_learning_curve(
    loss_history,
    title="Learning Curve",
    filename="learning_curve.png",
    save_path=".",
    save_fig=False,
    show_fig=True,
):
    """
    Plot the learning curve over training iterations.

    Parameters
    ----------
    loss_history : list or array-like
        Sequence of loss values across training iterations.
    title : str, optional
        Title for the plot (default is "Learning Curve").
    filename : str, optional
        Filename for the saved plot (default is "learning_curve.png").
    save_path : str, optional
        Directory path to save the image (default is current directory ".").
    save_fig : bool, optional
        If True, saves the figure to file (default is False).
    show_fig : bool, optional
        If True, displays the figure inline (default is True).
    """
    plt.figure(figsize=(8, 6))
    plt.plot(loss_history, label="Training Loss", color="blue")
    plt.title(title)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    if save_fig:
        save_fullpath = os.path.join(save_path, filename)
        plt.savefig(save_fullpath)
    if show_fig:
        plt.show()
    plt.close()


def plot_decision_boundary(
    model,
    X,
    y,
    title="Decision Boundary",
    filename="decision_boundary.png",
    save_path=".",
    save_fig=False,
    show_fig=True,
):
    """
    Plot the decision boundary for a 2D classification model.

    Parameters
    ----------
    model : object
        Trained model with a predict method.
    X : array-like, shape (n_samples, 2)
        Feature data.
    y : array-like, shape (n_samples,)
        Target labels.
    title : str, optional
        Title of the plot (default is "Decision Boundary").
    filename : str, optional
        Filename for the saved plot (default is "decision_boundary.png").
    save_path : str, optional
        Directory path to save the image (default is current directory ".").
    save_fig : bool, optional
        If True, saves the figure to file (default is False).
    show_fig : bool, optional
        If True, displays the figure inline (default is True).

    Raises
    ------
    ValueError
        If the feature space is not 2D.
    """
    if X.shape[1] != 2:
        raise ValueError("Decision boundary plotting requires 2D feature space")
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.4, cmap="coolwarm")
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors="k", marker="o", cmap="coolwarm")
    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    if save_fig:
        save_fullpath = os.path.join(save_path, filename)
        plt.savefig(save_fullpath)
    if show_fig:
        plt.show()
    plt.close()


def plot_two_decision_boundaries(
    model1,
    model2,
    X,
    y,
    labels=["Model 1", "Model 2"],
    title="Decision Boundaries Comparison",
    filename="decision_boundaries_comparison.png",
    save_path=".",
    save_fig=False,
    show_fig=True,
):
    """
    Plot decision boundaries of two models side-by-side for comparison (2D feature space).

    Parameters
    ----------
    model1 : object
        First trained model with a predict method.
    model2 : object
        Second trained model with a predict method.
    X : array-like, shape (n_samples, 2)
        Feature data.
    y : array-like, shape (n_samples,)
        Target labels.
    labels : list of str, optional
        Labels for the two models (default is ["Model 1", "Model 2"]).
    title : str, optional
        Title of the overall plot (default is "Decision Boundaries Comparison").
    filename : str, optional
        Filename for the saved plot (default is "decision_boundaries_comparison.png").
    save_path : str, optional
        Directory path to save the image (default is current directory ".").
    save_fig : bool, optional
        If True, saves the figure to file (default is False).
    show_fig : bool, optional
        If True, displays the figure inline (default is True).

    Raises
    ------
    ValueError
        If the feature space is not 2D.
    """
    if X.shape[1] != 2:
        raise ValueError("Decision boundary plotting requires 2D feature space")
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
    Z1 = model1.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    Z2 = model2.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].contourf(xx, yy, Z1, alpha=0.4, cmap="coolwarm")
    axes[0].scatter(X[:, 0], X[:, 1], c=y, edgecolors="k", marker="o", cmap="coolwarm")
    axes[0].set_title(f"{labels[0]} Decision Boundary")
    axes[1].contourf(xx, yy, Z2, alpha=0.4, cmap="coolwarm")
    axes[1].scatter(X[:, 0], X[:, 1], c=y, edgecolors="k", marker="o", cmap="coolwarm")
    axes[1].set_title(f"{labels[1]} Decision Boundary")
    plt.suptitle(title)
    if save_fig:
        save_fullpath = os.path.join(save_path, filename)
        plt.savefig(save_fullpath)
    if show_fig:
        plt.show()
    plt.close()


def plot_regression_line(
    model,
    X,
    y,
    title="Regression Line",
    filename="regression_line.png",
    save_path=".",
    save_fig=False,
    show_fig=True,
):
    """
    Plot the regression line for a 1D regression model along with the data points.

    Parameters
    ----------
    model : object
        Trained regression model with a predict method.
    X : array-like, shape (n_samples, 1)
        Feature data.
    y : array-like, shape (n_samples,)
        Target values.
    title : str, optional
        Title of the plot (default is "Regression Line").
    filename : str, optional
        Filename for the saved plot (default is "regression_line.png").
    save_path : str, optional
        Directory path to save the image (default is current directory ".").
    save_fig : bool, optional
        If True, saves the figure to file (default is False).
    show_fig : bool, optional
        If True, displays the figure inline (default is True).

    Raises
    ------
    ValueError
        If the provided feature data is not one-dimensional.
    """
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
    if save_fig:
        save_fullpath = os.path.join(save_path, filename)
        plt.savefig(save_fullpath)
    if show_fig:
        plt.show()
    plt.close()


def plot_two_regression_lines(
    model1,
    model2,
    X,
    y,
    labels=["Model 1", "Model 2"],
    title="Regression Lines Comparison",
    filename="regression_lines_comparison.png",
    save_path=".",
    save_fig=False,
    show_fig=True,
):
    """
    Plot regression lines from two models for 1D regression to compare performance.

    Parameters
    ----------
    model1 : object
        First trained regression model with a predict method.
    model2 : object
        Second trained regression model with a predict method.
    X : array-like, shape (n_samples, 1)
        Feature data.
    y : array-like, shape (n_samples,)
        Target values.
    labels : list of str, optional
        Labels to identify the models (default is ["Model 1", "Model 2"]).
    title : str, optional
        Title of the plot (default is "Regression Lines Comparison").
    filename : str, optional
        Filename for the saved plot (default is "regression_lines_comparison.png").
    save_path : str, optional
        Directory path to save the image (default is current directory ".").
    save_fig : bool, optional
        If True, saves the figure to file (default is False).
    show_fig : bool, optional
        If True, displays the figure inline (default is True).

    Raises
    ------
    ValueError
        If the provided feature data is not one-dimensional.
    """
    if X.shape[1] != 1:
        raise ValueError("Regression line plotting requires 1D feature space")
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, color="blue", label="Data")
    X_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    plt.plot(X_range, model1.predict(X_range), color="red", label=labels[0])
    plt.plot(X_range, model2.predict(X_range), color="green", label=labels[1])
    plt.title(title)
    plt.xlabel("Feature")
    plt.ylabel("Target")
    plt.legend()
    plt.grid(True)
    if save_fig:
        save_fullpath = os.path.join(save_path, filename)
        plt.savefig(save_fullpath)
    if show_fig:
        plt.show()
    plt.close()


def plot_confusion_matrix(
    y_true,
    y_pred,
    title="Confusion Matrix",
    filename="confusion_matrix.png",
    save_path=".",
    save_fig=False,
    show_fig=True,
):
    """
    Plot the confusion matrix for classification results.

    Parameters
    ----------
    y_true : array-like, shape (n_samples,)
        True labels.
    y_pred : array-like, shape (n_samples,)
        Predicted labels.
    title : str, optional
        Title for the plot (default is "Confusion Matrix").
    filename : str, optional
        Filename for the saved plot (default is "confusion_matrix.png").
    save_path : str, optional
        Directory path to save the image (default is current directory ".").
    save_fig : bool, optional
        If True, saves the figure to file (default is False).
    show_fig : bool, optional
        If True, displays the figure inline (default is True).
    """
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    if save_fig:
        save_fullpath = os.path.join(save_path, filename)
        plt.savefig(save_fullpath)
    if show_fig:
        plt.show()
    plt.close()


def plot_prediction_errors(
    model,
    X,
    y,
    title="Prediction Errors",
    filename="prediction_errors.png",
    save_path=".",
    save_fig=False,
    show_fig=True,
):
    """
    Plot a histogram of prediction errors (residuals) for regression analysis.

    Parameters
    ----------
    model : object
        Trained model with a predict method.
    X : array-like, shape (n_samples, n_features)
        Feature data.
    y : array-like, shape (n_samples,)
        True target values.
    title : str, optional
        Title for the plot (default is "Prediction Errors").
    filename : str, optional
        Filename for the saved plot (default is "prediction_errors.png").
    save_path : str, optional
        Directory path to save the image (default is current directory ".").
    save_fig : bool, optional
        If True, saves the figure to file (default is False).
    show_fig : bool, optional
        If True, displays the figure inline (default is True).
    """
    predictions = model.predict(X)
    errors = predictions - y
    plt.figure(figsize=(8, 6))
    plt.hist(errors, bins=30, color="purple", alpha=0.7)
    plt.title(title)
    plt.xlabel("Error")
    plt.ylabel("Frequency")
    plt.grid(True)
    if save_fig:
        save_fullpath = os.path.join(save_path, filename)
        plt.savefig(save_fullpath)
    if show_fig:
        plt.show()
    plt.close()


def plot_feature_importance(
    model,
    feature_names,
    title="Feature Importance",
    filename="feature_importance.png",
    save_path=".",
    save_fig=False,
    show_fig=True,
):
    """
    Plot the feature importance based on the model's weights.

    Parameters
    ----------
    model : object
        Trained model that has a 'weights' attribute.
    feature_names : list of str
        Names of the features.
    title : str, optional
        Title for the plot (default is "Feature Importance").
    filename : str, optional
        Filename for the saved plot (default is "feature_importance.png").
    save_path : str, optional
        Directory path to save the image (default is current directory ".").
    save_fig : bool, optional
        If True, saves the figure to file (default is False).
    show_fig : bool, optional
        If True, displays the figure inline (default is True).
    """
    weights = model.weights[1:]  # Exclude bias term
    plt.figure(figsize=(10, 6))
    plt.bar(feature_names, np.abs(weights), color="green", alpha=0.7)
    plt.title(title)
    plt.xlabel("Features")
    plt.ylabel("Absolute Weight")
    plt.xticks(rotation=45)
    if save_fig:
        save_fullpath = os.path.join(save_path, filename)
        plt.savefig(save_fullpath)
    if show_fig:
        plt.show()
    plt.close()


def plot_actual_vs_predicted(
    y_true,
    y_pred,
    title="Actual vs. Predicted",
    filename="actual_vs_predicted.png",
    save_path=".",
    save_fig=False,
    show_fig=True,
):
    """
    Plot a scatter diagram of actual versus predicted values for regression analysis.

    A reference line representing the ideal prediction (y = x) is also plotted.

    Parameters
    ----------
    y_true : array-like
        True target values.
    y_pred : array-like
        Predicted target values.
    title : str, optional
        Title for the plot (default is "Actual vs. Predicted").
    filename : str, optional
        Filename for the saved plot (default is "actual_vs_predicted.png").
    save_path : str, optional
        Directory path to save the image (default is current directory ".").
    save_fig : bool, optional
        If True, saves the figure to file (default is False).
    show_fig : bool, optional
        If True, displays the figure inline (default is True).
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.7, color="blue", label="Predictions")
    plt.plot(
        [np.min(y_true), np.max(y_true)],
        [np.min(y_true), np.max(y_true)],
        "r--",
        lw=2,
        label="Ideal Fit",
    )
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    if save_fig:
        save_fullpath = os.path.join(save_path, filename)
        plt.savefig(save_fullpath)
    if show_fig:
        plt.show()
    plt.close()


def plot_residuals_vs_predicted(
    y_true,
    y_pred,
    title="Residuals vs. Predicted",
    filename="residuals_vs_predicted.png",
    save_path=".",
    save_fig=False,
    show_fig=True,
):
    """
    Plot residuals versus predicted values to diagnose patterns such as heteroscedasticity or model misspecification.

    Parameters
    ----------
    y_true : array-like
        True target values.
    y_pred : array-like
        Predicted target values.
    title : str, optional
        Title for the plot (default is "Residuals vs. Predicted").
    filename : str, optional
        Filename for the saved plot (default is "residuals_vs_predicted.png").
    save_path : str, optional
        Directory path to save the image (default is current directory ".").
    save_fig : bool, optional
        If True, saves the figure to file (default is False).
    show_fig : bool, optional
        If True, displays the figure inline (default is True).
    """
    residuals = y_true - y_pred
    plt.figure(figsize=(8, 6))
    plt.scatter(y_pred, residuals, alpha=0.7, color="green")
    plt.axhline(y=0, color="black", linestyle="--", linewidth=2)
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.title(title)
    plt.grid(True)
    if save_fig:
        save_fullpath = os.path.join(save_path, filename)
        plt.savefig(save_fullpath)
    if show_fig:
        plt.show()
    plt.close()


def plot_qq_residuals(
    y_true,
    y_pred,
    title="Q-Q Plot of Residuals",
    filename="qq_residuals.png",
    save_path=".",
    save_fig=False,
    show_fig=True,
):
    """
    Generate a Q-Q plot of residuals to assess their normality.

    Parameters
    ----------
    y_true : array-like
        True target values.
    y_pred : array-like
        Predicted target values.
    title : str, optional
        Title for the plot (default is "Q-Q Plot of Residuals").
    filename : str, optional
        Filename for the saved plot (default is "qq_residuals.png").
    save_path : str, optional
        Directory path to save the image (default is current directory ".").
    save_fig : bool, optional
        If True, saves the figure to file (default is False).
    show_fig : bool, optional
        If True, displays the figure inline (default is True).
    """
    import scipy.stats as stats

    residuals = y_true - y_pred
    plt.figure(figsize=(8, 6))
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title(title)
    if save_fig:
        save_fullpath = os.path.join(save_path, filename)
        plt.savefig(save_fullpath)
    if show_fig:
        plt.show()
    plt.close()


def plot_residual_histogram(
    y_true,
    y_pred,
    title="Residual Histogram",
    filename="residual_histogram.png",
    save_path=".",
    save_fig=False,
    show_fig=True,
    bin_num=30,
):
    """
    Plot a histogram of residuals to visualize their distribution.

    Parameters
    ----------
    y_true : array-like
        True target values.
    y_pred : array-like
        Predicted target values.
    title : str, optional
        Title for the histogram (default is "Residual Histogram").
    filename : str, optional
        Filename for the saved plot (default is "residual_histogram.png").
    save_path : str, optional
        Directory path to save the image (default is current directory ".").
    save_fig : bool, optional
        If True, saves the figure to file (default is False).
    show_fig : bool, optional
        If True, displays the figure inline (default is True).
    bin_num : int, default=30
        number of bins in the histogram
    """
    residuals = y_true - y_pred
    plt.figure(figsize=(8, 6))
    plt.hist(residuals, bins=bin_num, color="purple", alpha=0.7)
    plt.title(title)
    plt.xlabel("Residual")
    plt.ylabel("Frequency")
    plt.grid(True)
    if save_fig:
        save_fullpath = os.path.join(save_path, filename)
        plt.savefig(save_fullpath)
    if show_fig:
        plt.show()
    plt.close()


def plot_scatter_for_regression(
    X,
    y,
    feature_index=0,
    title="Feature vs. Target",
    filename="scatter_regression.png",
    save_path=".",
    save_fig=False,
    show_fig=True,
):
    """
    Plot a scatter plot of a selected feature against the target variable for regression datasets.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Feature data.
    y : array-like, shape (n_samples,)
        Target values.
    feature_index : int, optional
        Index of the feature to plot (default is 0).
    title : str, optional
        Title for the plot (default is "Feature vs. Target").
    filename : str, optional
        Filename for the saved plot (default is "scatter_regression.png").
    save_path : str, optional
        Directory path to save the image (default is current directory ".").
    save_fig : bool, optional
        If True, saves the figure to file (default is False).
    show_fig : bool, optional
        If True, displays the figure inline (default is True).

    Raises
    ------
    ValueError
        If feature_index is out of bounds.
    """
    if feature_index >= X.shape[1]:
        raise ValueError("Feature index out of bounds")
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, feature_index], y, alpha=0.7, color="blue")
    plt.xlabel(f"Feature {feature_index + 1}")
    plt.ylabel("Target")
    plt.title(title)
    plt.grid(True)
    if save_fig:
        save_fullpath = os.path.join(save_path, filename)
        plt.savefig(save_fullpath)
    if show_fig:
        plt.show()
    plt.close()


def plot_roc_curve(
    y_true,
    y_scores,
    title="ROC Curve",
    filename="roc_curve.png",
    save_path=".",
    save_fig=False,
    show_fig=True,
):
    """
    Plot the Receiver Operating Characteristic (ROC) curve for binary classification.

    Parameters
    ----------
    y_true : array-like, shape (n_samples,)
        True binary labels.
    y_scores : array-like, shape (n_samples,)
        Predicted probabilities for the positive class.
    title : str, optional
        Title for the plot (default is "ROC Curve").
    filename : str, optional
        Filename for the saved plot (default is "roc_curve.png").
    save_path : str, optional
        Directory path to save the image (default is current directory ".").
    save_fig : bool, optional
        If True, saves the figure to file (default is False).
    show_fig : bool, optional
        If True, displays the figure inline (default is True).
    """
    from sklearn.metrics import auc, roc_curve

    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(
        fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc:.2f})"
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    if save_fig:
        save_fullpath = os.path.join(save_path, filename)
        plt.savefig(save_fullpath)
    if show_fig:
        plt.show()
    plt.close()


def plot_precision_recall_curve(
    y_true,
    y_scores,
    title="Precision-Recall Curve",
    filename="precision_recall_curve.png",
    save_path=".",
    save_fig=False,
    show_fig=True,
):
    """
    Plot the Precision-Recall curve for binary classification.

    Parameters
    ----------
    y_true : array-like, shape (n_samples,)
        True binary labels.
    y_scores : array-like, shape (n_samples,)
        Predicted probabilities for the positive class.
    title : str, optional
        Title for the plot (default is "Precision-Recall Curve").
    filename : str, optional
        Filename for the saved plot (default is "precision_recall_curve.png").
    save_path : str, optional
        Directory path to save the image (default is current directory ".").
    save_fig : bool, optional
        If True, saves the figure to file (default is False).
    show_fig : bool, optional
        If True, displays the figure inline (default is True).
    """
    from sklearn.metrics import precision_recall_curve

    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color="b", lw=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.grid(True)
    if save_fig:
        save_fullpath = os.path.join(save_path, filename)
        plt.savefig(save_fullpath)
    if show_fig:
        plt.show()
    plt.close()


def plot_scatter_for_classification(
    X,
    y,
    feature_indices=(0, 1),
    title="Classification Scatter Plot",
    filename="scatter_classification.png",
    save_path=".",
    save_fig=False,
    show_fig=True,
):
    """
    Plot a scatter plot of two selected features colored by class for classification datasets.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Feature data.
    y : array-like, shape (n_samples,)
        Target labels.
    feature_indices : tuple of int, optional
        Indices of the two features to plot (default is (0, 1)).
    title : str, optional
        Title for the plot (default is "Classification Scatter Plot").
    filename : str, optional
        Filename for the saved plot (default is "scatter_classification.png").
    save_path : str, optional
        Directory path to save the image (default is current directory ".").
    save_fig : bool, optional
        If True, saves the figure to file (default is False).
    show_fig : bool, optional
        If True, displays the figure inline (default is True).

    Raises
    ------
    ValueError
        If exactly two feature indices are not provided or if indices are out of bounds.
    """
    if len(feature_indices) != 2:
        raise ValueError("Exactly two feature indices are required")
    if max(feature_indices) >= X.shape[1]:
        raise ValueError("Feature index out of bounds")
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        X[:, feature_indices[0]],
        X[:, feature_indices[1]],
        c=y,
        cmap="coolwarm",
        alpha=0.7,
    )
    plt.xlabel(f"Feature {feature_indices[0] + 1}")
    plt.ylabel(f"Feature {feature_indices[1] + 1}")
    plt.title(title)
    plt.legend(*scatter.legend_elements(), title="Classes")
    if save_fig:
        save_fullpath = os.path.join(save_path, filename)
        plt.savefig(save_fullpath)
    if show_fig:
        plt.show()
    plt.close()


def plot_pairplot(
    data,
    hue=None,
    title="Pairplot",
    filename="pairplot.png",
    save_path=".",
    save_fig=False,
    show_fig=True,
):
    """
    Generate a pairplot for exploratory data analysis, showing pairwise relationships between features.

    Parameters
    ----------
    data : pandas DataFrame
        Dataset to plot.
    hue : str, optional
        Column name in data to map plot aspects to different colors (default is None).
    title : str, optional
        Title for the plot (default is "Pairplot").
    filename : str, optional
        Filename for the saved plot (default is "pairplot.png").
    save_path : str, optional
        Directory path to save the image (default is current directory ".").
    save_fig : bool, optional
        If True, saves the figure to file (default is False).
    show_fig : bool, optional
        If True, displays the figure inline (default is True).
    """
    plt.figure(figsize=(10, 10))
    sns.pairplot(data, hue=hue)
    plt.suptitle(title, y=1.02)
    if save_fig:
        save_fullpath = os.path.join(save_path, filename)
        plt.savefig(save_fullpath)
    if show_fig:
        plt.show()
    plt.close()


def plot_elbow_method(
    k_range,
    inertias,
    title="Elbow Method",
    filename="elbow_method.png",
    save_path=".",
    save_fig=False,
    show_fig=True,
):
    """
    Plot the elbow method curve for K-Means clustering.

    Parameters
    ----------
    k_range : list or array-like
        Range of k values.
    inertias : list or array-like
        Corresponding inertias for each k.
    title : str, optional
        Title of the plot (default is "Elbow Method").
    filename : str, optional
        Filename for the saved plot (default is "elbow_method.png").
    save_path : str, optional
        Directory path to save the image (default is current directory ".").
    save_fig : bool, optional
        If True, saves the figure to file (default is False).
    show_fig : bool, optional
        If True, displays the figure inline (default is True).

    Raises
    ------
    ValueError
        If k_range and inertias are not of the same length.
    """
    # Input validation
    if len(k_range) != len(inertias):
        raise ValueError("k_range and inertias must be of the same length")

    plt.figure(figsize=(8, 6))
    plt.plot(k_range, inertias, marker="o")  # Fixed marker parameter
    plt.title(title)
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Inertia")
    plt.grid(True)
    if save_fig:
        save_fullpath = os.path.join(save_path, filename)
        plt.savefig(save_fullpath)
    if show_fig:
        plt.show()
    plt.close()


def plot_clusters(
    X,
    labels,
    centroids,
    title="K-Means Clustering",
    filename="clusters.png",
    save_path=".",
    save_fig=False,
    show_fig=True,
):
    """
    Plot the clusters and centroids for 2D data, including a legend for clusters and centroids.

    Parameters
    ----------
    X : array-like, shape (n_samples, 2)
        Feature data.
    labels : array-like, shape (n_samples,)
        Cluster labels for each point.
    centroids : array-like, shape (n_clusters, 2)
        Coordinates of cluster centroids.
    title : str, optional
        Title of the plot (default is "K-Means Clustering").
    filename : str, optional
        Filename for the saved plot (default is "clusters.png").
    save_path : str, optional
        Directory path to save the image (default is current directory ".").
    save_fig : bool, optional
        If True, saves the figure to file (default is False).
    show_fig : bool, optional
        If True, displays the figure inline (default is True).

    Raises
    ------
    ValueError
        If the feature space is not 2D.
    """
    if X.shape[1] != 2:
        raise ValueError("Plotting clusters requires 2D feature space")

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="viridis", alpha=0.6)
    centroid_scatter = plt.scatter(
        centroids[:, 0], centroids[:, 1], c="red", marker="x", s=100
    )

    # Get legend elements for clusters
    handles, numeric_labels = scatter.legend_elements()
    cluster_labels = [f"Cluster {label}" for label in numeric_labels]

    # Add centroid to the legend
    handles.append(centroid_scatter)
    cluster_labels.append("Centroids")

    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend(handles, cluster_labels)

    if save_fig:
        save_fullpath = os.path.join(save_path, filename)
        plt.savefig(save_fullpath)
    if show_fig:
        plt.show()
    plt.close()
