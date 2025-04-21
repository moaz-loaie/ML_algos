import numpy as np
import pandas as pd


def load_data(file_path):
    """
    Load a CSV dataset into a pandas DataFrame.

    Parameters
    ----------
    file_path : str
        The path to the CSV file to be loaded.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the loaded data.
    """
    return pd.read_csv(file_path)


def split_data(X, y, test_size=0.2):
    """
    Split data into training and test sets.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Input feature matrix.
    y : array-like, shape (n_samples,)
        Target array.
    test_size : float, optional
        Proportion of the dataset to include as the test set (default is 0.2).

    Returns
    -------
    tuple
        A tuple containing four elements: (X_train, X_test, y_train, y_test) where
        - X_train and y_train represent the training features and targets.
        - X_test and y_test represent the test features and targets.
    """
    indices = np.random.permutation(len(X))
    test_set_size = int(len(X) * test_size)
    test_indices = indices[:test_set_size]
    train_indices = indices[test_set_size:]
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]


def shuffle_data_pandas(data):
    """
    Shuffle data using pandas sample method.

    Args:
        data (pd.DataFrame): Input DataFrame to shuffle

    Returns:
        pd.DataFrame: Shuffled DataFrame

    Raises:
        TypeError: If input is not a pandas DataFrame
        ValueError: If DataFrame is empty
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    if data.empty:
        raise ValueError("DataFrame is empty")

    return data.sample(frac=1).reset_index(drop=True)


def shuffle_data_numpy(data):
    """
    Shuffle data using numpy permutation.

    Args:
        data (pd.DataFrame): Input DataFrame to shuffle

    Returns:
        pd.DataFrame: Shuffled DataFrame

    Raises:
        TypeError: If input is not a pandas DataFrame
        ValueError: If DataFrame is empty
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    if data.empty:
        raise ValueError("DataFrame is empty")

    shuffled_indices = np.random.permutation(len(data))
    return data.iloc[shuffled_indices].reset_index(drop=True)


def normalize(X):
    """
    Scale features to the range [0, 1].

    Each feature column in X is transformed so that its minimum value becomes 0 and its maximum becomes 1.

    Parameters
    ----------
    X : np.ndarray
        Input data matrix where rows represent samples and columns represent features.

    Returns
    -------
    np.ndarray
        Scaled data with each feature normalized to the [0, 1] range.

    Notes
    -----
    If a feature has constant value (zero variance), a division-by-zero error may occur.
    """
    min_val = X.min(axis=0)
    max_val = X.max(axis=0)
    return (X - min_val) / (max_val - min_val)


def handle_missing_values(df, strategy="mean"):
    """
    Handle missing values in a DataFrame using a specified strategy.

    Supported strategies:
    - 'mean': Replace missing values with the column mean.
    - 'median': Replace missing values with the column median.
    - 'mode': Replace missing values with the column mode.
    - 'drop': Remove rows that contain any missing values.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame that may contain missing values.
    strategy : str, optional
        Strategy to handle missing values (default is 'mean'). Must be one of
        {'mean', 'median', 'mode', 'drop'}.

    Returns
    -------
    pd.DataFrame
        DataFrame with missing values handled according to the specified strategy.

    Raises
    ------
    ValueError
        If the provided strategy is not one of 'mean', 'median', 'mode', or 'drop'.
    """
    if strategy == "drop":
        return df.dropna()
    elif strategy in ["mean", "median"]:
        return df.fillna(df.mean() if strategy == "mean" else df.median())
    elif strategy == "mode":
        # Use the first mode in case there are multiple modes
        return df.fillna(df.mode().iloc[0])
    else:
        raise ValueError("Strategy must be 'mean', 'median', 'mode', or 'drop'")


def encode_categorical(df):
    """
    Encode categorical columns in a DataFrame using pandas' factorize.

    Each categorical column (columns of type 'object' or 'category') is replaced
    with numerical codes.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with categorical columns.

    Returns
    -------
    pd.DataFrame
        A copy of the original DataFrame with all categorical columns encoded.
    """
    df_encoded = df.copy()
    for col in df_encoded.select_dtypes(include=["object", "category"]).columns:
        df_encoded[col], _ = pd.factorize(df_encoded[col].astype(str))
    return df_encoded


def drop_columns(df, columns):
    """
    Drop specified columns from a DataFrame.

    This function removes the columns listed in 'columns' from the DataFrame. If a column
    is not present, it is ignored.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    columns : list of str
        List of column names to drop.

    Returns
    -------
    pd.DataFrame
        DataFrame with the specified columns removed.
    """
    return df.drop(columns=columns, errors="ignore")


def feature_target_split(df, target_column):
    """
    Split a DataFrame into features and target.

    Separates the target column from the DataFrame and returns the remaining columns
    as features.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    target_column : str
        Name of the target column.

    Returns
    -------
    tuple
        A tuple (X, y) where:
        - X is a DataFrame of features,
        - y is a Series containing the target variable.
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y
