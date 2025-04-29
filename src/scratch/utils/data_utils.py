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
    X : array-like or pd.DataFrame
        Input feature matrix.
    y : array-like or pd.Series
        Target array.
    test_size : float, optional
        Proportion of the dataset to include as the test set (default is 0.2).

    Returns
    -------
    tuple
        (X_train, X_test, y_train, y_test) where:
        - X_train and y_train represent the training features and targets.
        - X_test and y_test represent the test features and targets.
    """
    indices = np.random.permutation(len(X))
    test_set_size = int(len(X) * test_size)
    test_indices = indices[:test_set_size]
    train_indices = indices[test_set_size:]

    # Use .iloc if X is a DataFrame; otherwise, index directly
    if isinstance(X, pd.DataFrame):
        X_train = X.iloc[train_indices]
        X_test = X.iloc[test_indices]
    else:
        X_train = X[train_indices]
        X_test = X[test_indices]

    if isinstance(y, (pd.Series, pd.DataFrame)):
        y_train = y.iloc[train_indices]
        y_test = y.iloc[test_indices]
    else:
        y_train = y[train_indices]
        y_test = y[test_indices]

    return X_train, X_test, y_train, y_test


def shuffle_data_pandas(data):
    """
    Shuffle data using pandas sample method.

    Parameters
    ----------
    data : pd.DataFrame
        Input DataFrame to shuffle.

    Returns
    -------
    pd.DataFrame
        Shuffled DataFrame.

    Raises
    ------
    TypeError
        If input is not a pandas DataFrame.
    ValueError
        If DataFrame is empty.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    if data.empty:
        raise ValueError("DataFrame is empty")

    return data.sample(frac=1).reset_index(drop=True)


def shuffle_data_numpy(data):
    """
    Shuffle data using numpy permutation.

    Parameters
    ----------
    data : pd.DataFrame
        Input DataFrame to shuffle.

    Returns
    -------
    pd.DataFrame
        Shuffled DataFrame.

    Raises
    ------
    TypeError
        If input is not a pandas DataFrame.
    ValueError
        If DataFrame is empty.
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
    For features with a constant value (zero variance), the denominator is set to 1 to avoid division-by-zero.

    Parameters
    ----------
    X : np.ndarray or pd.DataFrame
        Input data matrix where rows represent samples and columns represent features.

    Returns
    -------
    np.ndarray or pd.DataFrame
        Scaled data with each feature normalized to the [0, 1] range.
    """
    min_val = X.min(axis=0)
    max_val = X.max(axis=0)
    denom = max_val - min_val
    # Avoid division by zero: Replace zeros in denominator with one
    if isinstance(denom, pd.Series):
        denom[denom == 0] = 1
    else:
        denom[denom == 0] = 1
    return (X - min_val) / denom


def handle_missing_values(df, strategy="mean"):
    """
    Handle missing values in a DataFrame using a specified strategy.

    Supported strategies:
    - 'mean': Replace missing values with the column mean.
    - 'median': Replace missing values with the column median.
    - 'mode': Replace missing values with the column mode.
    - 'drop': Remove rows that contain any missing values.
    - 'differentiated': For numeric columns, replace missing values with the median;
        for categorical columns, replace with the mode.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame that may contain missing values.
    strategy : str, optional
        Strategy to handle missing values (default is 'mean'). Must be one of
        {'mean', 'median', 'mode', 'drop', 'differentiated'}.

    Returns
    -------
    pd.DataFrame
        DataFrame with missing values handled according to the specified strategy.

    Raises
    ------
    ValueError
        If the provided strategy is not one of 'mean', 'median', 'mode', 'drop', or 'differentiated'.
    """
    if strategy == "drop":
        return df.dropna()
    elif strategy in ["mean", "median"]:
        return df.fillna(df.mean() if strategy == "mean" else df.median())
    elif strategy == "mode":
        # Use the first mode in case there are multiple modes
        return df.fillna(df.mode().iloc[0])
    elif strategy == "differentiated":
        df_filled = df.copy()
        numeric_cols = df_filled.select_dtypes(include=[np.number]).columns
        categorical_cols = df_filled.select_dtypes(
            include=["object", "category"]
        ).columns
        for col in numeric_cols:
            df_filled[col] = df_filled[col].fillna(df_filled[col].median())
        for col in categorical_cols:
            df_filled[col] = df_filled[col].fillna(df_filled[col].mode().iloc[0])
        return df_filled
    else:
        raise ValueError(
            "Strategy must be 'mean', 'median', 'mode', 'drop', or 'differentiated'"
        )


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
