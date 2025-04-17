import numpy as np
import pandas as pd


def load_data(file_path):
    """Load a CSV dataset into a pandas DataFrame."""
    return pd.read_csv(file_path)


def split_data(X, y, test_size=0.2):
    """Split data into training and test sets."""
    indices = np.random.permutation(len(X))
    test_set_size = int(len(X) * test_size)
    test_indices = indices[:test_set_size]
    train_indices = indices[test_set_size:]
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]


def normalize(X):
    """Normalize features to have zero mean and unit variance."""
    return (X - X.mean(axis=0)) / X.std(axis=0)


def handle_missing_values(df, strategy='mean'):
    """
    Handle missing values in the DataFrame.
    Strategies: 'mean', 'median', 'mode', 'drop'.
    """
    if strategy == 'drop':
        return df.dropna()
    elif strategy in ['mean', 'median']:
        return df.fillna(df.mean() if strategy == 'mean' else df.median())
    elif strategy == 'mode':
        return df.fillna(df.mode().iloc[0])
    else:
        raise ValueError("Strategy must be 'mean', 'median', 'mode', or 'drop'")
    

def encode_categorical(df):
    """
    Encode categorical columns in the DataFrame using pandas factorize.
    Returns the DataFrame with encoded columns.
    """
    df_encoded = df.copy()
    for col in df_encoded.select_dtypes(include=['object', 'category']).columns:
        df_encoded[col], _ = pd.factorize(df_encoded[col].astype(str))
    return df_encoded


def drop_columns(df, columns):
    """
    Drop specified columns from the DataFrame.
    Useful for removing irrelevant or redundant features.
    """
    return df.drop(columns=columns, errors='ignore')


def feature_target_split(df, target_column):
    """
    Split DataFrame into features (X) and target (y).
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y
