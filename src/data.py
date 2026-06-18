"""Data loading and preprocessing for housing price regression."""

import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split


def load_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """Load the California housing dataset from sklearn.

    Returns a tuple of (X, y) where X is a DataFrame of features
    and y is a Series of median house values (in $100k units).
    """
    housing = fetch_california_housing(as_frame=True)
    X = housing.data.copy()
    y = housing.target.copy()
    y.name = "MedHouseVal"
    return X, y


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split X and y into train and test sets.

    Args:
        X: Feature DataFrame.
        y: Target Series.
        test_size: Fraction of data held out for testing.
        random_state: Seed for reproducibility.

    Returns:
        (X_train, X_test, y_train, y_test)
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def preprocess(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Scale features using standard normalization fit on the training set.

    Fits a StandardScaler on X_train and applies it to both X_train and X_test.
    Returns scaled DataFrames with the original column names preserved.

    Args:
        X_train: Training feature DataFrame.
        X_test: Test feature DataFrame.

    Returns:
        (X_train_scaled, X_test_scaled)
    """
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index,
    )
    return X_train_scaled, X_test_scaled
