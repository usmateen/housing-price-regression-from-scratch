"""Model training wrappers for linear, random forest, and gradient boosting."""

import pandas as pd
import numpy as np
from sklearn.base import RegressorMixin


def train_linear(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> RegressorMixin:
    """Fit a Ridge regression model on the training data.

    Uses cross-validated alpha selection via RidgeCV with a log-spaced grid.

    Args:
        X_train: Scaled training features.
        y_train: Training targets.

    Returns:
        Fitted RidgeCV model.
    """
    raise NotImplementedError("implement in later commit")


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 200,
    random_state: int = 42,
) -> RegressorMixin:
    """Fit a RandomForestRegressor on the training data.

    Args:
        X_train: Training features.
        y_train: Training targets.
        n_estimators: Number of trees.
        random_state: Seed for reproducibility.

    Returns:
        Fitted RandomForestRegressor.
    """
    raise NotImplementedError("implement in later commit")


def train_gradient_boosting(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 300,
    learning_rate: float = 0.05,
    random_state: int = 42,
) -> RegressorMixin:
    """Fit a HistGradientBoostingRegressor on the training data.

    Uses sklearn's HistGradientBoostingRegressor for speed on larger datasets.

    Args:
        X_train: Training features.
        y_train: Training targets.
        n_estimators: Number of boosting iterations.
        learning_rate: Step size shrinkage.
        random_state: Seed for reproducibility.

    Returns:
        Fitted HistGradientBoostingRegressor.
    """
    raise NotImplementedError("implement in later commit")
