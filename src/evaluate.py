"""Evaluation metrics and cross-validation utilities."""

import pandas as pd
import numpy as np
from sklearn.base import RegressorMixin


def compute_metrics(
    model: RegressorMixin,
    X: pd.DataFrame,
    y: pd.Series,
) -> dict[str, float]:
    """Compute RMSE, MAE, and R^2 for a fitted model on (X, y).

    Args:
        model: Any fitted sklearn regressor.
        X: Feature DataFrame.
        y: True target values.

    Returns:
        Dict with keys "rmse", "mae", "r2".
    """
    raise NotImplementedError("implement in later commit")


def cross_validate(
    model: RegressorMixin,
    X: pd.DataFrame,
    y: pd.Series,
    cv: int = 5,
) -> dict[str, float]:
    """Run k-fold cross-validation and return mean +/- std for RMSE and R^2.

    Args:
        model: Unfitted sklearn regressor (cloned internally per fold).
        X: Full feature DataFrame.
        y: Full target Series.
        cv: Number of folds.

    Returns:
        Dict with keys "rmse_mean", "rmse_std", "r2_mean", "r2_std".
    """
    raise NotImplementedError("implement in later commit")
