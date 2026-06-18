"""Plotting utilities: feature importance and residual diagnostics."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.base import RegressorMixin


def plot_feature_importance(
    model: RegressorMixin,
    feature_names: list[str],
    output_path: Path | None = None,
) -> plt.Figure:
    """Plot horizontal bar chart of feature importances.

    Works with any model that has a .feature_importances_ attribute
    (RandomForest, GradientBoosting) or .coef_ (linear models).

    Args:
        model: Fitted regressor.
        feature_names: Column names in the same order as the training data.
        output_path: If provided, save the figure to this path.

    Returns:
        Matplotlib Figure object.
    """
    raise NotImplementedError("implement in later commit")


def plot_residuals(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "model",
    output_path: Path | None = None,
) -> plt.Figure:
    """Plot residuals vs predicted values and a Q-Q plot for normality check.

    Args:
        y_true: Ground truth target values.
        y_pred: Model predictions.
        model_name: Used in the plot title.
        output_path: If provided, save the figure to this path.

    Returns:
        Matplotlib Figure object.
    """
    raise NotImplementedError("implement in later commit")
