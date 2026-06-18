"""End-to-end training script for housing price regression.

Usage:
    python train.py

Trains three models (linear baseline, random forest, gradient boosting),
prints a metrics table to stdout, and saves diagnostic plots to outputs/.
"""

from pathlib import Path

from src.data import load_dataset, split_data, preprocess
from src.features import engineer_features
from src.models import train_linear, train_random_forest, train_gradient_boosting
from src.evaluate import compute_metrics, cross_validate
from src.plot import plot_feature_importance, plot_residuals

OUTPUT_DIR = Path("outputs")


def main() -> None:
    """Run the full training pipeline."""
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Load and split
    X, y = load_dataset()
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Feature engineering
    X_train = engineer_features(X_train)
    X_test = engineer_features(X_test)

    # Scale (linear model needs it; trees don't but it doesn't hurt)
    X_train_scaled, X_test_scaled = preprocess(X_train, X_test)

    # Train
    linear = train_linear(X_train_scaled, y_train)
    rf = train_random_forest(X_train, y_train)
    gb = train_gradient_boosting(X_train, y_train)

    # Evaluate
    models = {"linear": (linear, X_test_scaled), "random_forest": (rf, X_test), "gradient_boosting": (gb, X_test)}
    print(f"{'model':<20} {'rmse':>8} {'mae':>8} {'r2':>8}")
    print("-" * 48)
    for name, (model, X_eval) in models.items():
        metrics = compute_metrics(model, X_eval, y_test)
        print(f"{name:<20} {metrics['rmse']:>8.4f} {metrics['mae']:>8.4f} {metrics['r2']:>8.4f}")

    # Plots
    plot_feature_importance(rf, list(X_train.columns), OUTPUT_DIR / "rf_importance.png")
    plot_residuals(y_test, gb.predict(X_test), "gradient_boosting", OUTPUT_DIR / "gb_residuals.png")
    print(f"\nPlots saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
