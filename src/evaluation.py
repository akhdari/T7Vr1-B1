"""Regression metrics used by the later modeling phase."""

from math import sqrt

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    """Compute standard regression metrics with RMSE named explicitly."""
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }
