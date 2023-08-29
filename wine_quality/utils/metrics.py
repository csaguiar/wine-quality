import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def eval_metrics(
    actual: np.ndarray, pred: np.ndarray
) -> (float, float, float):
    """
    Calculates the root mean squared error (RMSE), mean absolute error (MAE),
    and R-squared (R2) between the actual and predicted values.

    Args:
        actual (np.ndarray): Array of actual values.
        pred (np.ndarray): Array of predicted values.

    Returns:
        Tuple of floats: RMSE, MAE, and R2.
    """
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2
