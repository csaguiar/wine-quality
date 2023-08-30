import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.base import BaseEstimator
from wine_quality.utils import metrics as metrics_utils
import optuna


MODELS = {
    "elastic_net": ElasticNet
}


def build_model(model_name: str, params: dict) -> BaseEstimator:
    """
    Build the model based on the model name and parameters.

    Args:
        model_name (str): The name of the model.
        params (dict): The parameters of the model.

    Returns:
        BaseEstimator: The model.
    """
    model_class = MODELS.get(model_name)

    if model_class is None:
        raise ValueError(f"Unknown model: {model_name}")

    model = model_class(**params)
    return model


def available_models() -> list:
    """
    Get the list of available models.

    Returns:
        list: The list of available models.
    """
    return list(MODELS.keys())


def get_default_params(model_name: str, trial: optuna.trial.Trial) -> dict:
    if model_name == "elastic_net":
        return {
            "alpha": trial.suggest_float("alpha", 0.05, 1.0, step=0.05),
            "l1_ratio": trial.suggest_float("l1_ratio", 0.05, 1.0, step=0.05),
        }


def train_and_evaluate(
    model: BaseEstimator, x_train: np.ndarray, y_train: np.ndarray,
    x_test: np.ndarray, y_test: np.ndarray
) -> (float, float, float):
    """
    Train and evaluate the model.

    Args:
        model (BaseEstimator): The model to train and evaluate.
        x_train (np.ndarray): The training input data.
        y_train (np.ndarray): The training target data.
        x_test (np.ndarray): The testing input data.
        y_test (np.ndarray): The testing target data.

    Returns:
        tuple: A tuple containing the RMSE, MAE, and R2 score of the model.
    """
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    rmse, mae, r2 = metrics_utils.eval_metrics(y_test, y_pred)
    return rmse, mae, r2
