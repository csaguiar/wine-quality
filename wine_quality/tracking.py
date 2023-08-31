import mlflow
from optuna.integration.mlflow import MLflowCallback


TRACKING_URI = "http://127.0.0.1:5000"


def initialize_experiment():
    """
    Initializes MLflow tracking by setting the tracking URI and creating an
    MLflow callback.

    Returns:
        An MLflow callback object.
    """
    mlflow.set_tracking_uri(TRACKING_URI)
    tracking_uri = mlflow.get_tracking_uri()

    mlflc = MLflowCallback(
        tracking_uri=tracking_uri,
        metric_name="rmse",
    )

    return mlflc


def log_model(model, name):
    """
    Logs a trained machine learning model to MLflow.

    Args:
        model: A trained machine learning model.
        name: A name for the model.

    Returns:
        None
    """
    mlflow.sklearn.log_model(model, name)
