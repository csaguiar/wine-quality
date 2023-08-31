import numpy as np
import wine_quality.utils.model as model_utils
import wine_quality.utils.data as data_utils
import optuna
import mlflow
from optuna.integration.mlflow import MLflowCallback
from datetime import datetime

TRACKING_URI = "http://127.0.0.1:5000"
mlflow.set_tracking_uri(TRACKING_URI)
tracking_uri = mlflow.get_tracking_uri()

mlflc = MLflowCallback(
    tracking_uri=tracking_uri,
    metric_name="rmse",
)


def objective(
    trial: optuna.trial.Trial, x_train: np.ndarray, y_train: np.ndarray,
    x_test: np.ndarray, y_test: np.ndarray
) -> float:
    """
    Optuna objective function for hyperparameter tuning of a regression model.

    Args:
        trial: An Optuna `Trial` object used to sample hyperparameters.
        x_train: A numpy array of shape `(n_samples, n_features)` containing
            the training data.
        y_train: A numpy array of shape `(n_samples,)` containing the target
            values for the training data.
        x_test: A numpy array of shape `(n_samples, n_features)` containing
            the test data.
        y_test: A numpy array of shape `(n_samples,)` containing the target
            values for the test data.

    Returns:
        The root mean squared error (RMSE) of the regression model on the test
            data.
    """
    available_models = model_utils.available_models()
    model_name = trial.suggest_categorical("model_name", available_models)
    params = model_utils.get_default_params(model_name, trial)
    model = model_utils.build_model(model_name, params)
    mlflow.sklearn.log_model(model, "model")
    model.fit(x_train, y_train)
    rmse, _, _ = model_utils.evaluate(model, x_test, y_test)
    return rmse


if __name__ == "__main__":
    model_name = "elastic_net"
    date_run = datetime.now().strftime("%Y%m%d_%H%M%S")
    df = data_utils.load_data()
    df = data_utils.clean_data(df)
    x, y = data_utils.prepare_data(df)
    x_train, x_test, y_train, y_test = data_utils.split_data(x, y)

    @mlflc.track_in_mlflow()
    def optimize(trial):
        return objective(trial, x_train, y_train, x_test, y_test)

    study = optuna.create_study(
        direction="minimize",
        study_name=f"elastic_net_{date_run}"
    )
    study.optimize(optimize, n_trials=100, callbacks=[mlflc])
