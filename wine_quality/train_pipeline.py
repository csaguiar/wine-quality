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
    params = {
        "alpha": trial.suggest_float("alpha", 0.0, 1.0, step=0.1),
        "l1_ratio": trial.suggest_float("l1_ratio", 0.0, 1.0, step=0.1),
    }
    reg_model = model_utils.build_model(model_name, params)
    rmse, _, _ = model_utils.train_and_evaluate(
        reg_model, x_train, y_train, x_test, y_test
    )
    return rmse


if __name__ == "__main__":
    model_name = "elastic_net"
    date_run = datetime.now().strftime("%Y%m%d_%H%M%S")
    df = data_utils.load_data()
    df = data_utils.clean_data(df)
    x, y = data_utils.prepare_data(df)
    x_train, x_test, y_train, y_test = data_utils.split_data(x, y)

    def optimize(trial):
        return objective(trial, x_train, y_train, x_test, y_test)

    study = optuna.create_study(
        direction="minimize",
        study_name=f"elastic_net_{date_run}"
    )
    study.optimize(optimize, n_trials=100, callbacks=[mlflc])
