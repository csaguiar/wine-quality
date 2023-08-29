import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import logging


logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

CSV_URL = "https://raw.githubusercontent.com/mlflow/mlflow/master/tests/datasets/winequality-red.csv"  # noqa


def load_data() -> pd.DataFrame:
    """
    Load the wine-quality csv file from the URL.

    Returns:
        pd.DataFrame: The wine-quality dataset.
    """
    try:
        data = pd.read_csv(CSV_URL, sep=";")
    except Exception as e:
        logger.exception(
            "Unable to download training & test CSV, "
            "check your internet connection. Error: %s", e
        )

    return data


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the wine-quality dataset.

    Args:
        data (pd.DataFrame): The wine-quality dataset.

    Returns:
        pd.DataFrame: The cleaned wine-quality dataset.
    """
    return data


def prepare_data(data: pd.DataFrame) -> (np.ndarray, np.ndarray):
    """
    Prepare the wine-quality dataset for training.

    Args:
        data (pd.DataFrame): The wine-quality dataset.

    Returns:
        np.ndarray: The features of the wine-quality dataset.
        np.ndarray: The labels of the wine-quality dataset.
    """
    x = data.drop(["quality"], axis=1)
    y = data[["quality"]]

    return x, y


def split_data(
        x: np.ndarray, y: np.ndarray
) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    """
    Split the wine-quality dataset into training and testing sets.

    Args:
        x (np.ndarray): The features of the wine-quality dataset.
        y (np.ndarray): The labels of the wine-quality dataset.

    Returns:
        np.ndarray: The training features.
        np.ndarray: The testing features.
        np.ndarray: The training labels.
        np.ndarray: The testing labels.
    """
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.25)

    return train_x, test_x, train_y, test_y
