import joblib
import numpy as np
import pandas as pd


try:
    model = joblib.load("models/xgb_model.pkl")
except FileNotFoundError:
    raise RuntimeError("Model file not found. Ensure 'xgb_model.pkl' exists in the 'models' directory.")


def predict(data: pd.DataFrame) -> str:
    """
    Predicts fraud or not fraud from transaction data.

    Args:
        data (pd.DataFrame): Processed data

    Returns:
        int: 0 or 1
    """
    if not isinstance(data, (pd.DataFrame, np.ndarray)):
        raise ValueError("Input to predict must be a pandas DataFrame or NumPy array")

    try:
        prediction_class = model.predict(data)[0]
    except Exception as e:
        raise RuntimeError(f"Prediction failed: {e}")

    return prediction_class