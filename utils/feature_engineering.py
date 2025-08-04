import joblib
import pandas as pd
import numpy as np
from functools import lru_cache
from pydantic import BaseModel


TIME_MAP = {'Morning': 0, 'Afternoon': 1, 'Evening': 2, 'Night': 3}
DAY_MAP = {'Mon': 0, 'Tue': 1, 'Wed': 2, 'Thu': 3, 'Fri': 4, 'Sat': 5, 'Sun': 6}
ONEHOT_COLS = ['transaction_type', 'location', 'device_type']


def get_time_of_day(hour: int) -> str:
    if 5 <= hour < 12:
        return 'Morning'
    elif 12 <= hour < 17:
        return 'Afternoon'
    elif 17 <= hour < 21:
        return 'Evening'
    else:
        return 'Night'


@lru_cache()
def load_encoder():
    return joblib.load("models/onehot_encoder.pkl")

@lru_cache()
def load_scaler():
    return joblib.load("models/robust_scaler.pkl")


def feature_engineering(data: BaseModel) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        mod_data = data
    elif isinstance(data, dict):
        mod_data = pd.DataFrame([data])
    else:
        mod_data = pd.DataFrame([data.model_dump()])

    encoder = load_encoder()
    scaler = load_scaler()

    # One-hot encoding
    encoded_df = pd.DataFrame(
        encoder.transform(mod_data[ONEHOT_COLS]),
        columns=encoder.get_feature_names_out(ONEHOT_COLS)
    )
    mod_data = pd.concat([mod_data, encoded_df], axis=1)

    # Scaling txn amount
    mod_data["transaction_amount"] = scaler.transform(mod_data[["transaction_amount"]])

    # Cyclic encoding
    hour = mod_data["transaction_time"].dt.hour.iloc[0]
    time_of_day = get_time_of_day(hour)
    mod_data["time_sin"] = np.sin(2 * np.pi * TIME_MAP[time_of_day] / 4)
    mod_data["time_cos"] = np.cos(2 * np.pi * TIME_MAP[time_of_day] / 4)

    day_of_week = mod_data["transaction_time"].dt.strftime("%a").iloc[0]
    mod_data['day_sin'] = np.sin(2 * np.pi * DAY_MAP[day_of_week] / 7)
    mod_data['day_cos'] = np.cos(2 * np.pi * DAY_MAP[day_of_week] / 7)

    mod_data["foreign_or_high_risk"] = (
        mod_data["is_foreign_transaction"] + mod_data["is_high_risk_country"]
    )

    mod_data = mod_data.drop(ONEHOT_COLS + ["transaction_time"], axis=1)

    return mod_data
