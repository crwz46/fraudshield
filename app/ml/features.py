"""
Feature engineering pipeline for fraud detection.
Transforms raw transactions into ML-ready features.
"""

import numpy as np
import pandas as pd
from typing import Tuple
from sklearn.preprocessing import StandardScaler, LabelEncoder


class FeatureEngineer:
    def __init__(self):
        self.scalers: dict = {}
        self.encoders: dict = {}
        self._fitted = False

    def fit_transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        return self._transform(df, fit=True)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self._transform(df, fit=False)[0]

    def _transform(self, df: pd.DataFrame, fit: bool) -> Tuple[pd.DataFrame, pd.Series]:
        data = df.copy()

        # Temporal features
        data["hour_sin"] = np.sin(2 * np.pi * data["hour_of_day"] / 24)
        data["hour_cos"] = np.cos(2 * np.pi * data["hour_of_day"] / 24)
        data["day_sin"] = np.sin(2 * np.pi * data["day_of_week"] / 7)
        data["day_cos"] = np.cos(2 * np.pi * data["day_of_week"] / 7)

        # Ratio features — high value signals
        data["amount_to_avg_ratio"] = data["amount"] / (data["avg_transaction_amount_7d"] + 0.01)
        data["distance_to_amount_ratio"] = data["distance_from_home_km"] / (data["amount"] + 0.01)
        data["velocity_density"] = data["transactions_last_24h"] / (data["days_since_last_transaction"] + 1)
        data["failed_success_ratio"] = data["failed_attempts_last_hour"] / (data["transactions_last_24h"] + 0.01)

        # Interaction features — fraud pattern signatures
        data["night_high_amount"] = ((data["hour_of_day"] < 6) | (data["hour_of_day"] > 23)) & (data["amount"] > 500)
        data["new_device_high_amount"] = (data["age_days"] < 30) & (data["amount"] > 1000)
        data["foreign_large_txn"] = (~data["ip_country_match"].astype(bool)) & (data["amount"] > 2000)
        data["rapid_small_txns"] = (data["velocity_last_hour"] > 5) & (data["amount"] < 30)

        # Log transform skewed features
        data["amount_log"] = np.log1p(data["amount"])
        data["distance_log"] = np.log1p(data["distance_from_home_km"])
        data["age_log"] = np.log1p(data["age_days"])
        data["tenure_log"] = np.log1p(data["account_tenure_days"])

        # Encode categoricals
        cat_cols = ["merchant_category", "merchant_country", "card_type", "device_id"]
        for col in cat_cols:
            if col in data.columns:
                if fit:
                    self.encoders[col] = LabelEncoder()
                    data[col + "_encoded"] = self.encoders[col].fit_transform(data[col].astype(str))
                else:
                    data[col + "_encoded"] = data[col].astype(str).map(
                        lambda x: self.encoders[col].transform([x])[0]
                        if x in self.encoders[col].classes_ else -1
                    )

        # Select features for model
        feature_cols = [
            "amount_log", "distance_log", "age_log", "tenure_log",
            "hour_sin", "hour_cos", "day_sin", "day_cos",
            "amount_to_avg_ratio", "distance_to_amount_ratio",
            "velocity_density", "failed_success_ratio",
            "night_high_amount", "new_device_high_amount",
            "foreign_large_txn", "rapid_small_txns",
            "card_present", "ip_country_match",
            "transactions_last_24h", "velocity_last_hour",
            "days_since_last_transaction", "failed_attempts_last_hour",
        ] + [c + "_encoded" for c in cat_cols if c + "_encoded" in data.columns]

        # Ensure all feature columns exist
        for col in feature_cols:
            if col not in data.columns:
                data[col] = 0

        X = data[feature_cols].fillna(0)

        # Scale numeric features
        numeric_cols = ["amount_log", "distance_log", "age_log", "tenure_log",
                        "amount_to_avg_ratio", "distance_to_amount_ratio",
                        "velocity_density", "failed_success_ratio",
                        "transactions_last_24h", "velocity_last_hour",
                        "days_since_last_transaction", "failed_attempts_last_hour"]

        if fit:
            self.scaler = StandardScaler()
            X[numeric_cols] = self.scaler.fit_transform(X[numeric_cols])
            self._fitted = True
        else:
            X[numeric_cols] = self.scaler.transform(X[numeric_cols])

        y = data["is_fraud"] if "is_fraud" in data.columns else None

        return X, y

    def get_feature_names(self) -> list:
        return [
            "amount_log", "distance_log", "age_log", "tenure_log",
            "hour_sin", "hour_cos", "day_sin", "day_cos",
            "amount_to_avg_ratio", "distance_to_amount_ratio",
            "velocity_density", "failed_success_ratio",
            "night_high_amount", "new_device_high_amount",
            "foreign_large_txn", "rapid_small_txns",
            "card_present", "ip_country_match",
            "transactions_last_24h", "velocity_last_hour",
            "days_since_last_transaction", "failed_attempts_last_hour",
            "merchant_category_encoded", "merchant_country_encoded",
            "card_type_encoded", "device_id_encoded",
        ]
