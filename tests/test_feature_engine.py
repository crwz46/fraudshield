import pytest
import numpy as np
import pandas as pd
from app.ml.features import FeatureEngineer


@pytest.fixture
def sample_transactions():
    np.random.seed(42)
    n = 50
    return pd.DataFrame({
        "transaction_id": [f"txn_{i}" for i in range(n)],
        "amount": np.random.exponential(100, n),
        "merchant_category": np.random.choice(["retail", "food", "travel", "entertainment"], n),
        "merchant_country": np.random.choice(["US", "DE", "JP", "BR"], n),
        "card_present": np.random.choice([True, False], n),
        "distance_from_home_km": np.random.exponential(50, n),
        "hour_of_day": np.random.randint(0, 24, n),
        "day_of_week": np.random.randint(0, 7, n),
        "days_since_last_transaction": np.random.randint(0, 60, n),
        "transactions_last_24h": np.random.randint(0, 20, n),
        "avg_transaction_amount_7d": np.random.exponential(100, n),
        "velocity_last_hour": np.random.randint(0, 10, n),
        "device_id": np.random.choice(["d1", "d2", "d3", "d4", "d5"], n),
        "ip_country_match": np.random.choice([True, False], n),
        "card_type": np.random.choice(["visa", "mastercard", "amex"], n),
        "age_days": np.random.randint(1, 1000, n),
        "account_tenure_days": np.random.randint(1, 2000, n),
        "failed_attempts_last_hour": np.random.randint(0, 5, n),
        "is_fraud": np.random.choice([0, 1], n, p=[0.95, 0.05]),
    })


class TestFeatureEngineer:
    def test_fit_transform_returns_features_and_labels(self, sample_transactions):
        fe = FeatureEngineer()
        X, y = fe.fit_transform(sample_transactions)
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert len(X) == len(sample_transactions)
        assert len(y) == len(sample_transactions)

    def test_feature_count(self, sample_transactions):
        fe = FeatureEngineer()
        X, _ = fe.fit_transform(sample_transactions)
        # Should have ~26 features (temporal, ratio, interaction, log, encoded, scalars)
        assert X.shape[1] >= 20, f"Expected >=20 features, got {X.shape[1]}"

    def test_temporal_features_are_cyclic(self, sample_transactions):
        fe = FeatureEngineer()
        X, _ = fe.fit_transform(sample_transactions)
        for col in ["hour_sin", "hour_cos", "day_sin", "day_cos"]:
            assert col in X.columns
            assert X[col].min() >= -1.0
            assert X[col].max() <= 1.0

    def test_categorical_encoding(self, sample_transactions):
        fe = FeatureEngineer()
        X, _ = fe.fit_transform(sample_transactions)
        for col in ["merchant_category_encoded", "merchant_country_encoded", "card_type_encoded"]:
            assert col in X.columns, f"Missing encoded column: {col}"

    def test_transform_without_fit_raises_error(self, sample_transactions):
        fe = FeatureEngineer()
        with pytest.raises((AttributeError, KeyError)):
            fe.transform(sample_transactions.drop(columns=["is_fraud"]))

    def test_unknown_category_gets_minus_one(self, sample_transactions):
        fe = FeatureEngineer()
        X_train, _ = fe.fit_transform(sample_transactions)
        new_txn = sample_transactions.iloc[0:1].copy()
        new_txn["merchant_category"] = "unknown_cat_xyz"
        X_test = fe.transform(new_txn.drop(columns=["is_fraud"]))
        # Unknown category should be encoded as -1
        assert X_test["merchant_category_encoded"].iloc[0] == -1
