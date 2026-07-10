import pytest
import numpy as np
import pandas as pd
from app.ml.train import ModelTrainer


@pytest.fixture
def mini_training_data(tmp_path):
    np.random.seed(42)
    n = 500
    df = pd.DataFrame({
        "amount": np.random.exponential(100, n),
        "merchant_category": np.random.choice(["retail", "food", "travel"], n),
        "merchant_country": np.random.choice(["US", "DE", "JP"], n),
        "card_present": np.random.choice([True, False], n),
        "distance_from_home_km": np.random.exponential(50, n),
        "hour_of_day": np.random.randint(0, 24, n),
        "day_of_week": np.random.randint(0, 7, n),
        "days_since_last_transaction": np.random.randint(0, 60, n),
        "transactions_last_24h": np.random.randint(0, 20, n),
        "avg_transaction_amount_7d": np.random.exponential(100, n),
        "velocity_last_hour": np.random.randint(0, 10, n),
        "device_id": np.random.choice(["d1", "d2", "d3"], n),
        "ip_country_match": np.random.choice([True, False], n),
        "card_type": np.random.choice(["visa", "mastercard"], n),
        "age_days": np.random.randint(1, 1000, n),
        "account_tenure_days": np.random.randint(1, 2000, n),
        "failed_attempts_last_hour": np.random.randint(0, 5, n),
        "is_fraud": np.random.choice([0, 1], n, p=[0.9, 0.1]),
    })
    csv_path = tmp_path / "train_data.csv"
    df.to_csv(csv_path, index=False)
    return str(csv_path)


class TestModelTrainer:
    def test_xgboost_training(self, mini_training_data):
        trainer = ModelTrainer(model_type="xgboost")
        metrics = trainer.train(mini_training_data, test_size=0.3)
        assert "roc_auc" in metrics
        assert "f1_score" in metrics
        assert 0 <= metrics["roc_auc"] <= 1
        assert "cv_mean" in metrics
        assert "cv_scores" in metrics
        assert len(metrics["cv_scores"]) == 5
        assert "confusion_matrix" in metrics

    def test_catboost_training(self, mini_training_data):
        trainer = ModelTrainer(model_type="catboost")
        metrics = trainer.train(mini_training_data, test_size=0.3)
        assert "roc_auc" in metrics
        assert "average_precision" in metrics
        assert 0 <= metrics["roc_auc"] <= 1
        assert "cv_mean" in metrics

    def test_model_predicts_proba(self, mini_training_data):
        trainer = ModelTrainer(model_type="xgboost")
        trainer.train(mini_training_data, test_size=0.3)
        # Predict on a single row
        df = pd.read_csv(mini_training_data)
        sample = df.drop(columns=["is_fraud"]).iloc[0:1]
        preds, probs = trainer.predict(sample)
        assert len(preds) == 1
        assert len(probs) == 1
        assert 0 <= probs[0] <= 1

    def test_feature_importance(self, mini_training_data):
        trainer = ModelTrainer(model_type="xgboost")
        trainer.train(mini_training_data, test_size=0.3)
        importance = trainer.feature_importance()
        assert "features" in importance
        assert len(importance["features"]) > 0
        assert "name" in importance["features"][0]
        assert "importance" in importance["features"][0]
        # Check sorted descending
        vals = [f["importance"] for f in importance["features"]]
        assert all(vals[i] >= vals[i + 1] for i in range(len(vals) - 1))
