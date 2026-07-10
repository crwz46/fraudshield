import pytest
import pandas as pd
from pathlib import Path
from data.generator import generate_transactions


class TestDataGenerator:
    def test_generates_csv(self):
        csv_path = generate_transactions(num_normal=50, num_fraud=10)
        path = Path(csv_path)
        assert path.exists()
        assert path.suffix == ".csv"

    def test_fraud_and_normal_count(self):
        csv_path = generate_transactions(num_normal=100, num_fraud=20)
        df = pd.read_csv(csv_path)
        # 100 normal + 5 false_alarm (100//20) + 10 main fraud + 10 stealth = 125
        assert len(df) == 125
        assert df["is_fraud"].sum() == 20
        assert (df["is_fraud"] == 0).sum() == 105

    def test_all_expected_columns(self):
        csv_path = generate_transactions(num_normal=10, num_fraud=2)
        df = pd.read_csv(csv_path)
        expected = {
            "transaction_id", "amount", "merchant_category", "merchant_country",
            "card_present", "distance_from_home_km", "hour_of_day", "day_of_week",
            "days_since_last_transaction", "transactions_last_24h",
            "avg_transaction_amount_7d", "velocity_last_hour", "device_id",
            "ip_country_match", "card_type", "age_days", "account_tenure_days",
            "failed_attempts_last_hour", "is_fraud", "timestamp",
        }
        assert expected.issubset(set(df.columns)), f"Missing columns: {expected - set(df.columns)}"

    def test_no_null_values(self):
        csv_path = generate_transactions(num_normal=50, num_fraud=10)
        df = pd.read_csv(csv_path)
        assert df.isnull().sum().sum() == 0

    def test_fraud_transactions_scattered_across_patterns(self):
        csv_path = generate_transactions(num_normal=100, num_fraud=20)
        df = pd.read_csv(csv_path)
        fraud_txns = df[df["is_fraud"] == 1]
        assert len(fraud_txns) == 20
        # Fraud txns have distinct patterns: fast-small, geographic, midnight, card-testing, refund
        txn_ids = fraud_txns["transaction_id"].values
        prefixes = set(t.split("_")[0] + "_" + t.split("_")[1] for t in txn_ids)
        assert len(prefixes) > 1, f"Expected multiple fraud patterns, got: {prefixes}"
