"""
Synthetic fraud detection data generator.
Now with noise injection + stealth fraud for realistic ROC-AUC (~0.85-0.95).
"""

import csv
import random
import math
from datetime import datetime, timedelta
from pathlib import Path


def generate_transactions(
    num_normal: int = 10000,
    num_fraud: int = 200,
    output_path: str = "./data/synthetic",
    seed: int = 42,
):
    random.seed(seed)
    output = Path(output_path)
    output.mkdir(parents=True, exist_ok=True)

    transactions = []
    base_time = datetime(2025, 1, 1, 0, 0, 0)

    # Normal transactions
    for i in range(num_normal):
        t = _normal_transaction(base_time, i)
        transactions.append(t)

    # Fraudulent transactions — injected patterns
    patterns = [
        ("fast_small_staircase", _fast_small_staircase),
        ("large_geographic_hop", _large_geographic_hop),
        ("midnight_splash", _midnight_splash),
        ("card_testing", _card_testing),
        ("refund_cycle", _refund_cycle),
    ]
    main_fraud = num_fraud - max(num_fraud // 5, 10)  # 80% main patterns
    stealth_count = num_fraud - main_fraud              # 20% stealth

    per_pattern = main_fraud // len(patterns)
    for name, pattern_fn in patterns:
        for j in range(per_pattern):
            idx = hash(name) % 100000 + j
            t = pattern_fn(base_time, idx)
            if name != "fast_small_staircase":
                _add_noise(t)
            transactions.append(t)

    # Stealth fraud — looks exactly like normal transactions but is fraud
    for j in range(stealth_count):
        t = _normal_transaction(base_time, 900000 + j)
        t["transaction_id"] = f"txn_stealth_{j:08d}"
        t["is_fraud"] = 1
        t["days_since_last_transaction"] = random.randint(0, 2)
        t["failed_attempts_last_hour"] = random.randint(0, 1)
        transactions.append(t)

    # False alarm normals — look fraud-like but are legit
    for j in range(num_normal // 20):
        idx = random.randint(0, num_normal - 1)
        t = _normal_transaction(base_time, 800000 + j)
        t["transaction_id"] = f"txn_fa_{j:08d}"
        t["is_fraud"] = 0
        t["amount"] = round(random.uniform(300, 2000), 2)
        t["velocity_last_hour"] = random.randint(4, 10)
        t["hour_of_day"] = random.randint(0, 5)
        t["ip_country_match"] = random.choice([True, False])
        t["failed_attempts_last_hour"] = random.randint(0, 3)
        t["days_since_last_transaction"] = 0
        transactions.append(t)

    random.shuffle(transactions)

    fieldnames = [
        "transaction_id", "timestamp", "amount", "merchant_category",
        "merchant_country", "card_present", "distance_from_home_km",
        "hour_of_day", "day_of_week", "days_since_last_transaction",
        "transactions_last_24h", "avg_transaction_amount_7d",
        "velocity_last_hour", "device_id", "ip_country_match",
        "card_type", "age_days", "account_tenure_days",
        "failed_attempts_last_hour", "is_fraud",
    ]

    csv_path = output / "transactions.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(transactions)

    fraud_count = sum(1 for t in transactions if t["is_fraud"])
    print(f"Generated {len(transactions)} transactions ({fraud_count} fraud, {stealth_count} stealth)")
    print(f"Saved to {csv_path}")
    return csv_path


def _add_noise(t: dict, noise_level: float = 0.15):
    """Add gaussian noise to numeric features to blur fraud boundaries."""
    for key in ["amount", "distance_from_home_km", "avg_transaction_amount_7d"]:
        if key in t:
            noise = t[key] * random.gauss(0, noise_level)
            t[key] = round(max(0.01, t[key] + noise), 2)

    if random.random() < 0.2:
        t["ip_country_match"] = not t["ip_country_match"]
    if random.random() < 0.15:
        t["card_present"] = not t["card_present"]
    if random.random() < 0.15:
        t["hour_of_day"] = random.randint(0, 23)
    if random.random() < 0.1:
        t["merchant_category"] = random.choice(["grocery", "restaurant", "retail", "entertainment", "transport"])
    if random.random() < 0.1:
        t["merchant_country"] = "US"


def _normal_transaction(base, i):
    hour = random.randint(6, 23)
    return {
        "transaction_id": f"txn_{i:08d}",
        "timestamp": (base + timedelta(hours=i * random.randint(1, 6), minutes=random.randint(0, 59))).isoformat(),
        "amount": round(random.uniform(5, 500), 2),
        "merchant_category": random.choice(["grocery", "restaurant", "retail", "entertainment", "transport", "health", "utility"]),
        "merchant_country": "US",
        "card_present": random.choice([True, False]),
        "distance_from_home_km": round(random.uniform(0, 30), 1),
        "hour_of_day": hour,
        "day_of_week": random.randint(0, 6),
        "days_since_last_transaction": random.randint(0, 14),
        "transactions_last_24h": random.randint(0, 5),
        "avg_transaction_amount_7d": round(random.uniform(20, 200), 2),
        "velocity_last_hour": random.randint(0, 3),
        "device_id": f"dev_{random.randint(1, 500):04d}",
        "ip_country_match": True,
        "card_type": random.choice(["visa", "mastercard", "amex"]),
        "age_days": random.randint(365, 3650),
        "account_tenure_days": random.randint(30, 2000),
        "failed_attempts_last_hour": 0,
        "is_fraud": 0,
    }


def _fast_small_staircase(base, i):
    hour = random.randint(1, 5)
    return {
        "transaction_id": f"txn_fss_{i:08d}",
        "timestamp": (base + timedelta(hours=i * 2, minutes=random.randint(0, 5))).isoformat(),
        "amount": round(random.uniform(1, 30), 2),
        "merchant_category": "retail",
        "merchant_country": "US",
        "card_present": False,
        "distance_from_home_km": round(random.uniform(50, 200), 1),
        "hour_of_day": hour,
        "day_of_week": random.randint(0, 6),
        "days_since_last_transaction": 0,
        "transactions_last_24h": random.randint(10, 30),
        "avg_transaction_amount_7d": round(random.uniform(40, 80), 2),
        "velocity_last_hour": random.randint(5, 15),
        "device_id": f"dev_{random.randint(1, 10):04d}",
        "ip_country_match": True,
        "card_type": "visa",
        "age_days": random.randint(30, 180),
        "account_tenure_days": random.randint(1, 30),
        "failed_attempts_last_hour": random.randint(0, 2),
        "is_fraud": 1,
    }


def _large_geographic_hop(base, i):
    countries = ["RU", "CN", "NG", "BR", "IN"]
    return {
        "transaction_id": f"txn_lgh_{i:08d}",
        "timestamp": (base + timedelta(hours=i * 3)).isoformat(),
        "amount": round(random.uniform(2000, 15000), 2),
        "merchant_category": "electronics",
        "merchant_country": random.choice(countries),
        "card_present": False,
        "distance_from_home_km": round(random.uniform(5000, 15000), 1),
        "hour_of_day": random.randint(0, 23),
        "day_of_week": random.randint(0, 6),
        "days_since_last_transaction": random.randint(0, 3),
        "transactions_last_24h": 1,
        "avg_transaction_amount_7d": round(random.uniform(30, 100), 2),
        "velocity_last_hour": 1,
        "device_id": f"dev_{random.randint(501, 600):04d}",
        "ip_country_match": False,
        "card_type": random.choice(["visa", "mastercard"]),
        "age_days": random.randint(365, 1000),
        "account_tenure_days": random.randint(100, 500),
        "failed_attempts_last_hour": random.randint(0, 1),
        "is_fraud": 1,
    }


def _midnight_splash(base, i):
    return {
        "transaction_id": f"txn_ms_{i:08d}",
        "timestamp": (base + timedelta(hours=i * 4)).isoformat(),
        "amount": round(random.uniform(500, 5000), 2),
        "merchant_category": random.choice(["entertainment", "retail"]),
        "merchant_country": "US",
        "card_present": False,
        "distance_from_home_km": round(random.uniform(10, 100), 1),
        "hour_of_day": random.choice([0, 1, 2, 3, 4]),
        "day_of_week": random.randint(0, 6),
        "days_since_last_transaction": 0,
        "transactions_last_24h": random.randint(5, 15),
        "avg_transaction_amount_7d": round(random.uniform(30, 80), 2),
        "velocity_last_hour": random.randint(3, 8),
        "device_id": f"dev_{random.randint(1, 50):04d}",
        "ip_country_match": True,
        "card_type": random.choice(["amex", "visa"]),
        "age_days": random.randint(90, 500),
        "account_tenure_days": random.randint(7, 90),
        "failed_attempts_last_hour": random.randint(1, 5),
        "is_fraud": 1,
    }


def _card_testing(base, i):
    return {
        "transaction_id": f"txn_ct_{i:08d}",
        "timestamp": (base + timedelta(minutes=i * random.randint(1, 10))).isoformat(),
        "amount": round(random.uniform(0.5, 5), 2),
        "merchant_category": "retail",
        "merchant_country": "US",
        "card_present": False,
        "distance_from_home_km": round(random.uniform(0, 10), 1),
        "hour_of_day": random.randint(1, 6),
        "day_of_week": random.randint(0, 6),
        "days_since_last_transaction": 0,
        "transactions_last_24h": random.randint(15, 50),
        "avg_transaction_amount_7d": round(random.uniform(50, 150), 2),
        "velocity_last_hour": random.randint(10, 30),
        "device_id": f"dev_{random.randint(1, 5):04d}",
        "ip_country_match": True,
        "card_type": "visa",
        "age_days": random.randint(1, 30),
        "account_tenure_days": random.randint(0, 7),
        "failed_attempts_last_hour": random.randint(3, 10),
        "is_fraud": 1,
    }


def _refund_cycle(base, i):
    return {
        "transaction_id": f"txn_rc_{i:08d}",
        "timestamp": (base + timedelta(hours=i * random.randint(1, 8))).isoformat(),
        "amount": round(random.uniform(100, 800), 2),
        "merchant_category": "retail",
        "merchant_country": "US",
        "card_present": True,
        "distance_from_home_km": round(random.uniform(0, 20), 1),
        "hour_of_day": random.randint(9, 18),
        "day_of_week": random.randint(0, 6),
        "days_since_last_transaction": random.randint(0, 2),
        "transactions_last_24h": random.randint(3, 8),
        "avg_transaction_amount_7d": round(random.uniform(30, 80), 2),
        "velocity_last_hour": random.randint(1, 3),
        "device_id": f"dev_{random.randint(1, 100):04d}",
        "ip_country_match": True,
        "card_type": "mastercard",
        "age_days": random.randint(100, 500),
        "account_tenure_days": random.randint(30, 200),
        "failed_attempts_last_hour": 0,
        "is_fraud": 1,
    }


if __name__ == "__main__":
    generate_transactions()
