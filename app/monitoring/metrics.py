"""
Prometheus metrics for model serving monitoring.
Tracks prediction volume, latency, fraud rate, and model health.
"""

import time
from prometheus_client import Counter, Histogram, Gauge, generate_latest

predictions_total = Counter(
    "fraudshield_predictions_total",
    "Total predictions served",
    ["model_version", "risk_level"],
)

predictions_latency = Histogram(
    "fraudshield_prediction_latency_seconds",
    "Prediction latency in seconds",
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
)

fraud_rate_gauge = Gauge(
    "fraudshield_fraud_rate",
    "Current fraud rate in predictions",
)

model_health_gauge = Gauge(
    "fraudshield_model_health",
    "Model health status (1=healthy, 0=unhealthy)",
)

feature_drift_gauge = Gauge(
    "fraudshield_feature_drift",
    "Number of drifted features",
    ["feature"],
)

prediction_counter = Counter(
    "fraudshield_prediction_class_total",
    "Predictions by class",
    ["predicted_class"],
)


def track_prediction(latency_seconds: float, model_version: str, risk_level: str, is_fraud: bool):
    predictions_total.labels(model_version=model_version, risk_level=risk_level).inc()
    predictions_latency.observe(latency_seconds)
    pred_class = "fraud" if is_fraud else "normal"
    prediction_counter.labels(predicted_class=pred_class).inc()


def update_fraud_rate(rate: float):
    fraud_rate_gauge.set(rate)


def update_model_health(healthy: bool):
    model_health_gauge.set(1 if healthy else 0)


def update_feature_drift(feature: str, score: float):
    feature_drift_gauge.labels(feature=feature).set(score)


def get_metrics():
    return generate_latest().decode()
