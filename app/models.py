from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


class TransactionRequest(BaseModel):
    transaction_id: str
    amount: float
    merchant_category: str
    merchant_country: str
    card_present: bool
    distance_from_home_km: float
    hour_of_day: int = Field(ge=0, le=23)
    day_of_week: int = Field(ge=0, le=6)
    days_since_last_transaction: int
    transactions_last_24h: int
    avg_transaction_amount_7d: float
    velocity_last_hour: int
    device_id: str
    ip_country_match: bool
    card_type: str
    age_days: int
    account_tenure_days: int
    failed_attempts_last_hour: int


class BatchRequest(BaseModel):
    transactions: list[TransactionRequest]


class PredictionResult(BaseModel):
    transaction_id: str
    is_fraud: bool
    fraud_probability: float
    risk_level: str
    model_version: str
    processing_time_ms: float


class PredictionResponse(BaseModel):
    predictions: list[PredictionResult]
    batch_stats: dict


class ModelMetrics(BaseModel):
    model_type: str
    roc_auc: float
    f1_score: float
    fraud_count: int
    total_test: int
    trained_at: str
    threshold: float


class DriftReport(BaseModel):
    feature: str
    drift_score: float
    is_drifted: bool
    reference_mean: float
    current_mean: float
    reference_std: float
    current_std: float


class MonitoringSnapshot(BaseModel):
    total_predictions: int
    fraud_rate: float
    avg_confidence: float
    model_version: str
    timestamp: str
    service_health: str


class FeatureImportance(BaseModel):
    features: list[dict]


class TrainRequest(BaseModel):
    model_type: str = "xgboost"
    test_size: float = 0.2


class HealthResponse(BaseModel):
    status: str
    version: str
    model_loaded: bool
    model_type: Optional[str] = None
    predictions_served: int
    uptime: str
