import time
import pandas as pd
from fastapi import APIRouter, HTTPException

from app.config import settings
from app.ml.train import ModelTrainer
from app.ml.features import FeatureEngineer
from app.models import (
    TransactionRequest, BatchRequest, PredictionResponse,
    PredictionResult, ModelMetrics, FeatureImportance,
)
from app.monitoring.drift import DriftDetector
from app.monitoring import metrics as prom_metrics

router = APIRouter(prefix="/predict", tags=["Prediction"])

trainer = ModelTrainer(model_type="xgboost")
drift_detector = DriftDetector()
predictions_served = 0

try:
    trainer._load_model()
    model_loaded = True
except (FileNotFoundError, Exception):
    model_loaded = False
    print("WARNING: No trained model found. Run training first.")


def _to_dataframe(req: TransactionRequest) -> pd.DataFrame:
    return pd.DataFrame([req.model_dump()])


def _risk_level(prob: float) -> str:
    if prob >= 0.8:
        return "critical"
    elif prob >= 0.5:
        return "high"
    elif prob >= 0.2:
        return "medium"
    else:
        return "low"


@router.post("/single", response_model=PredictionResult)
async def predict_single(req: TransactionRequest):
    global predictions_served
    if not model_loaded:
        raise HTTPException(503, "Model not loaded. Train first.")

    start = time.time()
    df = _to_dataframe(req)
    preds, probs = trainer.predict(df)
    latency = (time.time() - start) * 1000

    risk = _risk_level(probs[0])
    predictions_served += 1
    prom_metrics.track_prediction(latency / 1000, settings.app_version, risk, bool(preds[0]))
    prom_metrics.update_fraud_rate(probs[0])

    return PredictionResult(
        transaction_id=req.transaction_id,
        is_fraud=bool(preds[0]),
        fraud_probability=round(float(probs[0]), 4),
        risk_level=risk,
        model_version=settings.app_version,
        processing_time_ms=round(latency, 2),
    )


@router.post("/batch", response_model=PredictionResponse)
async def predict_batch(req: BatchRequest):
    global predictions_served
    if not model_loaded:
        raise HTTPException(503, "Model not loaded. Train first.")

    start = time.time()
    df = pd.DataFrame([t.model_dump() for t in req.transactions])
    preds, probs = trainer.predict(df)
    latency = (time.time() - start) * 1000

    results = []
    fraud_count = 0
    for i, txn in enumerate(req.transactions):
        risk = _risk_level(probs[i])
        is_fraud = bool(preds[i])
        if is_fraud:
            fraud_count += 1
        prom_metrics.track_prediction(latency / 1000 / len(req.transactions), settings.app_version, risk, is_fraud)
        results.append(PredictionResult(
            transaction_id=txn.transaction_id,
            is_fraud=is_fraud,
            fraud_probability=round(float(probs[i]), 4),
            risk_level=risk,
            model_version=settings.app_version,
            processing_time_ms=round(latency / len(req.transactions), 2),
        ))

    predictions_served += len(req.transactions)
    prom_metrics.update_fraud_rate(fraud_count / max(len(req.transactions), 1))

    return PredictionResponse(
        predictions=results,
        batch_stats={
            "total": len(results),
            "fraud_detected": fraud_count,
            "fraud_rate": round(fraud_count / max(len(results), 1), 4),
            "avg_latency_ms": round(latency / len(req.transactions), 2),
        },
    )


@router.get("/drift")
async def check_drift(n_samples: int = 100):
    from data.generator import generate_transactions
    import tempfile

    csv_path = generate_transactions(num_normal=n_samples, num_fraud=max(n_samples // 10, 10))
    df = pd.read_csv(csv_path)
    fe = FeatureEngineer()
    X, _ = fe.fit_transform(df)
    reports = drift_detector.detect_drift(X.values, fe.get_feature_names())

    for r in reports:
        if "error" not in r:
            prom_metrics.update_feature_drift(r["feature"], r["drift_score"])

    return {"drift_report": reports, "total_features": len(reports), "drifted": sum(1 for r in reports if r.get("is_drifted"))}


@router.get("/metrics")
async def get_metrics():
    return prom_metrics.get_metrics()


@router.get("/importance", response_model=FeatureImportance)
async def feature_importance():
    return trainer.feature_importance()
