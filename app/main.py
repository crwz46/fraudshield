"""
FraudShield — ML Production Pipeline for Fraud Detection
FastAPI serving with Prometheus monitoring, drift detection, and retraining API.
"""

import time
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.config import settings
from app.api import predict
from app.models import HealthResponse


start_time = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"FraudShield v{settings.app_version} starting — model loaded: {predict.model_loaded}")
    yield
    print("FraudShield shutting down")


app = FastAPI(
    title="FraudShield API",
    description="Production-grade ML pipeline for fraud detection with XGBoost/CatBoost, "
                "drift monitoring, and retraining automation",
    version=settings.app_version,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict.router)


@app.get("/", tags=["System"])
async def root():
    return {"service": "FraudShield", "version": settings.app_version, "status": "running"}


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    uptime_secs = time.time() - start_time
    return HealthResponse(
        status="healthy" if predict.model_loaded else "degraded",
        version=settings.app_version,
        model_loaded=predict.model_loaded,
        model_type="xgboost" if predict.model_loaded else None,
        predictions_served=predict.predictions_served,
        uptime=f"{uptime_secs:.0f}s",
    )
