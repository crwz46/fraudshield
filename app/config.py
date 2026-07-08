from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    app_name: str = "FraudShield"
    app_version: str = "1.0.0"
    debug: bool = True

    # Model
    model_path: str = "./models/fraud_xgboost.joblib"
    model_threshold: float = 0.5
    feature_store_path: str = "./data/features"

    # MLflow
    mlflow_tracking_uri: str = "http://mlflow:5000"
    experiment_name: str = "fraudshield"

    # Monitoring
    drift_alert_threshold: float = 0.15
    monitoring_window: int = 1000

    # Serving
    batch_size: int = 32
    max_request_size: int = 1000

    # Data
    data_path: str = "./data"
    synthetic_data_path: str = "./data/synthetic"


settings = Settings()
