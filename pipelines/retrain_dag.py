"""
Airflow DAG for automated model retraining.
Triggers on schedule or when drift alert fires.
Runs data generation -> feature engineering -> training -> evaluation -> deployment.
"""

from datetime import datetime, timedelta
from pathlib import Path
import json

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator

default_args = {
    "owner": "fraudshield",
    "depends_on_past": False,
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

dag = DAG(
    "fraudshield_retrain",
    default_args=default_args,
    description="Retrain fraud detection model on schedule or drift",
    schedule_interval="0 6 * * 1",  # Every Monday 6am
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["fraudshield", "ml"],
)


def generate_data(**context):
    """Generate synthetic training data."""
    from data.generator import generate_transactions
    csv_path = generate_transactions(num_normal=20000, num_fraud=400)
    context["ti"].xcom_push(key="csv_path", value=str(csv_path))
    print(f"Data generated: {csv_path}")


def train_model(**context):
    """Train XGBoost and CatBoost models."""
    csv_path = context["ti"].xcom_pull(key="csv_path", task_ids="generate_data")

    from app.ml.train import ModelTrainer

    for model_type in ["xgboost", "catboost"]:
        print(f"\n=== Training {model_type} ===")
        trainer = ModelTrainer(model_type=model_type)
        metrics = trainer.train(csv_path)
        with open(f"models/metrics_{model_type}.json", "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"{model_type} ROC-AUC: {metrics['roc_auc']}")


def evaluate_and_promote(**context):
    """Compare models and promote best one."""
    best_model = "xgboost"
    best_score = 0.0

    for model_type in ["xgboost", "catboost"]:
        metrics_path = Path(f"models/metrics_{model_type}.json")
        if metrics_path.exists():
            with open(metrics_path) as f:
                metrics = json.load(f)
            score = metrics.get("roc_auc", 0)
            print(f"{model_type}: ROC-AUC = {score}")
            if score > best_score:
                best_score = score
                best_model = model_type

    print(f"Promoting best model: {best_model} (ROC-AUC: {best_score})")
    with open("models/promoted_model.txt", "w") as f:
        f.write(f"latest\t{best_model}\troc_auc\t{best_score}\t{datetime.now().isoformat()}")


def update_reference_stats(**context):
    """Update drift reference stats after retraining."""
    import numpy as np
    import pandas as pd
    from app.ml.features import FeatureEngineer
    from app.monitoring.drift import compute_reference_stats

    csv_path = context["ti"].xcom_pull(key="csv_path", task_ids="generate_data")
    df = pd.read_csv(csv_path)
    fe = FeatureEngineer()
    X, _ = fe.fit_transform(df)
    compute_reference_stats(X.values, fe.get_feature_names())


generate_task = PythonOperator(
    task_id="generate_data",
    python_callable=generate_data,
    dag=dag,
)

train_task = PythonOperator(
    task_id="train_models",
    python_callable=train_model,
    dag=dag,
)

evaluate_task = PythonOperator(
    task_id="evaluate_and_promote",
    python_callable=evaluate_and_promote,
    dag=dag,
)

reference_task = PythonOperator(
    task_id="update_reference_stats",
    python_callable=update_reference_stats,
    dag=dag,
)

generate_task >> train_task >> evaluate_task >> reference_task
