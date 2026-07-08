"""
Model training pipeline for FraudShield.
Trains XGBoost and CatBoost models with hyperparameter tuning.
"""

import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, f1_score, precision_recall_curve

from app.config import settings
from app.ml.features import FeatureEngineer


class ModelTrainer:
    def __init__(self, model_type: str = "xgboost"):
        self.model_type = model_type
        self.model = None
        self.feature_engineer = FeatureEngineer()
        self.metrics: dict = {}

    def load_data(self, csv_path: str) -> pd.DataFrame:
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} transactions ({df['is_fraud'].sum()} fraud / {len(df) - df['is_fraud'].sum()} normal)")
        return df

    def train(self, csv_path: str, test_size: float = 0.2) -> dict:
        df = self.load_data(csv_path)
        X, y = self.feature_engineer.fit_transform(df)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        if self.model_type == "xgboost":
            self._train_xgboost(X_train, y_train)
        elif self.model_type == "catboost":
            self._train_catboost(X_train, y_train)
        else:
            raise ValueError(f"Unsupported model: {self.model_type}")

        self._evaluate(X_test, y_test)
        self._save_model()
        return self.metrics

    def _train_xgboost(self, X_train, y_train):
        import xgboost as xgb
        self.model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=(len(y_train) - y_train.sum()) / y_train.sum(),
            eval_metric="auc",
            early_stopping_rounds=20,
            random_state=42,
            n_jobs=-1,
        )
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train)],
            verbose=False,
        )
        print(f"XGBoost trained: {self.model.n_estimators} estimators")

    def _train_catboost(self, X_train, y_train):
        from catboost import CatBoostClassifier
        self.model = CatBoostClassifier(
            iterations=300,
            depth=6,
            learning_rate=0.05,
            scale_pos_weight=(len(y_train) - y_train.sum()) / y_train.sum(),
            eval_metric="AUC",
            early_stopping_rounds=30,
            random_seed=42,
            verbose=False,
        )
        self.model.fit(X_train, y_train, eval_set=[(X_train, y_train)])
        print(f"CatBoost trained: {getattr(self.model, 'tree_count_', self.model.get_param('iterations'))} iterations")

    def _evaluate(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)[:, 1]

        self.metrics = {
            "model_type": self.model_type,
            "roc_auc": round(roc_auc_score(y_test, y_prob), 4),
            "f1_score": round(f1_score(y_test, y_pred), 4),
            "classification_report": classification_report(y_test, y_pred, output_dict=True),
            "fraud_count": int(y_test.sum()),
            "total_test": len(y_test),
            "fraud_rate": round(float(y_test.mean()), 4),
            "threshold": settings.model_threshold,
            "trained_at": datetime.now(timezone.utc).isoformat(),
        }
        print(f"\nModel Evaluation ({self.model_type}):")
        print(f"  ROC-AUC:  {self.metrics['roc_auc']}")
        print(f"  F1 Score: {self.metrics['f1_score']}")
        print(f"  Fraud in test: {self.metrics['fraud_count']}/{self.metrics['total_test']}")

    def _save_model(self):
        model_dir = Path("models")
        model_dir.mkdir(exist_ok=True)

        model_path = model_dir / f"fraud_{self.model_type}.joblib"
        joblib.dump(self.model, model_path)

        fe_path = model_dir / "feature_engineer.joblib"
        joblib.dump(self.feature_engineer, fe_path)

        metrics_path = model_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(self.metrics, f, indent=2)

        print(f"\nModel saved to {model_path}")
        print(f"Feature engineer saved to {fe_path}")

    def predict(self, features: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        if self.model is None:
            self._load_model()
        X = self.feature_engineer.transform(features)
        preds = self.model.predict(X)
        probs = self.model.predict_proba(X)[:, 1]
        return preds, probs

    def _load_model(self):
        import joblib
        model_path = Path(f"models/fraud_{self.model_type}.joblib")
        fe_path = Path("models/feature_engineer.joblib")
        if model_path.exists() and fe_path.exists():
            self.model = joblib.load(model_path)
            self.feature_engineer = joblib.load(fe_path)
            print(f"Loaded model from {model_path}")
        else:
            raise FileNotFoundError("No trained model found. Run training first.")

    def feature_importance(self) -> dict:
        if self.model is None:
            self._load_model()
        importances = self.model.feature_importances_
        names = self.feature_engineer.get_feature_names()
        sorted_idx = np.argsort(importances)[::-1]
        return {
            "features": [
                {"name": names[i], "importance": round(float(importances[i]), 4)}
                for i in sorted_idx[:20]
            ]
        }


if __name__ == "__main__":
    from data.generator import generate_transactions
    csv_path = generate_transactions()
    trainer = ModelTrainer(model_type="xgboost")
    metrics = trainer.train(str(csv_path))
    print(json.dumps(metrics, indent=2))
