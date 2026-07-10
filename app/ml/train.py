"""
Model training pipeline for FraudShield.
Trains XGBoost and CatBoost with cross-validation, threshold tuning, and full metrics.
"""

import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    classification_report, roc_auc_score, f1_score,
    precision_recall_curve, confusion_matrix, average_precision_score,
)

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
        n_fraud = df['is_fraud'].sum()
        print(f"Loaded {len(df)} transactions ({n_fraud} fraud / {len(df) - n_fraud} normal)")
        return df

    def train(self, csv_path: str, test_size: float = 0.2, cv_folds: int = 5) -> dict:
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

        cv_scores = self._cross_validate(X, y, cv_folds)
        self._evaluate(X_test, y_test, cv_scores)
        self._save_model()
        return self.metrics

    def _train_xgboost(self, X_train, y_train):
        import xgboost as xgb
        scale_pos = max(1, (len(y_train) - y_train.sum()) / max(y_train.sum(), 1))
        self.model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos,
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
        print(f"XGBoost trained: {self.model.n_estimators} estimators, scale_pos={scale_pos:.1f}")

    def _train_catboost(self, X_train, y_train):
        from catboost import CatBoostClassifier
        scale_pos = max(1, (len(y_train) - y_train.sum()) / max(y_train.sum(), 1))
        self.model = CatBoostClassifier(
            iterations=400,
            depth=5,
            learning_rate=0.05,
            scale_pos_weight=scale_pos,
            eval_metric="AUC",
            early_stopping_rounds=30,
            random_seed=42,
            verbose=False,
        )
        self.model.fit(X_train, y_train, eval_set=[(X_train, y_train)])
        print(f"CatBoost trained: scale_pos={scale_pos:.1f}")

    def _cross_validate(self, X, y, folds: int = 5) -> list[float]:
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
        scores = []
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_fold, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_fold, y_val = y.iloc[train_idx], y.iloc[val_idx]

            if self.model_type == "xgboost":
                import xgboost as xgb
                m = xgb.XGBClassifier(
                    n_estimators=100, max_depth=4, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8,
                    scale_pos_weight=max(1, (len(y_fold) - y_fold.sum()) / max(y_fold.sum(), 1)),
                    eval_metric="auc", random_state=42, n_jobs=-1,
                )
                m.fit(X_fold, y_fold, eval_set=[(X_fold, y_fold)], verbose=False)
            else:
                from catboost import CatBoostClassifier
                m = CatBoostClassifier(
                    iterations=100, depth=4, learning_rate=0.05,
                    scale_pos_weight=max(1, (len(y_fold) - y_fold.sum()) / max(y_fold.sum(), 1)),
                    eval_metric="AUC", random_seed=42, verbose=False,
                )
                m.fit(X_fold, y_fold, eval_set=[(X_fold, y_fold)], verbose=False)

            proba = m.predict_proba(X_val)[:, 1]
            score = roc_auc_score(y_val, proba)
            scores.append(round(score, 4))
            print(f"  Fold {fold + 1}: ROC-AUC = {score:.4f}")

        print(f"  CV mean: {np.mean(scores):.4f} +/- {np.std(scores):.4f}")
        return scores

    def _evaluate(self, X_test, y_test, cv_scores: list[float]):
        y_prob = self.model.predict_proba(X_test)[:, 1]
        y_pred = self.model.predict(X_test)

        # Threshold tuning — find best F1 threshold
        precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)
        f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-8)
        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_idx] if len(thresholds) > best_idx else 0.5
        y_pred_tuned = (y_prob >= best_threshold).astype(int)

        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        self.metrics = {
            "model_type": self.model_type,
            "roc_auc": round(roc_auc_score(y_test, y_prob), 4),
            "average_precision": round(average_precision_score(y_test, y_prob), 4),
            "f1_score": round(f1_score(y_test, y_pred), 4),
            "f1_tuned": round(f1_score(y_test, y_pred_tuned), 4),
            "best_threshold": round(float(best_threshold), 4),
            "cv_scores": [float(s) for s in cv_scores],
            "cv_mean": round(float(np.mean(cv_scores)), 4),
            "cv_std": round(float(np.std(cv_scores)), 4),
            "confusion_matrix": {"tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)},
            "precision": round(float(precisions[best_idx]), 4),
            "recall": round(float(recalls[best_idx]), 4),
            "fraud_count": int(y_test.sum()),
            "total_test": len(y_test),
            "fraud_rate": round(float(y_test.mean()), 4),
            "threshold_default": settings.model_threshold,
            "trained_at": datetime.now(timezone.utc).isoformat(),
        }
        print(f"\nModel Evaluation ({self.model_type}):")
        print(f"  ROC-AUC:          {self.metrics['roc_auc']}")
        print(f"  Avg Precision:    {self.metrics['average_precision']}")
        print(f"  F1 (default th):  {self.metrics['f1_score']}")
        print(f"  F1 (tuned th):    {self.metrics['f1_tuned']}")
        print(f"  Best threshold:   {self.metrics['best_threshold']}")
        print(f"  CV mean +/- std:  {self.metrics['cv_mean']} +/- {self.metrics['cv_std']}")
        print(f"  Confusion: TP={tp} FP={fp} FN={fn} TN={tn}")
        print(f"  Fraud: {self.metrics['fraud_count']}/{self.metrics['total_test']}")

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
    print("\n=== Training XGBoost ===")
    tx = ModelTrainer(model_type="xgboost")
    mx = tx.train(str(csv_path))
    print("\n=== Training CatBoost ===")
    tc = ModelTrainer(model_type="catboost")
    mc = tc.train(str(csv_path))
    print(f"\nXGBoost  ROC-AUC: {mx['roc_auc']}, F1: {mx['f1_score']}")
    print(f"CatBoost ROC-AUC: {mc['roc_auc']}, F1: {mc['f1_score']}")
