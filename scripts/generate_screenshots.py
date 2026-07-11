"""
Generate evaluation charts and capture API terminal output for README screenshots.
"""

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Ensure screenshots directory
SCREENSHOTS = Path("screenshots")
SCREENSHOTS.mkdir(exist_ok=True)

# ── 1. Confusion Matrix ──
def plot_confusion_matrix():
    import matplotlib.pyplot as plt
    import seaborn as sns

    with open("models/metrics.json") as f:
        metrics = json.load(f)

    cm = metrics["confusion_matrix"]
    matrix = [[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]]
    model = metrics["model_type"]
    roc = metrics["roc_auc"]
    f1 = metrics["f1_score"]

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Predicted Normal", "Predicted Fraud"],
                yticklabels=["Actual Normal", "Actual Fraud"],
                ax=ax, cbar=False)
    ax.set_title(f"Confusion Matrix — {model.upper()}\nROC-AUC: {roc} | F1: {f1}", fontsize=13, pad=15)
    plt.tight_layout()
    path = SCREENSHOTS / "confusion_matrix.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved {path}")


# ── 2. Feature Importance ──
def plot_feature_importance():
    import matplotlib.pyplot as plt

    with open("models/metrics.json") as f:
        metrics = json.load(f)

    from app.ml.train import ModelTrainer
    trainer = ModelTrainer(model_type=metrics["model_type"])
    importance = trainer.feature_importance()
    top10 = importance["features"][:10]

    names = [f["name"] for f in top10][::-1]
    scores = [f["importance"] for f in top10][::-1]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(names, scores, color="steelblue", edgecolor="white")
    ax.set_xlabel("Importance", fontsize=11)
    ax.set_title(f"Top 10 Features — {metrics['model_type'].upper()}", fontsize=13, pad=10)
    for bar, score in zip(bars, scores):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{score:.3f}", va="center", fontsize=9)
    plt.tight_layout()
    path = SCREENSHOTS / "feature_importance.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved {path}")


# ── 3. ROC Curve ──
def plot_roc_curve():
    import matplotlib.pyplot as plt
    import joblib
    import pandas as pd
    import numpy as np
    from sklearn.metrics import roc_curve, auc
    from app.ml.features import FeatureEngineer

    with open("models/metrics.json") as f:
        metrics = json.load(f)

    model = joblib.load(f"models/fraud_{metrics['model_type']}.joblib")
    fe = joblib.load("models/feature_engineer.joblib")

    df = pd.read_csv("data/synthetic/transactions.csv")
    X, y = fe.transform(df), df["is_fraud"]
    y_prob = model.predict_proba(X)[:, 1]

    fpr, tpr, _ = roc_curve(y, y_prob)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random (AUC = 0.5)")
    ax.fill_between(fpr, tpr, alpha=0.15, color="darkorange")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title(f"ROC Curve — {metrics['model_type'].upper()}", fontsize=13, pad=10)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = SCREENSHOTS / "roc_curve.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved {path}")


# ── 4. Terminal output capture (API health + predict) ──
def capture_api_output():
    import time
    import requests

    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "8765"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    time.sleep(4)

    lines = []
    try:
        # Health check
        r = requests.get("http://127.0.0.1:8765/health", timeout=5)
        lines.append(">>> curl http://localhost:8765/health")
        lines.append(json.dumps(r.json(), indent=2))
        lines.append("")

        # Predict single
        payload = {
            "transaction_id": "txn_demo_001",
            "amount": 250.00,
            "merchant_category": "retail",
            "merchant_country": "US",
            "card_present": True,
            "distance_from_home_km": 5.0,
            "hour_of_day": 14,
            "day_of_week": 3,
            "days_since_last_transaction": 2,
            "transactions_last_24h": 3,
            "avg_transaction_amount_7d": 75.0,
            "velocity_last_hour": 1,
            "device_id": "d_iphone_01",
            "ip_country_match": True,
            "card_type": "visa",
            "age_days": 365,
            "account_tenure_days": 730,
            "failed_attempts_last_hour": 0,
        }
        r2 = requests.post("http://127.0.0.1:8765/predict/single", json=payload, timeout=5)
        lines.append(">>> curl -X POST http://localhost:8765/predict/single -H 'Content-Type: application/json' -d '{...}'")
        lines.append(json.dumps(r2.json(), indent=2))
        lines.append("")

        # Feature importance
        r3 = requests.get("http://127.0.0.1:8765/predict/importance", timeout=5)
        lines.append(">>> curl http://localhost:8765/predict/importance")
        lines.append(json.dumps(r3.json(), indent=2)[:500] + "\n...")

        result = "\n".join(lines)
    except Exception as e:
        result = f"Error: {e}"
    finally:
        proc.terminate()
        proc.wait()

    path = SCREENSHOTS / "api_output.txt"
    with open(path, "w") as f:
        f.write(result)
    print(f"[OK] Saved {path}")


# ── 5. Training metrics table ──
def save_metrics_summary():
    with open("models/metrics.json") as f:
        m = json.load(f)

    summary = f"""
{'='*60}
  FRAUDSHIELD — MODEL EVALUATION SUMMARY
{'='*60}

  Model Type:       {m['model_type'].upper()}
  ROC-AUC:          {m['roc_auc']}
  Avg Precision:    {m['average_precision']}
  F1 (default th):  {m['f1_score']}
  F1 (tuned th):    {m['f1_tuned']}
  Best Threshold:   {m['best_threshold']}

  Cross-Validation (5-fold):
    Scores:  {', '.join(str(s) for s in m['cv_scores'])}
    Mean:    {m['cv_mean']} +/- {m['cv_std']}

  Confusion Matrix:
    TP: {m['confusion_matrix']['tp']}   FP: {m['confusion_matrix']['fp']}
    FN: {m['confusion_matrix']['fn']}   TN: {m['confusion_matrix']['tn']}

  Test Set: {m['total_test']} transactions ({m['fraud_count']} fraud)
  Trained:  {m['trained_at']}
{'='*60}
"""
    path = SCREENSHOTS / "metrics_summary.txt"
    with open(path, "w") as f:
        f.write(summary)
    print(f"[OK] Saved {path}")
    print(summary)


if __name__ == "__main__":
    plot_confusion_matrix()
    plot_feature_importance()
    plot_roc_curve()
    capture_api_output()
    save_metrics_summary()
    print(f"\nAll screenshots saved to {SCREENSHOTS.resolve()}")
