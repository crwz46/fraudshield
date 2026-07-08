"""
Model drift detection — monitors feature distribution shifts
using Population Stability Index (PSI) and KL divergence.
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime, timezone

from app.config import settings


class DriftDetector:
    def __init__(self):
        self.reference_stats: dict = {}
        self.drift_history_path = Path("data/drift_history.json")
        self.drift_history_path.parent.mkdir(parents=True, exist_ok=True)

    def compute_reference(self, features: np.ndarray, feature_names: list[str]):
        for i, name in enumerate(feature_names):
            col = features[:, i]
            self.reference_stats[name] = {
                "mean": float(np.mean(col)),
                "std": float(np.std(col)),
                "p1": float(np.percentile(col, 1)),
                "p5": float(np.percentile(col, 5)),
                "p25": float(np.percentile(col, 25)),
                "p50": float(np.percentile(col, 50)),
                "p75": float(np.percentile(col, 75)),
                "p95": float(np.percentile(col, 95)),
                "p99": float(np.percentile(col, 99)),
                "min": float(np.min(col)),
                "max": float(np.max(col)),
            }
        print(f"Reference stats computed for {len(feature_names)} features")

    def detect_drift(self, current_features: np.ndarray, feature_names: list[str]) -> list[dict]:
        if not self.reference_stats:
            print("No reference stats. Loading from saved...")
            self._load_reference()
            if not self.reference_stats:
                return [{"error": "No reference statistics available. Train model first."}]

        drift_reports = []
        for i, name in enumerate(feature_names):
            if name not in self.reference_stats:
                continue

            ref = self.reference_stats[name]
            curr_col = current_features[:, i]

            # PSI (Population Stability Index)
            psi = self._compute_psi(curr_col, ref)

            # Mean shift
            current_mean = float(np.mean(curr_col))
            current_std = float(np.std(curr_col))
            z_score = abs(current_mean - ref["mean"]) / (ref["std"] + 1e-8)

            is_drifted = psi > settings.drift_alert_threshold

            drift_reports.append({
                "feature": name,
                "drift_score": round(psi, 4),
                "is_drifted": bool(is_drifted),
                "reference_mean": round(ref["mean"], 4),
                "current_mean": round(current_mean, 4),
                "reference_std": round(ref["std"], 4),
                "current_std": round(current_std, 4),
                "z_score": round(z_score, 4),
            })

        # Save to history
        self._save_history(drift_reports)

        drifted = sum(1 for d in drift_reports if d["is_drifted"])
        print(f"Drift detection: {drifted}/{len(drift_reports)} features drifted")
        return drift_reports

    def _compute_psi(self, current: np.ndarray, ref: dict, bins: int = 10) -> float:
        """Population Stability Index."""
        if len(current) < 2:
            return 0.0
        hist_ref, edges = np.histogram(
            np.random.normal(ref["mean"], ref["std"], 10000),
            bins=bins, range=(ref["p1"], ref["p99"])
        )
        hist_curr, _ = np.histogram(current, bins=bins, range=(ref["p1"], ref["p99"]))

        psi = 0.0
        for i in range(bins):
            p_i = hist_ref[i] / max(hist_ref.sum(), 1) + 1e-8
            q_i = hist_curr[i] / max(hist_curr.sum(), 1) + 1e-8
            psi += (q_i - p_i) * np.log(q_i / p_i)

        return float(psi)

    def _save_history(self, reports: list[dict]):
        history = []
        if self.drift_history_path.exists():
            with open(self.drift_history_path) as f:
                history = json.load(f)

        history.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "drifted_features": [r["feature"] for r in reports if r["is_drifted"]],
            "total_drifted": sum(1 for r in reports if r["is_drifted"]),
            "reports": reports,
        })

        with open(self.drift_history_path, "w") as f:
            json.dump(history[-100:], f)

    def _load_reference(self):
        ref_path = Path("models/reference_stats.json")
        if ref_path.exists():
            with open(ref_path) as f:
                self.reference_stats = json.load(f)


def compute_reference_stats(features: np.ndarray, feature_names: list[str]):
    detector = DriftDetector()
    detector.compute_reference(features, feature_names)
    ref_path = Path("models/reference_stats.json")
    with open(ref_path, "w") as f:
        json.dump(detector.reference_stats, f, indent=2)
    print(f"Reference stats saved to {ref_path}")
