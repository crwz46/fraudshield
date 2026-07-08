import pytest
import numpy as np
from app.monitoring.drift import DriftDetector, compute_reference_stats
from app.monitoring import metrics as prom_metrics


class TestDriftDetector:
    @pytest.fixture
    def detector(self):
        det = DriftDetector()
        # Set up reference stats directly
        det.reference_stats = {
            "amount_log": {"mean": 4.5, "std": 1.2, "p1": 2.0, "p5": 2.5,
                           "p25": 3.5, "p50": 4.5, "p75": 5.5, "p95": 6.5,
                           "p99": 7.0, "min": 1.0, "max": 8.0},
            "hour_sin": {"mean": 0.0, "std": 0.7, "p1": -1.0, "p5": -0.9,
                         "p25": -0.5, "p50": 0.0, "p75": 0.5, "p95": 0.9,
                         "p99": 1.0, "min": -1.0, "max": 1.0},
        }
        return det

    def test_detect_drift_returns_reports(self, detector):
        features = np.random.randn(100, 2) * np.array([1.2, 0.7]) + np.array([4.5, 0.0])
        reports = detector.detect_drift(features, ["amount_log", "hour_sin"])
        assert len(reports) == 2
        assert "drift_score" in reports[0]
        assert "is_drifted" in reports[0]

    def test_drift_detected_with_shifted_distribution(self, detector):
        # Shift distribution significantly
        features = np.random.randn(100, 2) * np.array([2.0, 1.5]) + np.array([6.0, 0.5])
        reports = detector.detect_drift(features, ["amount_log", "hour_sin"])
        # PSI should be high
        assert reports[0]["drift_score"] > 0.05

    def test_drift_not_detected_with_similar_distribution(self, detector):
        features = np.random.randn(100, 2) * np.array([1.1, 0.7]) + np.array([4.5, 0.0])
        reports = detector.detect_drift(features, ["amount_log", "hour_sin"])
        # PSI should be low
        assert reports[0]["drift_score"] < 0.5  # Not a tight bound, just sanity


class TestPrometheusMetrics:
    def test_metrics_generate_valid_output(self):
        prom_metrics.track_prediction(0.05, "1.0.0", "low", False)
        prom_metrics.update_fraud_rate(0.02)
        prom_metrics.update_model_health(True)
        output = prom_metrics.get_metrics()
        assert "fraudshield_predictions_total" in output
        assert "fraudshield_fraud_rate" in output
        assert "fraudshield_model_health" in output
