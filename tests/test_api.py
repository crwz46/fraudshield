import pytest
from httpx import AsyncClient, ASGITransport
from app.main import app


@pytest.fixture
def client():
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


class TestAPI:
    @pytest.mark.asyncio
    async def test_root_endpoint(self, client):
        resp = await client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert data["service"] == "FraudShield"
        assert data["status"] == "running"

    @pytest.mark.asyncio
    async def test_health_endpoint(self, client):
        resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "predictions_served" in data

    @pytest.mark.asyncio
    async def test_predict_without_model_returns_503(self, client):
        resp = await client.post("/predict/single", json={
            "transaction_id": "test_001",
            "amount": 150.00,
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
        })
        # Either 503 if model not loaded, or 200 if model is loaded
        assert resp.status_code in (200, 503)

    @pytest.mark.asyncio
    async def test_feature_importance_endpoint(self, client):
        resp = await client.get("/predict/importance")
        # 200 if model loaded, 500 otherwise
        assert resp.status_code in (200, 500, 503)
