from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_predict_demo_or_model():
    payload = {
        "airline": "AA",
        "origin_airport": "LAX",
        "destination_airport": "PBI",
        "month": 1,
        "day": 1,
        "day_of_week": 4,
        "scheduled_departure_hhmm": 10,
        "scheduled_time_minutes": 300,
        "distance": 2330,
        "cancelled": 0,
        "diverted": 0
    }
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "delay_probability" in data
