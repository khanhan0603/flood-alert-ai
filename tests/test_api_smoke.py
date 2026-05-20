from fastapi.testclient import TestClient

from src.api import app


client = TestClient(app)


def test_health_endpoint_loads_models():
    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["features"] > 0
    assert payload["models"] == ["lead1d", "lead2d", "lead3d"]


def test_predict_endpoint_accepts_minimal_payload():
    response = client.post(
        "/predict",
        json={"province": "ci_smoke", "lat": 16.0, "lon": 108.0},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["province"] == "ci_smoke"
    assert payload["overall_risk"] in {"LOW", "MEDIUM", "HIGH"}
    assert set(payload["forecast"]) == {"day_1", "day_2", "day_3"}

