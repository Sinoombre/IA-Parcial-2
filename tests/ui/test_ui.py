from ui.app import DEFAULTS, app


def test_ui_health_declares_development_only():
    client = app.test_client()
    response = client.get("/api/health")
    payload = response.get_json()
    assert response.status_code == 200
    assert payload["mode"] == "development_only"
    assert payload["reserved_evaluation_executed"] is False


def test_ui_predicts_without_reservation_access():
    client = app.test_client()
    response = client.post("/api/predict", json=DEFAULTS)
    payload = response.get_json()
    assert response.status_code == 200
    assert 0.0 <= payload["p_churn"] <= 1.0
    assert payload["reserved_evaluation_executed"] is False
    assert "causalidad" in payload["disclaimer"]
