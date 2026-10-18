from ui.app import DEFAULTS, app


def test_seeded_profiles_are_reproducible_and_use_only_development_data(monkeypatch):
    import pandas as pd

    from churnshield.features import ALLOWED_PREDICTORS

    development = pd.DataFrame(
        [
            {
                **DEFAULTS,
                "Tenure Months": tenure,
                "Monthly Charges": monthly,
                "Total Charges": total,
                "CustomerID": f"DEV-{index}",
                "Churn Value": index % 2,
            }
            for index, (tenure, monthly, total) in enumerate(
                [
                    (1, 10.0, 10.0),
                    (12, 30.0, 360.0),
                    (36, 60.0, 2160.0),
                    (72, 90.0, 6480.0),
                ]
            )
        ]
    )
    monkeypatch.setattr("ui.app.load_dev", lambda: (development, {}))
    client = app.test_client()

    first = client.get("/api/sample?seed=41")
    second = client.get("/api/sample?seed=41")

    assert first.status_code == 200
    assert first.get_json() == second.get_json()
    payload = first.get_json()
    assert payload["source"] == "development_only"
    assert payload["reserved_evaluation_executed"] is False
    assert len(payload["profiles"]) == 3
    for profile in payload["profiles"]:
        assert set(profile["values"]) == set(ALLOWED_PREDICTORS)
        assert "CustomerID" not in profile["values"]
        assert "Churn Value" not in profile["values"]
    group = client.get("/api/sample?seed=41&count=4").get_json()
    assert group["count"] == 4
    assert len({tuple(profile["values"].items()) for profile in group["profiles"]}) == 4


def test_seeded_profiles_reject_invalid_seed():
    response = app.test_client().get("/api/sample?seed=-1")
    assert response.status_code == 400
    assert "semilla" in response.get_json()["error"]


def test_seeded_profiles_reject_invalid_group_size():
    response = app.test_client().get("/api/sample?seed=41&count=501")
    assert response.status_code == 400
    assert "grupo" in response.get_json()["error"]


def test_ui_rejects_non_finite_numeric_values():
    response = app.test_client().post(
        "/api/predict",
        json={**DEFAULTS, "Monthly Charges": float("nan")},
    )
    assert response.status_code == 400
    assert "número finito" in response.get_json()["error"]


def test_ui_predicts_group_without_exposing_training_labels(monkeypatch):
    import numpy as np

    class FakeModel:
        def predict_proba(self, features):
            return np.array([[0.8, 0.2], [0.2, 0.8]])

    monkeypatch.setattr("ui.app._load_model", lambda: FakeModel())
    monkeypatch.setattr("ui.app._model_card", lambda: {"mean_cltv_train": 840.0})
    response = app.test_client().post(
        "/api/predict",
        json={
            "profiles": [
                {"name": "Perfil 01", "values": DEFAULTS},
                {"name": "Perfil 02", "values": DEFAULTS},
            ]
        },
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["count"] == 2
    assert payload["mean_p_churn"] == 0.5
    assert payload["high_risk_count"] == 1
    assert payload["reserved_evaluation_executed"] is False
    assert len(payload["predictions"]) == 2
    assert "Churn Value" not in str(payload)


def test_ui_health_declares_development_only():
    client = app.test_client()
    response = client.get("/api/health")
    payload = response.get_json()
    assert response.status_code == 200
    assert payload["mode"] == "development_only"
    assert payload["reserved_evaluation_executed"] is False


def test_ui_offers_individual_and_group_prediction_without_theme_switch():
    response = app.test_client().get("/")
    html = response.get_data(as_text=True)
    assert response.status_code == 200
    assert 'id="predict-form"' in html
    assert 'id="group-size"' in html
    assert 'id="probability-chart"' in html
    assert "Generar y calcular grupo" in html
    assert "Probar una persona por separado" in html
    assert 'id="analyze-group"' not in html
    assert "ENTORNO DE DESARROLLO" not in html
    assert 'id="theme-toggle"' not in html


def test_ui_predicts_without_reservation_access():
    client = app.test_client()
    response = client.post("/api/predict", json=DEFAULTS)
    payload = response.get_json()
    assert response.status_code == 200
    assert 0.0 <= payload["p_churn"] <= 1.0
    assert payload["reserved_evaluation_executed"] is False
    assert "causalidad" in payload["disclaimer"]
