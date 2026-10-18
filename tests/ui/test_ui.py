import pytest

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


@pytest.mark.parametrize("monthly_charges", ["not-a-number", 10**1000])
def test_ui_rejects_unconvertible_numeric_values(monthly_charges):
    response = app.test_client().post(
        "/api/predict",
        json={**DEFAULTS, "Monthly Charges": monthly_charges},
    )

    assert response.status_code == 400
    assert "Monthly Charges debe ser un número finito" in response.get_json()["error"]


@pytest.mark.parametrize(
    ("body", "content_type", "message"),
    [
        ("{malformed", "application/json", "JSON válido"),
        ('["not", "an object"]', "application/json", "objeto"),
        ("{}", "text/plain", "debe ser JSON"),
    ],
)
def test_ui_rejects_invalid_json_request_bodies(body, content_type, message):
    response = app.test_client().post(
        "/api/predict",
        data=body,
        content_type=content_type,
    )

    assert response.status_code == 400
    assert message in response.get_json()["error"]


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
    assert 'class="background-picker"' not in html


def test_ui_predicts_without_reservation_access():
    client = app.test_client()
    response = client.post("/api/predict", json=DEFAULTS)
    payload = response.get_json()
    assert response.status_code == 200
    assert 0.0 <= payload["p_churn"] <= 1.0
    assert payload["reserved_evaluation_executed"] is False
    assert "causalidad" in payload["disclaimer"]


def test_ui_reports_missing_model_card_without_using_a_fallback(monkeypatch, tmp_path):
    monkeypatch.setattr("ui.app.CARD_PATH", tmp_path / "missing-model-card.json")

    response = app.test_client().post("/api/predict", json=DEFAULTS)

    assert response.status_code == 503
    assert "model-card.json" in response.get_json()["error"]


def test_ui_reuses_cached_model_and_reloads_when_artifact_changes(
    monkeypatch, tmp_path
):
    import importlib

    ui_module = importlib.import_module("ui.app")
    model_path = tmp_path / "model.joblib"
    model_path.write_bytes(b"first model")
    loaded_paths = []

    def load_model(path):
        loaded_paths.append(path)
        return object()

    monkeypatch.setattr(ui_module, "MODEL_PATH", model_path)
    monkeypatch.setattr(ui_module.joblib, "load", load_model)
    ui_module._load_model_from_path.cache_clear()
    try:
        first = ui_module._load_model()
        assert ui_module._load_model() is first
        assert len(loaded_paths) == 1

        model_path.write_bytes(b"updated model artifact")
        second = ui_module._load_model()
        assert second is not first
        assert len(loaded_paths) == 2
    finally:
        ui_module._load_model_from_path.cache_clear()
