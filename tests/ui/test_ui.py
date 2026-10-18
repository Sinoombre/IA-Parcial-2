import json
import hashlib
from importlib import import_module

import numpy as np

from ui.app import DEFAULTS, app
from churnshield.features import ALLOWED_PREDICTORS
from churnshield.settings import AGENT_CONFIG_PATH, THRESHOLDS_CONFIG_PATH


def _valid_model_card(mean_cltv=840.0):
    def sha256(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    return {
        "mean_cltv_train": mean_cltv,
        "features": list(ALLOWED_PREDICTORS),
        "reserved_evaluation_executed": False,
        "config_sha256": sha256(AGENT_CONFIG_PATH),
        "thresholds_sha256": sha256(THRESHOLDS_CONFIG_PATH),
    }


def test_seeded_profiles_are_reproducible_and_use_only_development_data(monkeypatch):
    import pandas as pd

    from churnshield import data

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
    monkeypatch.setattr(
        data,
        "_read_data_rows",
        lambda indices, *, split: (_ for _ in ()).throw(
            AssertionError(f"Unexpected data split access: {split}")
        ),
    )
    monkeypatch.setattr(
        data,
        "load_reserved",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("UI accessed reserved rows")
        ),
    )
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
    monkeypatch.setattr(
        "ui.app._model_card",
        _valid_model_card,
    )
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


def test_ui_predicts_without_reservation_access(monkeypatch):
    import numpy as np

    class FakeModel:
        def predict_proba(self, features):
            return np.array([[0.8, 0.2]])

    app.config["TESTING"] = True
    client = app.test_client()
    module = import_module("ui.app")
    monkeypatch.setattr(module, "_load_model", lambda: FakeModel())
    monkeypatch.setattr(
        module,
        "_model_card",
        _valid_model_card,
    )
    response = client.post("/api/predict", json=DEFAULTS)
    payload = response.get_json()
    assert response.status_code == 200
    assert 0.0 <= payload["p_churn"] <= 1.0
    assert payload["reserved_evaluation_executed"] is False
    assert "causalidad" in payload["disclaimer"]


def test_ui_rejects_non_json_and_empty_json_payloads():
    client = app.test_client()

    responses = [
        client.post("/api/predict", data="plain text", content_type="text/plain"),
        client.post("/api/predict", data='{"invalid"', content_type="application/json"),
        client.post("/api/predict", json={}),
    ]

    assert [response.status_code for response in responses] == [400, 400, 400]


def test_ui_defaults_match_the_canonical_feature_contract():
    assert set(DEFAULTS) == set(ALLOWED_PREDICTORS)


def test_ui_returns_503_if_model_or_card_is_missing(monkeypatch, tmp_path):
    module = import_module("ui.app")
    client = app.test_client()
    monkeypatch.setattr(module, "MODEL_PATH", tmp_path / "missing.joblib")
    monkeypatch.setattr(module, "CARD_PATH", tmp_path / "model_card.json")

    missing_model = client.post("/api/predict", json=DEFAULTS)
    assert missing_model.status_code == 503

    (tmp_path / "model.joblib").write_bytes(b"fake")
    monkeypatch.setattr(module, "MODEL_PATH", tmp_path / "model.joblib")
    monkeypatch.setattr(module.joblib, "load", lambda _: object())
    missing_card = client.post("/api/predict", json=DEFAULTS)
    assert missing_card.status_code == 503


def test_ui_reloads_card_contract_and_rejects_mismatched_features(
    monkeypatch, tmp_path
):
    from churnshield.features import ALLOWED_PREDICTORS

    module = import_module("ui.app")
    model_path = tmp_path / "model.joblib"
    card_path = tmp_path / "model_card.json"
    model_path.write_bytes(b"fake")
    card_path.write_text(
        json.dumps(
            {
                "mean_cltv_train": 840.0,
                "features": [*ALLOWED_PREDICTORS, "Phone Service"],
                "reserved_evaluation_executed": False,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(module, "MODEL_PATH", model_path)
    monkeypatch.setattr(module, "CARD_PATH", card_path)
    monkeypatch.setattr(module.joblib, "load", lambda _: object())

    response = app.test_client().post("/api/predict", json=DEFAULTS)

    assert response.status_code == 503


def test_ui_loads_updated_model_when_artifact_hash_changes(monkeypatch, tmp_path):
    module = import_module("ui.app")
    model_path = tmp_path / "model.joblib"
    card_path = tmp_path / "model_card.json"
    model_path.write_bytes(b"model-v1")

    def write_card():
        card = _valid_model_card()
        card["model_sha256"] = hashlib.sha256(model_path.read_bytes()).hexdigest()
        card_path.write_text(json.dumps(card), encoding="utf-8")

    class FakeModel:
        def __init__(self, probability):
            self.probability = probability

        def predict_proba(self, features):
            return np.array([[1.0 - self.probability, self.probability]])

    monkeypatch.setattr(module, "MODEL_PATH", model_path)
    monkeypatch.setattr(module, "CARD_PATH", card_path)
    monkeypatch.setattr(
        module.joblib,
        "load",
        lambda path: FakeModel(0.2 if path.read_bytes() == b"model-v1" else 0.8),
    )
    write_card()
    client = app.test_client()

    first_response = client.post("/api/predict", json=DEFAULTS)
    first = first_response.get_json()
    model_path.write_bytes(b"model-v2")
    write_card()
    second_response = client.post("/api/predict", json=DEFAULTS)
    second = second_response.get_json()

    assert first_response.status_code == second_response.status_code == 200
    assert first["p_churn"] == 0.2
    assert second["p_churn"] == 0.8
